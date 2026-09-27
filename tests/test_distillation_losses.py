"""偏好优化和蒸馏损失的数值、梯度、掩码及低精度测试。"""

import math
import unittest

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence

from llm.dpo_loss import dpo_loss
from llm.kl_loss import kl_loss


class DistillationLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_dpo_matches_binary_cross_entropy_and_freezes_reference(self):
        chosen = torch.tensor([-1., -1000., -3.], dtype=torch.float64, requires_grad=True)
        rejected = torch.tensor([-1000., -1., -2.], dtype=torch.float64, requires_grad=True)
        ref_chosen = torch.tensor([-2., -2., -3.], dtype=torch.float64, requires_grad=True)
        ref_rejected = torch.tensor([-2., -2., -2.], dtype=torch.float64, requires_grad=True)
        loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.3)
        margin = 0.3 * (chosen - rejected - ref_chosen.detach() + ref_rejected.detach())
        expected = F.binary_cross_entropy_with_logits(margin, torch.ones_like(margin))
        torch.testing.assert_close(loss, expected)

        grads = torch.autograd.grad(loss, (chosen, rejected, ref_chosen, ref_rejected), allow_unused=True)
        expected_grads = torch.autograd.grad(expected, (chosen, rejected))
        for actual, reference in zip(grads[:2], expected_grads):
            torch.testing.assert_close(actual, reference)
            self.assertTrue(torch.isfinite(actual).all())
        self.assertIsNone(grads[2])
        self.assertIsNone(grads[3])
        self.assertTrue((grads[0] <= 0).all())
        self.assertTrue((grads[1] >= 0).all())
        torch.testing.assert_close(
            dpo_loss(chosen, rejected, chosen.detach(), rejected.detach()),
            torch.tensor(math.log(2), dtype=torch.float64),
        )

    def test_dpo_token_masks_match_sequence_log_probabilities(self):
        chosen = torch.tensor([[-0.5, -1., float('-inf')], [-2., float('nan'), float('-inf')]], requires_grad=True)
        rejected = torch.tensor([[-1., -2., 0., float('-inf')], [-1., -1., -1., float('nan')]], requires_grad=True)
        ref_chosen = torch.tensor([[-1., -1., float('-inf')], [-2.5, 0., float('nan')]])
        ref_rejected = torch.tensor([[-1.5, -2., 0., float('nan')], [-1., -1., -1.5, float('-inf')]])
        chosen_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        rejected_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        actual = dpo_loss(chosen, rejected, ref_chosen, ref_rejected,
                          chosen_mask=chosen_mask, rejected_mask=rejected_mask)
        expected = dpo_loss(torch.tensor([-1.5, -2.]), torch.tensor([-3., -3.]),
                            torch.tensor([-2., -2.5]), torch.tensor([-3.5, -3.5]))
        torch.testing.assert_close(actual, expected)
        for grad, mask in zip(torch.autograd.grad(actual, (chosen, rejected)), (chosen_mask, rejected_mask)):
            self.assertTrue(torch.isfinite(grad).all())
            torch.testing.assert_close(grad[~mask.bool()], torch.zeros_like(grad[~mask.bool()]))

    def test_dpo_half_precision_long_responses(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                chosen = torch.full((2, 128), -1000., dtype=dtype, requires_grad=True)
                rejected = torch.full((2, 128), -1008., dtype=dtype, requires_grad=True)
                reference = torch.full((2, 128), -1000., dtype=dtype)
                actual = dpo_loss(chosen, rejected, reference, reference)
                expected = dpo_loss(chosen.float(), rejected.float(), reference.float(), reference.float())
                torch.testing.assert_close(actual, expected)
                self.assertTrue(torch.isfinite(actual))
                for grad in torch.autograd.grad(actual, (chosen, rejected)):
                    self.assertTrue(torch.isfinite(grad).all())

    def test_kl_matches_categorical_values_gradients_and_reductions(self):
        p_logits = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        q_logits = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        for reduction in ('none', 'sum', 'mean'):
            with self.subTest(reduction=reduction):
                actual = kl_loss(p_logits.log_softmax(-1), q_logits.log_softmax(-1), mask, reduction)
                expected = kl_divergence(Categorical(logits=p_logits), Categorical(logits=q_logits))
                expected = expected.masked_fill(~mask, 0)
                if reduction == 'sum':
                    expected = expected.sum()
                elif reduction == 'mean':
                    expected = expected.sum() / mask.sum()
                torch.testing.assert_close(actual, expected)
                upstream = torch.randn_like(actual)
                actual_grads = torch.autograd.grad(actual, (p_logits, q_logits), upstream)
                expected_grads = torch.autograd.grad(expected, (p_logits, q_logits), upstream)
                for a, b in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(a, b)

    def test_kl_zero_probabilities_and_masked_infinite_divergence(self):
        log_p = torch.tensor([[float('-inf'), 0.], [float('-inf'), 0.]], requires_grad=True)
        log_q = torch.tensor([[-math.log(2), -math.log(2)], [0., float('-inf')]], requires_grad=True)
        mask = torch.tensor([True, False])
        loss = kl_loss(log_p, log_q, mask)
        torch.testing.assert_close(loss, torch.tensor(math.log(2)))
        for grad in torch.autograd.grad(loss, (log_p, log_q)):
            self.assertTrue(torch.isfinite(grad).all())
            torch.testing.assert_close(grad[1], torch.zeros_like(grad[1]))
        self.assertTrue(torch.isposinf(kl_loss(log_p, log_q)))
        torch.testing.assert_close(kl_loss(log_p, log_p), torch.tensor(0.))
        torch.testing.assert_close(kl_loss(log_p, log_q, torch.zeros_like(mask)), torch.tensor(0.))

    def test_invalid_shapes_and_parameters(self):
        log_probs = torch.zeros(2)
        logits = torch.zeros(2, 3, 4)
        invalid_calls = (
            lambda: dpo_loss(log_probs, log_probs, log_probs, log_probs, beta=0),
            lambda: dpo_loss(log_probs, log_probs, log_probs[:1], log_probs),
            lambda: dpo_loss(log_probs, log_probs, log_probs, log_probs, chosen_mask=torch.ones(2)),
            lambda: kl_loss(logits, logits, reduction='invalid'),
            lambda: kl_loss(logits, logits[:, :1]),
            lambda: kl_loss(logits, logits, token_mask=torch.ones(2)),
        )
        for index, call in enumerate(invalid_calls):
            with self.subTest(index=index), self.assertRaises(ValueError):
                call()


if __name__ == '__main__':
    unittest.main()
