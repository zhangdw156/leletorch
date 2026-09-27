"""OPD 前向分布损失与采样反向 KL 策略梯度的数值、梯度回归测试。"""

import unittest

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence

from llm.opd_fwd_kl import opd_fwd_kl
from llm.opd_rev_kl import opd_rev_kl


class OPDLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_forward_has_no_temperature_squared_multiplier(self):
        mask = torch.tensor([[True, True, False], [True, False, False]])
        for temperature in (0.5, 1.0, 2.0):
            with self.subTest(temperature=temperature):
                student = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
                teacher = torch.randn_like(student, requires_grad=True)
                actual = opd_fwd_kl(student, teacher, mask, temperature)
                expected = kl_divergence(
                    Categorical(logits=teacher.detach() / temperature),
                    Categorical(logits=student / temperature),
                )[mask].mean()
                torch.testing.assert_close(actual, expected)
                student_grad, teacher_grad = torch.autograd.grad(actual, (student, teacher), allow_unused=True)
                torch.testing.assert_close(student_grad, torch.autograd.grad(expected, student)[0])
                self.assertIsNone(teacher_grad)

    def test_forward_topk_matches_dense_reference_and_preserves_support_gradients(self):
        for topk in (2, 5):
            for normalize in (False, True):
                with self.subTest(topk=topk, normalize=normalize):
                    student = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
                    teacher = torch.randn_like(student, requires_grad=True)
                    mask = torch.tensor([[True, True, False], [True, False, False]])
                    ids = teacher.detach().topk(topk, dim=-1).indices
                    support = F.one_hot(ids, num_classes=5).sum(-2).bool()
                    if normalize:
                        teacher_dist = Categorical(logits=teacher.detach().masked_fill(~support, -torch.inf))
                        student_dist = Categorical(logits=student.masked_fill(~support, -torch.inf))
                        expected = kl_divergence(teacher_dist, student_dist)[mask].mean()
                    else:
                        dense_terms = F.kl_div(
                            student.log_softmax(-1), teacher.detach().softmax(-1), reduction='none',
                        )
                        expected = dense_terms.masked_fill(~support, 0).sum(-1)[mask].mean()
                    actual = opd_fwd_kl(student, teacher, mask, topk=topk, normalize_topk=normalize)
                    torch.testing.assert_close(actual, expected)
                    gradient, teacher_grad = torch.autograd.grad(actual, (student, teacher), allow_unused=True)
                    torch.testing.assert_close(gradient, torch.autograd.grad(expected, student)[0])
                    self.assertIsNone(teacher_grad)
                    outside = ~support & mask.unsqueeze(-1)
                    if topk < 5:
                        if normalize:
                            torch.testing.assert_close(gradient[outside], torch.zeros_like(gradient[outside]))
                        else:
                            self.assertTrue((gradient[outside] > 0).all())
                    else:
                        torch.testing.assert_close(actual, opd_fwd_kl(student, teacher, mask))

    def test_original_topk_sum_can_be_negative(self):
        student = torch.tensor([[0.8, 0.2]], dtype=torch.float64).log().requires_grad_()
        teacher = torch.tensor([[0.6, 0.4]], dtype=torch.float64).log()
        loss = opd_fwd_kl(student, teacher, topk=1)
        self.assertLess(loss.item(), 0)
        gradient = torch.autograd.grad(loss, student)[0]
        self.assertNotEqual(gradient[0, 1].item(), 0)
        normalized = opd_fwd_kl(student, teacher, topk=1, normalize_topk=True)
        torch.testing.assert_close(normalized, torch.zeros_like(normalized))

    def test_forward_equal_distributions_and_masked_padding(self):
        student = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
        loss = opd_fwd_kl(student, student.detach())
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        torch.testing.assert_close(torch.autograd.grad(loss, student)[0], torch.zeros_like(student))
        mask = torch.tensor([[True, False, False], [True, False, False]])
        for kwargs in ({}, {'topk': 2}, {'topk': 2, 'normalize_topk': True}):
            with self.subTest(kwargs=kwargs):
                padded_student = student.detach().masked_fill(~mask.unsqueeze(-1), -torch.inf).requires_grad_()
                teacher = torch.randn_like(student).masked_fill(~mask.unsqueeze(-1), torch.nan)
                loss = opd_fwd_kl(padded_student, teacher, mask, **kwargs)
                expected = opd_fwd_kl(padded_student[mask], teacher[mask], **kwargs)
                torch.testing.assert_close(loss, expected)
                gradient = torch.autograd.grad(loss, padded_student)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                torch.testing.assert_close(gradient[~mask], torch.zeros_like(gradient[~mask]))
                empty_student = torch.full((2, 3, 5), -torch.inf, requires_grad=True)
                loss = opd_fwd_kl(empty_student, torch.full_like(empty_student, torch.nan),
                                  torch.zeros_like(mask), **kwargs)
                self.assertEqual(loss.item(), 0)
                torch.testing.assert_close(torch.autograd.grad(loss, empty_student)[0], torch.zeros_like(empty_student))

    def test_forward_half_precision_large_logits(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                student = torch.tensor([[1000., -1000., 0.], [-1000., 1000., 0.]], dtype=dtype, requires_grad=True)
                teacher = -student.detach()
                loss = opd_fwd_kl(student, teacher, temperature=0.01)
                expected = F.kl_div(
                    (student.float() / 0.01).log_softmax(-1),
                    (teacher.float() / 0.01).log_softmax(-1), log_target=True, reduction='batchmean',
                )
                torch.testing.assert_close(loss, expected)
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(torch.autograd.grad(loss, student)[0]).all())

    def test_sampled_reverse_expected_gradient_matches_dense_kl(self):
        # 枚举全部采样 action，按固定 rollout 概率加权，验证无采样噪声的期望梯度。
        for clip_range in (None, 0.2):
            with self.subTest(clip_range=clip_range):
                student = torch.randn(1, 5, dtype=torch.float64, requires_grad=True)
                rollout = student.detach().clone().requires_grad_()
                teacher = torch.randn_like(student, requires_grad=True)
                log_student, log_rollout, log_teacher = [x.log_softmax(-1) for x in (student, rollout, teacher)]
                losses = torch.stack([
                    opd_rev_kl(log_student[:, i:i+1], log_rollout[:, i:i+1], log_teacher[:, i:i+1],
                               clip_range=clip_range)
                    for i in range(5)
                ])
                expected_sample_loss = (rollout.detach().softmax(-1)[0] * losses).sum()
                dense_kl = kl_divergence(Categorical(logits=student), Categorical(logits=teacher.detach())).mean()
                gradients = torch.autograd.grad(expected_sample_loss, (student, rollout, teacher), allow_unused=True)
                torch.testing.assert_close(gradients[0], torch.autograd.grad(dense_kl, student)[0])
                self.assertIsNone(gradients[1])
                self.assertIsNone(gradients[2])

    def test_sampled_reverse_ppo_clipping_and_frozen_advantage(self):
        # 分别覆盖正/负 advantage 的裁剪区和未裁剪区；当前策略已偏离 rollout。
        ratios = torch.tensor([[1.5, 0.5, 1.5, 0.5]], dtype=torch.float64)
        student = (-3 + ratios.log()).requires_grad_()
        rollout = torch.full_like(student, -3, requires_grad=True)
        teacher = torch.tensor([[-2.5, -2.5, -3.5, -3.5]], dtype=torch.float64, requires_grad=True)
        loss = opd_rev_kl(student, rollout, teacher)
        torch.testing.assert_close(loss, torch.tensor((-0.6 - 0.25 + 0.75 + 0.4) / 4, dtype=loss.dtype))
        gradients = torch.autograd.grad(loss, (student, rollout, teacher), allow_unused=True)
        torch.testing.assert_close(gradients[0], torch.tensor([[0., -0.25, 0.75, 0.]], dtype=loss.dtype) / 4)
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])
        unclipped = opd_rev_kl(student, rollout, teacher, clip_range=None)
        expected_gradient = torch.tensor([[-0.75, -0.25, 0.75, 0.25]], dtype=loss.dtype) / 4
        torch.testing.assert_close(torch.autograd.grad(unclipped, student)[0], expected_gradient)

    def test_sampled_reverse_equal_models_and_masks(self):
        student = torch.tensor([[-2., -3., -4.]], requires_grad=True)
        loss = opd_rev_kl(student, student.detach(), student.detach())
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(torch.autograd.grad(loss, student)[0], torch.zeros_like(student))
        mask = torch.tensor([[True, False, False], [True, True, False]])
        student = torch.full((2, 3), -2.).masked_fill(~mask, -torch.inf).requires_grad_()
        rollout = torch.full_like(student, -2.1).masked_fill(~mask, torch.nan).requires_grad_()
        teacher = torch.full_like(student, -1.5).masked_fill(~mask, -torch.inf).requires_grad_()
        loss = opd_rev_kl(student, rollout, teacher, mask)
        self.assertLess(loss.item(), 0)  # PG 代理损失可以为负，不能强制当作非负 KL。
        expected = opd_rev_kl(student[mask][None], rollout[mask][None], teacher[mask][None])
        torch.testing.assert_close(loss, expected)
        gradients = torch.autograd.grad(loss, (student, rollout, teacher), allow_unused=True)
        self.assertTrue(torch.isfinite(gradients[0]).all())
        torch.testing.assert_close(gradients[0][~mask], torch.zeros_like(gradients[0][~mask]))
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])
        empty_student = torch.full((2, 3), torch.nan, requires_grad=True)
        loss = opd_rev_kl(empty_student, torch.full_like(empty_student, -torch.inf),
                          torch.full_like(empty_student, torch.nan), torch.zeros_like(mask))
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(torch.autograd.grad(loss, empty_student)[0], torch.zeros_like(empty_student))

    def test_sampled_reverse_clips_extreme_ratios_before_exponentiation(self):
        # 有利方向的极端概率比应被裁剪；零 advantage 不应因 exp(999)*0 产生 NaN。
        student = torch.tensor([[-1., -1000., -1.]], requires_grad=True)
        rollout = torch.tensor([[-1000., -1., -1000.]])
        teacher = torch.tensor([[-999., -2., -1000.]])
        loss = opd_rev_kl(student, rollout, teacher)
        torch.testing.assert_close(loss, torch.tensor((-1.2 + 0.8) / 3))
        torch.testing.assert_close(torch.autograd.grad(loss, student)[0], torch.zeros_like(student))
        zero_loss = opd_rev_kl(student[:, 2:], rollout[:, 2:], teacher[:, 2:], clip_range=None)
        self.assertEqual(zero_loss.item(), 0)
        torch.testing.assert_close(torch.autograd.grad(zero_loss, student)[0], torch.zeros_like(student))

    def test_sampled_reverse_half_precision(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                student = torch.tensor([[-256., -512.]], dtype=dtype, requires_grad=True)
                rollout = student.detach().clone()
                teacher = torch.tensor([[-252., -516.]], dtype=dtype)
                loss = opd_rev_kl(student, rollout, teacher)
                reference = student.detach().double().requires_grad_()
                expected = opd_rev_kl(reference, rollout.double(), teacher.double())
                self.assertEqual(loss.dtype, torch.float32)
                torch.testing.assert_close(loss, expected.float())
                gradient = torch.autograd.grad(loss, student)[0]
                torch.testing.assert_close(gradient, torch.autograd.grad(expected, reference)[0].to(dtype))
                self.assertTrue(torch.isfinite(gradient).all())

    def test_invalid_opd_inputs(self):
        logits = torch.zeros(2, 3, 4)
        log_probs = torch.zeros(2, 3)
        invalid_calls = [
            lambda: opd_fwd_kl(logits, logits, normalize_topk=True),
            lambda: opd_rev_kl(logits, logits, logits),  # 旧的全词表接口必须明确拒绝。
            lambda: opd_rev_kl(log_probs, log_probs[:, :1], log_probs),
            lambda: opd_rev_kl(log_probs, log_probs, log_probs, token_mask=torch.ones(2, 1)),
            lambda: opd_rev_kl(log_probs, log_probs, torch.full_like(log_probs, -torch.inf)),
        ]
        for temperature in (0, float('inf'), float('nan')):
            invalid_calls.append(lambda t=temperature: opd_fwd_kl(logits, logits, temperature=t))
        for topk in (0, 5, 1.5, True):
            invalid_calls.append(lambda k=topk: opd_fwd_kl(logits, logits, topk=k))
        for clip_range in (-0.1, 1.0, float('inf'), float('nan')):
            invalid_calls.append(lambda c=clip_range: opd_rev_kl(log_probs, log_probs, log_probs, clip_range=c))
        for index, call in enumerate(invalid_calls):
            with self.subTest(index=index), self.assertRaises(ValueError):
                call()


if __name__ == '__main__':
    unittest.main()
