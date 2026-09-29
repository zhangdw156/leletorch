"""采样 KL 的解析值、梯度、分布期望及半精度裁剪回归测试。"""

import math
import unittest

import torch
from torch.distributions import Categorical, kl_divergence

from llm.grpo_kl_loss import grpo_kl_loss


class GRPOKLLossTests(unittest.TestCase):
    def test_estimators_match_analytic_values_and_gradients(self):
        # Q/P=[1/2, 1, 2]，均不触发 k3 裁剪；直接使用解析值和导数作参考。
        log_two = math.log(2)
        expected_values = {
            "k1": [log_two, 0.0, -log_two],
            "k2": [0.5 * log_two**2, 0.0, 0.5 * log_two**2],
            "k3": [log_two - 0.5, 0.0, 1.0 - log_two],
        }
        expected_gradients = {
            "k1": [1.0, 1.0, 1.0],
            "k2": [log_two, 0.0, -log_two],
            "k3": [0.5, 0.0, -1.0],
        }
        for mode, alias in (("k1", "kl"), ("k2", "mse"), ("k3", "low_var_kl")):
            for name in (mode, alias):
                with self.subTest(mode=name):
                    current = torch.full((1, 3), -2.0, dtype=torch.float64, requires_grad=True)
                    reference = torch.tensor(
                        [[-2 - log_two, -2.0, -2 + log_two]],
                        dtype=torch.float64, requires_grad=True,
                    )
                    terms = grpo_kl_loss(current, reference, name)
                    expected = torch.tensor([expected_values[mode]], dtype=torch.float64)
                    torch.testing.assert_close(terms, expected)
                    grad, ref_grad = torch.autograd.grad(terms.sum(), (current, reference))
                    expected_grad = torch.tensor([expected_gradients[mode]], dtype=torch.float64)
                    torch.testing.assert_close(grad, expected_grad)
                    # 与 verl 底层函数一致，不在内部偷偷改变参考输入的梯度行为。
                    torch.testing.assert_close(ref_grad, -expected_grad)

    def test_k1_and_k3_expectations_equal_full_kl(self):
        # 枚举所有 action 并按 P 加权，消除随机采样噪声，验证 KL 的方向。
        p = Categorical(probs=torch.tensor([0.8, 0.2], dtype=torch.float64))
        q = Categorical(probs=torch.tensor([0.5, 0.5], dtype=torch.float64))
        actions = torch.arange(2)
        expected = kl_divergence(p, q)
        for mode in ("k1", "k3"):
            with self.subTest(mode=mode):
                terms = grpo_kl_loss(p.log_prob(actions), q.log_prob(actions), mode)
                torch.testing.assert_close((p.probs * terms).sum(), expected)

    def test_masked_aggregation_preserves_group_shape_and_only_updates_valid_tokens(self):
        current = torch.tensor(
            [[[-1.0, -2.0, -0.5], [-0.8, -0.3, -1.2]]],
            dtype=torch.float64, requires_grad=True,
        )
        reference = torch.full_like(current, -1.0)
        mask = torch.tensor([[[True, True, False], [True, False, False]]])
        terms = grpo_kl_loss(current, reference)
        self.assertEqual(terms.shape, (1, 2, 3))
        torch.testing.assert_close(terms, grpo_kl_loss(current, reference, "k3"))
        loss = (terms * mask).sum() / mask.sum()
        expected = grpo_kl_loss(current[mask], reference[mask], "k3").mean()
        torch.testing.assert_close(loss, expected)
        gradient = torch.autograd.grad(loss, current)[0]
        torch.testing.assert_close(gradient[~mask], torch.zeros_like(gradient[~mask]))
        # 全遮罩时仍然可以反向传播，且梯度为零。
        empty_loss = (grpo_kl_loss(current, reference) * torch.zeros_like(mask)).sum()
        self.assertEqual(empty_loss.item(), 0.0)
        torch.testing.assert_close(torch.autograd.grad(empty_loss, current)[0], torch.zeros_like(current))

    def test_k3_extreme_ratios_are_clipped_with_finite_gradients(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                # 含负/正超大 log-ratio，以及仅触发最终 kld 裁剪的 -12 和 +3。
                current = torch.tensor([[-1, -1, -1, -100, -1000, -3]], dtype=dtype, requires_grad=True)
                reference = torch.tensor([[-1000, -100, -13, -97, -1, -3]], dtype=dtype)
                terms = grpo_kl_loss(current, reference)
                expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
                self.assertEqual(terms.dtype, expected_dtype)
                expected = torch.tensor([[10, 10, 10, 10, 10, 0]], dtype=expected_dtype)
                torch.testing.assert_close(terms, expected)
                gradient = torch.autograd.grad(terms.sum(), current)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                torch.testing.assert_close(gradient, torch.zeros_like(current))

    def test_half_precision_k1_and_k2_compute_difference_and_square_in_float32(self):
        for dtype in (torch.float16, torch.bfloat16):
            for mode, expected_value, expected_grad in (("k1", -1000.0, 1.0), ("k2", 500000.0, -1000.0)):
                with self.subTest(dtype=dtype, mode=mode):
                    current = torch.tensor([-1000.0], dtype=dtype, requires_grad=True)
                    reference = torch.tensor([0.0], dtype=dtype)
                    actual = grpo_kl_loss(current, reference, mode)
                    torch.testing.assert_close(actual, torch.tensor([expected_value]))
                    gradient = torch.autograd.grad(actual.sum(), current)[0]
                    torch.testing.assert_close(gradient, torch.tensor([expected_grad], dtype=dtype))

    def test_invalid_inputs_and_unsupported_estimators(self):
        logprob = torch.zeros(2, 3)
        for mode in ("full", "unknown", "k3+"):
            with self.subTest(mode=mode), self.assertRaises(NotImplementedError):
                grpo_kl_loss(logprob, logprob, mode)
        for reference in (torch.zeros(2, 1), logprob.long(), torch.zeros(2, 3, device="meta")):
            with self.subTest(shape=reference.shape, dtype=reference.dtype), self.assertRaises(ValueError):
                grpo_kl_loss(logprob, reference)
        with self.assertRaises(ValueError):
            grpo_kl_loss(logprob.long(), logprob)


if __name__ == "__main__":
    unittest.main()
