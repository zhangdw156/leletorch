"""Softmax、缓存因果遮罩、RoPE 默认位置和 RMSNorm 半精度的回归测试。"""

import unittest

import torch
import torch.nn.functional as F

from llm.mha import MHA
from llm.rmsnorm import RMSNorm
from llm.rope import apply_rope
from llm.softmax import softmax


class LLMRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_softmax_zero_logits(self):
        torch.testing.assert_close(softmax(torch.zeros(4)), torch.full((4,), 0.25))

    def test_softmax_matches_torch_values_and_gradients(self):
        # 大的公共偏移验证数值稳定性，适中的相对差值保留非零梯度。
        x = (torch.randn(2, 3, 5, dtype=torch.float64) * 3 + 1000).requires_grad_()
        for dim in (0, 1, -1):
            with self.subTest(dim=dim):
                actual, expected = softmax(x, dim), torch.softmax(x, dim=dim)
                torch.testing.assert_close(actual, expected)
                upstream = torch.randn_like(actual)
                actual_grad = torch.autograd.grad(actual, x, upstream)[0]
                expected_grad = torch.autograd.grad(expected, x, upstream)[0]
                torch.testing.assert_close(actual_grad, expected_grad)

    def test_softmax_half_precision_large_vocabulary(self):
        # 70000 个 exp(0) 的和超过 float16 的最大有限值。
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.zeros(2, 70000, dtype=dtype)
                actual = softmax(x)
                expected = torch.softmax(x.float(), dim=-1).to(dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(actual).all())

    def test_mha_cached_chunks_match_full_attention(self):
        model = MHA(12, 3).double()
        x = torch.randn(2, 8, 12, dtype=torch.float64)
        reference = torch.nn.MultiheadAttention(12, 3, batch_first=True).double()
        with torch.no_grad():
            reference.in_proj_weight.copy_(torch.cat([
                model.q_proj.weight, model.k_proj.weight, model.v_proj.weight,
            ]))
            reference.in_proj_bias.copy_(torch.cat([
                model.q_proj.bias, model.k_proj.bias, model.v_proj.bias,
            ]))
            reference.out_proj.load_state_dict(model.o_proj.state_dict())

        mask = torch.ones(8, 8, dtype=torch.bool).triu(diagonal=1)
        expected, _ = reference(x, x, x, attn_mask=mask, need_weights=False)
        full, full_k, full_v = model(x, causal=True)
        torch.testing.assert_close(full, expected)

        # 同时覆盖整段、逐 token、多 token 分块续算。
        for lengths in ((8,), (1,) * 8, (3, 3, 2)):
            with self.subTest(lengths=lengths):
                outputs = []
                past_k = past_v = None
                start = 0
                for length in lengths:
                    out, past_k, past_v = model(
                        x[:, start:start + length], causal=True,
                        past_k=past_k, past_v=past_v,
                    )
                    outputs.append(out)
                    start += length
                torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
                torch.testing.assert_close(past_k, full_k)
                torch.testing.assert_close(past_v, full_v)

    def test_mha_cached_query_cannot_see_future_tokens(self):
        model = MHA(12, 3).double()
        prefix = torch.randn(2, 3, 12, dtype=torch.float64)
        _, past_k, past_v = model(prefix, causal=True)
        chunk = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
        out, _, _ = model(chunk, causal=True, past_k=past_k, past_v=past_v)

        # 修改当前块的后续 token，不得改变首个新 token 的输出。
        changed = chunk.detach().clone()
        changed[:, 1:] += 10
        changed_out, _, _ = model(changed, causal=True, past_k=past_k, past_v=past_v)
        torch.testing.assert_close(out[:, 0], changed_out[:, 0])
        grad = torch.autograd.grad(out[:, 0].sum(), chunk)[0]
        torch.testing.assert_close(grad[:, 1:], torch.zeros_like(grad[:, 1:]))

    def test_rope_default_positions_match_explicit_positions(self):
        q = torch.randn(2, 3, 5, 8, dtype=torch.float64, requires_grad=True)
        k = torch.randn(2, 1, 5, 8, dtype=torch.float64, requires_grad=True)
        positions = torch.arange(5).unsqueeze(0).expand(2, -1)
        for rotary_dim in (4, 8):
            with self.subTest(rotary_dim=rotary_dim):
                actual = apply_rope(q, k, rotary_dim=rotary_dim, scaling_factor=2)
                expected = apply_rope(
                    q, k, position_ids=positions, rotary_dim=rotary_dim, scaling_factor=2,
                )
                for output, reference, original in zip(actual, expected, (q, k)):
                    torch.testing.assert_close(output, reference)
                    torch.testing.assert_close(output[:, :, 0], original[:, :, 0])
                    torch.testing.assert_close(output[..., rotary_dim:], original[..., rotary_dim:])
                actual_grad = torch.autograd.grad(sum(t.sum() for t in actual), (q, k))
                expected_grad = torch.autograd.grad(sum(t.sum() for t in expected), (q, k))
                for a, b in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(a, b)

    def test_rmsnorm_half_precision_large_inputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            for weight_dtype in (torch.float32, dtype):
                with self.subTest(dtype=dtype, weight_dtype=weight_dtype):
                    model = RMSNorm(4).to(weight_dtype)
                    x = torch.full((2, 4), 1000.0, dtype=dtype)
                    result = model(x)
                    self.assertEqual(result.dtype, x.dtype)
                    self.assertTrue(torch.isfinite(result).all())
                    torch.testing.assert_close(result, torch.ones_like(x))

    def test_rmsnorm_matches_float64_reference_and_gradients(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model = RMSNorm(4).to(dtype)
                with torch.no_grad():
                    model.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
                x = torch.tensor([[1000., -2000., 3000., -4000.]], dtype=dtype, requires_grad=True)
                reference_x = x.detach().double().requires_grad_()
                reference_weight = model.weight.detach().double().requires_grad_()
                actual = model(x)
                expected = F.rms_norm(reference_x, (4,), reference_weight, eps=model.eps)
                torch.testing.assert_close(actual, expected.to(dtype))

                actual_grads = torch.autograd.grad(actual.sum(), (x, model.weight))
                expected_grads = torch.autograd.grad(expected.sum(), (reference_x, reference_weight))
                for a, b in zip(actual_grads, expected_grads):
                    self.assertTrue(torch.isfinite(a).all())
                    torch.testing.assert_close(a, b.to(dtype))


if __name__ == "__main__":
    unittest.main()
