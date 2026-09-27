"""GQA 与 PyTorch SDPA 的数值、梯度、缓存和半精度回归测试。"""

import unittest

import torch
import torch.nn.functional as F

from llm.gqa import GQA
from llm.mha import MHA
from llm.rope import apply_rope


def sdpa_reference(model, x, causal=False, position_ids=None, query_start=0):
    """显式复制 KV head 后调用标准 SDPA，不复用 GQA 的分组广播计算。"""
    B, L, D = x.shape
    q = model.q_proj(x).view(B, L, model.num_heads, model.head_dim).transpose(1, 2)
    k = model.k_proj(x).view(B, L, model.num_kv_heads, model.head_dim).transpose(1, 2)
    v = model.v_proj(x).view(B, L, model.num_kv_heads, model.head_dim).transpose(1, 2)
    if model.use_rope:
        q, k = apply_rope(q, k, position_ids=position_ids)
    k = k.repeat_interleave(model.num_kv_groups, dim=1)
    v = v.repeat_interleave(model.num_kv_groups, dim=1)
    mask = None
    if causal:
        mask = torch.arange(L, device=x.device)[None, :] <= torch.arange(
            query_start, L, device=x.device,
        )[:, None]
    out = F.scaled_dot_product_attention(q[:, :, query_start:], k, v, attn_mask=mask)
    out = out.transpose(1, 2).contiguous().view(B, L - query_start, D)
    return model.o_proj(out)


class GQATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)

    def test_matches_sdpa_outputs_and_gradients(self):
        for num_kv_heads in (1, 2, 4):
            for causal in (False, True):
                for use_rope in (False, True):
                    with self.subTest(kv_heads=num_kv_heads, causal=causal, rope=use_rope):
                        model = GQA(16, 4, num_kv_heads, use_rope=use_rope).double()
                        x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
                        actual, k, v = model(x, causal=causal)
                        expected = sdpa_reference(model, x, causal=causal)
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(k.shape, (2, num_kv_heads, 5, 4))
                        self.assertEqual(v.shape, k.shape)
                        upstream = torch.randn_like(actual)
                        parameters = (x, *model.parameters())
                        actual_grads = torch.autograd.grad(actual, parameters, upstream)
                        expected_grads = torch.autograd.grad(expected, parameters, upstream)
                        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                            torch.testing.assert_close(actual_grad, expected_grad)

    def test_equal_query_and_kv_heads_match_existing_mha(self):
        for use_rope in (False, True):
            with self.subTest(rope=use_rope):
                model = GQA(16, 4, 4, use_rope=use_rope).double()
                reference = MHA(16, 4, use_rope=use_rope).double()
                reference.load_state_dict(model.state_dict())
                x = torch.randn(2, 5, 16, dtype=torch.float64)
                for causal in (False, True):
                    for actual, expected in zip(model(x, causal), reference(x, causal)):
                        torch.testing.assert_close(actual, expected)

    def test_cached_chunks_match_full_attention_and_keep_compact_cache(self):
        for num_kv_heads in (1, 2, 4):
            for use_rope in (False, True):
                model = GQA(16, 4, num_kv_heads, use_rope=use_rope).double()
                x = torch.randn(2, 7, 16, dtype=torch.float64)
                full, full_k, full_v = model(x, causal=True)
                for lengths in ((7,), (1,) * 7, (2, 3, 2)):
                    with self.subTest(kv_heads=num_kv_heads, rope=use_rope, lengths=lengths):
                        outputs = []
                        k = v = None
                        start = 0
                        for length in lengths:
                            out, k, v = model(
                                x[:, start:start + length], causal=True, past_k=k, past_v=v,
                            )
                            outputs.append(out)
                            start += length
                            self.assertEqual(k.shape, (2, num_kv_heads, start, 4))
                            self.assertEqual(v.shape, k.shape)
                            self.assertEqual(k.numel() * model.num_kv_groups, 2 * 4 * start * 4)
                        torch.testing.assert_close(torch.cat(outputs, dim=1), full)
                        torch.testing.assert_close(k, full_k)
                        torch.testing.assert_close(v, full_v)

    def test_cached_chunks_preserve_gradients_to_prefix_and_parameters(self):
        model = GQA(16, 4, 2, use_rope=True).double()
        x = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
        full, _, _ = model(x, causal=True)
        _, k, v = model(x[:, :2], causal=True)
        suffix, _, _ = model(x[:, 2:], causal=True, past_k=k, past_v=v)
        upstream = torch.randn_like(suffix)
        parameters = (x, *model.parameters())
        actual_grads = torch.autograd.grad(suffix, parameters, upstream)
        expected_grads = torch.autograd.grad(full[:, 2:], parameters, upstream)
        self.assertGreater(actual_grads[0][:, :2].abs().sum().item(), 0)
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected)

    def test_cached_query_cannot_see_future_tokens(self):
        model = GQA(16, 4, 2, use_rope=True).double()
        _, k, v = model(torch.randn(2, 3, 16, dtype=torch.float64), causal=True)
        chunk = torch.randn(2, 3, 16, dtype=torch.float64, requires_grad=True)
        out, _, _ = model(chunk, causal=True, past_k=k, past_v=v)
        changed = chunk.detach().clone()
        changed[:, 1:] += 10
        changed_out, _, _ = model(changed, causal=True, past_k=k, past_v=v)
        torch.testing.assert_close(out[:, 0], changed_out[:, 0])
        grad = torch.autograd.grad(out[:, 0].sum(), chunk)[0]
        torch.testing.assert_close(grad[:, 1:], torch.zeros_like(grad[:, 1:]))

    def test_custom_rope_positions_and_noncausal_cached_attention(self):
        model = GQA(16, 4, 2, use_rope=True).double()
        # 不同 batch 使用不同的起点；非连续输入也应保留正确的 token/head 顺序。
        x = torch.randn(6, 2, 16, dtype=torch.float64).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        positions = torch.tensor([7, 29])[:, None] + torch.arange(6)[None, :]
        for causal in (False, True):
            with self.subTest(causal=causal):
                full, full_k, full_v = model(x, causal=causal, position_ids=positions)
                expected = sdpa_reference(model, x, causal=causal, position_ids=positions)
                torch.testing.assert_close(full, expected)
                _, k, v = model(x[:, :2], causal=causal, position_ids=positions[:, :2])
                suffix, k, v = model(
                    x[:, 2:], causal=causal, past_k=k, past_v=v, position_ids=positions[:, 2:],
                )
                expected_suffix = sdpa_reference(
                    model, x, causal=causal, position_ids=positions, query_start=2,
                )
                torch.testing.assert_close(suffix, expected_suffix)
                torch.testing.assert_close(k, full_k)
                torch.testing.assert_close(v, full_v)

    def test_half_precision_large_dot_products_stay_finite(self):
        for dtype in (torch.float16, torch.bfloat16):
            for autocast in (False, True):
                with self.subTest(dtype=dtype, autocast=autocast):
                    model = GQA(8, 4, 2)
                    with torch.no_grad():
                        # Q/K 元素有限，但单次 QK 点积约为 2e6，超出 float16 范围。
                        model.q_proj.weight.zero_()
                        model.k_proj.weight.zero_()
                        model.q_proj.bias.fill_(1000)
                        model.k_proj.bias.fill_(1000)
                    if not autocast:
                        model = model.to(dtype)
                    x = torch.randn(2, 5, 8, dtype=torch.float32 if autocast else dtype)
                    with torch.autocast("cpu", dtype=dtype, enabled=autocast):
                        out, k, v = model(x, causal=True)
                        _, past_k, past_v = model(x[:, :2], causal=True)
                        suffix, _, _ = model(x[:, 2:], causal=True, past_k=past_k, past_v=past_v)
                    self.assertEqual(out.dtype, dtype)
                    self.assertEqual(k.dtype, dtype)
                    self.assertEqual(v.dtype, dtype)
                    self.assertTrue(torch.isfinite(out).all())
                    # 改变块大小可能改变 Linear 的半精度舍入，尤其在输出接近 0 时。
                    eps = torch.finfo(dtype).eps
                    torch.testing.assert_close(suffix, out[:, 2:], rtol=2 * eps, atol=eps)
                    # 所有可见 QK 分数相等，结果应为 V 的因果前缀均值。
                    mean_v = v.float().cumsum(dim=-2) / torch.arange(1, 6)[None, None, :, None]
                    merged = mean_v.repeat_interleave(2, dim=1).transpose(1, 2).reshape(2, 5, 8)
                    with torch.autocast("cpu", dtype=dtype, enabled=autocast):
                        expected = model.o_proj(merged.to(dtype))
                    torch.testing.assert_close(out, expected)
                    out.float().square().mean().backward()
                    for parameter in model.parameters():
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_half_precision_rope_keeps_cache_dtype(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = GQA(16, 4, 2, use_rope=True).to(dtype)
                x = torch.randn(2, 6, 16).to(dtype)
                full, full_k, full_v = model(x, causal=True)
                prefix, k, v = model(x[:, :3], causal=True)
                suffix, k, v = model(x[:, 3:], causal=True, past_k=k, past_v=v)
                self.assertEqual(k.dtype, dtype)
                self.assertEqual(v.dtype, dtype)
                # 线性投影的整段/分块计算会有低精度舍入差异，容限按 dtype 精度设置。
                eps = torch.finfo(dtype).eps
                chunked = torch.cat([prefix, suffix], dim=1)
                torch.testing.assert_close(chunked, full, rtol=2 * eps, atol=eps)
                torch.testing.assert_close(k, full_k, rtol=2 * eps, atol=eps)
                torch.testing.assert_close(v, full_v, rtol=2 * eps, atol=eps)
                reference = GQA(16, 4, 2, use_rope=True).double()
                reference.load_state_dict(model.state_dict())
                expected = sdpa_reference(reference, x.double(), causal=True).to(dtype)
                torch.testing.assert_close(full, expected, rtol=2 * eps, atol=eps)
                torch.testing.assert_close(chunked, expected, rtol=2 * eps, atol=eps)

    def test_invalid_configuration_input_and_cache(self):
        for args in ((0, 4, 2), (16, 0, 2), (16, 4, 0), (16, 3, 1),
                     (16, 4, 3), (16, 4, 8), (16, 4, -1), (16, 4, 2.0), (16, True, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                GQA(*args)
        with self.assertRaisesRegex(ValueError, "even head_dim"):
            GQA(12, 4, 2, use_rope=True)
        # 未启用 RoPE 时，奇数 head_dim 合法。
        self.assertEqual(GQA(12, 4, 2)(torch.randn(1, 2, 12))[0].shape, (1, 2, 12))
        model = GQA(16, 4, 2)
        x = torch.randn(2, 3, 16)
        _, k, v = model(x)
        for bad_x in (torch.zeros(2, 16), torch.zeros(2, 0, 16), torch.zeros(2, 3, 8), x.long()):
            with self.subTest(shape=bad_x.shape), self.assertRaises(ValueError):
                model(bad_x)
        for kwargs in (
            {"past_k": k}, {"past_v": v},
            {"past_k": k[:, :1], "past_v": v[:, :1]},
            {"past_k": k[:1], "past_v": v[:1]},
            {"past_k": k, "past_v": v[:, :, :1]},
            {"past_k": k.double(), "past_v": v},
            {"past_k": k, "past_v": v.double()},
            {"past_k": k.to("meta"), "past_v": v},
            {"position_ids": torch.arange(3).expand(2, -1)},
        ):
            with self.subTest(keys=tuple(kwargs)), self.assertRaises(ValueError):
                model(x, **kwargs)
        model.use_rope = True
        for positions in (torch.arange(3), torch.empty(2, 3, device="meta")):
            with self.assertRaises(ValueError):
                model(x, position_ids=positions)


if __name__ == "__main__":
    unittest.main()
