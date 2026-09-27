"""Transformer 的训练、位置编码、因果性和逐层 KV 缓存集成测试。"""

import unittest

import torch
import torch.nn.functional as F

from llm.mha import MHA
from llm.transformer import Transformer


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def make_model(self):
        return Transformer(vocab_size=19, hidden_size=16, num_layers=2, num_heads=4)

    def test_mha_rope_matches_independent_attention_reference(self):
        model = MHA(16, 4, use_rope=True).double()
        x = torch.randn(2, 5, 16, dtype=torch.float64)
        positions = torch.arange(5).expand(2, -1) + torch.tensor([[3], [8]])
        q, k, v = [
            projection(x).reshape(2, 5, 4, 4).transpose(1, 2)
            for projection in (model.q_proj, model.k_proj, model.v_proj)
        ]

        # 使用复数乘法独立表达相邻特征对的二维旋转，避免只比较自身实现。
        frequencies = 10000.0 ** (-torch.arange(0, 4, 2, dtype=torch.float64) / 4)
        angles = positions[..., None] * frequencies
        rotation = torch.polar(torch.ones_like(angles), angles)[:, None]

        def rotate(tensor):
            pairs = tensor.reshape(2, 4, 5, 2, 2).contiguous()
            rotated = torch.view_as_complex(pairs) * rotation
            return torch.view_as_real(rotated).flatten(-2)

        expected = F.scaled_dot_product_attention(rotate(q), rotate(k), v, is_causal=True)
        expected = model.o_proj(expected.transpose(1, 2).reshape(2, 5, 16))
        actual, new_k, new_v = model(x, causal=True, position_ids=positions)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(new_k, rotate(k), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(new_v, v)

    def test_language_model_loss_and_optimizer_step(self):
        model = self.make_model().double()
        tokens = torch.randint(0, 19, (2, 7))
        labels = tokens.clone()
        labels[:, :3] = -100
        logits, loss, cache = model(tokens, labels=labels)
        expected = F.cross_entropy(
            logits[:, :-1].reshape(-1, 19), labels[:, 1:].reshape(-1), ignore_index=-100,
        )
        self.assertEqual(logits.shape, (2, 7, 19))
        self.assertIsNone(cache)
        torch.testing.assert_close(loss, expected)
        upstream = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        reference = torch.autograd.grad(expected, logits, retain_graph=True)[0]
        torch.testing.assert_close(upstream, reference)

        previous_weight = model.lm_head.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(previous_weight, model.lm_head.weight))

    def test_cached_chunks_match_full_sequence_at_every_layer(self):
        model = self.make_model().double().eval()
        tokens = torch.randint(0, 19, (2, 8))
        with torch.no_grad():
            full, loss, full_cache = model(tokens, use_cache=True)
            self.assertIsNone(loss)
            for lengths in ((8,), (1,) * 8, (3, 3, 2)):
                with self.subTest(lengths=lengths):
                    cache = None
                    outputs = []
                    offset = 0
                    for length in lengths:
                        logits, _, cache = model(
                            tokens[:, offset:offset + length], past_key_values=cache,
                            use_cache=True,
                        )
                        offset += length
                        outputs.append(logits)
                        self.assertEqual(len(cache), 2)
                        for key, value in cache:
                            self.assertEqual(key.shape, (2, 4, offset, 4))
                            self.assertEqual(value.shape, key.shape)
                    torch.testing.assert_close(torch.cat(outputs, dim=1), full)
                    for actual_pair, expected_pair in zip(cache, full_cache):
                        for actual, expected in zip(actual_pair, expected_pair):
                            torch.testing.assert_close(actual, expected)

    def test_future_tokens_do_not_change_prefix_logits(self):
        model = self.make_model().double().eval()
        tokens = torch.randint(0, 19, (2, 8))
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 1) % 19
        with torch.no_grad():
            original, _, _ = model(tokens)
            altered, _, _ = model(changed)
            torch.testing.assert_close(original[:, :5], altered[:, :5])
            _, _, prefix_cache = model(tokens[:, :3], use_cache=True)
            saved_cache = tuple((key.clone(), value.clone()) for key, value in prefix_cache)
            original_chunk, _, _ = model(tokens[:, 3:], past_key_values=prefix_cache)
            altered_chunk, _, cache = model(changed[:, 3:], past_key_values=prefix_cache)
            self.assertIsNone(cache)
            torch.testing.assert_close(original_chunk[:, :2], altered_chunk[:, :2])
            torch.testing.assert_close(original_chunk, original[:, 3:])
            for actual_pair, expected_pair in zip(prefix_cache, saved_cache):
                for actual, expected in zip(actual_pair, expected_pair):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_half_precision_training_and_cache(self):
        tokens = torch.randint(0, 19, (2, 6))
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self.make_model().to(dtype)
                logits, loss, _ = model(tokens, labels=tokens)
                self.assertEqual(logits.dtype, dtype)
                self.assertEqual(loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(logits).all())
                reference_loss = F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, 19), tokens[:, 1:].reshape(-1),
                )
                torch.testing.assert_close(loss, reference_loss)
                loss.backward()
                for parameter in model.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                model.eval()
                with torch.no_grad():
                    _, _, cache = model(tokens[:, :3], use_cache=True)
                    suffix, _, cache = model(tokens[:, 3:], past_key_values=cache, use_cache=True)
                    torch.testing.assert_close(suffix, logits[:, 3:])
                    for key, value in cache:
                        self.assertEqual(key.dtype, dtype)
                        self.assertEqual(value.dtype, dtype)

    def test_no_prediction_targets_produce_differentiable_zero(self):
        for length in (1, 5):
            with self.subTest(length=length):
                model = self.make_model()
                tokens = torch.randint(0, 19, (2, length))
                labels = tokens if length == 1 else torch.full_like(tokens, -100)
                _, loss, _ = model(tokens, labels=labels)
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                for parameter in model.parameters():
                    torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))

    def test_invalid_configuration_and_caches(self):
        for config in (
            {"vocab_size": 0}, {"vocab_size": 19, "num_layers": 0},
            {"vocab_size": 19, "hidden_size": 12, "num_heads": 4},
            {"vocab_size": 19, "intermediate_size": 0},
            {"vocab_size": 19, "num_heads": 0},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                Transformer(**config)
        model = self.make_model()
        tokens = torch.randint(0, 19, (2, 4))
        _, _, cache = model(tokens, use_cache=True)
        shorter_pair = tuple(tensor[:, :, :-1] for tensor in cache[1])
        wrong_heads = tuple(tensor[:, :1] for tensor in cache[0])
        wrong_dtype = tuple(tensor.double() for tensor in cache[0])
        for invalid_cache in (
            (), (cache[0],), (cache[0], shorter_pair), (wrong_heads, cache[1]),
            (wrong_dtype, cache[1]), (None, cache[1]), ((None, None), cache[1]),
        ):
            with self.subTest(cache=str(type(invalid_cache))), self.assertRaises(ValueError):
                model(tokens, past_key_values=invalid_cache)
        with self.assertRaises(ValueError):
            model(tokens, labels=tokens, past_key_values=cache)
        with self.assertRaises(ValueError):
            model(tokens.float())
        with self.assertRaises(ValueError):
            model(tokens[:, :0])


if __name__ == "__main__":
    unittest.main()
