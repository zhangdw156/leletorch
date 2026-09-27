"""MoE 稀疏分派与稠密参考的输出、梯度、负载统计和低精度回归测试。"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm.moe import MoE
from llm.transformer import Transformer


class MoETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_sparse_moe_matches_dense_all_expert_values_and_gradients(self):
        for top_k in (1, 2, 4):
            for normalize in (False, True):
                with self.subTest(top_k=top_k, normalize=normalize):
                    model = MoE(4, num_experts=4, top_k=top_k, normalize_topk=normalize).double()
                    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
                    actual, actual_aux = model(x, return_aux_loss=True)

                    # 稠密 oracle 计算所有专家，再用权重矩阵合并；不复用稀疏分派过程。
                    probabilities = torch.softmax(model.router(x), dim=-1)
                    weights, ids = probabilities.topk(top_k, dim=-1)
                    if normalize:
                        weights = weights / weights.sum(dim=-1, keepdim=True)
                    dense_weights = torch.zeros_like(probabilities).scatter(-1, ids, weights)
                    all_outputs = torch.stack([expert(x) for expert in model.experts], dim=-2)
                    expected = (dense_weights.unsqueeze(-1) * all_outputs).sum(dim=-2)
                    assignment_fraction = F.one_hot(ids, num_classes=4).double().mean(dim=(0, 1, 2))
                    expected_aux = 4 * (assignment_fraction * probabilities.mean(dim=(0, 1))).sum()
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(actual_aux, expected_aux)

                    upstream = torch.randn_like(actual)
                    parameters = (x, *model.parameters())
                    actual_grads = torch.autograd.grad(
                        (actual * upstream).sum() + 0.03 * actual_aux, parameters, allow_unused=True,
                    )
                    expected_grads = torch.autograd.grad(
                        (expected * upstream).sum() + 0.03 * expected_aux, parameters, allow_unused=True,
                    )
                    for parameter, actual_grad, expected_grad in zip(parameters, actual_grads, expected_grads):
                        # 未选专家在稀疏图中无梯度，在稠密 oracle 中得到零梯度。
                        if actual_grad is None:
                            actual_grad = torch.zeros_like(parameter)
                        if expected_grad is None:
                            expected_grad = torch.zeros_like(parameter)
                        torch.testing.assert_close(actual_grad, expected_grad)

    def test_only_selected_experts_receive_their_assigned_tokens(self):
        model = MoE(2, num_experts=5, top_k=2)
        with torch.no_grad():
            model.router.weight.copy_(torch.tensor([[2., 0.], [1., 0.], [-2., 0.], [-1., 0.], [0., 0.]]))
        x = torch.tensor([[1., 0.], [2., 0.], [-1., 0.]], requires_grad=True)
        calls = {}
        handles = [
            expert.register_forward_pre_hook(lambda module, args, i=i: calls.update({i: args[0].shape[0]}))
            for i, expert in enumerate(model.experts)
        ]
        try:
            model(x).sum().backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, {0: 2, 1: 2, 2: 1, 3: 1})
        self.assertEqual(sum(calls.values()), x.shape[0] * model.top_k)
        for parameter in model.experts[4].parameters():
            self.assertIsNone(parameter.grad)

    def test_single_expert_is_swiglu_and_aux_is_one(self):
        model = MoE(4, num_experts=1, top_k=1).double()
        x = torch.randn(2, 3, 4, dtype=torch.float64)
        out, aux = model(x, return_aux_loss=True)
        torch.testing.assert_close(out, model.experts[0](x))
        torch.testing.assert_close(aux, torch.ones_like(aux))
        router_grad = torch.autograd.grad(out.sum() + aux, model.router.weight)[0]
        torch.testing.assert_close(router_grad, torch.zeros_like(router_grad))

    def test_top_one_router_gradient_depends_on_gate_normalization(self):
        model = MoE(2, num_experts=3, top_k=1)
        # 恒等专家隔离门控行为：主损失的变化全部来自路由权重。
        model.experts = nn.ModuleList([nn.Identity() for _ in range(3)])
        with torch.no_grad():
            model.router.weight.copy_(torch.tensor([[1., 0.], [0., 0.], [-1., 0.]]))
        x = torch.tensor([[1., 1.], [2., 1.]])
        out, aux = model(x, return_aux_loss=True)
        main_grad = torch.autograd.grad(out.sum(), model.router.weight, retain_graph=True)[0]
        torch.testing.assert_close(main_grad, torch.zeros_like(main_grad))
        aux_grad = torch.autograd.grad(aux, model.router.weight)[0]
        self.assertGreater(aux_grad.norm().item(), 0)
        # 所有 token 选专家 0，因此 f=[1,0,0]；均衡损失应精确为 3*mean(p_0)。
        expected_aux = 3 * torch.softmax(model.router(x), dim=-1)[:, 0].mean()
        torch.testing.assert_close(aux, expected_aux)

        model.normalize_topk = False
        main_grad = torch.autograd.grad(model(x).sum(), model.router.weight)[0]
        self.assertGreater(main_grad.norm().item(), 0)

    def test_masked_noncontiguous_input_ignores_padding_and_restores_order(self):
        model = MoE(4, num_experts=3, top_k=2).double()
        mask = torch.tensor([[True, False, True], [False, True, False]]).transpose(0, 1)
        x = torch.randn(2, 3, 4, dtype=torch.float64).transpose(0, 1)
        x = x.masked_fill(~mask.unsqueeze(-1), torch.nan).transpose(0, 1).contiguous().transpose(0, 1)
        x.requires_grad_()
        self.assertFalse(x.is_contiguous())
        out, aux = model(x, mask, return_aux_loss=True)
        expected, expected_aux = model(x[mask], return_aux_loss=True)
        torch.testing.assert_close(out[mask], expected)
        torch.testing.assert_close(aux, expected_aux)
        torch.testing.assert_close(out[~mask], torch.zeros_like(out[~mask]))
        gradient = torch.autograd.grad(out.sum() + aux, x)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient[~mask], torch.zeros_like(gradient[~mask]))

    def test_empty_or_fully_masked_input_has_differentiable_zero_output(self):
        for shape in ((0, 4), (2, 0, 4), (2, 3, 4)):
            with self.subTest(shape=shape):
                model = MoE(4, num_experts=3, top_k=2)
                x = torch.full(shape, torch.nan, requires_grad=True)
                mask = torch.zeros(shape[:-1], dtype=torch.bool)
                out, aux = model(x, mask, return_aux_loss=True)
                self.assertEqual(out.shape, x.shape)
                torch.testing.assert_close(out, torch.zeros_like(out))
                self.assertEqual(aux.item(), 0)
                (out.sum() + aux).backward()
                torch.testing.assert_close(x.grad, torch.zeros_like(x))
                torch.testing.assert_close(model.router.weight.grad, torch.zeros_like(model.router.weight))
                self.assertTrue(all(p.grad is None for expert in model.experts for p in expert.parameters()))
                # 通常训练输入本身无需梯度，默认输出也应支持空批次的 backward。
                model(x.detach(), mask).sum().backward()

    def test_low_precision_and_autocast_keep_router_stable(self):
        reference = MoE(4, num_experts=4, top_k=2)
        x = torch.randn(2, 3, 4)
        expected = reference(x)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = MoE(4, num_experts=4, top_k=2).to(dtype)
                model.load_state_dict(reference.state_dict())
                inputs = x.to(dtype).requires_grad_()
                out, aux = model(inputs, return_aux_loss=True)
                self.assertEqual(out.dtype, dtype)
                self.assertEqual(aux.dtype, torch.float32)
                torch.testing.assert_close(out.float(), expected, rtol=0.03, atol=0.003)
                (out.float().square().mean() + 0.01 * aux).backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

        # FP16 下直接线性投影会溢出；专家置零以单独验证路由投影的数值稳定性。
        model = MoE(4, num_experts=3, top_k=2).half()
        with torch.no_grad():
            model.router.weight.copy_(torch.tensor([[1000.] * 4, [999.] * 4, [-1000.] * 4]))
            for expert in model.experts:
                for parameter in expert.parameters():
                    parameter.zero_()
        inputs = torch.full((2, 4), 1000., dtype=torch.float16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.float16):
            out, aux = model(inputs, return_aux_loss=True)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(aux))
        (out.float().sum() + aux).backward()
        self.assertTrue(torch.isfinite(model.router.weight.grad).all())

    def test_moe_can_replace_transformer_ffn_and_preserve_cached_inference(self):
        model = Transformer(vocab_size=19, hidden_size=16, num_layers=2, num_heads=4).double()
        model.layers[0].ffn = MoE(16, num_experts=3, top_k=2).double()
        tokens = torch.randint(0, 19, (2, 6))
        full, loss, _ = model(tokens, labels=tokens)
        loss.backward()
        self.assertTrue(torch.isfinite(model.layers[0].ffn.router.weight.grad).all())
        with torch.no_grad():
            _, _, cache = model(tokens[:, :3], use_cache=True)
            suffix, _, _ = model(tokens[:, 3:], past_key_values=cache)
        torch.testing.assert_close(suffix, full[:, 3:])

    def test_invalid_configuration_and_inputs(self):
        for config in (
            {'hidden_size': 0}, {'hidden_size': 4, 'num_experts': 0},
            {'hidden_size': 4, 'num_experts': 2, 'top_k': 3},
            {'hidden_size': 4, 'top_k': 0}, {'hidden_size': 4, 'top_k': 1.5},
            {'hidden_size': 4, 'intermediate_size': -1},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                MoE(**config)
        model = MoE(4)
        for x in (torch.zeros(2, 3), torch.zeros(2, 4, dtype=torch.long), torch.tensor(1.)):
            with self.assertRaises(ValueError):
                model(x)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 4), token_mask=torch.ones(2, 1))


if __name__ == '__main__':
    unittest.main()
