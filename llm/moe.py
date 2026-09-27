"""稀疏 top-k Mixture of Experts：逐 token 路由、专家计算和加权合并。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .softmax import softmax
    from .swiglu import SwiGLU
else:
    from softmax import softmax
    from swiglu import SwiGLU


class MoE(nn.Module):
    """
    Token-choice top-k MoE，每个专家复用已有的 SwiGLU 前馈网络。

    router: D -> E，为每个 token 选择 K 个专家；专家输出按路由权重求和。
    normalize_topk=True 时，所选专家权重之和为 1；False 保留全专家 softmax
    中的原始概率，此时选中权重之和通常小于 1。
    当 K=1 且重新归一化时，权重恒为 1，主损失无法训练 router；可使用辅助
    均衡损失训练 router，或设 normalize_topk=False 保留门控概率的梯度。

    本实现不限制专家容量，不丢弃有效 token；用单机循环展示稀疏分派过程。
    MoE 只提供前馈变换，归一化与残差连接由外层 TransformerBlock 负责。
    """

    def __init__(
        self, hidden_size, num_experts=8, top_k=2,
        intermediate_size=None, normalize_topk=True,
    ):
        super().__init__()
        for name, value in (
            ("hidden_size", hidden_size), ("num_experts", num_experts), ("top_k", top_k),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if top_k > num_experts:
            raise ValueError("top_k must not exceed num_experts")
        if intermediate_size is not None and (
            not isinstance(intermediate_size, int) or isinstance(intermediate_size, bool)
            or intermediate_size <= 0
        ):
            raise ValueError("intermediate_size must be a positive integer")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_topk = normalize_topk
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(hidden_size, intermediate_size) for _ in range(num_experts)
        ])

    def forward(self, x, token_mask=None, return_aux_loss=False):
        """
        x: [..., D]，常用 [B, L, D]；输出形状、dtype 与 x 相同。
        token_mask: [...]，1/True 表示有效 token；padding 不参与路由或统计，输出为 0。
        默认返回 out；return_aux_loss=True 时返回 (out, aux_loss)。

        aux_loss = E * sum_e(f_e * p_e)，其中 f_e 为专家 e 获得的分派数 / (N*K)，
        p_e 为有效 token 的平均全专家路由概率，N 为有效 token 数。
        f_e 来自离散 top-k 选择，不可导；梯度通过 p_e 回传给 router。
        均匀分派时辅助项为 1；其权重由调用方配置，不自动加进主损失。
        没有有效 token 时 out 与 aux_loss 都为可反向传播的 0。
        """
        if x.ndim < 1 or x.shape[-1] != self.hidden_size or not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor of shape [..., hidden_size]")
        flat_x = x.reshape(-1, self.hidden_size)
        if token_mask is None:
            keep = torch.ones(flat_x.shape[0], dtype=torch.bool, device=x.device)
        else:
            if token_mask.shape != x.shape[:-1] or token_mask.device != x.device:
                raise ValueError("token_mask must match x's token dimensions and device")
            keep = token_mask.reshape(-1).bool()
        token_ids = keep.nonzero(as_tuple=True)[0]
        # 在任何计算之前移除 padding，允许无效位置包含 NaN 等占位值。
        active_x = flat_x.index_select(0, token_ids)
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        if active_x.shape[0] == 0:
            # 空切片求和保留参数计算图，不会发生半精度大张量求和溢出。
            aux_loss = self.router.weight.reshape(-1)[:0].sum().to(compute_dtype)
            out = flat_x.masked_fill(~keep[:, None], 0.0).reshape_as(x) + aux_loss.to(x.dtype)
            if return_aux_loss:
                return out, aux_loss
            return out

        # router 的线性投影和 softmax 均使用 FP32；double 输入保持 FP64。
        # 显式关闭局部 autocast，避免先在半精度投影中溢出、再转 FP32 已经太晚。
        with torch.autocast(device_type=x.device.type, enabled=False):
            router_logits = F.linear(active_x.to(compute_dtype), self.router.weight.to(compute_dtype))
            selected_logits, selected_experts = router_logits.topk(self.top_k, dim=-1)
            router_probs = softmax(router_logits) if return_aux_loss or not self.normalize_topk else None
            if self.normalize_topk:
                # 在选中的 logits 上直接 softmax，等价于全概率取 top-k 后重新归一化。
                routing_weights = softmax(selected_logits)
            else:
                routing_weights = router_probs.gather(-1, selected_experts)

        combined = torch.zeros_like(active_x, dtype=compute_dtype)
        for expert_id, expert in enumerate(self.experts):
            # 每个 token 可能进入多个专家；slot 指示该专家在该 token 的 top-k 中的位置。
            rows, slots = (selected_experts == expert_id).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            expert_out = expert(active_x.index_select(0, rows))
            weighted_out = expert_out.to(compute_dtype) * routing_weights[rows, slots, None]
            # 将同一 token 的不同专家贡献累加，保留专家、输入和路由权重的梯度。
            combined.index_add_(0, rows, weighted_out)

        # 恢复 token 的原始位置；被屏蔽的 token 保持为 0。
        flat_out = torch.zeros_like(flat_x).index_copy(0, token_ids, combined.to(x.dtype))
        out = flat_out.reshape_as(x)
        if not return_aux_loss:
            return out

        assignments = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts)
        fraction = assignments.to(compute_dtype) / (active_x.shape[0] * self.top_k)
        aux_loss = self.num_experts * (fraction * router_probs.mean(dim=0)).sum()
        return out, aux_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    moe = MoE(hidden_size=16, num_experts=4, top_k=2)
    x = torch.randn(2, 5, 16, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    out, aux_loss = moe(x, token_mask=mask, return_aux_loss=True)
    # 0.01 是此示例选择的辅助损失权重，可由训练配置调整。
    loss = out[mask].square().mean() + 0.01 * aux_loss
    loss.backward()
    print(f"output: {tuple(out.shape)}, aux_loss: {aux_loss.item():.4f}")
    print(f"router grad norm: {moe.router.weight.grad.norm().item():.4f}")
