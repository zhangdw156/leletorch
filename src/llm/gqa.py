"""分组查询注意力（Grouped-Query Attention）：多个 Q head 共享一组 K/V。"""

import math

import torch
import torch.nn as nn

if __package__:
    from .rope import apply_rope
    from .softmax import softmax
else:
    from rope import apply_rope
    from softmax import softmax


class GQA(nn.Module):
    """
    Hq 个查询头、Hkv 个键值头，每 G=Hq/Hkv 个相邻查询头共享一个 KV head。

    第 h 个查询头计算 softmax(Q_h K_gᵀ / sqrt(Dh)) V_g，其中 g=h//G。
    Hkv=Hq 时退化为 MHA，Hkv=1 时为 MQA；Dh=hidden_size/Hq。
    Q 投影宽度仍为 hidden_size，K/V 投影宽度缩小为 Hkv*Dh，缓存也只存
    Hkv 个头，相比相同 hidden_size、Hq 的 MHA，K/V 缓存元素数为 Hkv/Hq。
    复用已有 RoPE 和 softmax；归一化、残差连接由外层网络负责。
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, use_rope=False):
        super().__init__()
        for name, value in (
            ("hidden_size", hidden_size), ("num_heads", num_heads),
            ("num_kv_heads", num_kv_heads),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope
        if use_rope and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")

        # 与 MHA 保持相同的投影命名和 bias 设置，便于比较或迁移同形状权重。
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, causal=False, past_k=None, past_v=None, position_ids=None):
        """
        x: [B, L, D]；past_k、past_v: [B, Hkv, S, Dh]，缓存需成对提供。
        返回 out: [B, L, D]，new_k、new_v: [B, Hkv, S+L, Dh]。

        causal=True 时按序列顺序遮罩未来 token，支持单 token 和多 token 续算。
        use_rope=True 时，position_ids: [B, L] 默认使用 S 到 S+L-1；自定义
        位置时调用方需自行提供各块对应的位置，因果遮罩仍按 token 的先后顺序计算。
        缓存中的 K 已旋转，每次只旋转新 Q/K；缓存不 detach，训练时仍可回传梯度。
        缓存的设备、投影 dtype 和位置编码设置应在各次调用中保持一致。
        """
        if (x.ndim != 3 or x.shape[-1] != self.hidden_size or x.shape[1] == 0
                or not x.is_floating_point()):
            raise ValueError("x must be floating-point with shape [B, L, hidden_size] and L > 0")
        B, L, D = x.shape
        Hq, Hkv, G, Dh = self.num_heads, self.num_kv_heads, self.num_kv_groups, self.head_dim
        if (past_k is None) != (past_v is None):
            raise ValueError("past_k and past_v must be provided together")
        past_len = 0
        if past_k is not None:
            if (past_k.ndim != 4 or past_k.shape != past_v.shape
                    or past_k.shape[:2] != (B, Hkv) or past_k.shape[-1] != Dh):
                raise ValueError("past_k and past_v must have shape [B, Hkv, S, Dh]")
            if past_k.device != x.device or past_v.device != x.device:
                raise ValueError("KV caches and x must be on the same device")
            past_len = past_k.shape[-2]

        # Q: [B, Hq, L, Dh]；K/V: [B, Hkv, L, Dh]，不先扩展到 Hq 个头。
        q = self.q_proj(x).view(B, L, Hq, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, Hkv, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, Hkv, Dh).transpose(1, 2)
        if self.use_rope:
            if position_ids is None:
                position_ids = torch.arange(past_len, past_len + L, device=x.device)
                position_ids = position_ids.unsqueeze(0).expand(B, -1)
            elif position_ids.shape != (B, L) or position_ids.device != x.device:
                raise ValueError("position_ids must have shape [B, L] on x.device")
            q_dtype, k_dtype = q.dtype, k.dtype
            q, k = apply_rope(q, k, position_ids=position_ids)
            q, k = q.to(q_dtype), k.to(k_dtype)
        elif position_ids is not None:
            raise ValueError("position_ids requires use_rope=True")

        if past_k is not None:
            if past_k.dtype != k.dtype or past_v.dtype != v.dtype:
                raise ValueError("KV cache dtypes must match the current projected K/V")
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        # 返回的缓存始终只包含 Hkv 个头，不存储为每个 Q head 复制的 K/V。
        new_k, new_v = k, v

        # 将查询头拆成 [Hkv, G]，通过广播让组内 G 个 Q head 共享 K/V。
        q = q.reshape(B, Hkv, G, L, Dh)
        compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        # 点积前升精度；局部关闭 autocast，防止 matmul 又被降回半精度。
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = torch.matmul(
                q.to(compute_dtype), k.to(compute_dtype).unsqueeze(2).transpose(-1, -2),
            ) / math.sqrt(Dh)
            # scores: [B, Hkv, G, L, S+L]；第 i 个新 token 只能看到 key <= S+i。
            if causal:
                q_pos = past_len + torch.arange(L, device=x.device)
                k_pos = torch.arange(k.shape[-2], device=x.device)
                mask = k_pos[None, :] <= q_pos[:, None]
                scores = scores.masked_fill(~mask, float("-inf"))
            attn = softmax(scores, dim=-1)
            out = torch.matmul(attn, v.to(compute_dtype).unsqueeze(2))

        # [B, Hkv, G, L, Dh] -> [B, L, D]；保留原有查询头顺序再做输出投影。
        out = out.to(q.dtype).reshape(B, Hq, L, Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out), new_k, new_v


if __name__ == "__main__":
    torch.manual_seed(0)
    model = GQA(hidden_size=32, num_heads=8, num_kv_heads=2, use_rope=True).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        full, full_k, _ = model(x, causal=True)
        prefix, k, v = model(x[:, :3], causal=True)
        suffix, k, v = model(x[:, 3:], causal=True, past_k=k, past_v=v)
        chunked = torch.cat([prefix, suffix], dim=1)
        torch.testing.assert_close(chunked, full)
        torch.testing.assert_close(k, full_k)
    print(f"output: {tuple(full.shape)}, KV cache: {tuple(k.shape)}")
    print(f"KV cache / MHA: {model.num_kv_heads / model.num_heads:.2f}")
    print(f"cached chunks max error: {(chunked - full).abs().max().item():.2e}")
