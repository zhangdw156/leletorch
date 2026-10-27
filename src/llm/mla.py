import math

import torch
import torch.nn as nn

if __package__:
    from .rope import apply_rope
else:
    from rope import apply_rope


class MLA(nn.Module):
    """多头潜在注意力：低秩 KV 压缩、可选 Q 压缩和解耦 RoPE。"""

    def __init__(
        self, hidden_size, num_heads, kv_lora_rank=None,
        rope_head_dim=None, q_lora_rank=None,
    ):
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if kv_lora_rank is None:
            kv_lora_rank = self.head_dim
        if rope_head_dim is None:
            rope_head_dim = self.head_dim
        if kv_lora_rank <= 0 or rope_head_dim <= 0 or rope_head_dim % 2 != 0:
            raise ValueError("kv_lora_rank must be positive and rope_head_dim positive and even")
        if q_lora_rank is not None and q_lora_rank <= 0:
            raise ValueError("q_lora_rank must be positive")
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim

        q_input_dim = hidden_size if q_lora_rank is None else q_lora_rank
        self.q_down_proj = (
            nn.Identity() if q_lora_rank is None
            else nn.Linear(hidden_size, q_lora_rank, bias=False)
        )
        self.q_proj = nn.Linear(
            q_input_dim, num_heads * (self.head_dim + rope_head_dim), bias=False
        )
        self.kv_down_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        # 无 bias 才能直接将 K/V 的上投影吸收到后面的矩阵乘法中
        self.k_up_proj = nn.Linear(kv_lora_rank, hidden_size, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, hidden_size, bias=False)
        self.k_rope_proj = nn.Linear(hidden_size, rope_head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, causal=False, past_kv=None, past_k_rope=None, position_ids=None):
        """
        x: [B, L, D]；position_ids: [B, L]，默认从缓存长度开始计数。
        past_kv: [B, S, R]，压缩的 KV；past_k_rope: [B, 1, S, Dr]。
        返回 out, new_kv, new_k_rope；缓存仅保存低秩表示和共享 RoPE key。
        """
        B, L, D = x.shape
        H, Dh, Dr = self.num_heads, self.head_dim, self.rope_head_dim
        if D != self.hidden_size or L == 0:
            raise ValueError("x must have shape [B, L, hidden_size] with L > 0")
        if (past_kv is None) != (past_k_rope is None):
            raise ValueError("past_kv and past_k_rope must be provided together")
        past_len = 0 if past_kv is None else past_kv.shape[1]
        if past_kv is not None:
            if (past_kv.shape != (B, past_len, self.kv_lora_rank)
                    or past_k_rope.shape != (B, 1, past_len, Dr)):
                raise ValueError("invalid compressed KV cache shapes")

        q = self.q_proj(self.q_down_proj(x)).view(B, L, H, Dh + Dr).transpose(1, 2)
        q_content, q_rope = q.split([Dh, Dr], dim=-1)
        kv = self.kv_down_proj(x)
        k_rope = self.k_rope_proj(x).unsqueeze(1)

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + L, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(B, -1)
        elif position_ids.shape != (B, L):
            raise ValueError("position_ids must have shape [B, L]")
        q_rope, k_rope = apply_rope(q_rope, k_rope, position_ids=position_ids)
        q_rope, k_rope = q_rope.to(q.dtype), k_rope.to(q.dtype)

        if past_kv is not None:
            kv = torch.cat([past_kv, kv], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=-2)

        # q^T(W_k c) = (W_k^T q)^T c，无需展开并缓存每个 head 的 K
        k_weight = self.k_up_proj.weight.view(H, Dh, self.kv_lora_rank)
        q_latent = torch.einsum("bhld,hdr->bhlr", q_content, k_weight)
        scores = torch.einsum("bhlr,btr->bhlt", q_latent, kv)
        scores = scores + torch.matmul(q_rope, k_rope.transpose(-1, -2))
        scores = scores / math.sqrt(Dh + Dr)

        if causal:
            q_pos = past_len + torch.arange(L, device=x.device)
            k_pos = torch.arange(kv.shape[1], device=x.device)
            scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))

        if scores.dtype in (torch.float16, torch.bfloat16):
            scores = scores.float()
        scores_exp = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
        attn = (scores_exp / scores_exp.sum(dim=-1, keepdim=True)).to(kv.dtype)

        # 先对压缩表示加权求和，再上投影 V，避免展开整个历史序列的 V
        latent_out = torch.einsum("bhlt,btr->bhlr", attn, kv)
        v_weight = self.v_up_proj.weight.view(H, Dh, self.kv_lora_rank)
        out = torch.einsum("bhlr,hdr->bhld", latent_out, v_weight)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out), kv, k_rope
