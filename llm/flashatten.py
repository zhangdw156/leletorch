import math

import torch


def flash_attention(q, k, v, causal=False, block_size=64, query_offset=0):
    """
    用分块和在线 softmax 演示 FlashAttention 的前向计算。

    q: [B, H, Lq, D]，k: [B, H, Lk, D]，v: [B, H, Lk, Dv]
    返回 [B, H, Lq, Dv]，不构造完整的 Lq x Lk 注意力矩阵。
    causal 时仅允许 key_index <= query_offset + query_index；
    使用 KV cache 时，将 query_offset 设为已有缓存的长度。
    这是 PyTorch 教学实现，非融合内核；autograd 仍会保存各块中间量。
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be four-dimensional tensors")
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
        raise ValueError("q, k and v must have matching batch, head and KV lengths")
    if q.shape[-1] != k.shape[-1] or min(q.shape[-2:]) == 0 or k.shape[-2] == 0:
        raise ValueError("q and k must have matching, non-empty head dimensions and lengths")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if not isinstance(query_offset, int) or query_offset < 0:
        raise ValueError("query_offset must be a non-negative integer")
    if q.dtype != k.dtype or q.dtype != v.dtype or not q.is_floating_point():
        raise ValueError("q, k and v must have the same floating-point dtype")

    output_dtype = q.dtype
    # 半精度输入使用 float32 累加，同时保留 float64 输入的精度
    if q.dtype in (torch.float16, torch.bfloat16):
        q, k, v = q.float(), k.float(), v.float()

    scale = 1.0 / math.sqrt(q.shape[-1])
    blocks = []
    for q_start in range(0, q.shape[-2], block_size):
        q_block = q[..., q_start:q_start + block_size, :]
        row_shape = q_block.shape[:-1] + (1,)
        row_max = q.new_full(row_shape, float("-inf"))
        row_sum = q.new_zeros(row_shape)
        numerator = q.new_zeros(q_block.shape[:-1] + (v.shape[-1],))

        for k_start in range(0, k.shape[-2], block_size):
            if causal and k_start >= query_offset + q_start + q_block.shape[-2]:
                break
            k_block = k[..., k_start:k_start + block_size, :]
            v_block = v[..., k_start:k_start + block_size, :]
            scores = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale

            if causal:
                q_pos = query_offset + q_start + torch.arange(
                    q_block.shape[-2], device=q.device
                )
                k_pos = k_start + torch.arange(k_block.shape[-2], device=q.device)
                scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))

            # 最大值变化时，重新缩放之前累计的分母和加权和
            new_max = torch.maximum(row_max, scores.max(dim=-1, keepdim=True).values)
            correction = torch.exp(row_max - new_max)
            weights = torch.exp(scores - new_max)
            row_sum = correction * row_sum + weights.sum(dim=-1, keepdim=True)
            numerator = correction * numerator + torch.matmul(weights, v_block)
            row_max = new_max

        blocks.append(numerator / row_sum)

    return torch.cat(blocks, dim=-2).to(output_dtype)
