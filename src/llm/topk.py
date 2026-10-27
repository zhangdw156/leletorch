import torch


def top_k_filter(logits, k):
    """logits: [..., V]，只保留最大的 k 项，其余置为 -inf，不修改输入。"""
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    if not isinstance(k, int) or not 1 <= k <= logits.shape[-1]:
        raise ValueError("k must be an integer in [1, vocab_size]")

    # 按索引保留恰好 k 项，避免相同 logits 导致候选数超过 k
    values, indices = torch.topk(logits, k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, indices, values)


def top_k_sampling(logits, k, temperature=1.0, generator=None):
    """从 top-k 候选中采样，输入 [..., V]，返回 token 索引 [...]。"""
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")

    # 半精度 logits 转为 float32，降低温度缩放和 softmax 的溢出风险
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    filtered = top_k_filter(logits / temperature, k)
    probs = torch.softmax(filtered, dim=-1)
    samples = torch.multinomial(
        probs.reshape(-1, probs.shape[-1]), 1, generator=generator
    )
    return samples.reshape(logits.shape[:-1])
