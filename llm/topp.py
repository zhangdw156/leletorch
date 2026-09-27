import torch


def top_p_filter(logits, p):
    """logits: [..., V]，保留累计概率达到 p 的最小候选集，不修改输入。"""
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    if p == 1:
        return logits.clone()

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    # 用前缀（不含当前项）决定是否删除，保留让累计概率首次达到 p 的项
    prob_logits = sorted_logits
    if logits.dtype in (torch.float16, torch.bfloat16):
        prob_logits = sorted_logits.float()
    cumulative_probs = torch.softmax(prob_logits, dim=-1).cumsum(dim=-1)
    remove = torch.zeros_like(sorted_logits, dtype=torch.bool)
    remove[..., 1:] = cumulative_probs[..., :-1] >= p

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, sorted_indices, sorted_logits)


def top_p_sampling(logits, p, temperature=1.0, generator=None):
    """Nucleus sampling：输入 [..., V]，返回 token 索引 [...]。"""
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")

    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    # 温度会改变累计概率，因此必须先缩放，再筛选 top-p 候选
    filtered = top_p_filter(logits / temperature, p)
    probs = torch.softmax(filtered, dim=-1)
    samples = torch.multinomial(
        probs.reshape(-1, probs.shape[-1]), 1, generator=generator
    )
    return samples.reshape(logits.shape[:-1])
