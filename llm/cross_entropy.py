import torch


def cross_entropy(logits, targets):
    """
    logits:  [B, C]
    targets: [B]
    """
    logits_max = logits.max(dim=-1, keepdim=True).values

    # 数值稳定
    shifted_logits = logits - logits_max

    # log softmax
    log_sum_exp = torch.log(
        torch.exp(shifted_logits).sum(dim=-1, keepdim=True)
    )

    log_probs = shifted_logits - log_sum_exp

    # 取正确类别的 log probability
    target_log_probs = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    )

    loss = -target_log_probs.mean()

    return loss