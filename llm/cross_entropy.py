import torch


def cross_entropy(logits, targets):
    """
    多分类交叉熵：正确类别的负对数概率，在 batch 上取平均。
    logits:  [B, C]，未经 softmax 的分类分数
    targets: [B]，取值为 0 到 C-1 的整数类别索引
    返回标量 loss。
    """
    logits_max = logits.max(dim=-1, keepdim=True).values

    # 每行减去最大值不改变 softmax，且能避免 exp 溢出
    shifted_logits = logits - logits_max

    # log_softmax(z)_i = z_i - log(sum_j exp(z_j))
    log_sum_exp = torch.log(
        torch.exp(shifted_logits).sum(dim=-1, keepdim=True)
    )

    log_probs = shifted_logits - log_sum_exp

    # gather 按标签取出每个样本正确类别的 log probability，结果为 [B, 1]
    target_log_probs = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    )

    # 概率越接近 1，对应的负对数损失越接近 0
    loss = -target_log_probs.mean()

    return loss
