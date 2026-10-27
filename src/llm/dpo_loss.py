import torch


def _sequence_log_prob(log_prob, token_mask):
    """将有效回复 token 的对数概率相加，得到每条回复的对数概率。"""
    if log_prob.ndim not in (1, 2) or log_prob.shape[0] == 0:
        raise ValueError("log_prob must have shape [B] or [B, L] with B > 0")
    if not log_prob.is_floating_point():
        raise ValueError("log_prob must be a floating-point tensor")
    if log_prob.dtype in (torch.float16, torch.bfloat16):
        log_prob = log_prob.float()
    if token_mask is not None:
        if log_prob.ndim != 2 or token_mask.shape != log_prob.shape:
            raise ValueError("token_mask is only supported for log_prob with shape [B, L]")
        # 先替换 padding，避免 -inf * 0 产生 NaN；prompt 位置也应置为 False
        log_prob = log_prob.masked_fill(~token_mask.bool(), 0.0)
    return log_prob if log_prob.ndim == 1 else log_prob.sum(dim=-1)


def dpo_loss(
    chosen_log_prob,
    rejected_log_prob,
    ref_chosen_log_prob,
    ref_rejected_log_prob,
    beta=0.1,
    chosen_mask=None,
    rejected_mask=None,
):
    """
    Direct Preference Optimization（DPO），返回 batch 平均损失。

    chosen/rejected 是偏好对中的优选/劣选回复，ref_* 来自固定参考策略。
    对数概率可以是整条回复的 [B]，也可以是目标 token 的 [B, L]；
    chosen 与 rejected 可有不同的 L，但各自必须与参考策略形状一致。
    mask 为 [B, L]，1/True 表示有效回复 token；不传时使用所有 token。
    输入应已对齐下一 token 标签，且只统计回复部分，序列概率取和而非平均。
    """
    if not 0 < beta < float("inf"):
        raise ValueError("beta must be finite and positive")
    if chosen_log_prob.shape != ref_chosen_log_prob.shape:
        raise ValueError("chosen and reference chosen log probabilities must match")
    if rejected_log_prob.shape != ref_rejected_log_prob.shape:
        raise ValueError("rejected and reference rejected log probabilities must match")

    chosen = _sequence_log_prob(chosen_log_prob, chosen_mask)
    rejected = _sequence_log_prob(rejected_log_prob, rejected_mask)
    ref_chosen = _sequence_log_prob(ref_chosen_log_prob.detach(), chosen_mask)
    ref_rejected = _sequence_log_prob(ref_rejected_log_prob.detach(), rejected_mask)
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected batches must have the same size")

    # z = β * [log(πθ(y_w|x)/πref(y_w|x)) - log(πθ(y_l|x)/πref(y_l|x))]
    logits = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
    # -log(sigmoid(z)) = log(1 + exp(-z))，用 logaddexp 避免大幅度 z 溢出
    return torch.logaddexp(torch.zeros_like(logits), -logits).mean()
