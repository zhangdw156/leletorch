import math

import torch


def opd_rev_kl(
    student_log_probs, rollout_log_probs, teacher_log_probs,
    token_mask=None, clip_range=0.2,
):
    """
    PG OPD 采样反向 KL：固定 token 级 advantage，再用 PPO 代理损失更新学生。

    三组输入均为 [B, L]，是同一批学生采样 token 的 log probability，已对齐
    token 与前缀；不接收全词表 logits，也不在本函数内做温度缩放或乘 T²。
    student_log_probs: 当前可训练学生的 log probability，保留梯度。
    rollout_log_probs: 生成轨迹时的学生 log probability，整个更新期间固定。
    teacher_log_probs: 固定教师对这些 token 的 log probability。
    token_mask: [B, L]，1/True 标记回复中的有效 token；仅在这些位置求平均。
    clip_range: PPO 概率比裁剪半径，默认 0.2；None 表示不裁剪概率比。

    对 a~pi_rollout，log pi_rollout(a)-log pi_teacher(a) 是反向 KL 的采样项。
    A = stop_grad(log pi_teacher(a) - log pi_rollout(a))，
    r = exp(log pi_current(a) - stop_grad(log pi_rollout(a)))，
    loss = -mean(min(r*A, clip(r, 1-eps, 1+eps)*A))。
    返回的是优化用的代理损失，其数值不等于 KL；只向当前学生传播梯度。
    在 current=rollout 时，其期望梯度与固定前缀上的全词表反向 KL 一致。
    多轮复用轨迹时保留原始 rollout 概率；概率应与实际采样分布一致。

    参考 slime 的 OPD advantage 和 verl / verl-agent 的 PG-OPD 更新路径。
    此处使用标准 PPO 裁剪，不含额外的 dual-clip、任务奖励或 advantage 标准化。
    """
    tensors = (student_log_probs, rollout_log_probs, teacher_log_probs)
    if (student_log_probs.ndim != 2
            or any(tensor.shape != student_log_probs.shape for tensor in tensors)):
        raise ValueError("all log probabilities must have the same shape [B, L]")
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise ValueError("log probabilities must be floating-point tensors")
    if any(tensor.device != student_log_probs.device for tensor in tensors):
        raise ValueError("all log probabilities must be on the same device")
    if clip_range is not None and (not math.isfinite(clip_range) or not 0 <= clip_range < 1):
        raise ValueError("clip_range must be None or finite and in [0, 1)")
    if token_mask is None:
        keep = torch.ones_like(student_log_probs, dtype=torch.bool)
    else:
        if token_mask.shape != student_log_probs.shape or token_mask.device != student_log_probs.device:
            raise ValueError("token_mask must match log probability shape and device")
        keep = token_mask.bool()

    # 固定采样策略与教师；低精度先转 FP32，再清理 padding，避免 inf-inf 或 0*NaN。
    student, rollout, teacher = [
        (tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor)
        .masked_fill(~keep, 0.0)
        for tensor in (student_log_probs, rollout_log_probs.detach(), teacher_log_probs.detach())
    ]
    if any(not torch.isfinite(tensor).all() for tensor in (student, rollout, teacher)):
        raise ValueError("active sampled-token log probabilities must be finite")

    advantage = teacher - rollout
    log_ratio = student - rollout
    if clip_range is not None:
        # min(r*A, clip(r)*A)：A>=0 只限制上界，A<0 只限制下界。
        # 先在 log ratio 上选定裁剪分支，避免 exp 溢出后才裁剪造成 inf*0 梯度。
        log_ratio = torch.where(
            advantage >= 0,
            log_ratio.clamp(max=math.log1p(clip_range)),
            log_ratio.clamp(min=math.log1p(-clip_range)),
        )
    log_ratio = log_ratio.masked_fill(advantage == 0, 0.0)
    objective = log_ratio.exp() * advantage
    return -objective.masked_fill(~keep, 0.0).sum() / keep.sum().clamp_min(1)
