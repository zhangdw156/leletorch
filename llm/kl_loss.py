import torch


def kl_loss(log_p, log_q, token_mask=None, reduction="mean"):
    """
    KL(P || Q) = sum_v P(v) * [log P(v) - log Q(v)]，通常不对称。

    log_p/log_q: 相同形状的 [..., V]，必须是沿词表归一化后的对数概率。
    token_mask: [...]，1/True 表示有效位置；不传时使用所有位置。
    reduction: 'none' 返回 [...]；'sum' 求和；'mean' 按有效位置数平均。
    全部位置被遮罩时返回 0；本函数不截断 P 或 Q 的梯度。
    """
    if log_p.shape != log_q.shape or log_p.ndim < 1 or log_p.shape[-1] == 0:
        raise ValueError("log_p and log_q must have the same shape [..., V] with V > 0")
    if not log_p.is_floating_point() or not log_q.is_floating_point():
        raise ValueError("log_p and log_q must be floating-point tensors")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be 'none', 'sum' or 'mean'")
    if log_p.dtype in (torch.float16, torch.bfloat16):
        log_p = log_p.float()
    if log_q.dtype in (torch.float16, torch.bfloat16):
        log_q = log_q.float()

    if token_mask is not None:
        if token_mask.shape != log_p.shape[:-1]:
            raise ValueError("token_mask must match the dimensions before the vocabulary")
        keep = token_mask.bool()
        # 在求差之前清除无效位置，避免先算出 inf/NaN 再乘 0
        log_p = log_p.masked_fill(~keep.unsqueeze(-1), 0.0)
        log_q = log_q.masked_fill(~keep.unsqueeze(-1), 0.0)

    p = torch.exp(log_p)
    # 约定 P(v)=0 时该项为 0，包括 P(v)=Q(v)=0，避免 0*(-inf) 或 inf-inf
    has_mass = p > 0
    safe_log_p = torch.where(has_mass, log_p, 0.0)
    safe_log_q = torch.where(has_mass, log_q, 0.0)
    loss_token = (p * (safe_log_p - safe_log_q)).sum(dim=-1)

    if reduction == "none":
        return loss_token
    total = loss_token.sum()
    if reduction == "sum":
        return total
    count = max(loss_token.numel(), 1) if token_mask is None else keep.sum().clamp_min(1)
    return total / count


def _distillation_log_probs(student_logits, teacher_logits, token_mask, temperature):
    """对齐蒸馏输入、冻结教师，并在温度缩放后稳定计算对数概率。"""
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")
    if (student_logits.shape != teacher_logits.shape
            or student_logits.ndim < 1 or student_logits.shape[-1] == 0):
        raise ValueError("student and teacher logits must have the same shape [..., V]")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise ValueError("student and teacher logits must be floating-point tensors")

    teacher_logits = teacher_logits.detach()
    if student_logits.dtype in (torch.float16, torch.bfloat16):
        student_logits = student_logits.float()
    if teacher_logits.dtype in (torch.float16, torch.bfloat16):
        teacher_logits = teacher_logits.float()
    if token_mask is not None:
        if token_mask.shape != student_logits.shape[:-1]:
            raise ValueError("token_mask must match the dimensions before the vocabulary")
        # 在 log_softmax 前处理 padding，允许被遮罩位置存放 -inf 等占位值
        drop = ~token_mask.bool().unsqueeze(-1)
        student_logits = student_logits.masked_fill(drop, 0.0)
        teacher_logits = teacher_logits.masked_fill(drop, 0.0)

    student_log_prob = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_prob = torch.log_softmax(teacher_logits / temperature, dim=-1)
    return student_log_prob, teacher_log_prob
