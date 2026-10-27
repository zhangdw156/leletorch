import torch

if __package__:
    from .kl_loss import _distillation_log_probs, kl_loss
else:
    from kl_loss import _distillation_log_probs, kl_loss


def opd_fwd_kl(
    student_logits, teacher_logits, token_mask=None, temperature=1.0,
    topk=None, normalize_topk=False,
):
    """
    GKD OPD 前向 KL：在学生采样轨迹的固定前缀上直接优化 KL(teacher || student)。

    师生 logits: [B, L, V]（也支持 [..., V]），来自同一条学生采样轨迹的前缀；
    词表及 token 顺序必须一致，本函数不负责生成轨迹。
    token_mask: [B, L]，仅标记有效回复位置，损失按有效 token 数平均。
    教师分布固定；采样得到的前缀视作固定输入，只对学生 logits 求梯度。
    temperature: 师生 logits 的可选共同温度，默认 1；不额外乘 T²。
        这是分布计算的温度，轨迹采样温度由调用方单独控制。
    topk: 默认 None，计算全词表 KL；设为 K 时只在教师的 top-k 词表项上求和。
    normalize_topk: False 保留原始全词表概率，此时是截断 KL 和，可能为负；
        True 则分别在教师支持集上重新归一化师生概率，计算条件分布的 KL。
        不对截断和做 clamp，保留其梯度；K=V 时恢复全词表 KL。

    参考 verl / verl-agent 的 forward_kl_topk；采样反向 KL 见 opd_rev_kl.py。
    """
    student_log_prob, teacher_log_prob = _distillation_log_probs(
        student_logits, teacher_logits, token_mask, temperature,
    )
    if topk is None:
        if normalize_topk:
            raise ValueError("normalize_topk requires topk")
        return kl_loss(teacher_log_prob, student_log_prob, token_mask)
    if (not isinstance(topk, int) or isinstance(topk, bool)
            or not 0 < topk <= student_logits.shape[-1]):
        raise ValueError("topk must be an integer in [1, vocab_size]")

    # 支持集由固定教师选择；学生仍使用全词表 softmax，保留对支持集外 logits 的梯度。
    teacher_log_prob, teacher_ids = teacher_log_prob.topk(topk, dim=-1)
    student_log_prob = student_log_prob.gather(-1, teacher_ids)
    if normalize_topk:
        teacher_log_prob = torch.log_softmax(teacher_log_prob, dim=-1)
        student_log_prob = torch.log_softmax(student_log_prob, dim=-1)
        return kl_loss(teacher_log_prob, student_log_prob, token_mask)

    # 原始 top-k 概率之和通常小于 1，不能当作完整归一化分布传给 kl_loss。
    teacher_prob = teacher_log_prob.exp()
    has_mass = teacher_prob > 0
    safe_teacher = torch.where(has_mass, teacher_log_prob, 0.0)
    safe_student = torch.where(has_mass, student_log_prob, 0.0)
    loss_token = (teacher_prob * (safe_teacher - safe_student)).sum(dim=-1)
    if token_mask is None:
        return loss_token.sum() / max(loss_token.numel(), 1)
    keep = token_mask.bool()
    return loss_token.masked_fill(~keep, 0.0).sum() / keep.sum().clamp_min(1)
