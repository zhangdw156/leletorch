if __package__:
    from .kl_loss import _distillation_log_probs, kl_loss
else:
    from kl_loss import _distillation_log_probs, kl_loss


def opd_fwd_kl(student_logits, teacher_logits, token_mask=None, temperature=1.0):
    """
    On-Policy Distillation 前向 KL：T² * KL(teacher_T || student_T)。

    师生 logits: [B, L, V]（也支持 [..., V]），来自同一条学生采样轨迹的前缀；
    词表及 token 顺序必须一致，本函数计算全词表 KL，不负责生成轨迹。
    token_mask: [B, L]，仅标记有效回复位置，损失按有效 token 数平均。
    教师分布固定；采样得到的前缀视作固定输入，只对学生 logits 求梯度。
    T² 用于补偿温度缩放造成的梯度尺度变化，默认 T=1。
    """
    student_log_prob, teacher_log_prob = _distillation_log_probs(
        student_logits, teacher_logits, token_mask, temperature,
    )
    # 教师高概率 token 获得较大权重；保留教师熵，因此同分布时损失为 0
    return temperature ** 2 * kl_loss(teacher_log_prob, student_log_prob, token_mask)
