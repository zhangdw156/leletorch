if __package__:
    from .kl_loss import _distillation_log_probs, kl_loss
else:
    from kl_loss import _distillation_log_probs, kl_loss


def opd_rev_kl(student_logits, teacher_logits, token_mask=None, temperature=1.0):
    """
    On-Policy Distillation 反向 KL：T² * KL(student_T || teacher_T)。

    师生 logits: [B, L, V]（也支持 [..., V]），来自同一条学生采样轨迹的前缀；
    词表及 token 顺序必须一致，本函数计算全词表 KL，不负责生成轨迹。
    token_mask: [B, L]，仅标记有效回复位置，损失按有效 token 数平均。
    教师分布及采样前缀固定；T² 补偿温度缩放，默认 T=1。
    """
    student_log_prob, teacher_log_prob = _distillation_log_probs(
        student_logits, teacher_logits, token_mask, temperature,
    )
    # 学生概率作为权重，但不能 detach：概率权重和 log probability 都需参与求导
    return temperature ** 2 * kl_loss(student_log_prob, teacher_log_prob, token_mask)
