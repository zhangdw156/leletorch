"""参考 verl 的 kl_penalty_forward，实现采样 token 的 KL 正则项。

源码：https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
"""

import torch


def grpo_kl_loss(logprob, ref_logprob, kl_penalty="low_var_kl"):
    """
    logprob/ref_logprob: 当前策略/参考模型对同一批生成 token 的对数概率。
    输入形状相同，通常为 [B, L] 或 [B, G, L]；输入应为有限值，且没有词表维。
    返回同形状的逐 token KL 项；response mask、聚合及 KL 系数由调用方处理。
    float16/bfloat16 输入转为 float32 计算并返回，避免 exp 和平方溢出。

    记 P 为当前策略，Q 为参考模型，r=Q(a|s)/P(a|s)：
    - kl/k1: log(P/Q)，单个采样项可能为负。
    - mse/k2: 0.5*log(P/Q)^2，是 KL 的近似，并非无偏的 KL 数值估计。
    - low_var_kl/k3: r-log(r)-1，先裁剪 log(r) 到 [-20, 20]，
      再将结果裁剪到 [-10, 10]，与 verl 的基础 k3 分支一致。

    未裁剪且 a~P 时，k1/k3 的期望等于 KL(P||Q)，固定 token 上的自动微分
    梯度不等同于全词表 KL 的梯度。这里没有使用 k3+ 的替代梯度。
    与 verl 的辅助函数一致，本函数不 detach 输入；参考模型应由调用方冻结。
    full 不受支持：完整词表的 KL 请使用 kl_loss.py 中的 kl_loss。
    """
    if kl_penalty == "full":
        raise NotImplementedError("full-vocabulary KL is provided by llm.kl_loss.kl_loss")
    if kl_penalty not in ("kl", "k1", "mse", "k2", "low_var_kl", "k3"):
        raise NotImplementedError(f"Unsupported KL penalty: {kl_penalty}")
    if logprob.shape != ref_logprob.shape:
        raise ValueError("logprob and ref_logprob must have the same shape")
    if not logprob.is_floating_point() or not ref_logprob.is_floating_point():
        raise ValueError("logprob and ref_logprob must be floating-point tensors")
    if logprob.device != ref_logprob.device:
        raise ValueError("logprob and ref_logprob must be on the same device")

    # 在求差、平方或 exp 之前升精度；exp(20) 已超过 float16 的有限范围。
    if logprob.dtype in (torch.float16, torch.bfloat16):
        logprob = logprob.float()
    if ref_logprob.dtype in (torch.float16, torch.bfloat16):
        ref_logprob = ref_logprob.float()

    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # k3：概率比使用 ref/current，注意与 PPO 的 current/old 概率比区分。
    kl = ref_logprob - logprob
    kl = torch.clamp(kl, min=-20, max=20)
    ratio = torch.exp(kl)
    kld = ratio - kl - 1
    return torch.clamp(kld, min=-10, max=10)


if __name__ == "__main__":
    logprob = torch.tensor([[-0.2, -1.0, -0.5], [-0.7, -0.3, 0.0]], requires_grad=True)
    ref_logprob = torch.tensor([[-0.4, -0.8, -0.9], [-0.5, -0.6, 0.0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    kl_token = grpo_kl_loss(logprob, ref_logprob)
    # 示例采用 token-mean；训练时将 kl_loss_coef * loss 加到 GRPO 策略损失中。
    loss = (kl_token * mask).sum() / mask.sum().clamp_min(1)
    loss.backward()
    print(f"per-token KL: {kl_token.detach()}")
    print(f"masked KL loss: {loss.item():.6f}")
