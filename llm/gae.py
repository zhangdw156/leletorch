import torch


@torch.no_grad()
def gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    广义优势估计（Generalized Advantage Estimation）。

    rewards: [..., T]，每一步的奖励
    values:  [..., T+1]，包含轨迹末尾状态的 bootstrap value
    dones:   [..., T]，当前转移是否真正终止；轨迹末尾仅因时间截断时为 False
    返回 advantages, returns，形状均为 [..., T]，不参与梯度计算。
    """
    if rewards.ndim < 1 or dones.shape != rewards.shape:
        raise ValueError("rewards and dones must have shape [..., T]")
    if values.shape != rewards.shape[:-1] + (rewards.shape[-1] + 1,):
        raise ValueError("values must have shape [..., T+1]")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("gamma and lam must be in [0, 1]")

    advantages = torch.zeros_like(values[..., :-1])
    last_advantage = torch.zeros_like(values[..., -1])

    for t in reversed(range(rewards.shape[-1])):
        # 终止状态不 bootstrap，也不把下一条 episode 的优势传回来
        non_terminal = 1.0 - dones[..., t].to(values.dtype)
        delta = (
            rewards[..., t]
            + gamma * values[..., t + 1] * non_terminal
            - values[..., t]
        )
        last_advantage = delta + gamma * lam * non_terminal * last_advantage
        advantages[..., t] = last_advantage

    returns = advantages + values[..., :-1]
    return advantages, returns
