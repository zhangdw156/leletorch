import torch

def grpo_loss(
    log_prob,  # [B,G,L]
    old_log_prob,  # [B,G,L]
    rewards,  # [B,G]
    token_mask,  # [B,G,L]
    clip_low=0.2,
    clip_high=0.28,
    eps=1e-8
):
    """
    GRPO 的裁剪策略损失，此处不包含 KL 正则项。
    B 为提示数量，G 为每个提示采样的回复数，L 为回复的 token 数。
    token_mask 中 1 表示有效 token，0 表示 padding 等不计入损失的位置。
    调用方应将 old_log_prob 和 rewards 从计算图中分离，作为固定的参考值。
    """
    # 对同一提示下的 G 个奖励做组内标准化；unbiased=False 使用总体标准差
    mean=rewards.mean(dim=-1,keepdim=True)
    std=rewards.std(dim=-1, keepdim=True, unbiased=False)
    
    advantage=(rewards-mean)/(std+eps)
    
    # 每个回复只有一个优势值，扩维后广播到该回复的全部 token
    advantage=advantage.unsqueeze(-1)  # [B,G,1]
    
    # 新旧策略概率比：π_new / π_old = exp(log π_new - log π_old)
    ratio=torch.exp(log_prob-old_log_prob)
    
    # 比较原始目标与裁剪目标，限制单次更新幅度；上下界允许不对称
    obj1=ratio*advantage
    obj2=torch.clamp(ratio,1-clip_low,1+clip_high)*advantage
    
    # 取负号，将最大化策略目标转成最小化损失
    loss_token=-torch.minimum(obj1,obj2)
    
    # 先按每条回复的有效长度平均，再对 B、G 平均；clamp_min 避免空掩码除零
    loss=(loss_token*token_mask).sum(dim=-1)/token_mask.sum(dim=-1).clamp_min(1)
    loss=loss.mean()

    return loss
