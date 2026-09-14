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
    mean=rewards.mean(dim=-1,keepdim=True)
    std=rewards.std(dim=-1, keepdim=True, unbiased=False)
    
    advantage=(rewards-mean)/(std+eps)
    
    advantage=advantage.unsqueeze(-1)  # [B,G,1]
    
    ratio=torch.exp(log_prob-old_log_prob)
    
    obj1=ratio*advantage
    obj2=torch.clamp(ratio,1-clip_low,1+clip_high)*advantage
    
    loss_token=-torch.minimum(obj1,obj2)
    
    loss=(loss_token*token_mask).sum(dim=-1)/token_mask.sum(dim=-1).clamp_min(1)
    loss=loss.mean()

    return loss