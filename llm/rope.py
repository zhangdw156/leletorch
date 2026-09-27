import torch

def apply_rope(
    q,
    k,
    position_ids=None,
    base=10000.0,
    rotary_dim=None,
    scaling_factor=1.0
):
    """
    旋转位置编码（RoPE）：对 Q/K 的相邻特征对做二维旋转，输出形状不变。
    q: [B, Hq, L, D]，k: [B, Hk, L, D]，position_ids: [B, L]。
    未提供 position_ids 时使用 0 到 L-1；缓存续算时应显式传入带偏移的位置。
    rotary_dim 为参与旋转的前缀维数，应为不超过 D 的正偶数；其余维度直接保留。
    """
    B,_,L,D=q.shape
    if rotary_dim is None:
        rotary_dim=D
    assert rotary_dim%2==0
    
    device=q.device
    
    # 第 i 对特征的旋转频率 θ_i = base^(-2i/rotary_dim)
    i=torch.arange(rotary_dim//2,device=device,dtype=torch.float32)
    inv_freq=1.0/(base**(2*i/rotary_dim))
    
    # 默认所有 batch 使用相同的序列位置，位置张量与 q 位于同一设备
    if position_ids is None:
        position_ids=torch.arange(L,device=device).unsqueeze(0).expand(B,-1)
    # 位置除以 scaling_factor，实现线性位置缩放
    position_ids=position_ids/scaling_factor
    
    # 每个位置乘以各频率，得到角度 [B, L, rotary_dim/2]
    angles=position_ids[:,:,None]*inv_freq[None,None,:]
    
    # 插入 head 维，cos/sin 可广播到所有 Q/K head
    cos=torch.cos(angles)[:,None,:,:]
    sin=torch.sin(angles)[:,None,:,:]
    
    # 支持部分旋转：只变换前 rotary_dim 个特征
    q_rope,q_pass=q[...,:rotary_dim],q[...,rotary_dim:]
    k_rope,k_pass=k[...,:rotary_dim],k[...,rotary_dim:]
    
    # 将相邻的偶数、奇数下标特征组成二维向量
    q_even,q_odd=q_rope[...,0::2],q_rope[...,1::2]
    k_even,k_odd=k_rope[...,0::2],k_rope[...,1::2]
    
    # [x', y'] = [x*cosθ - y*sinθ, x*sinθ + y*cosθ]
    q_even_rot=q_even*cos-q_odd*sin
    q_odd_rot=q_even*sin+q_odd*cos
    
    k_even_rot=k_even*cos-k_odd*sin
    k_odd_rot=k_even*sin+k_odd*cos
    
    # 交错还原偶/奇特征的顺序，再拼回不参与旋转的后缀
    q_rope=torch.stack([q_even_rot,q_odd_rot],dim=-1).flatten(-2)
    k_rope=torch.stack([k_even_rot,k_odd_rot],dim=-1).flatten(-2)
    
    q=torch.cat([q_rope,q_pass],dim=-1)
    k=torch.cat([k_rope,k_pass],dim=-1)
    
    return q,k
