import torch

def apply_rope(
    q,
    k,
    position_ids=None,
    base=10000.0,
    rotary_dim=None,
    scaling_factor=1.0
):
    B,_,L,D=q.shape
    if rotary_dim is None:
        rotary_dim=D
    assert rotary_dim%2==0
    
    device=q.device
    
    i=torch.arange(rotary_dim//2,device=device,dtype=torch.float32)
    inv_freq=1.0/(base**(2*i/rotary_dim))
    
    position_ids=position_ids/scaling_factor
    
    angles=position_ids[:,:,None]*inv_freq[None,None,:]
    
    cos=torch.cos(angles)[:,None,:,:]
    sin=torch.sin(angles)[:,None,:,:]
    
    q_rope,q_pass=q[...,:rotary_dim],q[...,rotary_dim:]
    k_rope,k_pass=k[...,:rotary_dim],k[...,rotary_dim:]
    
    q_even,q_odd=q_rope[...,0::2],q_rope[...,1::2]
    k_even,k_odd=k_rope[...,0::2],k_rope[...,1::2]
    
    q_even_rot=q_even*cos-q_odd*sin
    q_odd_rot=q_even*sin+q_odd*cos
    
    k_even_rot=k_even*cos-k_odd*sin
    k_odd_rot=k_even*sin+k_odd*cos
    
    q_rope=torch.stack([q_even_rot,q_odd_rot],dim=-1).flatten(-2)
    k_rope=torch.stack([k_even_rot,k_odd_rot],dim=-1).flatten(-2)
    
    q=torch.cat([q_rope,q_pass],dim=-1)
    k=torch.cat([k_rope,k_pass],dim=-1)
    
    return q,k
    