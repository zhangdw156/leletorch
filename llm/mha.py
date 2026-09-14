import math
import torch
import torch.nn as nn


class MHA(nn.Module):
    def __init__(self,hidden_size,num_heads):
        super().__init__()
        assert hidden_size%num_heads==0
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.head_dim=hidden_size//num_heads
        
        self.q_proj=nn.Linear(hidden_size,hidden_size)
        self.k_proj=nn.Linear(hidden_size,hidden_size)
        self.v_proj=nn.Linear(hidden_size,hidden_size)
        
        self.o_proj=nn.Linear(hidden_size,hidden_size)
    
    def forward(self,x,causal=False,past_k=None,past_v=None):
        B,L,D=x.shape
        H=self.num_heads
        Dh=self.head_dim
        
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        
        q=q.view(B,L,H,Dh).transpose(1,2)
        k=k.view(B,L,H,Dh).transpose(1,2)
        v=v.view(B,L,H,Dh).transpose(1,2)
        
        if past_k is not None:
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        new_k = k
        new_v = v
        scores=torch.matmul(q,k.transpose(-1,-2))
        scores=scores/math.sqrt(Dh)
        
        if causal and past_k is None:
            mask=torch.tril(torch.ones(L,L,device=x.device,dtype=torch.bool))
            scores=scores.masked_fill(~mask,float("-inf"))
        
        scores_max=scores.max(dim=-1,keepdim=True).values
        scores_exp=torch.exp(scores-scores_max)
        attn=scores_exp/scores_exp.sum(dim=-1,keepdim=True)
        
        out=torch.matmul(attn,v)
        out=out.transpose(1,2).contiguous().view(B,L,D)
        
        out=self.o_proj(out)
        
        return out,new_k,new_v