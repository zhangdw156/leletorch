import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MHA(nn.Module):
    def __init__(self,hidden_dim,num_heads):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.q_proj=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.k_proj=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.v_proj=nn.Linear(self.hidden_dim,self.hidden_dim)
        self.o_proj=nn.Linear(self.hidden_dim,self.hidden_dim)

    def forward(self,x,causal=False,kv_cache=None):
        B,L,D=x.shape
        H=self.num_heads
        Dh=self.hidden_dim//self.num_heads
        q=self.q_proj(x).view(B,L,H,Dh).transpose(1,2)
        k=self.k_proj(x).view(B,L,H,Dh).transpose(1,2)
        v=self.v_proj(x).view(B,L,H,Dh).transpose(1,2)
        past_len=0
        if kv_cache:
            past_k,past_v=kv_cache
            past_len=past_k.shape[-2]
            k=torch.cat([past_k,k],dim=-2)
            v=torch.cat([past_v,v],dim=-2)
        kv_cache=(k,v)
        attn=q @ k.transpose(-1,-2)/math.sqrt(Dh)
        if causal:
            q_pos=past_len+torch.arange(L,device=x.device)
            k_pos=torch.arange(L+past_len,device=x.device)
            q_pos=q_pos[:,None]
            k_pos=k_pos[None,:]
            mask=k_pos<=q_pos
            attn=attn.masked_fill(~mask,float("-inf"))

        attn=F.softmax(attn,dim=-1)
        scores=attn @ v
        scores=scores.transpose(1,2).contiguous().view(B,L,D)
        out=self.o_proj(scores)
        return out,kv_cache
