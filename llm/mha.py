import math
import torch
import torch.nn as nn


class MHA(nn.Module):
    """多头自注意力：各 head 独立计算 softmax(QKᵀ / sqrt(Dh))V，再合并。"""

    def __init__(self,hidden_size,num_heads):
        super().__init__()
        # 将隐藏维度 D 均分到 H 个 head，每个 head 的维度为 Dh = D/H
        assert hidden_size%num_heads==0
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.head_dim=hidden_size//num_heads
        
        self.q_proj=nn.Linear(hidden_size,hidden_size)
        self.k_proj=nn.Linear(hidden_size,hidden_size)
        self.v_proj=nn.Linear(hidden_size,hidden_size)
        
        self.o_proj=nn.Linear(hidden_size,hidden_size)
    
    def forward(self,x,causal=False,past_k=None,past_v=None):
        """
        x: [B, L, D]；past_k、past_v: [B, H, S, Dh]，缓存需成对提供。
        返回 out: [B, L, D]，以及包含当前 token 的新 K/V 缓存 [B, H, S+L, Dh]。
        当前带缓存的因果推理分支适用于单 token 续算，多 token 续算见下方 TODO。
        """
        B,L,D=x.shape
        H=self.num_heads
        Dh=self.head_dim
        
        # 同一输入分别投影为查询 Q、键 K、值 V
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        
        # [B, L, D] -> [B, H, L, Dh]，让各 head 并行计算注意力
        q=q.view(B,L,H,Dh).transpose(1,2)
        k=k.view(B,L,H,Dh).transpose(1,2)
        v=v.view(B,L,H,Dh).transpose(1,2)
        
        if past_k is not None:
            # 只追加新 token 的 K/V，历史 token 无需再次做线性投影
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        new_k = k
        new_v = v
        # scores: [B, H, L, S+L]；除以 sqrt(Dh) 缓和点积随维度增大的幅度
        scores=torch.matmul(q,k.transpose(-1,-2))
        scores=scores/math.sqrt(Dh)
        
        # 无缓存时，下三角掩码让每个 token 只能访问自身及之前的位置
        # TODO: 有缓存且 L>1 时，还需按缓存长度 S 偏移构造因果掩码
        if causal and past_k is None:
            mask=torch.tril(torch.ones(L,L,device=x.device,dtype=torch.bool))
            scores=scores.masked_fill(~mask,float("-inf"))
        
        # 对 key 维度做数值稳定的 softmax，被遮罩的 -inf 位置得到 0 权重
        scores_max=scores.max(dim=-1,keepdim=True).values
        scores_exp=torch.exp(scores-scores_max)
        attn=scores_exp/scores_exp.sum(dim=-1,keepdim=True)
        
        # 对 V 加权求和，再拼接各 head；transpose 后先 contiguous 再 view
        out=torch.matmul(attn,v)
        out=out.transpose(1,2).contiguous().view(B,L,D)
        
        out=self.o_proj(out)
        
        return out,new_k,new_v
