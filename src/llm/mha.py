import math
import torch
import torch.nn as nn

if __package__:
    from .rope import apply_rope
    from .softmax import softmax
else:
    from rope import apply_rope
    from softmax import softmax


class MHA(nn.Module):
    """多头自注意力：各 head 独立计算 softmax(QKᵀ / sqrt(Dh))V，再合并。"""

    def __init__(self,hidden_size,num_heads,use_rope=False):
        super().__init__()
        # 将隐藏维度 D 均分到 H 个 head，每个 head 的维度为 Dh = D/H
        if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.head_dim=hidden_size//num_heads
        self.use_rope=use_rope
        if use_rope and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        
        self.q_proj=nn.Linear(hidden_size,hidden_size)
        self.k_proj=nn.Linear(hidden_size,hidden_size)
        self.v_proj=nn.Linear(hidden_size,hidden_size)
        
        self.o_proj=nn.Linear(hidden_size,hidden_size)
    
    def forward(self,x,causal=False,past_k=None,past_v=None,position_ids=None):
        """
        x: [B, L, D]；past_k、past_v: [B, H, S, Dh]，缓存需成对提供。
        返回 out: [B, L, D]，以及包含当前 token 的新 K/V 缓存 [B, H, S+L, Dh]。
        因果遮罩按缓存长度偏移，支持单 token 和多 token 分块续算。
        use_rope=True 时旋转新 Q/K，position_ids: [B, L] 默认从缓存长度开始。
        缓存中的 K 已完成旋转；每次调用需沿用同一实例和位置编码设置。
        """
        if x.ndim != 3 or x.shape[-1] != self.hidden_size or x.shape[1] == 0:
            raise ValueError("x must have shape [B, L, hidden_size] with L > 0")
        B,L,D=x.shape
        H=self.num_heads
        Dh=self.head_dim
        if (past_k is None) != (past_v is None):
            raise ValueError("past_k and past_v must be provided together")
        past_len=0
        if past_k is not None:
            if (past_k.ndim != 4 or past_k.shape != past_v.shape
                    or past_k.shape[:2] != (B,H) or past_k.shape[-1] != Dh):
                raise ValueError("past_k and past_v must have shape [B, H, S, Dh]")
            if past_k.device != x.device or past_v.device != x.device:
                raise ValueError("KV caches and x must be on the same device")
            past_len=past_k.shape[-2]
        
        # 同一输入分别投影为查询 Q、键 K、值 V
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        
        # [B, L, D] -> [B, H, L, Dh]，让各 head 并行计算注意力
        q=q.view(B,L,H,Dh).transpose(1,2)
        k=k.view(B,L,H,Dh).transpose(1,2)
        v=v.view(B,L,H,Dh).transpose(1,2)

        if self.use_rope:
            if position_ids is None:
                position_ids=torch.arange(past_len,past_len+L,device=x.device)
                position_ids=position_ids.unsqueeze(0).expand(B,-1)
            elif position_ids.shape != (B,L) or position_ids.device != x.device:
                raise ValueError("position_ids must have shape [B, L] on x.device")
            # 先旋转当前 Q/K，再追加历史 K；不能重复旋转缓存。
            q_dtype,k_dtype=q.dtype,k.dtype
            q,k=apply_rope(q,k,position_ids=position_ids)
            q,k=q.to(q_dtype),k.to(k_dtype)
        elif position_ids is not None:
            raise ValueError("position_ids requires use_rope=True")
        
        if past_k is not None:
            if past_k.dtype != k.dtype or past_v.dtype != v.dtype:
                raise ValueError("KV cache dtypes must match the current projected K/V")
            # 只追加新 token 的 K/V，历史 token 无需再次做线性投影
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        new_k = k
        new_v = v
        # scores: [B, H, L, S+L]；除以 sqrt(Dh) 缓和点积随维度增大的幅度
        scores=torch.matmul(q,k.transpose(-1,-2))
        scores=scores/math.sqrt(Dh)
        
        # 第 i 个新 token 的绝对位置是 S+i，只能访问 key 位置 <= S+i
        if causal:
            q_pos=past_len+torch.arange(L,device=x.device)
            k_pos=torch.arange(k.shape[-2],device=x.device)
            mask=k_pos[None,:]<=q_pos[:,None]
            scores=scores.masked_fill(~mask,float("-inf"))
        
        # 对 key 维度做数值稳定的 softmax，被遮罩的 -inf 位置得到 0 权重
        attn=softmax(scores,dim=-1)
        
        # 对 V 加权求和，再拼接各 head；transpose 后先 contiguous 再 view
        out=torch.matmul(attn,v)
        out=out.transpose(1,2).contiguous().view(B,L,D)
        
        out=self.o_proj(out)
        
        return out,new_k,new_v
