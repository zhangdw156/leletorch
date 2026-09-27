import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """均方根归一化：γ * x / sqrt(mean(x²) + eps)，不减均值、不加偏置。"""

    def __init__(self,hidden_size,eps=1e-6):
        super().__init__()
        
        # 每个隐藏特征一个可学习的缩放系数 γ
        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.eps=eps
    
    def forward(self,x):
        """x: [..., hidden_size]，仅沿最后一维归一化，输出类型与输入一致。"""
        input_dtype=x.dtype
        # 半精度先升到 float32，避免平方溢出；float64 输入保持原有精度
        if x.dtype in (torch.float16,torch.bfloat16):
            x=x.float()
        # 变量 variance 存的是均方 E[x²]，不是减去均值后的方差
        variance=x.pow(2).mean(-1,keepdim=True)
        # rsqrt(z) = 1/sqrt(z)，eps 避免全零输入导致除零
        x=x*torch.rsqrt(variance+self.eps)
        return (self.weight*x).to(input_dtype)
