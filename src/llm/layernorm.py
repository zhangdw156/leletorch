import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """沿最后一维做 LayerNorm：γ * (x - mean) / sqrt(var + eps) + β。"""

    def __init__(self,hidden_size,eps=1e-5):
        super().__init__()
        # γ、β 是逐特征的可学习参数，初始时不额外缩放或平移归一化结果
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        self.eps=eps
    
    def forward(self,x):
        """x: [..., hidden_size]，每个位置独立归一化，输出形状不变。"""
        mean=x.mean(dim=-1,keepdim=True)
        
        # 使用总体方差（除以 hidden_size），keepdim 便于与原输入广播
        var=((x-mean)**2).mean(dim=-1,keepdim=True)
        
        # eps 避免特征值相同、方差为 0 时出现除零
        x_norm=(x-mean)/torch.sqrt(var+self.eps)
        
        out=self.weight*x_norm+self.bias
        return out
