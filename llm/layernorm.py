import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        self.eps=eps
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        
        var=((x-mean)**2).mean(dim=-1,keepdim=True)
        
        x_norm=(x-mean)/torch.sqrt(var+self.eps)
        
        out=self.weight*x_norm+self.bias
        return out
        