import torch

def softmax(x,dim=-1):
    x_max=x.max(dim=dim,keepdim=True).values
    x_exp=torch.exp(x-x_max)
    return x_exp*torch.rsqrt(x_exp.sum(dim=dim,keepdim=True))