import torch

def softmax(x,dim=-1):
    """
    稳定 softmax，输入为浮点张量，输出形状和类型不变。
    标准公式：exp(x_i - max(x)) / sum_j exp(x_j - max(x))。
    """
    output_dtype=x.dtype
    # 半精度先升到 float32，避免候选数较多时指数和溢出
    if x.dtype in (torch.float16,torch.bfloat16):
        x=x.float()
    # 减去当前维度的最大值，避免指数溢出；keepdim 便于广播
    x_max=x.max(dim=dim,keepdim=True).values
    x_exp=torch.exp(x-x_max)
    # 除以指数和，使该维度上的概率和为 1
    return (x_exp/x_exp.sum(dim=dim,keepdim=True)).to(output_dtype)
