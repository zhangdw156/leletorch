import torch

def softmax(x,dim=-1):
    """
    稳定 softmax 的计算步骤，输入/输出形状相同；当前分母实现待修正。
    标准公式：exp(x_i - max(x)) / sum_j exp(x_j - max(x))。
    """
    # 减去当前维度的最大值，避免指数溢出；keepdim 便于广播
    x_max=x.max(dim=dim,keepdim=True).values
    x_exp=torch.exp(x-x_max)
    # TODO: 标准 softmax 应除以指数和；rsqrt 除以的是平方根，不能保证概率和为 1
    return x_exp*torch.rsqrt(x_exp.sum(dim=dim,keepdim=True))
