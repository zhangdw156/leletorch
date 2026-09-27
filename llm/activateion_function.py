"""常见激活函数的 PyTorch 实现，输入为浮点张量，默认保持输入形状。"""

import math

import torch


def sigmoid(x):
    """Sigmoid(x) = 1 / (1 + exp(-x))，输出范围 (0, 1)。"""
    # 分正负计算，使 exp 的输入始终非正，避免大负数导致溢出
    positive = x >= 0
    z = torch.exp(torch.where(positive, -x, x))
    return torch.where(positive, 1.0 / (1.0 + z), z / (1.0 + z))


def logsigmoid(x):
    """log(Sigmoid(x)) = -Softplus(-x)，避免先求 sigmoid 再取 log 的下溢。"""
    return -softplus(-x)


def tanh(x):
    """Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))，范围 (-1, 1)。"""
    # 使用稳定的双曲正切原语，避免直接计算指数的溢出与相减误差
    return torch.tanh(x)


def relu(x):
    """ReLU(x) = max(0, x)，保持输入形状不变。"""
    return torch.where(x <= 0, 0, x)


def leaky_relu(x, negative_slope=0.01):
    """正半轴为 x，负半轴为 negative_slope * x，保留负半轴梯度。"""
    return torch.where(x > 0, x, negative_slope * x)


def prelu(x, weight):
    """
    带可学习负斜率的 ReLU；weight 可传入 torch.nn.Parameter。
    weight 为标量或 [1] 时共享斜率；为 [C] 时沿 x 的通道维（第 1 维）广播。
    逐通道输入形状为 [B, C, ...]，与 torch.nn.PReLU 一致。
    """
    if weight.ndim > 1 or weight.numel() == 0:
        raise ValueError("weight must be a scalar or a non-empty vector")
    if weight.numel() == 1:
        weight = weight.reshape(())
    elif x.ndim < 2 or weight.numel() != x.shape[1]:
        raise ValueError("weight must contain one value or one value per channel")
    else:
        weight = weight.reshape((1, -1) + (1,) * (x.ndim - 2))
    return torch.where(x > 0, x, weight * x)


def hardtanh(x, min_val=-1.0, max_val=1.0):
    """将 x 截断到 [min_val, max_val]，端点处梯度取 0。"""
    if min_val >= max_val:
        raise ValueError("min_val must be smaller than max_val")
    return torch.where(x <= min_val, min_val, torch.where(x >= max_val, max_val, x))


def relu6(x):
    """ReLU6(x) = min(max(0, x), 6)，常用于移动端网络。"""
    return hardtanh(x, 0.0, 6.0)


def elu(x, alpha=1.0):
    """ELU：正半轴为 x，负半轴为 alpha * (exp(x) - 1)。"""
    # 不对正半轴求指数，避免未选中的分支溢出并污染反向传播
    negative_x = torch.where(x > 0, 0, x)
    return torch.where(x > 0, x, alpha * torch.expm1(negative_x))


def celu(x, alpha=1.0):
    """CELU：正半轴为 x，负半轴为 alpha * (exp(x / alpha) - 1)，一阶连续。"""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    negative_x = torch.where(x > 0, 0, x)
    return torch.where(x > 0, x, alpha * torch.expm1(negative_x / alpha))


def selu(x):
    """SELU(x) = scale * ELU(x, alpha)，使用自归一化网络中的固定常数。"""
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return scale * elu(x, alpha=alpha)


def softplus(x, beta=1.0, threshold=20.0):
    """Softplus(x) = log(1 + exp(beta*x)) / beta，是 ReLU 的平滑近似。"""
    if beta <= 0:
        raise ValueError("beta must be positive")
    scaled = beta * x
    # logaddexp(0, z) 稳定计算 log(1 + exp(z))；超过阈值时使用线性近似
    smooth = torch.logaddexp(torch.zeros_like(scaled), scaled) / beta
    return torch.where(scaled > threshold, x, smooth)


def softsign(x):
    """Softsign(x) = x / (1 + |x|)，输出范围 (-1, 1)。"""
    return x / (1.0 + torch.abs(x))


def gelu(x, approximate="none"):
    """GELU，支持精确计算和 tanh 近似。"""
    if approximate not in ("none", "tanh"):
        raise ValueError("approximate must be 'none' or 'tanh'")
    output_dtype = x.dtype
    # 半精度下用 float32 计算，避免三次项溢出后产生 NaN 梯度
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    if approximate == "tanh":
        out = 0.5 * x * (
            1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))
        )
    else:
        # GELU(x) = x * Φ(x)，Φ 是标准正态分布的累积分布函数
        out = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return out.to(output_dtype)


def silu(x):
    """SiLU(x) = x * sigmoid(x)，也叫 Swish（β=1）。"""
    return swish(x)


def swish(x, beta=1.0):
    """Swish(x) = x * sigmoid(beta*x)，beta=1 时等价于 SiLU。"""
    return x * sigmoid(beta * x)


def mish(x):
    """Mish(x) = x * tanh(Softplus(x))，平滑且非单调。"""
    return x * torch.tanh(softplus(x))


def hardsigmoid(x):
    """HardSigmoid(x) = ReLU6(x + 3) / 6，sigmoid 的分段线性近似。"""
    return relu6(x + 3.0) / 6.0


def hardswish(x):
    """HardSwish(x) = x * ReLU6(x + 3) / 6，SiLU 的分段近似。"""
    return x * hardsigmoid(x)


def softmax(x, dim=-1):
    """沿 dim 将 logits 转成概率，输出非负且在该维度上的和为 1。"""
    output_dtype = x.dtype
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    shifted = x - x.max(dim=dim, keepdim=True).values
    weights = torch.exp(shifted)
    return (weights / weights.sum(dim=dim, keepdim=True)).to(output_dtype)


def log_softmax(x, dim=-1):
    """稳定计算 log(Softmax(x))，避免先求 softmax 再取 log 的下溢。"""
    output_dtype = x.dtype
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    shifted = x - x.max(dim=dim, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
    return (shifted - log_sum_exp).to(output_dtype)


def softmin(x, dim=-1):
    """Softmin(x) = Softmax(-x)，较小的输入获得较大的概率。"""
    return softmax(-x, dim=dim)


def _split_gate(x, dim):
    if x.ndim == 0 or not -x.ndim <= dim < x.ndim:
        raise ValueError("dim must select an existing dimension")
    if x.shape[dim] % 2 != 0:
        raise ValueError("the gated dimension must have an even size")
    return torch.chunk(x, 2, dim=dim)


def glu(x, dim=-1):
    """沿 dim 将 x 分为 a、b，返回 a * sigmoid(b)，该维度大小减半。"""
    a, b = _split_gate(x, dim)
    return a * sigmoid(b)


def geglu(x, dim=-1, approximate="none"):
    """GEGLU：返回 a * GELU(b)；输入已拼接内容分支 a 和门控分支 b。"""
    a, b = _split_gate(x, dim)
    return a * gelu(b, approximate=approximate)


def swiglu(x, dim=-1):
    """SwiGLU：返回 a * SiLU(b)；只计算门控激活，不包含线性投影。"""
    a, b = _split_gate(x, dim)
    return a * silu(b)
