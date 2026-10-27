import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络：down_proj(SiLU(gate_proj(x)) * up_proj(x))。"""

    def __init__(self, hidden_size, intermediate_size=None, bias=False):
        super().__init__()
        if intermediate_size is None:
            # 三个投影，取约 8D/3 的中间维度以匹配普通 4D FFN 的参数量
            intermediate_size = 8 * hidden_size // 3

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        # x: [..., hidden_size]
        gate = self.gate_proj(x)
        gate = gate * torch.sigmoid(gate)
        hidden = gate * self.up_proj(x)
        return self.down_proj(hidden)
