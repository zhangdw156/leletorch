import torch.nn as nn


class FFN(nn.Module):
    """逐 token 的前馈网络：Linear -> GELU -> Linear。"""

    def __init__(self, hidden_size, intermediate_size=None, bias=True):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        # x: [..., hidden_size]，只变换最后一维
        hidden = self.activation(self.up_proj(x))
        return self.down_proj(hidden)
