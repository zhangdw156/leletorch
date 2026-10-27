"""用已有组件组装解码器式 Transformer，可直接运行本文件查看训练和缓存示例。"""

import torch
import torch.nn as nn

if __package__:
    from .cross_entropy import cross_entropy
    from .mha import MHA
    from .rmsnorm import RMSNorm
    from .swiglu import SwiGLU
else:
    from cross_entropy import cross_entropy
    from mha import MHA
    from rmsnorm import RMSNorm
    from swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-Norm 残差块：x += Attention(RMSNorm(x))；x += SwiGLU(RMSNorm(x))。"""

    def __init__(self, hidden_size, num_heads, intermediate_size=None, norm_eps=1e-6):
        super().__init__()
        if intermediate_size is not None and intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if not 0 < norm_eps < float("inf"):
            raise ValueError("norm_eps must be finite and positive")
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MHA(hidden_size, num_heads, use_rope=True)
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x, past_k=None, past_v=None, position_ids=None):
        """x: [B, L, D]；返回隐藏状态及本层更新后的 K/V 缓存。"""
        attn_out, new_k, new_v = self.attn(
            self.attn_norm(x), causal=True, past_k=past_k, past_v=past_v,
            position_ids=position_ids,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_k, new_v


class Transformer(nn.Module):
    """
    因果语言模型：Embedding -> N 个 TransformerBlock -> RMSNorm -> LM Head。

    每层复用 MHA + RoPE + RMSNorm + SwiGLU；head_dim 必须为偶数。
    输入为未 padding 的等长 token 序列，RoPE 提供位置信息，无需位置嵌入表。
    本示例不包含 dropout、tokenizer 或 attention padding mask。
    """

    def __init__(
        self, vocab_size, hidden_size=256, num_layers=4, num_heads=8,
        intermediate_size=None, norm_eps=1e-6,
    ):
        super().__init__()
        if vocab_size <= 0 or num_layers <= 0:
            raise ValueError("vocab_size and num_layers must be positive")
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, norm_eps)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, labels=None, past_key_values=None, use_cache=False):
        """
        input_ids: [B, L] 的 torch.long token ID；带缓存时只传新 token。
        labels: 可选的 [B, L]，与 input_ids 对齐，内部以 logits[:, :-1]
            预测 labels[:, 1:]；通常传 input_ids，自行屏蔽 prompt 标签为 -100。
            -100 只忽略损失，不屏蔽注意力；labels 不与历史缓存同时使用。
        past_key_values: 每层一个 (K, V)，各为 [B, H, S, Dh]，所有层 S 相同。
            位置从 S 自动接续；缓存仅适用于同一模型、同一 batch 的连续前缀。
        返回 (logits, loss, new_past_key_values)：logits 为 [B, L, vocab_size]，
            未传 labels 时 loss=None；use_cache=False 时第三项为 None。
            无有效预测标签（含 L=1）时 loss 为可反向传播的 0。
        推理请配合 eval() 和 torch.no_grad()，避免缓存保留历史计算图。
        """
        if (input_ids.ndim != 2 or input_ids.dtype != torch.long
                or input_ids.shape[0] == 0 or input_ids.shape[1] == 0):
            raise ValueError("input_ids must have shape [B, L] with B, L > 0 and dtype long")
        if labels is not None:
            if (labels.shape != input_ids.shape or labels.dtype != torch.long
                    or labels.device != input_ids.device):
                raise ValueError("labels must match input_ids shape, dtype, and device")
            if past_key_values is not None:
                raise ValueError("labels cannot be used with past_key_values")

        batch_size, seq_len = input_ids.shape
        past_len = 0
        if past_key_values is not None:
            if len(past_key_values) != self.num_layers:
                raise ValueError("past_key_values must contain one (K, V) pair per layer")
            for index, pair in enumerate(past_key_values):
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("each cache entry must be a (K, V) pair")
                key, value = pair
                if (not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor)
                        or key.ndim != 4 or key.shape != value.shape):
                    raise ValueError("cache K/V must be tensors of the same shape [B, H, S, Dh]")
                if index == 0:
                    past_len = key.shape[-2]
                attn = self.layers[index].attn
                if key.shape != (batch_size, attn.num_heads, past_len, attn.head_dim):
                    raise ValueError("all layers must have matching cache batch, heads, length, and head_dim")

        # 所有层共享当前绝对位置；过去的 key 已编码，无需重新计算。
        position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        x = self.embed_tokens(input_ids)
        new_past_key_values = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            past_k, past_v = (None, None) if past_key_values is None else past_key_values[index]
            x, new_k, new_v = layer(x, past_k, past_v, position_ids)
            if use_cache:
                new_past_key_values.append((new_k, new_v))
        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            # 展平有效的下一 token 预测，复用已有的二维分类交叉熵。
            shift_logits = logits[:, :-1].reshape(-1, self.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            keep = shift_labels != -100
            if keep.any():
                loss_logits = shift_logits[keep]
                if loss_logits.dtype in (torch.float16, torch.bfloat16):
                    loss_logits = loss_logits.float()
                loss = cross_entropy(loss_logits, shift_labels[keep])
            else:
                # 对空切片求和得到带计算图的 0，也避免半精度先求和再乘 0 溢出。
                loss = logits.reshape(-1)[:0].sum()

        cache = tuple(new_past_key_values) if use_cache else None
        return logits, loss, cache


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Transformer(vocab_size=128, hidden_size=32, num_layers=2, num_heads=4)
    tokens = torch.randint(0, 128, (2, 6))

    # 训练：第 t 个位置预测第 t+1 个 token。
    logits, loss, _ = model(tokens, labels=tokens)
    loss.backward()
    print(f"logits: {tuple(logits.shape)}, loss: {loss.item():.4f}")

    # 推理：先处理前缀，再一次追加多个新 token，与整段计算比较。
    model.eval()
    with torch.no_grad():
        full_logits, _, _ = model(tokens)
        _, _, cache = model(tokens[:, :3], use_cache=True)
        suffix_logits, _, cache = model(tokens[:, 3:], past_key_values=cache, use_cache=True)
        torch.testing.assert_close(suffix_logits, full_logits[:, 3:])
        print(f"cache layers: {len(cache)}, K shape: {tuple(cache[0][0].shape)}")
