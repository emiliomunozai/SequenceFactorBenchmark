"""
Vanilla Transformer encoder for sequence-to-sequence prediction.
Uses sinusoidal positional encoding and standard multi-head self-attention
with pre-norm residual blocks (GPT-2 style).

No causal mask — every position attends to the full sequence, which is
appropriate for tasks like copy, sorting, and reverse where the entire
input is available.
"""
import math

import torch
import torch.nn as nn

from seqfacben.registry import register_model


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LayerNorm -> MHA -> residual -> LayerNorm -> FFN -> residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


@register_model(
    "transformer",
    display_params=["d_model", "n_layers", "n_heads"],
    constructor_params=["vocab_size", "seq_len", "d_model", "n_layers", "n_heads"],
    param_defaults={"n_layers": 4, "n_heads": 4},
)
class TransformerEncoder(nn.Module):
    """Vanilla Transformer: embedding + sinusoidal PE -> N blocks -> linear head."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = SinusoidalPE(d_model, max_len=seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(self.embed(x))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
