"""Transformer over the full sequence, mean-pool, project to ``d_bottleneck``."""

import math

import torch
import torch.nn as nn

from sfb.models.codec import EncoderOutput, SequenceEncoder


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
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


class TransformerSequenceEncoder(SequenceEncoder):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        d_bottleneck: int,
        n_layers: int,
        n_heads: int,
    ):
        super().__init__(vocab_size, seq_len)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = SinusoidalPE(d_model, max_len=seq_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.to_bottleneck = nn.Linear(d_model, d_bottleneck)
        self.out_dim = d_bottleneck

    def encode(self, x):
        h = self.pe(self.embed(x))
        for block in self.blocks:
            h = block(h)
        h = self.norm(h).mean(dim=1)
        z = self.to_bottleneck(h)
        return EncoderOutput(z=z)
