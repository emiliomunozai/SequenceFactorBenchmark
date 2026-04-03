"""MLP decoder with shared latent conditioning across all positions."""

import torch
import torch.nn as nn

from sfb.models.bottleneck import SequenceDecoder
from sfb.registry import register_decoder


@register_decoder(
    "mlp",
    constructor_params=["d_model"],
)
class MLPSequenceDecoder(SequenceDecoder):
    def __init__(
        self,
        z_dim: int,
        seq_len: int,
        d_model: int,
        vocab_size: int,
    ):
        super().__init__()
        hidden_dim = max(d_model, z_dim * 2)
        self.from_z = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size),
        )

    def decode(self, z: torch.Tensor, seq_len: int):
        batch_size = z.size(0)
        base = self.from_z(z).unsqueeze(1)
        pos = self.pos_emb[:, :seq_len, :].expand(batch_size, -1, -1)
        return self.head(base + pos)
