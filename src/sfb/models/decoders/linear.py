"""MLP expansion from bottleneck to per-step logits."""

import torch
import torch.nn as nn

from sfb.models.codec import SequenceDecoder


class LinearSequenceDecoder(SequenceDecoder):
    """Expand ``z`` to a full sequence with an MLP, then predict vocab per step."""

    def __init__(
        self,
        z_dim: int,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        hidden_mult: int = 2,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        h = max(d_model * hidden_mult, z_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim, h),
            nn.GELU(),
            nn.Linear(h, seq_len * d_model),
        )
        self.head = nn.Linear(d_model, vocab_size)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        b = z.size(0)
        h = self.net(z).view(b, seq_len, self.d_model)
        return self.head(h)
