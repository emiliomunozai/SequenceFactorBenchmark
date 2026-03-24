"""Map ``z`` to per-step logits without a giant ``Linear(z_dim, seq_len * d_model)``."""

import torch
import torch.nn as nn

from sfb.models.codec import SequenceDecoder


class LinearSequenceDecoder(SequenceDecoder):
    """Project ``z`` to width ``d_model``, add learned position embeddings, then LM head.

    Parameter count is ``O(z_dim * d_model + seq_len * d_model + d_model * vocab_size)``
    instead of ``O(z_dim * seq_len * d_model)`` from a single full-sequence expansion matrix.
    """

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
        self.from_z = nn.Sequential(
            nn.Linear(z_dim, h),
            nn.GELU(),
            nn.Linear(h, d_model),
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.head = nn.Linear(d_model, vocab_size)

    def decode(self, z, seq_len: int):
        b = z.size(0)
        base = self.from_z(z).unsqueeze(1)
        pos = self.pos_emb[:, :seq_len, :].expand(b, -1, -1)
        return self.head(base + pos)
