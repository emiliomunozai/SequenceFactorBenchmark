"""LSTM decoder conditioned on the fixed bottleneck."""

import torch
import torch.nn as nn

from sfb.models.bottleneck import SequenceDecoder
from sfb.registry import register_decoder


@register_decoder(
    "lstm",
    constructor_params=["d_model", "n_layers"],
    param_defaults={"n_layers": 1},
)
class LSTMSequenceDecoder(SequenceDecoder):
    """LSTM states initialized from ``z``; learned per-step inputs."""

    def __init__(
        self,
        z_dim: int,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        n_layers: int = 1,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.cell = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.step_in = nn.Parameter(torch.zeros(1, 1, d_model))
        self.init_h = nn.Linear(z_dim, n_layers * d_model)
        self.init_c = nn.Linear(z_dim, n_layers * d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        b = z.size(0)
        h0 = self.init_h(z).view(b, self.n_layers, self.d_model).transpose(0, 1).contiguous()
        c0 = self.init_c(z).view(b, self.n_layers, self.d_model).transpose(0, 1).contiguous()
        inp = self.step_in.expand(b, seq_len, -1)
        out, _ = self.cell(inp, (h0, c0))
        return self.head(out)
