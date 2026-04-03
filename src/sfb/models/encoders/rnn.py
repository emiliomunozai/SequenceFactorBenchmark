"""Vanilla RNN encoder with a fixed bottleneck projection."""

from torch import nn

from sfb.models.bottleneck import EncoderOutput, SequenceEncoder
from sfb.registry import register_encoder


@register_encoder(
    "rnn",
    constructor_params=["d_model", "bottleneck_dim", "n_layers"],
    param_defaults={"n_layers": 1},
)
class RNNSequenceEncoder(SequenceEncoder):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        bottleneck_dim: int,
        n_layers: int,
    ):
        super().__init__(vocab_size, seq_len)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.RNN(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.to_bottleneck = nn.Linear(d_model, bottleneck_dim)
        self.out_dim = bottleneck_dim

    def encode(self, x):
        x = self.embed(x)
        _, h_n = self.rnn(x)
        z = self.to_bottleneck(h_n[-1])
        return EncoderOutput(z=z)
