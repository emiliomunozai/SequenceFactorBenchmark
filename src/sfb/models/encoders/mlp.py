"""MLP encoder that compresses the full sequence into a fixed bottleneck."""

from torch import nn

from sfb.models.bottleneck import EncoderOutput, SequenceEncoder
from sfb.registry import register_encoder


@register_encoder(
    "mlp",
    constructor_params=["d_model", "bottleneck_dim"],
)
class MLPSequenceEncoder(SequenceEncoder):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        bottleneck_dim: int,
    ):
        super().__init__(vocab_size, seq_len)
        self.embed = nn.Embedding(vocab_size, d_model)
        hidden_dim = max(d_model, bottleneck_dim * 2)
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.to_bottleneck = nn.Linear(seq_len * bottleneck_dim, bottleneck_dim)
        self.out_dim = bottleneck_dim

    def encode(self, x):
        batch_size = x.size(0)
        token_features = self.token_mlp(self.embed(x))
        z = self.to_bottleneck(token_features.reshape(batch_size, -1))
        return EncoderOutput(z=z)
