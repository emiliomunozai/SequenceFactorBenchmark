"""
Autoencoder: compresses the embedded sequence through a bottleneck, then
reconstructs per-position logits.  Tests whether the model can learn a
compact internal representation sufficient for the target task.
"""
from torch import nn
from seqfacben.registry import register_model


@register_model(
    "autoencoder",
    display_params=["d_model", "n_layers", "bottleneck_dim"],
    constructor_params=["vocab_size", "seq_len", "d_model", "n_layers", "bottleneck_dim"],
    param_defaults={"n_layers": 2, "bottleneck_dim": 16},
)
class Autoencoder(nn.Module):
    """Symmetric encoder-decoder with a per-position bottleneck."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int = 2,
        bottleneck_dim: int = 16,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        encoder_layers = []
        in_dim = d_model
        for i in range(n_layers):
            out_dim = bottleneck_dim if i == n_layers - 1 else max(in_dim // 2, bottleneck_dim)
            encoder_layers += [nn.Linear(in_dim, out_dim), nn.GELU()]
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(n_layers):
            out_dim = d_model if i == n_layers - 1 else min(in_dim * 2, d_model)
            decoder_layers += [nn.Linear(in_dim, out_dim), nn.GELU()]
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embed(x)      # [batch, seq_len, d_model]
        z = self.encoder(x)     # [batch, seq_len, bottleneck_dim]
        x = self.decoder(z)     # [batch, seq_len, d_model]
        return self.head(x)     # [batch, seq_len, vocab_size]
