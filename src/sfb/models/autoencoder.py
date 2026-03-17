"""
Autoencoder: compresses the entire sequence through a fixed-size
bottleneck, then reconstructs per-position logits.  Tests whether the
model can learn a compact internal representation sufficient for the
target task.
"""
from torch import nn
from sfb.registry import register_model


@register_model(
    "autoencoder",
    display_params=["d_model", "n_layers", "bottleneck_dim"],
    constructor_params=["vocab_size", "seq_len", "d_model", "n_layers", "bottleneck_dim"],
    param_defaults={"n_layers": 2, "bottleneck_dim": 16},
)
class Autoencoder(nn.Module):
    """Symmetric encoder-decoder with a sequence-level bottleneck."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int = 2,
        bottleneck_dim: int = 16,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.bottleneck_dim = bottleneck_dim

        self.embed = nn.Embedding(vocab_size, d_model)

        # Encoder: per-position MLP, then flatten + linear to fixed latent
        encoder_layers = []
        in_dim = d_model
        for i in range(n_layers - 1):
            out_dim = max(in_dim // 2, bottleneck_dim)
            encoder_layers += [nn.Linear(in_dim, out_dim), nn.GELU()]
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        # Compress entire sequence to a single latent vector
        self.seq_compress = nn.Linear(in_dim * seq_len, bottleneck_dim)

        # Decoder: expand latent back to full sequence, then per-position MLP
        self.seq_expand = nn.Linear(bottleneck_dim, in_dim * seq_len)
        decoder_layers = []
        for i in range(n_layers - 1):
            out_dim = min(in_dim * 2, d_model)
            decoder_layers += [nn.Linear(in_dim, out_dim), nn.GELU()]
            in_dim = out_dim
        # Final projection to d_model (no activation before head)
        decoder_layers.append(nn.Linear(in_dim, d_model))
        self.decoder = nn.Sequential(*decoder_layers)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        B, S = x.shape
        x = self.embed(x)                        # [B, S, d_model]
        x = self.encoder(x)                      # [B, S, in_dim]

        # Sequence-level bottleneck
        z = x.reshape(B, -1)                     # [B, S * in_dim]
        z = self.seq_compress(z)                 # [B, bottleneck_dim]
        z = self.seq_expand(z)                   # [B, S * in_dim]
        z = z.reshape(B, S, -1)                  # [B, S, in_dim]

        x = self.decoder(z)                      # [B, S, d_model]
        return self.head(x)                      # [B, S, vocab_size]