"""Symmetric MLP decoder paired with :class:`sfb.models.encoders.autoencoder.AutoencoderSequenceEncoder`."""

from torch import nn

from sfb.models.codec import SequenceDecoder


class AutoencoderSequenceDecoder(SequenceDecoder):
    def __init__(
        self,
        bottleneck_dim: int,
        seq_len: int,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        tail_dim: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.seq_expand = nn.Linear(bottleneck_dim, tail_dim * seq_len)
        decoder_layers = []
        in_dim = tail_dim
        for _ in range(n_layers - 1):
            out_dim = min(in_dim * 2, d_model)
            decoder_layers += [nn.Linear(in_dim, out_dim), nn.GELU()]
            in_dim = out_dim
        decoder_layers.append(nn.Linear(in_dim, d_model))
        self.decoder = nn.Sequential(*decoder_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def decode(self, z, seq_len):
        b = z.size(0)
        h = self.seq_expand(z).view(b, seq_len, -1)
        h = self.decoder(h)
        return self.head(h)
