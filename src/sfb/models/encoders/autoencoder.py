"""Per-step MLP + sequence-level linear compress to ``bottleneck_dim``."""

from torch import nn

from sfb.models.codec import EncoderOutput, SequenceEncoder


class AutoencoderSequenceEncoder(SequenceEncoder):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_layers: int,
        bottleneck_dim: int,
    ):
        super().__init__(vocab_size, seq_len)
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layers = []
        in_dim = d_model
        for _ in range(n_layers - 1):
            out_dim = max(in_dim // 2, bottleneck_dim)
            encoder_layers += [nn.Linear(in_dim, out_dim), nn.GELU()]
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.seq_compress = nn.Linear(in_dim * seq_len, bottleneck_dim)
        self.tail_dim = in_dim
        self.out_dim = bottleneck_dim

    def encode(self, x):
        b = x.size(0)
        h = self.embed(x)
        h = self.encoder(h)
        z = self.seq_compress(h.reshape(b, -1))
        return EncoderOutput(z=z)
