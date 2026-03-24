"""Flatten embeddings and compress to one bottleneck vector."""

from torch import nn

from sfb.models.codec import EncoderOutput, SequenceEncoder


class SimpleNNSequenceEncoder(SequenceEncoder):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        d_bottleneck: int,
    ):
        super().__init__(vocab_size, seq_len)
        self.embed = nn.Embedding(vocab_size, d_model)
        flat = d_model * seq_len
        self.compress = nn.Sequential(
            nn.Linear(flat, d_bottleneck * 2),
            nn.ReLU(),
            nn.Linear(d_bottleneck * 2, d_bottleneck),
        )
        self.out_dim = d_bottleneck

    def encode(self, x):
        b = x.size(0)
        e = self.embed(x).reshape(b, -1)
        z = self.compress(e)
        return EncoderOutput(z=z)
