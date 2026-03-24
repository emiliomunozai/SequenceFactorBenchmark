"""Shared per-position MLP, then a single linear from stacked features to ``z``.

Avoids ``Linear(seq_len * d_model, ...)``, which makes parameter count scale as
``O(seq_len * d_model * d_bottleneck)`` and dominates the model at long sequences.
"""

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
        h = max(d_bottleneck * 2, min(d_model, 128))
        self.token_mlp = nn.Sequential(
            nn.Linear(d_model, h),
            nn.ReLU(),
            nn.Linear(h, d_bottleneck),
        )
        self.to_bottleneck = nn.Linear(seq_len * d_bottleneck, d_bottleneck)
        self.out_dim = d_bottleneck

    def encode(self, x):
        b = x.size(0)
        h = self.token_mlp(self.embed(x))
        z = self.to_bottleneck(h.reshape(b, -1))
        return EncoderOutput(z=z)
