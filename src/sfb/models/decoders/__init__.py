"""
Decoders: bottleneck ``[B, D]`` → logits ``[B, L, V]``.

Add new decoder classes here and register them in :func:`build_sequence_decoder`.
"""

from sfb.models.codec import SequenceDecoder
from sfb.models.decoders.autoencoder import AutoencoderSequenceDecoder
from sfb.models.decoders.linear import LinearSequenceDecoder
from sfb.models.decoders.rnn import GRUSequenceDecoder, LSTMSequenceDecoder

DECODER_CHOICES = ("linear", "lstm", "gru")

__all__ = [
    "AutoencoderSequenceDecoder",
    "DECODER_CHOICES",
    "GRUSequenceDecoder",
    "LinearSequenceDecoder",
    "LSTMSequenceDecoder",
    "SequenceDecoder",
    "build_sequence_decoder",
]


def build_sequence_decoder(
    name: str,
    *,
    z_dim: int,
    seq_len: int,
    d_model: int,
    vocab_size: int,
    n_layers: int = 1,
) -> SequenceDecoder:
    key = (name or "linear").lower()
    if key == "linear":
        return LinearSequenceDecoder(z_dim, seq_len, d_model, vocab_size)
    if key == "lstm":
        return LSTMSequenceDecoder(z_dim, seq_len, d_model, vocab_size, n_layers=n_layers)
    if key == "gru":
        return GRUSequenceDecoder(z_dim, seq_len, d_model, vocab_size, n_layers=n_layers)
    raise ValueError(f"Unknown decoder {name!r}; use one of {DECODER_CHOICES}")
