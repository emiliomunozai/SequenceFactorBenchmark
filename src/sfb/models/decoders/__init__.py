"""
Decoders: bottleneck ``[B, D]`` → logits ``[B, L, V]``.
"""

from sfb.models.bottleneck import SequenceDecoder
from sfb.models.decoders.lstm import LSTMSequenceDecoder
from sfb.models.decoders.mlp import MLPSequenceDecoder
from sfb.models.decoders.rnn import RNNSequenceDecoder
from sfb.models.decoders.transformer import TransformerSequenceDecoder

__all__ = [
    "LSTMSequenceDecoder",
    "MLPSequenceDecoder",
    "RNNSequenceDecoder",
    "SequenceDecoder",
    "TransformerSequenceDecoder",
]
