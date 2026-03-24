"""
Encoder / decoder composition with an explicit bottleneck.

Every registered sequence model should funnel information through a fixed-size
per-batch tensor ``z`` (shape ``[B, D]``) before decoding to per-step logits.
Submodules are exposed as ``.encoder`` and ``.decoder`` for analysis and ablations.
Concrete stacks live under ``sfb.models.encoders`` and ``sfb.models.decoders``;
registered end-to-end models are wired in ``sfb.models.composed``.

Targets Python 3.12+ (see ``pyproject.toml`` ``requires-python``).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    """Bottleneck representation for one batch."""

    z: torch.Tensor  # [B, D]


class SequenceEncoder(nn.Module, ABC):
    """Maps token ids [B, L] to a bottleneck vector [B, D]."""

    vocab_size: int
    seq_len: int
    out_dim: int

    def __init__(self, vocab_size: int, seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.out_dim = 0

    @abstractmethod
    def encode(self, x: torch.Tensor) -> EncoderOutput:
        ...


class SequenceDecoder(nn.Module, ABC):
    """Maps bottleneck [B, D] to logits [B, L, V]."""

    @abstractmethod
    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        ...


class ComposedCodecModel(nn.Module):
    """``forward`` = encode → decode. Keeps encoder/decoder as named children."""

    def __init__(self, encoder: SequenceEncoder, decoder: SequenceDecoder):
        super().__init__()
        if encoder.out_dim <= 0:
            raise ValueError("encoder.out_dim must be set to z.size(-1) before building the decoder")
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder.encode(x).z
        return self.decoder.decode(z, self.encoder.seq_len)


def resolve_d_bottleneck(d_model: int, d_bottleneck: int | None) -> int:
    """If ``d_bottleneck`` is omitted, use ``d_model`` (sweep / CLI backward compatibility)."""
    return int(d_bottleneck) if d_bottleneck is not None else int(d_model)
