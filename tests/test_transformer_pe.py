"""Regression: sinusoidal PE must support odd d_model (even width only was previously broken)."""

import torch

from sfb.models.encoders.transformer import SinusoidalPE, TransformerSequenceEncoder


def test_sinusoidal_pe_odd_d_model_forward():
    pe = SinusoidalPE(d_model=15, max_len=32)
    x = torch.zeros(2, 10, 15)
    y = pe(x)
    assert y.shape == x.shape


def test_transformer_encoder_odd_d_model_build_and_forward():
    # n_heads must divide d_model; 63 / 3 == 21
    enc = TransformerSequenceEncoder(
        vocab_size=16, seq_len=8, d_model=63, d_bottleneck=16, n_layers=1, n_heads=3
    )
    x = torch.randint(0, 16, (4, 8))
    z = enc.encode(x).z
    assert z.shape == (4, 16)
