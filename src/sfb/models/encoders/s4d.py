"""S4D stack over the sequence, mean-pool, project to ``d_bottleneck``."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfb.models.codec import EncoderOutput, SequenceEncoder


class S4DLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, state_dim)))
        self.register_buffer(
            "A_imag",
            math.pi * torch.arange(state_dim).float().unsqueeze(0).expand(d_model, -1),
        )
        self.C = nn.Parameter(torch.randn(d_model, state_dim, 2) * state_dim ** -0.5)
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)

    def _kernel(self, L: int) -> torch.Tensor:
        dt = self.log_dt.exp()
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)
        dtA = A * dt.unsqueeze(-1)
        A_bar = torch.exp(dtA)
        C_tilde = torch.view_as_complex(self.C) * (A_bar - 1.0) / A
        powers = torch.arange(L, device=A.device, dtype=torch.float)
        V = A_bar.unsqueeze(-1) ** powers
        K = torch.einsum("dn, dnl -> dl", C_tilde, V)
        return 2.0 * K.real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        L = x.size(1)
        K = self._kernel(L)
        x_t = x.transpose(1, 2)
        fft_n = 2 * L
        y = torch.fft.irfft(
            torch.fft.rfft(x_t, n=fft_n) * torch.fft.rfft(K, n=fft_n).unsqueeze(0),
            n=fft_n,
        )[..., :L]
        y = y + x_t * self.D.unsqueeze(0).unsqueeze(-1)
        return F.gelu(y.transpose(1, 2)) + residual


class S4DSequenceEncoder(SequenceEncoder):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        d_bottleneck: int,
        n_layers: int,
    ):
        super().__init__(vocab_size, seq_len)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([S4DLayer(d_model) for _ in range(n_layers)])
        self.to_bottleneck = nn.Linear(d_model, d_bottleneck)
        self.out_dim = d_bottleneck

    def encode(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        z = self.to_bottleneck(h.mean(dim=1))
        return EncoderOutput(z=z)
