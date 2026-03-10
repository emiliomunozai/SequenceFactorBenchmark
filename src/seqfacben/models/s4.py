"""
S4D (Structured State Space - Diagonal) encoder.
Pure PyTorch — no external SSM library required.

Reference: Gu et al., "On the Parameterization and Initialization of
Diagonal State Space Models" (NeurIPS 2022).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from seqfacben.registry import register_model


class S4DLayer(nn.Module):
    """Single S4D layer: diagonal SSM convolution + residual + GELU."""

    def __init__(self, d_model: int, state_dim: int = 64,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()

        # Learnable log step-size (one per feature)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Diagonal A: S4D-Lin init — negative real (stability), fixed imaginary harmonics
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, state_dim)))
        self.register_buffer(
            "A_imag",
            math.pi * torch.arange(state_dim).float().unsqueeze(0).expand(d_model, -1),
        )

        # C (complex learnable); B is fixed to 1 and absorbed into the kernel
        self.C = nn.Parameter(torch.randn(d_model, state_dim, 2) * state_dim ** -0.5)

        # D: skip / feed-through connection
        self.D = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)

    def _kernel(self, L: int) -> torch.Tensor:
        """Compute causal SSM convolution kernel of length L via Vandermonde."""
        dt = self.log_dt.exp()
        A = torch.complex(-torch.exp(self.log_A_real), self.A_imag)

        dtA = A * dt.unsqueeze(-1)                        # [d, N]
        A_bar = torch.exp(dtA)                             # [d, N]
        # ZOH discretisation for B=1: B_bar = (A_bar - 1) / A
        C_tilde = torch.view_as_complex(self.C) * (A_bar - 1.0) / A  # [d, N]

        # Vandermonde: A_bar^k for k = 0 … L-1
        powers = torch.arange(L, device=A.device, dtype=torch.float)
        V = A_bar.unsqueeze(-1) ** powers                 # [d, N, L]

        K = torch.einsum("dn, dnl -> dl", C_tilde, V)     # [d, L]
        return 2.0 * K.real                                # conjugate-pair factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]"""
        residual = x
        x = self.norm(x)
        L = x.size(1)

        K = self._kernel(L)                                # [d, L]

        # FFT convolution (zero-pad to avoid circular aliasing)
        x_t = x.transpose(1, 2)                            # [B, d, L]
        fft_n = 2 * L
        y = torch.fft.irfft(
            torch.fft.rfft(x_t, n=fft_n) * torch.fft.rfft(K, n=fft_n).unsqueeze(0),
            n=fft_n,
        )[..., :L]                                         # [B, d, L]

        y = y + x_t * self.D.unsqueeze(0).unsqueeze(-1)    # skip connection
        return F.gelu(y.transpose(1, 2)) + residual


@register_model(
    "s4d",
    display_params=["d_model", "n_layers"],
    constructor_params=["vocab_size", "seq_len", "d_model", "n_layers"],
    param_defaults={"n_layers": 4},
)
class S4DEncoder(nn.Module):
    """S4D encoder: embedding -> stacked S4D layers -> linear head."""

    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([S4DLayer(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
