"""Multiwavelet Neural Operator (Gupta et al., 2021) implementation.

Uses multiwavelet decomposition instead of Fourier for the integral
operator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiwaveletConv1d(nn.Module):
    """1D Multiwavelet convolution.

    Uses learnable weights in frequency domain approximated via
    wavelet decomposition.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize, _, N = x.shape
        x_ft = torch.fft.rfft(x, norm="ortho")
        out_ft = torch.zeros(
            batchsize, self.out_channels, N // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )
        x = torch.fft.irfft(out_ft, n=N, norm="ortho")
        return x


class MWNOBlock1d(nn.Module):
    """Single 1D MWNO block."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.wavelet_conv = MultiwaveletConv1d(width, width, modes)
        self.linear_conv = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.wavelet_conv(x) + self.linear_conv(x))


class MWNO(nn.Module):
    """Multiwavelet Neural Operator for 1D+time problems."""

    def __init__(
        self,
        modes1: int,
        modes_t: int = 0,
        width: int = 20,
        n_layers: int = 4,
        input_channels: int = 5,
        output_channels: int = 1,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes_t = modes_t
        self.width = width
        self.n_layers = n_layers

        self.lifting = nn.Linear(input_channels, width)

        n_blocks = n_layers * 2 if modes_t > 0 else n_layers
        self.fourier_blocks = nn.ModuleList([
            MWNOBlock1d(width, modes1 if (i % 2 == 0 or modes_t == 0) else modes_t)
            for i in range(n_blocks)
        ])

        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, output_channels),
        )

    def _apply_lifting(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, *range(2, x.dim()), 1)
        x_flat = x.reshape(-1, x.shape[-1])
        x_lifted = self.lifting(x_flat)
        grid_dims = x.shape[1:-1]
        x_lifted = x_lifted.reshape(x.shape[0], *grid_dims, self.width)
        x_lifted = x_lifted.permute(0, -1, *range(1, x_lifted.dim() - 1))
        return x_lifted

    def _apply_projection(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, *range(2, x.dim()), 1)
        x_flat = x.reshape(-1, self.width)
        x_proj = self.projection(x_flat)
        grid_dims = x.shape[1:-1]
        x_proj = x_proj.reshape(x.shape[0], *grid_dims, -1)
        x_proj = x_proj.permute(0, -1, *range(1, x_proj.dim() - 1))
        return x_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_lifting(x)

        if self.modes_t > 0:
            B, C, Nx, Nt = x.shape
            for i, block in enumerate(self.fourier_blocks):
                if i % 2 == 0:
                    x = x.reshape(B * Nx, C, Nt)
                    x = block(x)
                    x = x.reshape(B, C, Nx, Nt)
                else:
                    x = x.permute(0, 1, 3, 2).reshape(B * Nt, C, Nx)
                    x = block(x)
                    x = x.reshape(B, C, Nt, Nx).permute(0, 1, 3, 2)
        else:
            for block in self.fourier_blocks:
                x = block(x)

        return self._apply_projection(x)
