"""Fourier Neural Operator (Li et al., 2021) implementation.

Supports 1D (temporal), 2D (1D spatial + time or 2D spatial), and 3D
(2D spatial + time) configurations via spectral convolutions.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier_layers import SpectralConv1d, SpectralConv2d


class FNOBlock1d(nn.Module):
    """Single 1D Fourier block: spectral conv + linear bypass + GELU."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.linear = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.linear(x))


class FNOBlock2d(nn.Module):
    """Single 2D Fourier block: spectral conv + linear bypass + GELU."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.linear = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.linear(x))


class FNO(nn.Module):
    """Fourier Neural Operator.

    Architecture (per paper Section 2.1 and Appendix C):
    1. Lifting layer: Linear(input_channels -> width)
    2. N Fourier layers (4 per Table C.7) with spectral convolutions
    3. Projection: Linear(width -> 128) + GELU + Linear(128 -> output)

    For 1D+time problems, the Fourier layers alternate between applying
    spectral convolution along the time dimension and the spatial dimension.

    Args:
        modes1: Fourier modes for first spatial dimension.
        modes2: Fourier modes for second spatial dimension (0 for 1D).
        modes_t: Fourier modes for time dimension (0 if time-stepping not used).
        width: Hidden channel width (20 per Table C.7).
        n_layers: Number of Fourier layers (4 per paper).
        input_channels: Number of input function channels.
        output_channels: Number of output channels (typically 1).
        ndim: Dimensionality ("1d", "2d", or "3d" for 2D+time).
    """

    def __init__(
        self,
        modes1: int,
        modes2: int = 0,
        modes_t: int = 0,
        width: int = 20,
        n_layers: int = 4,
        input_channels: int = 5,
        output_channels: int = 1,
        ndim: str = "1d",
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes_t = modes_t
        self.width = width
        self.n_layers = n_layers
        self.ndim = ndim

        self.lifting = nn.Linear(input_channels, width)

        if ndim == "1d":
            self.fourier_blocks = nn.ModuleList([
                FNOBlock1d(width, modes1) for _ in range(n_layers)
            ])
        elif ndim == "2d":
            if modes_t > 0:
                self.fourier_blocks = nn.ModuleList([
                    FNOBlock1d(width, modes1 if i % 2 == 0 else modes_t)
                    for i in range(n_layers * 2)
                ])
            else:
                self.fourier_blocks = nn.ModuleList([
                    FNOBlock2d(width, modes1, modes2)
                    for _ in range(n_layers)
                ])
        elif ndim == "3d":
            self.fourier_blocks = nn.ModuleList([
                FNOBlock2d(width, modes1, modes2)
                for _ in range(n_layers)
            ])

        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, output_channels),
        )

    def _apply_lifting(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise lifting to width-dimensional features.

        Args:
            x: (B, C, *grid_dims)

        Returns:
            (B, width, *grid_dims)
        """
        x = x.permute(0, *range(2, x.dim()), 1)  # (B, *grid, C)
        x_flat = x.reshape(-1, x.shape[-1])
        x_lifted = self.lifting(x_flat)
        grid_dims = x.shape[1:-1]
        x_lifted = x_lifted.reshape(x.shape[0], *grid_dims, self.width)
        x_lifted = x_lifted.permute(0, -1, *range(1, x_lifted.dim() - 1))
        return x_lifted

    def _apply_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Project from width back to output_channels.

        Args:
            x: (B, width, *grid_dims)

        Returns:
            (B, output_channels, *grid_dims)
        """
        x = x.permute(0, *range(2, x.dim()), 1)
        x_flat = x.reshape(-1, self.width)
        x_proj = self.projection(x_flat)
        grid_dims = x.shape[1:-1]
        x_proj = x_proj.reshape(x.shape[0], *grid_dims, -1)
        x_proj = x_proj.permute(0, -1, *range(1, x_proj.dim() - 1))
        return x_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, input_channels, *grid_dims).

        Returns:
            Output tensor (B, output_channels, *grid_dims).
        """
        x = self._apply_lifting(x)

        if self.ndim == "1d":
            for block in self.fourier_blocks:
                x = block(x)

        elif self.ndim == "2d":
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

        elif self.ndim == "3d":
            B, C, Nx, Ny, Nt = x.shape
            for block in self.fourier_blocks:
                x = x.permute(0, 1, 4, 2, 3).reshape(B * Nt, C, Nx, Ny)
                x = block(x)
                x = x.reshape(B, C, Nt, Nx, Ny).permute(0, 1, 3, 4, 2)

        return self._apply_projection(x)
