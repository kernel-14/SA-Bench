"""
Fourier Neural Operator (FNO) implementation.
Based on: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations"
This serves as the baseline model in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv1d(nn.Module):
    """1D Fourier integral operator layer."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], self.weights
        )
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):
    """2D Fourier integral operator layer."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNOBlock1d(nn.Module):
    """Single FNO block for 1D problems."""

    def __init__(self, width, modes):
        super().__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.conv(x) + self.w(x))


class FNOBlock2d(nn.Module):
    """Single FNO block for 2D problems."""

    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.conv(x) + self.w(x))


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator.
    Architecture: Lifting -> FNO blocks -> Projection
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 64,
        modes: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width
        self.modes = modes
        self.n_layers = n_layers

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # FNO blocks (shared backbone)
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, modes) for _ in range(n_layers)
        ])

        # Projection layer (adapter - problem specific)
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, n_output),
        )

    def get_backbone_params(self):
        """Return parameters of the shared backbone (FNO blocks)."""
        return list(self.fno_blocks.parameters())

    def get_adapter_params(self):
        """Return parameters of the problem-specific adapters."""
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx)
        # Transpose for linear layer: (batch, nx, n_input)
        x = x.permute(0, 2, 1)
        x = self.lifting(x)
        # Back to (batch, width, nx)
        x = x.permute(0, 2, 1)

        for block in self.fno_blocks:
            x = block(x)

        # Transpose for linear layer: (batch, nx, width)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        # Back to (batch, n_output, nx)
        x = x.permute(0, 2, 1)
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.
    Architecture: Lifting -> FNO blocks -> Projection
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_layers = n_layers

        # Lifting layer (adapter - problem specific)
        self.lifting = nn.Linear(n_input, width)

        # FNO blocks (shared backbone)
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(width, modes1, modes2) for _ in range(n_layers)
        ])

        # Projection layer (adapter - problem specific)
        self.projection = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, n_output),
        )

    def get_backbone_params(self):
        """Return parameters of the shared backbone (FNO blocks)."""
        return list(self.fno_blocks.parameters())

    def get_adapter_params(self):
        """Return parameters of the problem-specific adapters."""
        return list(self.lifting.parameters()) + list(self.projection.parameters())

    def forward(self, x):
        # x: (batch, n_input, nx, ny)
        # Transpose for linear layer: (batch, nx, ny, n_input)
        x = x.permute(0, 2, 3, 1)
        x = self.lifting(x)
        # Back to (batch, width, nx, ny)
        x = x.permute(0, 3, 1, 2)

        for block in self.fno_blocks:
            x = block(x)

        # Transpose for linear layer: (batch, nx, ny, width)
        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        # Back to (batch, n_output, nx, ny)
        x = x.permute(0, 3, 1, 2)
        return x
