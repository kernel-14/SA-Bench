"""
Fourier Neural Operator (FNO) baseline.

Reference: Li et al. (2021), "Fourier Neural Operator for Parametric PDEs"

Hyperparameters from Table 26 (1D) and Table 32 (2D):
  1D:
    - modes: 16
    - width: 64
    - n_layers: 4
    - mlp_expansion: 0.5
    - activation: GELU
    - padding mode: one-sided

  2D:
    - modes: 16
    - width: 64
    - n_layers: 4
    - mlp_expansion: 0.5
    - activation: GELU
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D Fourier integral operator layer."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

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

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x) + self.w(x)))


class FNO1D(nn.Module):
    """
    FNO for 1D PDE data (treats time-space as 2D).

    Input: [batch, C_in, T, X]
    Output: [batch, C_out, T, X]

    Hyperparameters (Table 26):
      modes=16, width=64, n_layers=4, mlp_expansion=0.5
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        mlp_hidden: int = 256,
        padding: int = 9,
    ):
        super().__init__()
        self.padding = padding

        self.fc0 = nn.Linear(in_channels, width)
        self.layers = nn.ModuleList([FNOBlock2d(width, modes, modes) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, T, X] → [B, C_out, T, X]"""
        # Permute to [B, T, X, C]
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, width, T, X]

        # Pad
        x = F.pad(x, [0, self.padding])

        for layer in self.layers:
            x = layer(x)

        # Remove padding
        x = x[..., :-self.padding]

        x = x.permute(0, 2, 3, 1)  # [B, T, X, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)  # [B, C_out, T, X]


class SpectralConv3d(nn.Module):
    """3D Fourier integral operator layer."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2
        )
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4
        )

        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


class FNOBlock3d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.conv = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.w = nn.Conv3d(width, width, 1)
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(x) + self.w(x)))


class FNO2D(nn.Module):
    """
    FNO for 2D PDE data (treats time-space as 3D).

    Input: [batch, C_in, T, H, W]
    Output: [batch, C_out, T, H, W]

    Hyperparameters (Table 32):
      modes=16, width=64, n_layers=4
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.layers = nn.ModuleList([FNOBlock3d(width, modes, modes, modes) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, T, H, W] → [B, C_out, T, H, W]"""
        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, width, T, H, W]

        for layer in self.layers:
            x = layer(x)

        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, width]
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 4, 1, 2, 3)  # [B, C_out, T, H, W]
