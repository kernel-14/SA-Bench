"""Fourier Neural Operator layers as described in Li et al. (2021)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution layer for FNO.

    Applies learned linear transform in Fourier space, keeping only
    the specified number of modes.
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
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.shape[-1] // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )

        x = torch.fft.irfft(out_ft, n=x.shape[-1])
        return x


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution layer for FNO."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        size_1, size_2 = x.shape[-2], x.shape[-1]

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, size_1, size_2 // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1,
        )

        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2,
        )

        x = torch.fft.irfft2(out_ft, s=(size_1, size_2))
        return x


class FourierBlock1d(nn.Module):
    """Single Fourier block for 1D FNO.

    Each block contains:
    - A spectral convolution (Fourier domain)
    - A local linear convolution (physical space)
    - Non-linear activation (GELU)
    """

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = F.gelu(x)
        return x


class FourierBlock2d(nn.Module):
    """Single Fourier block for 2D FNO."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = F.gelu(x)
        return x
