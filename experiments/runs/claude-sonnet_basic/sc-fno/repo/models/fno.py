"""
Fourier Neural Operator (FNO) implementation.
Based on: Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv1d(nn.Module):
    """1D Fourier layer: FFT -> linear transform in frequency domain -> IFFT."""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):
    """2D Fourier layer."""

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
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] =             self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] =             self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv3d(nn.Module):
    """3D Fourier layer."""

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] =             self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] =             self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] =             self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] =             self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator for ODEs and 1D PDEs.
    Input: (batch, T, in_channels)
    Output: (batch, T_out, out_channels)
    """

    def __init__(self, modes, width, in_channels, out_channels, num_layers=4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.fc0 = nn.Linear(in_channels, width)
        self.conv_layers = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(num_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(num_layers)
        ])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for 1D PDEs (space + time).
    Input: (batch, S, T, in_channels)
    Output: (batch, S, T_out, out_channels)
    """

    def __init__(self, modes1, modes2, width, in_channels, out_channels, num_layers=4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.fc0 = nn.Linear(in_channels, width)
        self.conv_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(num_layers)
        ])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator for 2D PDEs (x, y, t).
    Input: (batch, Sx, Sy, T, in_channels)
    Output: (batch, Sx, Sy, T_out, out_channels)
    """

    def __init__(self, modes1, modes2, modes3, width, in_channels, out_channels, num_layers=4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.fc0 = nn.Linear(in_channels, width)
        self.conv_layers = nn.ModuleList([
            SpectralConv3d(width, width, modes1, modes2, modes3) for _ in range(num_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv3d(width, width, 1) for _ in range(num_layers)
        ])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        for i in range(self.num_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
