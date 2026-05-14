
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes #Number of Fourier modes to retain

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # x: (batchsize, channels, x_dim)

        # 1. Fourier transform
        x_ft = torch.fft.rfft(x, dim=-1)

        # 2. Multiply relevant modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights1)

        # 3. Inverse Fourier transform
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to retain in x-direction
        self.modes2 = modes2 #Number of Fourier modes to retain in y-direction

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # x: (batchsize, channels, x_dim, y_dim)

        # 1. Fourier transform
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        # 2. Multiply relevant modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 3. Inverse Fourier transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-2, -1))
        return x

