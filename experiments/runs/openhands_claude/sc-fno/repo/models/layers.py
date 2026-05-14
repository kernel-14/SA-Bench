import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier integral operator: K(phi)v_t(x) = F^{-1}(R_phi * F(v_t))(x)."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, T)
        batch = x.shape[0]
        T = x.shape[-1]

        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, T//2+1)

        out_ft = torch.zeros(batch, self.out_channels, T // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, : self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes], self.weights
        )

        return torch.fft.irfft(out_ft, n=T, dim=-1)  # (batch, out_channels, T)


class SpectralConv2d(nn.Module):
    """2D Fourier integral operator for (space, time) or (x, y) domains."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, H, W)
        batch = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (batch, in_channels, H, W//2+1)

        out_ft = torch.zeros(batch, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        return torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))  # (batch, out_channels, H, W)


class SpectralConv3d(nn.Module):
    """3D Fourier integral operator for (x, y, time) domains."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, D1, D2, D3)
        batch = x.shape[0]
        D1, D2, D3 = x.shape[-3], x.shape[-2], x.shape[-1]

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))  # (batch, in_channels, D1, D2, D3//2+1)

        out_ft = torch.zeros(
            batch, self.out_channels, D1, D2, D3 // 2 + 1, dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3],
            self.weights2,
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3],
            self.weights3,
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3],
            self.weights4,
        )

        return torch.fft.irfftn(out_ft, s=(D1, D2, D3), dim=(-3, -2, -1))


class FourierLayer1d(nn.Module):
    """Single Fourier layer: spectral conv + linear bypass + activation."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral_conv(x) + self.w(x))


class FourierLayer2d(nn.Module):
    """Single 2D Fourier layer: spectral conv + linear bypass + activation."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral_conv(x) + self.w(x))


class FourierLayer3d(nn.Module):
    """Single 3D Fourier layer: spectral conv + linear bypass + activation."""

    def __init__(self, width: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.spectral_conv = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.w = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral_conv(x) + self.w(x))
