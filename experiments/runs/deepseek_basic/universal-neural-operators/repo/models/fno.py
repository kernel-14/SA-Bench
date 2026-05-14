"""Fourier Neural Operator (FNO) baseline implementation.

Based on: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D spectral convolution layer performing convolution in Fourier space."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
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

    def compl_mul2d(self, input, weights):
        # (batch, in_channels, x, y) -> (batch, in_channels, x, y) * (in_channels, out_channels, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.shape[-2],
            x.shape[-1] // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x


class FNOBlock(nn.Module):
    """A single FNO block with spectral convolution and linear skip connection."""

    def __init__(self, hidden_channels, modes1, modes2, activation=F.gelu):
        super().__init__()
        self.spectral = SpectralConv2d(hidden_channels, hidden_channels, modes1, modes2)
        self.linear = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.spectral(x) + self.linear(x))


class FNO(nn.Module):
    """Fourier Neural Operator with lifting, FNO blocks, and projection.

    As described in Section 3 of the paper:
    - Lifting layer: projects input functions to hidden representation
    - FNO blocks: integral kernel operators in Fourier space
    - Projection layer: maps hidden representation to output functions
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 4,
        modes1: int = 12,
        modes2: int = 12,
        spatial_dim: int = 2,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.modes1 = modes1
        self.modes2 = modes2

        # Lifting layer: input_channels -> hidden_channels
        self.lifting = nn.Linear(input_channels, hidden_channels)

        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(hidden_channels, modes1, modes2)
            for _ in range(n_layers)
        ])

        # Projection layer: hidden_channels -> output_channels
        self.projection = nn.Linear(hidden_channels, output_channels)

    def forward(self, x, grid=None):
        """
        Args:
            x: Input tensor of shape (batch, spatial_x, spatial_y, input_channels)
            grid: Optional grid coordinates
        Returns:
            Output tensor of shape (batch, spatial_x, spatial_y, output_channels)
        """
        batch, nx, ny, _ = x.shape

        # Lift to hidden representation
        v = self.lifting(x)  # (batch, nx, ny, hidden_channels)

        # Transpose for convolution: (batch, hidden, nx, ny)
        v = v.permute(0, 3, 1, 2)

        # Apply FNO blocks
        for block in self.fno_blocks:
            v = block(v)

        # Transpose back: (batch, nx, ny, hidden)
        v = v.permute(0, 2, 3, 1)

        # Project to output space
        out = self.projection(v)

        return out

    def get_lifting_params(self):
        return list(self.lifting.parameters())

    def get_projection_params(self):
        return list(self.projection.parameters())

    def get_core_params(self):
        return list(self.fno_blocks.parameters())
