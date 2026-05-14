"""
Fourier Neural Operator (FNO) implementation.

Based on: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", 2021.
Extended for the sensitivity-constrained framework from Behroozi, Shen & Kifer.

Supports 1D (temporal only), 2D (spatial + temporal), and 3D (2D spatial + temporal) FNO variants.
Uses spectral convolutions in the Fourier domain with learnable weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution layer for FNO.
    
    Applies learned linear transform on lower Fourier modes and filters out higher modes.
    """

    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution layer for FNO."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
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
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution layer for FNO."""

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
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
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO(nn.Module):
    """
    Fourier Neural Operator.
    
    Maps input functions (initial conditions + parameters) to output functions (solution paths).
    
    Architecture:
    - Lifting layer: input_dim -> width
    - N Fourier layers: SpectralConv + linear bypass with GELU activation
    - Projection layers: width -> 128 -> output_dim
    
    The input includes:
    - Spatial/temporal grid coordinates (x, t)
    - Initial conditions u0(x)
    - Parameters p (broadcast to match spatial/temporal dimensions)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        modes: list = None,
        width: int = 20,
        n_layers: int = 4,
        spatial_dims: int = 2,
        activation: str = 'gelu',
    ):
        super().__init__()
        if modes is None:
            modes = [8, 8]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.spatial_dims = spatial_dims

        # Lifting layer: pointwise linear
        self.fc0 = nn.Linear(input_dim, width)

        # Fourier layers
        self.conv_layers = nn.ModuleList()
        self.bypass_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.bypass_layers.append(nn.Conv1d(width, width, 1))

        if spatial_dims == 1:
            for _ in range(n_layers):
                self.conv_layers.append(SpectralConv1d(width, width, modes[0]))
        elif spatial_dims == 2:
            for _ in range(n_layers):
                self.conv_layers.append(SpectralConv2d(width, width, modes[0], modes[1]))
        elif spatial_dims == 3:
            for _ in range(n_layers):
                self.conv_layers.append(SpectralConv3d(width, width, modes[0], modes[1], modes[2]))

        # Projection layers
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.gelu

    def forward(self, x, params_tensor=None):
        """
        Args:
            x: Input tensor of shape (batch, *grid_dims, input_dim)
               For 1D: (batch, n_t, input_dim)
               For 2D: (batch, n_t, n_x, input_dim)
               For 3D: (batch, n_t, n_x, n_y, input_dim)
            params_tensor: Optional parameter tensor for computing Jacobians.
                          When None, returns only the solution.

        Returns:
            Output tensor of shape (batch, *grid_dims, output_dim)
        """
        # Lifting
        x = self.fc0(x)

        # Reshape for Fourier layers: move channel dimension to position 1
        if self.spatial_dims == 1:
            x = x.permute(0, 2, 1)
        elif self.spatial_dims == 2:
            x = x.permute(0, 3, 1, 2)
        elif self.spatial_dims == 3:
            x = x.permute(0, 4, 1, 2, 3)

        # Fourier layers with skip connections
        for i in range(self.n_layers):
            x_conv = self.conv_layers[i](x)
            x_bypass = self.bypass_layers[i](x)
            x = self.activation(x_conv + x_bypass)

        # Reshape back
        if self.spatial_dims == 1:
            x = x.permute(0, 2, 1)
        elif self.spatial_dims == 2:
            x = x.permute(0, 2, 3, 1)
        elif self.spatial_dims == 3:
            x = x.permute(0, 2, 3, 4, 1)

        # Projection
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x

    def count_params(self):
        """Count the number of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
