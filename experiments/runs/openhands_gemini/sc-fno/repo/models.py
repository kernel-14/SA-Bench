
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SpectralConv1d, SpectralConv2d

class FNO(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int, modes_x: int, modes_y: int = None,
                 num_fourier_layers: int = 4):
        super(FNO, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.num_fourier_layers = num_fourier_layers

        # Lifting layer: maps input features to hidden dimension 'width'
        # Expects input shape (batch_size, *grid_dims, in_channels)
        self.p = nn.Linear(in_channels, self.width)
        
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()

        if modes_y is None: # 1D spatial problem (or ODE treated as 1D time problem)
            self.spectral_conv_type = SpectralConv1d
            for _ in range(num_fourier_layers):
                self.convs.append(SpectralConv1d(self.width, self.width, self.modes_x))
                self.ws.append(nn.Conv1d(self.width, self.width, 1))
        else: # 2D spatial problem
            self.spectral_conv_type = SpectralConv2d
            for _ in range(num_fourier_layers):
                self.convs.append(SpectralConv2d(self.width, self.width, self.modes_x, self.modes_y))
                self.ws.append(nn.Conv2d(self.width, self.width, 1))

        # Projection layer: maps hidden dimension 'width' to output channels
        # Will operate on (batch_size, *grid_dims, width)
        self.q = nn.Linear(self.width, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to be (batch_size, *grid_dims, in_channels)
        # Example: (batch, time_steps, spatial_x, in_channels) for 2D-spatial PDE
        # Or: (batch, time_steps, in_channels) for 1D-spatial PDE/ODE
        
        # 1. Lifting layer
        # Output shape: (batch_size, *grid_dims, width)
        x = self.p(x)

        # Permute to (batch_size, width, *grid_dims) for convolution layers
        if self.modes_y is None: # 1D spatial / ODE: (batch, time_steps, width) -> (batch, width, time_steps)
            x = x.permute(0, 2, 1)
        else: # 2D spatial PDE: (batch, spatial_x, spatial_y, width) -> (batch, width, spatial_x, spatial_y)
            x = x.permute(0, 3, 1, 2)

        # 2. Fourier layers
        for i in range(self.num_fourier_layers):
            # Spectral convolution part
            x_conv = self.convs[i](x)
            # Linear convolution part (residual connection)
            x_w = self.ws[i](x)
            x = x_conv + x_w
            if i < self.num_fourier_layers - 1:
                x = F.gelu(x)

        # 3. Projection layer
        # Permute back to (batch_size, *grid_dims, width)
        if self.modes_y is None: # (batch, width, time_steps) -> (batch, time_steps, width)
            x = x.permute(0, 2, 1)
        else: # (batch, width, spatial_x, spatial_y) -> (batch, spatial_x, spatial_y, width)
            x = x.permute(0, 2, 3, 1)

        # Output shape: (batch_size, *grid_dims, out_channels)
        x = self.q(x)
        return x

