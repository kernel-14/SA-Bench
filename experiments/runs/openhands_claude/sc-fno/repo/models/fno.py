"""
FNO model variants for 1D (ODEs), 2D (1D PDEs), and 3D (2D PDEs like Navier-Stokes).

Architecture (shared by FNO and SC-FNO — they differ only in loss):
  1. Lifting layer: concatenated inputs → width channels
  2. n_layers Fourier layers
  3. Projection layer: width → output_dim

Input construction:
  - ODE (1D): [u(0:M), p_repeated] on temporal grid → u(M:N)
  - PDE 1D (2D): [u(x, 0:M), p_repeated] on (x, t) grid → u(x, M:N)
  - PDE 2D (3D): [u(x,y,0), p_repeated] on (x, y) grid → u(x, y, T_final)
"""

import torch
import torch.nn as nn

from .layers import FourierLayer1d, FourierLayer2d, FourierLayer3d


class FNO1d(nn.Module):
    """
    FNO for ODEs: maps (u[0:M], p) → u[M:N] on a 1D temporal grid.

    Input tensor shape: (batch, T_in, in_channels)
    where in_channels = 1 (u values) + n_params + 1 (time coordinate)
    Output tensor shape: (batch, T_out, 1)
    """

    def __init__(
        self,
        modes: int,
        width: int,
        in_channels: int,
        out_channels: int,
        n_layers: int = 4,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers

        self.fc0 = nn.Linear(in_channels, width)
        self.fourier_layers = nn.ModuleList(
            [FourierLayer1d(width, modes) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, in_channels)
        x = self.fc0(x)           # (batch, T, width)
        x = x.permute(0, 2, 1)   # (batch, width, T)

        for layer in self.fourier_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)   # (batch, T, width)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)           # (batch, T, out_channels)
        return x


class FNO2d(nn.Module):
    """
    FNO for 1D PDEs: maps (u[x, 0:M], p) → u[x, M:N] on a 2D (x, t) grid.

    Input tensor shape: (batch, Sx, T_in, in_channels)
    where in_channels = 1 (u values) + n_params + 2 (x, t coordinates)
    Output tensor shape: (batch, Sx, T_out, 1)
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        width: int,
        in_channels: int,
        out_channels: int,
        n_layers: int = 4,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers

        self.fc0 = nn.Linear(in_channels, width)
        self.fourier_layers = nn.ModuleList(
            [FourierLayer2d(width, modes1, modes2) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, Sx, T, in_channels)
        x = self.fc0(x)                  # (batch, Sx, T, width)
        x = x.permute(0, 3, 1, 2)       # (batch, width, Sx, T)

        for layer in self.fourier_layers:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)       # (batch, Sx, T, width)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                  # (batch, Sx, T, out_channels)
        return x


class FNO3d(nn.Module):
    """
    FNO for 2D PDEs (Navier-Stokes): maps (u[x,y,0], p) → u[x,y,T_final].

    Input tensor shape: (batch, Sx, Sy, T_in, in_channels)
    where in_channels = 1 (vorticity) + n_params + 3 (x, y, t coordinates)
    Output tensor shape: (batch, Sx, Sy, T_out, 1)
    """

    def __init__(
        self,
        modes1: int,
        modes2: int,
        modes3: int,
        width: int,
        in_channels: int,
        out_channels: int,
        n_layers: int = 4,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.n_layers = n_layers

        self.fc0 = nn.Linear(in_channels, width)
        self.fourier_layers = nn.ModuleList(
            [FourierLayer3d(width, modes1, modes2, modes3) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, Sx, Sy, T, in_channels)
        x = self.fc0(x)                     # (batch, Sx, Sy, T, width)
        x = x.permute(0, 4, 1, 2, 3)       # (batch, width, Sx, Sy, T)

        for layer in self.fourier_layers:
            x = layer(x)

        x = x.permute(0, 2, 3, 4, 1)       # (batch, Sx, Sy, T, width)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                     # (batch, Sx, Sy, T, out_channels)
        return x
