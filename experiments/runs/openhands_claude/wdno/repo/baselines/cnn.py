"""
CNN baseline for 1D PDE simulation and U-Net for 2D.

Reference: Hwang et al. (2022), "Solving PDE-Constrained Control Problems Using Operator Learning"

Hyperparameters from Table 28 (1D CNN):
  - Autoencoder of state: kernel=5, padding=2, activation=ELU, latent=256
  - Autoencoder of force: kernel=5, padding=2, activation=ELU, latent=256
  - Training: batch=5100, optimizer=Adam, lr=1e-3, epochs=500, scheduler=cosine

For 2D: U-Net (Ronneberger et al., 2015)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1D CNN (Hwang et al. 2022 style)
# ---------------------------------------------------------------------------

class ConvEncoder1D(nn.Module):
    """1D convolutional encoder for state or force."""

    def __init__(self, in_channels: int, latent_dim: int = 256, kernel_size: int = 5, padding: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size, padding=padding),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size, padding=padding),
            nn.ELU(),
            nn.Conv1d(128, 256, kernel_size, padding=padding),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, X] → [B, latent_dim]"""
        h = self.encoder(x).squeeze(-1)
        return self.fc(h)


class ConvDecoder1D(nn.Module):
    """1D convolutional decoder."""

    def __init__(self, latent_dim: int = 256, out_channels: int = 1, out_size: int = 120, kernel_size: int = 5, padding: int = 2):
        super().__init__()
        self.out_size = out_size
        self.fc = nn.Linear(latent_dim, 256 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(64, out_channels, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim] → [B, out_channels, out_size]"""
        h = self.fc(z).reshape(z.shape[0], 256, 8)
        out = self.decoder(h)
        return F.interpolate(out, size=self.out_size, mode="linear", align_corners=False)


class CNN1D(nn.Module):
    """
    CNN-based surrogate model for 1D PDE simulation.

    Architecture: VAE-based encoder-decoder with transition model.
    Input: u_0 (initial condition) + f (force sequence)
    Output: u_{[0,T]} (state trajectory)

    Hyperparameters (Table 28):
      kernel=5, padding=2, activation=ELU, latent=256
    """

    def __init__(
        self,
        nx: int = 120,
        nt: int = 80,
        latent_dim: int = 256,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.nx = nx
        self.nt = nt
        padding = kernel_size // 2

        # Encode initial condition
        self.u0_encoder = ConvEncoder1D(1, latent_dim, kernel_size, padding)

        # Encode force (2D: [T, X] → flatten to 1D)
        self.f_encoder = nn.Sequential(
            nn.Conv2d(1, 32, (3, kernel_size), padding=(1, padding)),
            nn.ELU(),
            nn.Conv2d(32, 64, (3, kernel_size), padding=(1, padding)),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.f_fc = nn.Linear(64, latent_dim)

        # Transition model: predict trajectory from latent
        self.transition = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, latent_dim * nt),
        )

        # Decode each time step
        self.decoder = ConvDecoder1D(latent_dim, 1, nx, kernel_size, padding)

    def forward(self, u0: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u0: [B, X]
            f: [B, T, X]
        Returns:
            u_pred: [B, T, X]
        """
        B = u0.shape[0]

        # Encode
        z_u0 = self.u0_encoder(u0.unsqueeze(1))  # [B, latent]
        z_f = self.f_encoder(f.unsqueeze(1))      # [B, 64, 1, 1]
        z_f = self.f_fc(z_f.squeeze(-1).squeeze(-1))  # [B, latent]

        z = torch.cat([z_u0, z_f], dim=-1)  # [B, 2*latent]

        # Predict trajectory latents
        z_traj = self.transition(z).reshape(B, self.nt, -1)  # [B, T, latent]

        # Decode each time step
        u_pred = torch.stack([
            self.decoder(z_traj[:, t]).squeeze(1)
            for t in range(self.nt)
        ], dim=1)  # [B, T, X]

        return u_pred


# ---------------------------------------------------------------------------
# U-Net for 2D PDE data
# ---------------------------------------------------------------------------

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2D(nn.Module):
    """
    3D U-Net for 2D PDE data (Ronneberger et al., 2015 adapted for 3D).

    Input: [B, C_in, T, H, W]
    Output: [B, C_out, T, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: List[int] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )
            self.ups.append(DoubleConv3D(feature * 2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)
