## models/p2vae.py
"""
P2VAE – Pretrained Physics Variational Autoencoder.

Compresses physical snapshots (3, 128, 128) into a compact latent grid
(16, 16, 16) using an architecture derived from SD-VAE [38].

The model supports two sizes by varying `base_dim`:
    - 16M parameters when base_dim = 64
    - 87M parameters when base_dim = 128

Typical usage:
    vae = P2VAE(base_dim=64)
    mu, logvar = vae.encode(x)
    z = vae.reparameterize(mu, logvar)
    x_recon = vae.decode(z)
    # Or all at once:
    x_recon, mu, logvar = vae(x)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ---------------------------------------------------------------------------
# Import the self-attention block for bottleneck attention
# (Assumes models/modules.py provides this class)
# ---------------------------------------------------------------------------
from models.modules import SelfAttentionBlock


# ---------------------------------------------------------------------------
# Basic convolutional helpers (internal to P2VAE)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual block with GroupNorm → SiLU → Conv sequence.
    If in_ch != out_ch, a 1×1 convolution aligns the residual dimension.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        num_groups: int = min(32, in_ch)  # avoid groups > channels
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        num_groups2: int = min(32, out_ch)
        self.norm2 = nn.GroupNorm(num_groups2, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Residual projection
        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(x))
        h = self.conv1(h)
        h = self.act2(self.norm2(h))
        h = self.conv2(h)
        return h + self.residual(x)


class Downsample(nn.Module):
    """Halves spatial resolution and doubles channels."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 2 * in_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Doubles spatial resolution and halves channels."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        out_ch = in_ch // 2
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Encoder of the P2VAE. Transforms a (B, 3, 128, 128) snapshot into
    a (B, 2*latent_dim, 16, 16) feature map from which μ and log(σ²) are split.
    """

    def __init__(
        self,
        base_dim: int = 64,
        latent_dim: int = 16,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_ch = base_dim
        for i, mult in enumerate(ch_mult):
            out_ch = base_dim * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            if i < len(ch_mult) - 1:
                self.down_blocks.append(Downsample(in_ch))
                in_ch *= 2  # Downsample doubles channels

        # Bottleneck: ResBlock → SelfAttention → ResBlock
        # (at spatial resolution 16×16 if ch_mult[-1] = 8)
        self.mid_block_1 = ResBlock(in_ch, in_ch)
        self.mid_attn = SelfAttentionBlock(in_ch)
        self.mid_block_2 = ResBlock(in_ch, in_ch)

        # Final mapping to μ + logvar (channels 2*latent_dim)
        num_groups = min(32, in_ch)
        self.norm_out = nn.GroupNorm(num_groups, in_ch)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(in_ch, 2 * latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, 3, 128, 128)
        h = self.conv_in(x)

        for block in self.down_blocks:
            h = block(h)

        h = self.mid_block_1(h)
        h = self.mid_attn(h)
        h = self.mid_block_2(h)

        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)  # (B, 2*latent_dim, 16, 16)

        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Decoder of the P2VAE. Transforms a latent tensor (B, latent_dim, 16, 16)
    back to a physical snapshot (B, 3, 128, 128).
    """

    def __init__(
        self,
        base_dim: int = 64,
        latent_dim: int = 16,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        out_channels: int = 3,
    ) -> None:
        super().__init__()

        # Inverse of ch_mult for upsampling
        in_ch_init = base_dim * ch_mult[-1]

        # Map from latent to highest channel count at 16×16
        self.conv_in = nn.Conv2d(latent_dim, in_ch_init, kernel_size=3, padding=1)

        # Bottleneck: ResBlock → SelfAttention → ResBlock
        self.mid_block_1 = ResBlock(in_ch_init, in_ch_init)
        self.mid_attn = SelfAttentionBlock(in_ch_init)
        self.mid_block_2 = ResBlock(in_ch_init, in_ch_init)

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_ch = in_ch_init
        for i in range(len(ch_mult) - 1, -1, -1):
            out_ch = base_dim * ch_mult[i]
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResBlock(in_ch, out_ch))
                in_ch = out_ch
            if i > 0:
                self.up_blocks.append(Upsample(in_ch))
                in_ch //= 2  # Upsample halves channels

        # Final output layer
        num_groups = min(32, in_ch)
        self.norm_out = nn.GroupNorm(num_groups, in_ch)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (B, latent_dim, 16, 16)
        h = self.conv_in(z)
        h = self.mid_block_1(h)
        h = self.mid_attn(h)
        h = self.mid_block_2(h)

        for block in self.up_blocks:
            h = block(h)

        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)  # (B, out_channels, 128, 128)
        return h


# ---------------------------------------------------------------------------
# Full P2VAE
# ---------------------------------------------------------------------------

class P2VAE(nn.Module):
    """
    Pretrained Physics Variational Autoencoder.

    Args:
        base_dim: Base channel dimension (64 → ~16M params, 128 → ~87M params).
        latent_dim: Dimensionality of the latent space (default 16).
        ch_mult: Channel multipliers for each resolution level.
        num_res_blocks: Number of residual blocks per resolution.
        in_channels: Number of input physical channels (default 3).
        out_channels: Number of reconstructed channels (same as in_channels).
    """

    def __init__(
        self,
        base_dim: int = 64,
        latent_dim: int = 16,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.base_dim = base_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            base_dim=base_dim,
            latent_dim=latent_dim,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
        )
        self.decoder = Decoder(
            base_dim=base_dim,
            latent_dim=latent_dim,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            out_channels=in_channels,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input snapshot to latent space.

        Args:
            x: (B, 3, 128, 128) physical field.

        Returns:
            mu, logvar: each (B, latent_dim, 16, 16)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to physical space.

        Args:
            z: (B, latent_dim, 16, 16) latent sample.

        Returns:
            x_recon: (B, 3, 128, 128) reconstructed snapshot.
        """
        return self.decoder(z)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard VAE reparameterisation trick.

        Args:
            mu: Mean tensor.
            logvar: Log variance tensor (same shape as mu).

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass (used during training).

        Args:
            x: (B, 3, 128, 128) input.

        Returns:
            x_recon: (B, 3, 128, 128)
            mu: (B, latent_dim, 16, 16)
            logvar: (B, latent_dim, 16, 16)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

