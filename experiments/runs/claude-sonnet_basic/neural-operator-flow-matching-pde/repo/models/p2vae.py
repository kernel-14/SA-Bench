"""
P2VAE: Pretrained Physics Variational Autoencoder

Based on the SD-VAE architecture (Rombach et al., 2022), adapted for PDE field compression.
Compresses physical field snapshots from c3p128 to c16p16 (12x compression rate).

P2VAE-16M uses base_dim=64
P2VAE-87M uses base_dim=128
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Residual block with group normalization."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block for VAE."""

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv.unbind(1)  # each (B, C, HW)
        # Scaled dot-product attention
        attn = torch.einsum("bci,bcj->bij", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DownsampleBlock(nn.Module):
    """Downsampling by factor 2 using strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """Upsampling by factor 2 using nearest-neighbor + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class P2VAEEncoder(nn.Module):
    """
    Encoder for P2VAE.

    Compresses spatial fields from (C_in, H, W) to latent (2*latent_channels, H/8, W/8).
    The factor of 2 is for mean and log-variance of the Gaussian posterior.

    For c3p128 -> c16p16: 3 downsampling stages (128 -> 64 -> 32 -> 16).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 64,
        latent_channels: int = 16,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_dim = base_dim
        self.latent_channels = latent_channels

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        current_channels = base_dim
        current_res = 128  # starting resolution

        for i, mult in enumerate(channel_multipliers):
            out_channels = base_dim * mult
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResidualBlock(current_channels, out_channels, num_groups))
                current_channels = out_channels
                if current_res in attention_resolutions:
                    block.append(AttentionBlock(current_channels, num_groups))
            self.down_blocks.append(block)
            if i < len(channel_multipliers) - 1:
                self.down_blocks.append(nn.ModuleList([DownsampleBlock(current_channels)]))
                current_res //= 2

        # Middle blocks
        self.mid_block1 = ResidualBlock(current_channels, current_channels, num_groups)
        self.mid_attn = AttentionBlock(current_channels, num_groups)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, num_groups)

        # Output
        self.norm_out = nn.GroupNorm(num_groups, current_channels)
        self.act_out = nn.SiLU()
        # Output 2*latent_channels for mean and log-variance
        self.conv_out = nn.Conv2d(current_channels, 2 * latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) input field

        Returns:
            moments: (B, 2*latent_channels, H/8, W/8) mean and log-variance
        """
        h = self.conv_in(x)

        for block in self.down_blocks:
            for layer in block:
                h = layer(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)
        return h


class P2VAEDecoder(nn.Module):
    """
    Decoder for P2VAE.

    Reconstructs spatial fields from latent (latent_channels, H/8, W/8) to (C_out, H, W).
    """

    def __init__(
        self,
        out_channels: int = 3,
        base_dim: int = 64,
        latent_channels: int = 16,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        num_groups: int = 32,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.base_dim = base_dim
        self.latent_channels = latent_channels

        # Compute the number of channels at the bottleneck
        bottleneck_channels = base_dim * channel_multipliers[-1]

        # Initial convolution from latent
        self.conv_in = nn.Conv2d(latent_channels, bottleneck_channels, 3, padding=1)

        # Middle blocks
        self.mid_block1 = ResidualBlock(bottleneck_channels, bottleneck_channels, num_groups)
        self.mid_attn = AttentionBlock(bottleneck_channels, num_groups)
        self.mid_block2 = ResidualBlock(bottleneck_channels, bottleneck_channels, num_groups)

        # Upsampling blocks (reverse of encoder)
        self.up_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        current_res = 16  # starting resolution for decoder

        reversed_multipliers = list(reversed(channel_multipliers))
        for i, mult in enumerate(reversed_multipliers):
            out_ch = base_dim * mult
            block = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                block.append(ResidualBlock(current_channels, out_ch, num_groups))
                current_channels = out_ch
                if current_res in attention_resolutions:
                    block.append(AttentionBlock(current_channels, num_groups))
            self.up_blocks.append(block)
            if i < len(reversed_multipliers) - 1:
                self.up_blocks.append(nn.ModuleList([UpsampleBlock(current_channels)]))
                current_res *= 2

        # Output
        self.norm_out = nn.GroupNorm(num_groups, current_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(current_channels, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_channels, H/8, W/8) latent code

        Returns:
            x_hat: (B, C_out, H, W) reconstructed field
        """
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for block in self.up_blocks:
            for layer in block:
                h = layer(h)

        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    """Diagonal Gaussian distribution for VAE reparameterization."""

    def __init__(self, deterministic: bool = False):
        super().__init__()
        self.deterministic = deterministic

    def forward(self, moments: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            moments: (B, 2*C, H, W) concatenated mean and log-variance

        Returns:
            z: sampled latent (B, C, H, W)
            kl: KL divergence per sample
        """
        mean, logvar = moments.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)

        if self.deterministic:
            return mean, torch.zeros_like(mean)

        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)

        # KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        return z, kl


class P2VAE(nn.Module):
    """
    Pretrained Physics Variational Autoencoder (P2VAE).

    Compresses PDE field snapshots from c3p128 to c16p16 (12x compression).
    Based on SD-VAE architecture (Rombach et al., 2022).

    Two variants:
    - P2VAE-16M: base_dim=64
    - P2VAE-87M: base_dim=128

    Training objective:
        L_VAE = 0.5 * E[||x - x_hat||^2] + beta * KL(q(y|x) || p(y))
    with beta = 1e-3.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_dim: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        num_groups: int = 32,
        kl_weight: float = 1e-3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.kl_weight = kl_weight

        self.encoder = P2VAEEncoder(
            in_channels=in_channels,
            base_dim=base_dim,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_groups=num_groups,
        )

        self.decoder = P2VAEDecoder(
            out_channels=in_channels,
            base_dim=base_dim,
            latent_channels=latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_groups=num_groups,
        )

        self.gaussian = DiagonalGaussian()

        # Learnable scaling factor for latent space (following SD-VAE practice)
        self.scale_factor = nn.Parameter(torch.ones(1))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input field to latent distribution.

        Args:
            x: (B, C, H, W) input field

        Returns:
            z: (B, latent_channels, H/8, W/8) sampled latent
            kl: KL divergence
        """
        moments = self.encoder(x)
        z, kl = self.gaussian(moments)
        return z, kl

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to field.

        Args:
            z: (B, latent_channels, H/8, W/8) latent code

        Returns:
            x_hat: (B, C, H, W) reconstructed field
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            x: (B, C, H, W) input field

        Returns:
            x_hat: (B, C, H, W) reconstructed field
            loss: VAE loss (reconstruction + KL)
        """
        z, kl = self.encode(x)
        x_hat = self.decode(z)

        # Reconstruction loss
        recon_loss = 0.5 * F.mse_loss(x_hat, x, reduction="mean")

        # KL loss
        kl_loss = kl.mean()

        loss = recon_loss + self.kl_weight * kl_loss
        return x_hat, loss

    def get_latent(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get latent representation (mean if deterministic, sampled otherwise).

        Args:
            x: (B, C, H, W) input field
            deterministic: if True, return mean without sampling

        Returns:
            z: (B, latent_channels, H/8, W/8) latent code
        """
        moments = self.encoder(x)
        mean, logvar = moments.chunk(2, dim=1)
        if deterministic:
            return mean
        std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))
        return mean + std * torch.randn_like(mean)


def P2VAE_16M(**kwargs) -> P2VAE:
    """P2VAE with ~16M parameters (base_dim=64)."""
    return P2VAE(base_dim=64, **kwargs)


def P2VAE_87M(**kwargs) -> P2VAE:
    """P2VAE with ~87M parameters (base_dim=128)."""
    return P2VAE(base_dim=128, **kwargs)
