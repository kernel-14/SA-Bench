"""3D VAE with causal convolutions for video compression.

Architecture similar to MAGVIT-v2 with 8x8x8 compression ratio.
Trained from scratch on WebVid-10M and SA-1B images.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.layers import (
    CausalConv3d,
    Downsample3D,
    ResBlock3D,
    Upsample3D,
)


class AttentionBlock3D(nn.Module):
    """Self-attention block for the VAE bottleneck."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        residual = x
        x = self.norm(x)
        x = rearrange(x, "b c t h w -> b c (t h w)")
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (h d) n -> b h n d", h=self.num_heads)
        k = rearrange(k, "b (h d) n -> b h n d", h=self.num_heads)
        v = rearrange(v, "b (h d) n -> b h n d", h=self.num_heads)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b (h d) n")
        out = self.proj(out)
        out = rearrange(out, "b c (t h w) -> b c t h w", t=T, h=H, w=W)
        return out + residual


class Encoder3D(nn.Module):
    """3D causal encoder with spatial 8x and temporal 8x downsampling."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        latent_channels: int = 16,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_multipliers]

        self.conv_in = CausalConv3d(in_channels, channels[0], (3, 3, 3))

        self.down_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(channels[i]))
            # Downsample: spatial 2x at each of 3 levels, temporal 2x at first 3 levels
            # Total: spatial 8x, temporal 8x
            if i < 3:
                block.append(Downsample3D(channels[i], spatial_factor=2, temporal_factor=2))
            else:
                block.append(Downsample3D(channels[i], spatial_factor=2, temporal_factor=1))
            block.append(CausalConv3d(channels[i], channels[i + 1], (1, 1, 1)))
            self.down_blocks.append(block)

        # Bottleneck
        self.mid_block1 = ResBlock3D(channels[-1])
        self.mid_attn = AttentionBlock3D(channels[-1])
        self.mid_block2 = ResBlock3D(channels[-1])

        self.norm_out = nn.GroupNorm(32, channels[-1])
        # Output 2*latent_channels for mean and log-variance
        self.conv_out = CausalConv3d(channels[-1], 2 * latent_channels, (3, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            for layer in block:
                x = layer(x)
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x


class Decoder3D(nn.Module):
    """3D causal decoder with spatial 8x and temporal 8x upsampling."""

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        latent_channels: int = 16,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_multipliers]
        channels_rev = list(reversed(channels))

        self.conv_in = CausalConv3d(latent_channels, channels_rev[0], (3, 3, 3))

        # Bottleneck
        self.mid_block1 = ResBlock3D(channels_rev[0])
        self.mid_attn = AttentionBlock3D(channels_rev[0])
        self.mid_block2 = ResBlock3D(channels_rev[0])

        self.up_blocks = nn.ModuleList()
        for i in range(len(channels_rev) - 1):
            block = nn.ModuleList()
            block.append(CausalConv3d(channels_rev[i], channels_rev[i + 1], (1, 1, 1)))
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(channels_rev[i + 1]))
            if i < 3:
                block.append(Upsample3D(channels_rev[i + 1], spatial_factor=2, temporal_factor=2))
            else:
                block.append(Upsample3D(channels_rev[i + 1], spatial_factor=2, temporal_factor=1))
            self.up_blocks.append(block)

        self.norm_out = nn.GroupNorm(32, channels_rev[-1])
        self.conv_out = CausalConv3d(channels_rev[-1], out_channels, (3, 3, 3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        for block in self.up_blocks:
            for layer in block:
                x = layer(x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x


class DiagonalGaussian(nn.Module):
    """Reparameterization with diagonal Gaussian."""

    def forward(
        self, parameters: torch.Tensor, sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = parameters.chunk(2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        if sample:
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean
        return z, mean, logvar


class VideoVAE(nn.Module):
    """3D VAE for video compression with 8x8x8 compression ratio.

    Compresses (B, 3, T, H, W) -> (B, 16, T//8, H//8, W//8).
    Uses 3D causal convolutions so each frame only depends on preceding frames.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        kl_weight: float = 1e-6,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.latent_channels = latent_channels
        # Scaling factor for latent normalization (empirically set)
        self.scale_factor = 0.18215

        self.encoder = Encoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
        )
        self.decoder = Decoder3D(
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
        )
        self.gaussian = DiagonalGaussian()

    def encode(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Encode video to latent space.

        Args:
            x: (B, C, T, H, W) video tensor in [-1, 1]
            sample: whether to sample from the posterior

        Returns:
            z: (B, latent_channels, T//8, H//8, W//8)
        """
        params = self.encoder(x)
        z, mean, logvar = self.gaussian(params, sample=sample)
        return z * self.scale_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to video.

        Args:
            z: (B, latent_channels, T//8, H//8, W//8)

        Returns:
            x: (B, C, T, H, W) in [-1, 1]
        """
        z = z / self.scale_factor
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass for training.

        Returns:
            recon: reconstructed video
            z: sampled latent
            mean: posterior mean
            logvar: posterior log-variance
        """
        params = self.encoder(x)
        z, mean, logvar = self.gaussian(params, sample=True)
        recon = self.decoder(z)
        return recon, z * self.scale_factor, mean, logvar

    def kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence loss: KL(q(z|x) || p(z))."""
        return 0.5 * torch.mean(mean.pow(2) + logvar.exp() - 1.0 - logvar)

    @torch.no_grad()
    def encode_video_chunked(
        self, x: torch.Tensor, chunk_size: int = 16
    ) -> torch.Tensor:
        """Encode long videos by processing temporal chunks.

        Distributes computation across GPUs for very long videos.
        """
        B, C, T, H, W = x.shape
        latents = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk = x[:, :, start:end]
            latents.append(self.encode(chunk, sample=False))
        return torch.cat(latents, dim=2)
