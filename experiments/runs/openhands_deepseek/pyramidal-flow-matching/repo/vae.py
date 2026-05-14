"""3D Variational Autoencoder with 8x8x8 compression (similar to MAGVIT-v2).

Features:
- 3D causal convolution (each frame depends only on preceding frames)
- Asymmetric encoder-decoder with KL regularization
- 8x spatial and 8x temporal downsampling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CausalConv3d(nn.Module):
    """3D convolution with causal temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__()
        self.pad_t = (kernel_size[0] - 1) * stride[0]
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=(0, padding[1], padding[2]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, self.pad_t, 0))
        return self.conv(x)


class ResBlock3D(nn.Module):
    """3D residual block with group normalization."""

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        use_causal: bool = True,
    ):
        super().__init__()
        out_channels = out_channels or channels
        self.norm1 = nn.GroupNorm(32, channels)
        if use_causal:
            self.conv1 = CausalConv3d(channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        else:
            self.conv1 = nn.Conv3d(channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if use_causal:
            self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        else:
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = None
        if channels != out_channels:
            self.skip = nn.Conv3d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.skip is not None:
            x = self.skip(x)
        return x + h


class Downsample3D(nn.Module):
    """3D downsampling (can be spatial or temporal)."""

    def __init__(
        self,
        channels: int,
        downsample_t: bool = False,
        downsample_spatial: bool = True,
    ):
        super().__init__()
        stride = (
            (2 if downsample_t else 1),
            2 if downsample_spatial else 1,
            2 if downsample_spatial else 1,
        )
        self.conv = CausalConv3d(channels, channels, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D upsampling with nearest-neighbor interpolation."""

    def __init__(
        self,
        channels: int,
        upsample_t: bool = False,
        upsample_spatial: bool = True,
    ):
        super().__init__()
        self.upsample_t = upsample_t
        self.upsample_spatial = upsample_spatial
        self.conv = CausalConv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_factor = (
            2 if self.upsample_t else 1,
            2 if self.upsample_spatial else 1,
            2 if self.upsample_spatial else 1,
        )
        x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")
        return self.conv(x)


class Encoder3D(nn.Module):
    """3D VAE Encoder with 8x8x8 compression."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample: Tuple[bool, ...] = (True, True, True),
        spatial_downsample: Tuple[bool, ...] = (True, True, True),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_levels = len(channel_multipliers)
        channels_list = [base_channels * m for m in channel_multipliers]

        self.conv_in = CausalConv3d(in_channels, channels_list[0], kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.down_blocks = nn.ModuleList()
        ch = channels_list[0]
        for level in range(self.num_levels):
            block = nn.ModuleList()
            out_ch = channels_list[level]
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(ch, out_ch, dropout=dropout))
                ch = out_ch
            if level < self.num_levels - 1:
                block.append(
                    Downsample3D(
                        ch,
                        downsample_t=temporal_downsample[level],
                        downsample_spatial=spatial_downsample[level],
                    )
                )
            self.down_blocks.append(block)

        self.mid_blocks = nn.ModuleList([
            ResBlock3D(ch, ch, dropout=dropout),
            ResBlock3D(ch, ch, dropout=dropout),
        ])

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = CausalConv3d(ch, latent_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            for layer in block:
                x = layer(x)
        for layer in self.mid_blocks:
            x = layer(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class Decoder3D(nn.Module):
    """3D VAE Decoder."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_upsample: Tuple[bool, ...] = (True, True, True),
        spatial_upsample: Tuple[bool, ...] = (True, True, True),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_levels = len(channel_multipliers)
        channels_list = [base_channels * m for m in reversed(channel_multipliers)]
        ch = channels_list[0]

        self.conv_in = CausalConv3d(latent_channels, ch, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.mid_blocks = nn.ModuleList([
            ResBlock3D(ch, ch, dropout=dropout),
            ResBlock3D(ch, ch, dropout=dropout),
        ])

        self.up_blocks = nn.ModuleList()
        for level in range(self.num_levels):
            block = nn.ModuleList()
            out_ch = channels_list[level]
            for _ in range(num_res_blocks):
                block.append(ResBlock3D(ch, out_ch, dropout=dropout))
                ch = out_ch
            if level < self.num_levels - 1:
                block.append(
                    Upsample3D(
                        ch,
                        upsample_t=temporal_upsample[level],
                        upsample_spatial=spatial_upsample[level],
                    )
                )
            self.up_blocks.append(block)

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = CausalConv3d(ch, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for layer in self.mid_blocks:
            x = layer(x)
        for block in self.up_blocks:
            for layer in block:
                x = layer(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class DiagonalGaussian(nn.Module):
    """Diagonal Gaussian distribution for KL regularization."""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = z.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mean + eps * std
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=[1, 2, 3, 4])
        return sample, kl.mean()


class VideoVAE(nn.Module):
    """3D Video VAE with 8x8x8 compression rate.

    Architecture similar to MAGVIT-v2, using 3D causal convolution
    and asymmetric encoder-decoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample: Tuple[bool, ...] = (True, True, True),
        spatial_downsample: Tuple[bool, ...] = (True, True, True),
        kl_weight: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.latent_channels = latent_channels

        self.encoder = Encoder3D(
            in_channels=in_channels,
            latent_channels=latent_channels * 2,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            temporal_downsample=temporal_downsample,
            spatial_downsample=spatial_downsample,
            dropout=dropout,
        )

        self.quant_conv = nn.Conv3d(latent_channels * 2, latent_channels * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        self.decoder = Decoder3D(
            out_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            temporal_upsample=temporal_downsample,
            spatial_upsample=spatial_downsample,
            dropout=dropout,
        )

        self.gaussian = DiagonalGaussian()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.quant_conv(h)
        z, kl = self.gaussian(h)
        return z, kl

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, kl = self.encode(x)
        recon = self.decode(z)
        return recon, z, kl

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon, z, kl = self.forward(x)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = kl * self.kl_weight
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    @torch.no_grad()
    def encode_latents(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = h.chunk(2, dim=1)
        return mean
