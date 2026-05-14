## Code: models/p2vae.py

```python
## models/p2vae.py
"""P2VAE: Pretrained Physics Variational Autoencoder.

Compresses PDE field snapshots from c3p128 (3 channels, 128×128 spatial)
to c16p16 (16 channels, 16×16 spatial latent grid) using an SD-VAE
architecture (Rombach et al. 2022).

Two size variants are supported via the base_dim parameter:
  - P2VAE-16M: base_dim=64  (config: p2vae.variants.p2vae_16m.base_dim)
  - P2VAE-87M: base_dim=128 (config: p2vae.variants.p2vae_87m.base_dim)

Architecture overview:
  Encoder: Conv → [ResnetBlock × num_res_blocks → Downsample] × 3 stages
           → ResnetBlock × num_res_blocks (no downsample at stage 3)
           → MidBlock (ResnetBlock → AttnBlock → ResnetBlock)
           → GroupNorm → SiLU → Conv → quant_conv
           Output: (B, 2*latent_channels, 16, 16)

  Decoder: post_quant_conv → Conv → MidBlock
           → [Upsample → ResnetBlock × num_res_blocks] × 3 stages
           → ResnetBlock × num_res_blocks (no upsample at stage 0)
           → GroupNorm → SiLU → Conv
           Output: (B, out_channels, 128, 128)

Training objective (paper Eq. L_VAE, Section 3.3):
  L_VAE = 0.5 * E||x - x̂||² + β * KL(q_ω(y|x) || p(y))
  where β = kl_weight = 1e-3 (config: p2vae.kl_weight)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GroupNorm group count — SD-VAE standard.
# All channel counts in this file are multiples of 32 by construction.
# ---------------------------------------------------------------------------
_GROUPNORM_GROUPS: int = 32


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResnetBlock(nn.Module):
    """SD-VAE residual block with GroupNorm and SiLU activations.

    Architecture:
        x → GroupNorm → SiLU → Conv2d(3×3) → GroupNorm → SiLU
          → Dropout → Conv2d(3×3) → + shortcut(x)

    The second Conv2d is zero-initialized to stabilize early training,
    following the SD-VAE convention.

    Attributes:
        norm1: GroupNorm applied before the first convolution.
        conv1: First 3×3 convolution projecting in_channels → out_channels.
        norm2: GroupNorm applied before the second convolution.
        dropout: Dropout layer (identity when dropout=0.0).
        conv2: Second 3×3 convolution, zero-initialized.
        shortcut: 1×1 convolution for channel-matching skip connection,
            or nn.Identity when in_channels == out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize ResnetBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            dropout: Dropout probability applied between norm2 and conv2.
                Set to 0.0 (no dropout) per config: p2vae.dropout = 0.0.
        """
        super().__init__()

        self.norm1: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GROUPNORM_GROUPS,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GROUPNORM_GROUPS,
            num_channels=out_channels,
            eps=1e-6,
            affine=True,
        )
        self.dropout: nn.Module = (
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        )
        self.conv2: nn.Conv2d = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Zero-initialize the output convolution to stabilize early training.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)  # type: ignore[arg-type]

        # Skip connection: 1×1 conv if channel counts differ, else identity.
        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H, W).
        """
        residual: Tensor = self.shortcut(x)

        h: Tensor = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class AttnBlock(nn.Module):
    """Single-head self-attention block for mid-resolution features.

    Implements the SD-VAE attention block: spatial positions are treated
    as sequence elements, and a single attention head computes scaled
    dot-product attention over them.

    Architecture:
        x → GroupNorm → reshape to (B, H*W, C)
          → Q, K, V projections (1×1 conv)
          → scaled dot-product attention
          → output projection (1×1 conv)
          → reshape to (B, C, H, W)
          → + x (residual)

    Scale factor: 1 / sqrt(channels), applied to Q before dot product.

    Attributes:
        norm: GroupNorm applied before attention.
        q: Query projection (1×1 Conv2d).
        k: Key projection (1×1 Conv2d).
        v: Value projection (1×1 Conv2d).
        proj_out: Output projection (1×1 Conv2d).
    """

    def __init__(self, channels: int) -> None:
        """Initialize AttnBlock.

        Args:
            channels: Number of input and output channels. Must be divisible
                by _GROUPNORM_GROUPS (32).
        """
        super().__init__()

        self.norm: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GROUPNORM_GROUPS,
            num_channels=channels,
            eps=1e-6,
            affine=True,
        )
        self.q: nn.Conv2d = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )
        self.k: nn.Conv2d = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )
        self.v: nn.Conv2d = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out: nn.Conv2d = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )

        self._channels: int = channels

    def forward(self, x: Tensor) -> Tensor:
        """Apply single-head self-attention with residual connection.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W) with the same spatial
            dimensions as the input.
        """
        residual: Tensor = x
        b, c, h, w = x.shape

        # Normalize.
        h_norm: Tensor = self.norm(x)

        # Compute Q, K, V via 1×1 convolutions.
        q: Tensor = self.q(h_norm)  # (B, C, H, W)
        k: Tensor = self.k(h_norm)  # (B, C, H, W)
        v: Tensor = self.v(h_norm)  # (B, C, H, W)

        # Reshape to sequence format: (B, H*W, C).
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
        k = k.reshape(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)  # (B, H*W, C)

        # Scaled dot-product attention.
        # scale = 1 / sqrt(C) applied to Q.
        scale: float = float(c) ** -0.5
        # attn_weights: (B, H*W, H*W)
        attn_weights: Tensor = torch.bmm(q * scale, k.permute(0, 2, 1))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum of values: (B, H*W, C).
        out: Tensor = torch.bmm(attn_weights, v)

        # Reshape back to spatial format: (B, C, H, W).
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        # Output projection and residual.
        out = self.proj_out(out)
        return out + residual


class Downsample(nn.Module):
    """Strided convolution for spatial downsampling (2×).

    Uses a learned 3×3 convolution with stride 2, following the SD-VAE
    convention. This is preferred over average pooling as it allows the
    model to learn the optimal downsampling kernel.

    Attributes:
        conv: 3×3 Conv2d with stride=2 and padding=1.
    """

    def __init__(self, channels: int) -> None:
        """Initialize Downsample.

        Args:
            channels: Number of input and output channels (unchanged by
                downsampling — only spatial resolution is halved).
        """
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Downsample spatial resolution by 2×.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H//2, W//2).
        """
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor interpolation followed by convolution for 2× upsampling.

    Avoids checkerboard artifacts from transposed convolutions by using
    nearest-neighbor interpolation followed by a learned 3×3 convolution,
    following the SD-VAE convention.

    Attributes:
        conv: 3×3 Conv2d applied after interpolation.
    """

    def __init__(self, channels: int) -> None:
        """Initialize Upsample.

        Args:
            channels: Number of input and output channels (unchanged by
                upsampling — only spatial resolution is doubled).
        """
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Upsample spatial resolution by 2×.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H*2, W*2).
        """
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """SD-VAE encoder: compresses c3p128 → c(2*latent_channels)p16.

    Architecture:
        1. Initial projection: Conv2d(in_channels → base_dim*ch_mult[0])
        2. For each stage i in range(len(channel_multipliers)):
             - num_res_blocks ResnetBlocks at current channel count
             - Downsample (except at the last stage)
        3. Mid-block: ResnetBlock → AttnBlock → ResnetBlock
        4. Output: GroupNorm → SiLU → Conv2d(→ 2*latent_channels)

    With channel_multipliers=[1,2,4,4] and base_dim=64:
        Stage 0: 64 ch, 128×128 → Downsample → 64×64
        Stage 1: 128 ch, 64×64  → Downsample → 32×32
        Stage 2: 256 ch, 32×32  → Downsample → 16×16
        Stage 3: 256 ch, 16×16  → no Downsample (last stage)
        Mid:     256 ch, 16×16
        Output:  32 ch (2*16), 16×16

    Attributes:
        conv_in: Initial projection convolution.
        down_blocks: ModuleList of per-stage blocks (ResnetBlocks + optional Downsample).
        mid_block: Mid-resolution block (ResnetBlock → AttnBlock → ResnetBlock).
        norm_out: Final GroupNorm before output convolution.
        conv_out: Output convolution producing 2*latent_channels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 64,
        latent_channels: int = 16,
        channel_multipliers: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the Encoder.

        Args:
            in_channels: Number of input channels. From config:
                p2vae.in_channels = 3.
            base_dim: Base channel count. From config:
                p2vae.variants.p2vae_16m.base_dim = 64 or
                p2vae.variants.p2vae_87m.base_dim = 128.
            latent_channels: Number of latent channels. From config:
                p2vae.latent_channels = 16. The encoder outputs
                2*latent_channels (mean + logvar concatenated).
            channel_multipliers: Per-stage channel multipliers. From config:
                p2vae.channel_multipliers = [1, 2, 4, 4].
            num_res_blocks: Number of ResnetBlocks per stage. From config:
                p2vae.num_res_blocks = 2.
            dropout: Dropout probability in ResnetBlocks. From config:
                p2vae.dropout = 0.0.
        """
        super().__init__()

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 4]

        self._in_channels: int = in_channels
        self._base_dim: int = base_dim
        self._latent_channels: int = latent_channels
        self._channel_multipliers: List[int] = channel_multipliers
        self._num_res_blocks: int = num_res_blocks
        self._num_stages: int = len(channel_multipliers)

        # Compute channel counts per stage.
        stage_channels: List[int] = [base_dim * m for m in channel_multipliers]

        # Initial projection: in_channels → stage_channels[0].
        self.conv_in: nn.Conv2d = nn.Conv2d(
            in_channels,
            stage_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Build per-stage down blocks.
        # Each stage contains num_res_blocks ResnetBlocks followed by
        # a Downsample (except the last stage).
        self.down_blocks: nn.ModuleList = nn.ModuleList()
        current_channels: int = stage_channels[0]

        for stage_idx in range(self._num_stages):
            target_channels: int = stage_channels[stage_idx]
            stage_modules: List[nn.Module] = []

            for block_idx in range(num_res_blocks):
                in_ch: int = current_channels if block_idx == 0 else target_channels
                stage_modules.append(
                    ResnetBlock(in_ch, target_channels, dropout=dropout)
                )

            current_channels = target_channels

            # Downsample after all stages except the last.
            if stage_idx < self._num_stages - 1:
                stage_modules.append(Downsample(current_channels))

            self.down_blocks.append(nn.Sequential(*stage_modules))

        # Mid-block at the bottleneck resolution (16×16).
        # ResnetBlock → AttnBlock → ResnetBlock
        self.mid_block: nn.Sequential = nn.Sequential(
            ResnetBlock(current_channels, current_channels, dropout=dropout),
            AttnBlock(current_channels),
            ResnetBlock(current_channels, current_channels, dropout=dropout),
        )

        # Output normalization and projection.
        self.norm_out: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GROUPNORM_GROUPS,
            num_channels=current_channels,
            eps=1e-6,
            affine=True,
        )
        # Output 2*latent_channels: first half = mu, second half = logvar.
        self.conv_out: nn.Conv2d = nn.Conv2d(
            current_channels,
            2 * latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._out_channels: int = current_channels

    def forward(self, x: Tensor) -> Tensor:
        """Encode input field to concatenated (mu, logvar) latent.

        Args:
            x: Input tensor of shape (B, in_channels, 128, 128).

        Returns:
            Tensor of shape (B, 2*latent_channels, 16, 16) containing
            concatenated mean and log-variance along the channel dimension.
            The first latent_channels channels are mu; the last
            latent_channels channels are logvar.
        """
        # Initial projection.
        h: Tensor = self.conv_in(x)

        # Downsampling stages.
        for down_block in self.down_blocks:
            h = down_block(h)

        # Mid-block at bottleneck resolution.
        h = self.mid_block(h)

        # Output projection.
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """SD-VAE decoder: reconstructs c3p128 from c(latent_channels)p16.

    Architecture (mirror of Encoder):
        1. Initial projection: Conv2d(latent_channels → stage_channels[-1])
        2. Mid-block: ResnetBlock → AttnBlock → ResnetBlock
        3. For each stage i in reverse(range(len(channel_multipliers))):
             - num_res_blocks ResnetBlocks at current channel count
             - Upsample (except at the last stage, i.e., stage 0)
        4. Output: GroupNorm → SiLU → Conv2d(→ out_channels)

    With channel_multipliers=[1,2,4,4] and base_dim=64:
        Input:   16 ch, 16×16
        Mid:     256 ch, 16×16
        Stage 3: 256 ch, 16×16 → Upsample → 32×32
        Stage 2: 256 ch, 32×32 → Upsample → 64×64
        Stage 1: 128 ch, 64×64 → Upsample → 128×128
        Stage 0: 64 ch, 128×128 → no Upsample (last stage)
        Output:  3 ch, 128×128

    Attributes:
        conv_in: Initial projection from latent to bottleneck channels.
        mid_block: Mid-resolution block (ResnetBlock → AttnBlock → ResnetBlock).
        up_blocks: ModuleList of per-stage blocks (ResnetBlocks + optional Upsample).
        norm_out: Final GroupNorm before output convolution.
        conv_out: Output convolution producing out_channels.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        base_dim: int = 64,
        out_channels: int = 3,
        channel_multipliers: Optional[List[int]] = None,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the Decoder.

        Args:
            latent_channels: Number of latent channels. From config:
                p2vae.latent_channels = 16.
            base_dim: Base channel count. From config:
                p2vae.variants.p2vae_16m.base_dim = 64 or
                p2vae.variants.p2vae_87m.base_dim = 128.
            out_channels: Number of output channels. From config:
                p2vae.out_channels = 3.
            channel_multipliers: Per-stage channel multipliers. From config:
                p2vae.channel_multipliers = [1, 2, 4, 4].
            num_res_blocks: Number of ResnetBlocks per stage. From config:
                p2vae.num_res_blocks = 2.
            dropout: Dropout probability in ResnetBlocks. From config:
                p2vae.dropout = 0.0.
        """
        super().__init__()

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 4]

        self._latent_channels: int = latent_channels
        self._base_dim: int = base_dim
        self._out_channels: int = out_channels
        self._channel_multipliers: List[int] = channel_multipliers
        self._num_res_blocks: int = num_res_blocks
        self._num_stages: int = len(channel_multipliers)

        # Compute channel counts per stage (same as encoder).
        stage_channels: List[int] = [base_dim * m for m in channel_multipliers]
        bottleneck_channels: int = stage_channels[-1]

        # Initial projection: latent_channels → bottleneck_channels.
        self.conv_in: nn.Conv2d = nn.Conv2d(
            latent_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Mid-block at bottleneck resolution (16×16).
        self.mid_block: nn.Sequential = nn.Sequential(
            ResnetBlock(bottleneck_channels, bottleneck_channels, dropout=dropout),
            AttnBlock(bottleneck_channels),
            ResnetBlock(bottleneck_channels, bottleneck_channels, dropout=dropout),
        )

        # Build per-stage up blocks in reverse order.
        # Stage order: [3, 2, 1, 0] (from bottleneck to full resolution).
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        current_channels: int = bottleneck_channels

        for stage_idx in reversed(range(self._num_stages)):
            target_channels: int = stage_channels[stage_idx]
            stage_modules: List[nn.Module] = []

            for block_idx in range(num_res_blocks):
                in_ch: int = current_channels if block_idx == 0 else target_channels
                stage_modules.append(
                    ResnetBlock(in_ch, target_channels, dropout=dropout)
                )

            current_channels = target_channels

            # Upsample after all stages except the last (stage 0 = full resolution).
            if stage_idx > 0:
                stage_modules.append(Upsample(current_channels))

            self.up_blocks.append(nn.Sequential(*stage_modules))

        # Output normalization and projection.
        self.norm_out: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GROUPNORM_GROUPS,
            num_channels=current_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv_out: nn.Conv2d = nn.Conv2d(
            current_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent representation to reconstructed field.

        Args:
            z: Latent tensor of shape (B, latent_channels, 16, 16).

        Returns:
            Reconstructed tensor of shape (B, out_channels, 128, 128).
        """
        # Initial projection from latent to bottleneck channels.
        h: Tensor = self.conv_in(z)

        # Mid-block at bottleneck resolution.
        h = self.mid_block(h)

        # Upsampling stages (reverse order: bottleneck → full resolution).
        for up_block in self.up_blocks:
            h = up_block(h)

        # Output projection.
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


# ---------------------------------------------------------------------------
# P2VAE
# ---------------------------------------------------------------------------


class P2VAE(nn.Module):
    """Pretrained Physics Variational Autoencoder (P2VAE).

    Compresses PDE field snapshots from c3p128 to c16p16 latent grids
    using an SD-VAE architecture. Supports two size variants:
      - P2VAE-16M: base_dim=64
      - P2VAE-87M: base_dim=128

    The VAE training objective (paper Section 3.3, Eq. L_VAE):
        L_VAE = 0.5 * E||x - x̂||² + β * KL(q_ω(y|x) || p(y))
    where β = kl_weight = 1e-3.

    For FMT training, get_latent() returns the posterior mean μ (no
    sampling), providing deterministic latents as described in the paper's
    Shared Knowledge: "p2vae.get_latent(x) always returns mu (the mean
    of the posterior) without sampling, used for deterministic FMT training."

    Attributes:
        encoder: Encoder module compressing c3p128 → c32p16 (mu+logvar).
        decoder: Decoder module reconstructing c16p16 → c3p128.
        quant_conv: 1×1 Conv2d projecting encoder output (2*latent_channels).
        post_quant_conv: 1×1 Conv2d projecting latent before decoding.
        latent_channels: Number of latent channels (16 from config).
        kl_weight: KL divergence weight β (1e-3 from config).
    """

    def __init__(self, config: Dict) -> None:
        """Initialize P2VAE from a configuration dictionary.

        The config dict should contain the keys from config.yaml under
        the 'p2vae' section, with the variant-specific 'base_dim' merged in.
        Expected keys:
            in_channels (int): Input channels. Default: 3.
            out_channels (int): Output channels. Default: 3.
            base_dim (int): Base channel count. Default: 64 (16M variant).
            latent_channels (int): Latent channels. Default: 16.
            channel_multipliers (list): Per-stage multipliers. Default: [1,2,4,4].
            num_res_blocks (int): ResNet blocks per stage. Default: 2.
            dropout (float): Dropout probability. Default: 0.0.
            kl_weight (float): KL divergence weight β. Default: 1e-3.

        Args:
            config: Configuration dictionary. All keys have defaults so
                partial configs are supported.
        """
        super().__init__()

        # Extract configuration with defaults matching config.yaml.
        in_channels: int = int(config.get("in_channels", 3))
        out_channels: int = int(config.get("out_channels", 3))
        base_dim: int = int(config.get("base_dim", 64))
        self.latent_channels: int = int(config.get("latent_channels", 16))
        channel_multipliers: List[int] = list(
            config.get("channel_multipliers", [1, 2, 4, 4])
        )
        num_res_blocks: int = int(config.get("num_res_blocks", 2))
        dropout: float = float(config.get("dropout", 0.0))
        self.kl_weight: float = float(config.get("kl_weight", 1e-3))

        # Encoder: c3p128 → c(2*latent_channels)p16
        self.encoder: Encoder = Encoder(
            in_channels=in_channels,
            base_dim=base_dim,
            latent_channels=self.latent_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        # Decoder: c(latent_channels)p16 → c3p128
        self.decoder: Decoder = Decoder(
            latent_channels=self.latent_channels,
            base_dim=base_dim,
            out_channels=out_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        # Quantization convolutions (1×1) for better-conditioned latent space.
        # quant_conv: applied to encoder output before splitting mu/logvar.
        self.quant_conv: nn.Conv2d = nn.Conv2d(
            2 * self.latent_channels,
            2 * self.latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # post_quant_conv: applied to z before decoding.
        self.post_quant_conv: nn.Conv2d = nn.Conv2d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=1,