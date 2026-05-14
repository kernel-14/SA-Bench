## Code: models/vae_3d.py

```python
## models/vae_3d.py
"""3D Causal Variational Autoencoder for Pyramidal Flow Matching.

Implements the 3D causal VAE described in the paper (Section 4.1, Appendix B):
- Architecture similar to MAGVIT-v2 with 3D causal convolutions
- 8×8×8 compression ratio (8× spatial in H and W, 8× temporal)
- Asymmetric encoder-decoder with KL regularization
- Trained from scratch on WebVid-10M and 6.9M SAM images

The causality constraint ensures each frame depends only on preceding frames,
which is critical for autoregressive video generation.

Tensor convention: videos are [B, C, T, H, W] in pixel space and
[B, latent_dim, T//8, H//8, W//8] in latent space.

Usage:
    from models.vae_3d import VAE3D

    vae = VAE3D(config)

    # Encode video to latent
    z = vae.encode(video_tensor)  # [B, 16, T//8, H//8, W//8]

    # Decode latent to video
    recon = vae.decode(z)  # [B, 3, T, H, W]

    # Full forward pass (training)
    outputs = vae.forward(video_tensor)
    loss = vae.vae_loss(
        outputs['recon'], video_tensor,
        outputs['mu'], outputs['logvar'],
        kl_weight=1e-6
    )
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)

## ---------------------------------------------------------------------------
## Constants
## ---------------------------------------------------------------------------
_GROUPNORM_NUM_GROUPS: int = 32
_LOGVAR_MIN: float = -30.0
_LOGVAR_MAX: float = 20.0
_DEFAULT_CHUNK_SIZE: int = 32   # Frames per chunk for long video encoding
_DEFAULT_CHUNK_OVERLAP: int = 4  # Overlap frames for causal continuity


## ---------------------------------------------------------------------------
## CausalConv3d
## ---------------------------------------------------------------------------

class CausalConv3d(nn.Module):
    """3D convolution with causal (left-only) temporal padding.

    Enforces the paper's causality constraint: each frame depends only on
    preceding frames. Standard nn.Conv3d with symmetric temporal padding
    would allow future frames to influence current frame encoding, which
    would break autoregressive generation.

    Temporal padding: (kernel_t - 1) zeros prepended, 0 appended.
    Spatial padding: symmetric, (kernel_s - 1) // 2 on each side.

    The padding is applied via F.pad before the convolution, with the
    underlying nn.Conv3d using padding=0 in the temporal dimension.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Tuple (k_t, k_h, k_w) of kernel sizes.
        stride: Tuple (s_t, s_h, s_w) of strides.
        temporal_pad: Number of zeros prepended in temporal dimension.
        spatial_pad_h: Symmetric padding in height dimension.
        spatial_pad_w: Symmetric padding in width dimension.
        conv: Underlying nn.Conv3d with zero temporal padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int, int] = 3,
        stride: int | Tuple[int, int, int] = 1,
        bias: bool = True,
    ) -> None:
        """Initializes CausalConv3d.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            kernel_size: Kernel size as int (applied to all dims) or
                tuple (k_t, k_h, k_w). Defaults to 3.
            stride: Stride as int (applied to all dims) or tuple
                (s_t, s_h, s_w). Defaults to 1.
            bias: Whether to include a bias term. Defaults to True.
        """
        super().__init__()

        # Normalize kernel_size and stride to tuples
        if isinstance(kernel_size, int):
            self.kernel_size: Tuple[int, int, int] = (
                kernel_size, kernel_size, kernel_size
            )
        else:
            self.kernel_size = tuple(kernel_size)  # type: ignore[assignment]

        if isinstance(stride, int):
            self.stride: Tuple[int, int, int] = (stride, stride, stride)
        else:
            self.stride = tuple(stride)  # type: ignore[assignment]

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels

        # ----------------------------------------------------------------
        # Compute padding amounts
        # ----------------------------------------------------------------
        k_t, k_h, k_w = self.kernel_size

        # Temporal: causal (left-only) padding
        # Prepend (k_t - 1) zeros so output[t] depends only on input[0..t]
        self.temporal_pad: int = k_t - 1

        # Spatial: symmetric padding (standard)
        self.spatial_pad_h: int = (k_h - 1) // 2
        self.spatial_pad_w: int = (k_w - 1) // 2

        # ----------------------------------------------------------------
        # Underlying conv with no temporal padding (we handle it manually)
        # ----------------------------------------------------------------
        self.conv: nn.Conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, self.spatial_pad_h, self.spatial_pad_w),
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies causal 3D convolution.

        Args:
            x: Input tensor of shape [B, C, T, H, W].

        Returns:
            Output tensor of shape [B, C_out, T_out, H_out, W_out] where:
                T_out = ceil(T / s_t) (with causal padding)
                H_out = ceil(H / s_h) (with symmetric padding)
                W_out = ceil(W / s_w) (with symmetric padding)
        """
        # F.pad takes padding in reverse dimension order:
        # (W_left, W_right, H_left, H_right, T_left, T_right)
        # We only need temporal padding here; spatial is handled by conv.
        if self.temporal_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_pad, 0))

        return self.conv(x)

    def extra_repr(self) -> str:
        """Returns string representation for debugging."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"temporal_pad={self.temporal_pad}"
        )


## ---------------------------------------------------------------------------
## ResBlock3D
## ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """3D residual block with causal convolutions.

    Structure:
        x → CausalConv3d → GroupNorm → SiLU
          → CausalConv3d → GroupNorm
          → + skip_connection(x)
          → SiLU

    If in_channels != out_channels, a 1×1×1 CausalConv3d is used for the
    skip connection. Otherwise, the skip connection is the identity.

    The second conv's output projection is zero-initialized to improve
    training stability at the start of training.

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        conv1: First 3×3×3 causal convolution.
        norm1: GroupNorm after conv1.
        conv2: Second 3×3×3 causal convolution.
        norm2: GroupNorm after conv2.
        skip_conv: 1×1×1 conv for channel projection (or None if same channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """Initializes ResBlock3D.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
        """
        super().__init__()

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels

        # ----------------------------------------------------------------
        # Main path: conv → norm → act → conv → norm
        # ----------------------------------------------------------------
        self.norm1: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GROUPNORM_NUM_GROUPS,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv1: CausalConv3d = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )

        self.norm2: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GROUPNORM_NUM_GROUPS,
            num_channels=out_channels,
            eps=1e-6,
            affine=True,
        )
        self.conv2: CausalConv3d = CausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )

        # ----------------------------------------------------------------
        # Skip connection: identity or 1×1×1 projection
        # ----------------------------------------------------------------
        if in_channels != out_channels:
            self.skip_conv: Optional[CausalConv3d] = CausalConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            )
        else:
            self.skip_conv = None

        # ----------------------------------------------------------------
        # Zero-initialize the output conv for training stability
        # ----------------------------------------------------------------
        nn.init.zeros_(self.conv2.conv.weight)
        if self.conv2.conv.bias is not None:
            nn.init.zeros_(self.conv2.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the residual block.

        Args:
            x: Input tensor of shape [B, in_channels, T, H, W].

        Returns:
            Output tensor of shape [B, out_channels, T, H, W].
        """
        # Skip connection
        if self.skip_conv is not None:
            skip: Tensor = self.skip_conv(x)
        else:
            skip = x

        # Main path: pre-norm style
        h: Tensor = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + skip


## ---------------------------------------------------------------------------
## Encoder3D
## ---------------------------------------------------------------------------

class Encoder3D(nn.Module):
    """3D causal encoder for the VAE.

    Maps pixel video [B, C_in, T, H, W] to latent parameters
    [B, 2*latent_channels, T//8, H//8, W//8] (mu and logvar concatenated).

    Architecture:
        1. Initial projection: CausalConv3d(C_in, base_channels)
        2. Four resolution levels with channel_multipliers
        3. Three stride-2 downsampling operations (at levels 0→1, 1→2, 2→3)
        4. Middle block with two ResBlock3Ds
        5. Output projection to 2*latent_channels

    Attributes:
        in_channels: Number of input pixel channels (3 for RGB).
        latent_channels: Number of latent channels (16 from config).
        base_channels: Base channel count (128 from config).
        channel_multipliers: Per-level channel multipliers ([1,2,4,4]).
        num_res_blocks: Number of ResBlock3Ds per level (2 from config).
        channels_per_level: Computed channel counts per level.
        initial_conv: Initial projection convolution.
        down_blocks: nn.ModuleList of downsampling stages.
        middle_block: Middle residual blocks.
        output_norm: Final GroupNorm before output projection.
        output_conv: Final 1×1×1 convolution to 2*latent_channels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        base_channels: int = 128,
        channel_multipliers: Optional[List[int]] = None,
        num_res_blocks: int = 2,
    ) -> None:
        """Initializes Encoder3D.

        Args:
            in_channels: Number of input pixel channels. Defaults to 3 (RGB).
            latent_channels: Number of latent channels. Defaults to 16.
            base_channels: Base channel count. Defaults to 128.
            channel_multipliers: Per-level channel multipliers.
                Defaults to [1, 2, 4, 4] (4 levels).
            num_res_blocks: Number of ResBlock3Ds per level. Defaults to 2.
        """
        super().__init__()

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 4]

        self.in_channels: int = in_channels
        self.latent_channels: int = latent_channels
        self.base_channels: int = base_channels
        self.channel_multipliers: List[int] = channel_multipliers
        self.num_res_blocks: int = num_res_blocks
        self.num_levels: int = len(channel_multipliers)

        # Compute channel counts per level
        self.channels_per_level: List[int] = [
            base_channels * m for m in channel_multipliers
        ]

        # ----------------------------------------------------------------
        # Initial projection: C_in → base_channels
        # ----------------------------------------------------------------
        self.initial_conv: CausalConv3d = CausalConv3d(
            in_channels=in_channels,
            out_channels=self.channels_per_level[0],
            kernel_size=3,
            stride=1,
        )

        # ----------------------------------------------------------------
        # Downsampling stages
        # Each stage: [optional channel proj] + res_blocks + [downsample]
        # Downsampling happens at transitions between levels 0→1, 1→2, 2→3
        # Level 3 has no downsampling (it's the bottleneck level)
        # ----------------------------------------------------------------
        self.down_blocks: nn.ModuleList = nn.ModuleList()

        for level_idx in range(self.num_levels):
            in_ch: int = self.channels_per_level[level_idx]
            out_ch: int = self.channels_per_level[level_idx]

            stage_modules: List[nn.Module] = []

            # Channel projection from previous level (except first level)
            if level_idx > 0:
                prev_ch: int = self.channels_per_level[level_idx - 1]
                stage_modules.append(
                    CausalConv3d(
                        in_channels=prev_ch,
                        out_channels=in_ch,
                        kernel_size=1,
                        stride=1,
                    )
                )

            # Residual blocks at this level
            for _ in range(num_res_blocks):
                stage_modules.append(ResBlock3D(in_ch, out_ch))

            # Downsampling (stride-2 conv) at all levels except the last
            # 3 downsampling ops → 8× spatial and 8× temporal compression
            if level_idx < self.num_levels - 1:
                stage_modules.append(
                    CausalConv3d(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        kernel_size=3,
                        stride=2,  # 2× in all dimensions (T, H, W)
                    )
                )

            self.down_blocks.append(nn.Sequential(*stage_modules))

        # ----------------------------------------------------------------
        # Middle block: two ResBlock3Ds at the bottleneck resolution
        # ----------------------------------------------------------------
        bottleneck_ch: int = self.channels_per_level[-1]
        self.middle_block: nn.Sequential = nn.Sequential(
            ResBlock3D(bottleneck_ch, bottleneck_ch),
            ResBlock3D(bottleneck_ch, bottleneck_ch),
        )

        # ----------------------------------------------------------------
        # Output projection: bottleneck_ch → 2*latent_channels
        # ----------------------------------------------------------------
        self.output_norm: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GROUPNORM_NUM_GROUPS,
            num_channels=bottleneck_ch,
            eps=1e-6,
            affine=True,
        )
        self.output_conv: CausalConv3d = CausalConv3d(
            in_channels=bottleneck_ch,
            out_channels=2 * latent_channels,
            kernel_size=1,
            stride=1,
        )

        # Zero-initialize output conv for stable training start
        nn.init.zeros_(self.output_conv.conv.weight)
        if self.output_conv.conv.bias is not None:
            nn.init.zeros_(self.output_conv.conv.bias)

        logger.info(
            "Encoder3D initialized: in_channels=%d, latent_channels=%d, "
            "base_channels=%d, channel_multipliers=%s, num_res_blocks=%d, "
            "channels_per_level=%s",
            in_channels,
            latent_channels,
            base_channels,
            channel_multipliers,
            num_res_blocks,
            self.channels_per_level,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encodes pixel video to latent parameters.

        Args:
            x: Pixel video tensor of shape [B, C_in, T, H, W].
                Should be normalized to [-1, 1].

        Returns:
            Tensor of shape [B, 2*latent_channels, T//8, H//8, W//8]
            containing concatenated mu and logvar along the channel dim.
        """
        # Initial projection
        h: Tensor = self.initial_conv(x)

        # Downsampling stages
        for down_block in self.down_blocks:
            h = down_block(h)

        # Middle block
        h = self.middle_block(h)

        # Output projection
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)

        return h  # [B, 2*latent_channels, T//8, H//8, W//8]


## ---------------------------------------------------------------------------
## Decoder3D
## ---------------------------------------------------------------------------

class Decoder3D(nn.Module):
    """3D causal decoder for the VAE.

    Maps latent [B, latent_channels, T//8, H//8, W//8] back to pixel video
    [B, C_out, T, H, W].

    Architecture (mirror of encoder with nearest-neighbor upsampling):
        1. Input projection: latent_channels → bottleneck_channels
        2. Middle block with two ResBlock3Ds
        3. Four resolution levels (reverse order of encoder)
        4. Three nearest-neighbor upsampling operations
        5. Output projection to C_out with Tanh activation

    Attributes:
        latent_channels: Number of latent channels (16 from config).
        out_channels: Number of output pixel channels (3 for RGB).
        base_channels: Base channel count (128 from config).
        channel_multipliers: Per-level channel multipliers ([1,2,4,4]).
        num_res_blocks: Number of ResBlock3Ds per level (2 from config).
        channels_per_level: Computed channel counts per level.
        input_conv: Initial projection from latent_channels.
        middle_block: Middle residual blocks.
        up_blocks: nn.ModuleList of upsampling stages.
        output_norm: Final GroupNorm before output projection.
        output_conv: Final convolution to C_out.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Optional[List[int]] = None,
        num_res_blocks: int = 2,
    ) -> None:
        """Initializes Decoder3D.

        Args:
            latent_channels: Number of latent channels. Defaults to 16.
            out_channels: Number of output pixel channels. Defaults to 3 (RGB).
            base_channels: Base channel count. Defaults to 128.
            channel_multipliers: Per-level channel multipliers.
                Defaults to [1, 2, 4, 4] (4 levels, reversed for decoder).
            num_res_blocks: Number of ResBlock3Ds per level. Defaults to 2.
        """
        super().__init__()

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 4]

        self.latent_channels: int = latent_channels
        self.out_channels: int = out_channels
        self.base_channels: int = base_channels
        self.channel_multipliers: List[int] = channel_multipliers
        self.num_res_blocks: int = num_res_blocks
        self.num_levels: int = len(channel_multipliers)

        # Compute channel counts per level (same as encoder)
        self.channels_per_level: List[int] = [
            base_channels * m for m in channel_multipliers
        ]

        # Bottleneck is the last (highest-channel) level
        bottleneck_ch: int = self.channels_per_level[-1]

        # ----------------------------------------------------------------
        # Input projection: latent_channels → bottleneck_channels
        # ----------------------------------------------------------------
        self.input_conv: CausalConv3d = CausalConv3d(
            in_channels=latent_channels,
            out_channels=bottleneck_ch,
            kernel_size=1,
            stride=1,
        )

        # ----------------------------------------------------------------
        # Middle block: two ResBlock3Ds at bottleneck resolution
        # ----------------------------------------------------------------
        self.middle_block: nn.Sequential = nn.Sequential(
            ResBlock3D(bottleneck_ch, bottleneck_ch),
            ResBlock3D(bottleneck_ch, bottleneck_ch),
        )

        # ----------------------------------------------------------------
        # Upsampling stages (reverse order of encoder)
        # Level order: 3 → 2 → 1 → 0
        # Upsampling happens between levels: 3→2, 2→1, 1→0
        # ----------------------------------------------------------------
        self.up_blocks: nn.ModuleList = nn.ModuleList()

        # Iterate levels in reverse: from bottleneck (level 3) to finest (level 0)
        for level_idx in range(self.num_levels - 1, -1, -1):
            in_ch: int = self.channels_per_level[level_idx]
            out_ch: int = self.channels_per_level[level_idx]

            stage_modules: List[nn.Module] = []

            # Residual blocks at this level
            for _ in range(num_res_blocks):
                stage_modules.append(ResBlock3D(in_ch, out_ch))

            # Upsampling (nearest-neighbor + conv) at all levels except level 0
            # Upsampling happens AFTER the res blocks at each level
            if level_idx > 0:
                # Nearest-neighbor upsample is applied in forward() before
                # the next level's channel projection. We store the post-upsample
                # conv here as part of this stage.
                next_ch: int = self.channels_per_level[level_idx - 1]
                # Channel projection after upsampling
                stage_modules.append(
                    CausalConv3d(
                        in_channels=out_ch,
                        out_channels=next_ch,
                        kernel_size=3,
                        stride=1,
                    )
                )

            self.up_blocks.append(nn.Sequential(*stage_modules))

        # ----------------------------------------------------------------
        # Output projection: base_channels → C_out
        # ----------------------------------------------------------------
        self.output_norm: nn.GroupNorm = nn.GroupNorm(
            num_groups=_GROUPNORM_NUM_GROUPS,
            num_channels=self.channels_per_level[0],
            eps=1e-6,
            affine=True,
        )
        self.output_conv: CausalConv3d = CausalConv3d(
            in_channels=self.channels_per_level[0],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
        )

        logger.info(
            "Decoder3D initialized: latent_channels=%d, out_channels=%d, "
            "base_channels=%d, channel_multipliers=%s, num_res_blocks=%d",
            latent_channels,
            out_channels,
            base_channels,
            channel_multipliers,
            num_res_blocks,
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decodes latent to pixel video.

        Args:
            z: Latent tensor of shape [B, latent_channels, T//8, H//8, W//8].

        Returns:
            Reconstructed pixel video of shape [B, C_out, T, H, W],
            with values in [-1, 1] (Tanh output).
        """
        # Input projection
        h: Tensor = self.input_conv(z)

        # Middle block
        h = self.middle_block(h)

        # Upsampling stages (reverse order: level 3 → 2 → 1 → 0)
        for level_rev_idx, up_block in enumerate(self.up_blocks):
            # Determine the actual level index (reversed)
            level_idx: int = self.num_levels - 1 - level_rev_idx

            # Apply residual blocks (and channel projection if not last level)
            # The up_block contains: res_blocks + [channel_proj_conv if level > 0]
            # We need to apply nearest-neighbor upsample BEFORE the channel proj conv
            # but AFTER the res blocks.
            # To handle this cleanly, we split the up_block into res_blocks and
            # the optional channel proj conv.

            # Count the modules: num_res_blocks res blocks + optionally 1 channel proj
            num_modules: int = len(up_block)
            has_upsample: bool = (level_idx > 0)

            if has_upsample:
                # Apply res blocks (all but last module)
                for i in range(num_modules - 1):
                    h = up_block[i](h)

                # Apply nearest-neighbor 3D upsampling (2× in T, H, W)
                h = F.interpolate(
                    h,
                    scale_factor=(2.0, 2.0, 2.0),
                    mode="nearest",
                )

                # Apply channel projection conv (last module)
                h = up_block[-1](h)
            else:
                # Level 0: no upsampling, just apply all res blocks
                h = up_block(h)

        # Output projection
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)

        # Tanh to bound output to [-1, 1]
        h = torch.tanh(h)

        return h  # [B, C_out, T, H, W]


## ---------------------------------------------------------------------------
## VAE3D
## ---------------------------------------------------------------------------

class VAE3D(nn.Module):
    """3D Causal Variational Autoencoder for video compression.

    Implements the 3D VAE described in the paper (Section 4.1, Appendix B):
    - MAGVIT-v2-style architecture with 3D causal convolutions
    - 8×8×8 compression ratio (8× spatial, 8× temporal)
    - Asymmetric encoder-decoder with KL regularization
    - Supports chunked encoding for long videos (config.vae.scatter_long_videos)

    The VAE compresses pixel videos to a compact latent space that the
    PyramidFlowModel operates on. The latent z (or mu for deterministic
    encoding) is what gets downsampled/upsampled in the spatial pyramid.

    Attributes:
        latent_channels: Number of latent channels (16 from config).
        kl_weight: KL divergence loss weight (1e-6 from config).
        scatter_long_videos: Whether to use chunked encoding for long videos.
        spatial_compression: Spatial compression factor (8 from config).
        temporal_compression: Temporal compression factor (8 from config).
        encoder: Encoder3D module.
        decoder: Decoder3D module.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes VAE3D from the project config.

        Reads all required values from configs/default.yaml via the
        omegaconf DictConfig (or plain dict) passed as ``config``.

        Args:
            config: Project configuration dictionary. Expected keys under
                config['vae']:
                - latent_channels (int): 16
                - base_channels (int): 128
                - channel_multipliers (list[int]): [1, 2, 4, 4]
                - num_res_blocks (int): 2
                - kl_weight (float): 1e-6
                - scatter_long_videos (bool): True
                - spatial_compression (int): 8
                - temporal_compression (int): 8
                - pretrained_path (str or None): path to pretrained VAE
        """
        super().__init__()

        # ----------------------------------------------------------------
        # Parse config
        # ----------------------------------------------------------------
        vae_cfg: Dict[str, Any] = config.get("vae", {})

        self.latent_channels: int = int(vae_cfg.get("latent_channels", 16))
        self.base_channels: int = int(vae_cfg.get("base_channels", 128))
        self.channel_multipliers: List[int] = list(
            vae_cfg.get("channel_multipliers", [1, 2, 4, 4])
        )
        self.num_res_blocks: int = int(vae_cfg.get("num_res_blocks", 2))
        self.kl_weight: float = float(vae_cfg.get("kl_weight", 1.0e-6))
        self.scatter_long_videos: bool = bool(
            vae_cfg.get("scatter_long_videos", True)
        )
        self.spatial_compression: int = int(
            vae_cfg.get("spatial_compression", 8)
        )
        self.temporal_compression: int = int(
            vae_cfg.get("temporal_compression", 8)
        )

        # Input/output pixel channels (RGB = 3)
        self._pixel_channels: int = 3

        # ----------------------------------------------------------------
        # Build encoder and decoder