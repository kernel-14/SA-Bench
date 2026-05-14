## models/frvae/decoder.py
"""VQ-GAN style decoder for the FR-VAE.

Reconstructs full-resolution RGB images (256×256×3) from the aggregated
quantized latent feature map f_tilde ∈ R^(B×768×16×16) produced by the
ResidualQuantizer.

Architecture overview (16×16 → 256×256, 4 upsampling stages of 2×):
    f_tilde [B, 768, 16, 16]
        → initial_conv (768→512)
        → AttentionBlock at 16×16 (global context)
        → Stage 1: ResBlock(512→512) + ResBlock(512→256) + Upsample×2  → [B, 256, 32, 32]
        → Stage 2: ResBlock(256→256) + ResBlock(256→128) + Upsample×2  → [B, 128, 64, 64]
        → Stage 3: ResBlock(128→128) + ResBlock(128→64)  + Upsample×2  → [B, 64, 128, 128]
        → Stage 4: ResBlock(64→64)                       + Upsample×2  → [B, 64, 256, 256]
        → GroupNorm(32, 64) + SiLU()
        → out_conv: Conv2d(64→3, 3×3) + tanh
    x_hat [B, 3, 256, 256]

Design references:
    - VQ-GAN decoder: Esser et al. 2021 (Taming Transformers)
    - GroupNorm: Wu & He 2018
    - SiLU activation: Hendrycks & Gimpel 2016
    - Nearest-neighbor upsample + conv: avoids checkerboard artifacts
      vs. ConvTranspose2d (Odena et al. 2016)

Config values used:
    config.frvae.latent_channels = 768   (input channel dimension)
    config.frvae.image_size      = 256   (output spatial resolution)
    config.frvae.latent_spatial_size = 16 (input spatial resolution)
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper: safe GroupNorm group count
# ---------------------------------------------------------------------------

def _safe_group_norm(num_channels: int, preferred_groups: int = 32) -> nn.GroupNorm:
    """Create a GroupNorm layer with a valid group count.

    GroupNorm requires num_channels % num_groups == 0. This helper finds
    the largest divisor of num_channels that is ≤ preferred_groups.

    Args:
        num_channels: Number of channels to normalize.
        preferred_groups: Preferred number of groups (default 32, VQ-GAN standard).

    Returns:
        nn.GroupNorm instance with a valid group count.
    """
    num_groups: int = preferred_groups
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6)


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual convolutional block following the VQ-GAN decoder pattern.

    Architecture:
        GroupNorm → SiLU → Conv2d(3×3) → GroupNorm → SiLU → Conv2d(3×3)
        + skip connection (1×1 conv if in_channels ≠ out_channels)

    Attributes:
        norm1: First GroupNorm layer.
        conv1: First 3×3 convolution.
        norm2: Second GroupNorm layer.
        conv2: Second 3×3 convolution.
        skip: Skip connection (1×1 conv or Identity).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the ResBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()

        self.norm1: nn.GroupNorm = _safe_group_norm(in_channels)
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2: nn.GroupNorm = _safe_group_norm(out_channels)
        self.conv2: nn.Conv2d = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )

        # Skip connection: 1×1 conv when channel dimensions differ, else identity.
        if in_channels != out_channels:
            self.skip: nn.Module = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.skip = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following VQ-GAN conventions.

        conv1 and conv2 use Kaiming normal initialization (appropriate for
        SiLU activations). The skip 1×1 conv uses Kaiming normal as well.
        The final conv2 output is scaled down slightly to stabilize early
        training (zero-init of the last layer in each residual branch is a
        common trick from ResNet literature).
        """
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="linear")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="linear")
        # Zero-initialize the bias of the last conv in the residual branch
        # so the block starts as a near-identity mapping.
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        if isinstance(self.skip, nn.Conv2d):
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity="linear")

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H, W).
        """
        # Main branch: norm → act → conv → norm → act → conv
        h: Tensor = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # Residual connection.
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Attention Block (at bottleneck resolution 16×16)
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """Lightweight self-attention block for global context at low resolution.

    Applied once at the 16×16 bottleneck to capture long-range spatial
    dependencies before upsampling. Uses single-head attention for simplicity
    (the spatial resolution is small enough that multi-head is not critical).

    Architecture:
        GroupNorm → reshape to sequence → QKV projection → scaled dot-product
        attention → output projection → reshape back → residual add

    Attributes:
        norm: GroupNorm normalization.
        qkv: 1×1 conv projecting channels to 3×channels (Q, K, V).
        proj_out: 1×1 conv projecting attended features back to channels.
    """

    def __init__(self, channels: int) -> None:
        """Initialize the AttentionBlock.

        Args:
            channels: Number of input/output channels.
        """
        super().__init__()

        self.channels: int = channels
        self.norm: nn.GroupNorm = _safe_group_norm(channels)

        # QKV projection: maps C → 3C (Q, K, V concatenated along channel dim).
        self.qkv: nn.Conv2d = nn.Conv2d(
            channels, 3 * channels, kernel_size=1, bias=False
        )

        # Output projection: maps attended C → C.
        self.proj_out: nn.Conv2d = nn.Conv2d(
            channels, channels, kernel_size=1, bias=True
        )

        # Scale factor for dot-product attention: 1 / sqrt(C).
        self._scale: float = 1.0 / math.sqrt(channels)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize attention weights.

        QKV projection uses Xavier uniform (appropriate for attention).
        Output projection is zero-initialized so the block starts as identity.
        """
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention with residual connection.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W) with global context.
        """
        B, C, H, W = x.shape
        N: int = H * W  # Sequence length (256 at 16×16 bottleneck)

        # Normalize input.
        h: Tensor = self.norm(x)  # (B, C, H, W)

        # Compute Q, K, V via 1×1 conv.
        qkv: Tensor = self.qkv(h)  # (B, 3C, H, W)

        # Reshape to sequence format: (B, 3C, H, W) → (B, 3, C, N)
        qkv = qkv.reshape(B, 3, C, N)

        # Split into Q, K, V: each (B, C, N)
        q: Tensor = qkv[:, 0]  # (B, C, N)
        k: Tensor = qkv[:, 1]  # (B, C, N)
        v: Tensor = qkv[:, 2]  # (B, C, N)

        # Scaled dot-product attention.
        # attn_weights: (B, N, N) — attention from each position to all others.
        # q.permute(0,2,1): (B, N, C)
        # k: (B, C, N)
        attn_weights: Tensor = torch.bmm(
            q.permute(0, 2, 1),  # (B, N, C)
            k,                    # (B, C, N)
        ) * self._scale           # (B, N, N)

        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, N, N)

        # Weighted sum of values.
        # v: (B, C, N), attn_weights.permute(0,2,1): (B, N, N)
        # attended: (B, C, N)
        attended: Tensor = torch.bmm(
            v,                              # (B, C, N)
            attn_weights.permute(0, 2, 1),  # (B, N, N)
        )  # (B, C, N)

        # Reshape back to spatial: (B, C, N) → (B, C, H, W)
        attended = attended.reshape(B, C, H, W)

        # Output projection.
        attended = self.proj_out(attended)  # (B, C, H, W)

        # Residual connection.
        return x + attended


# ---------------------------------------------------------------------------
# Upsample Block
# ---------------------------------------------------------------------------

class UpsampleBlock(nn.Module):
    """2× spatial upsampling block using nearest-neighbor + convolution.

    Nearest-neighbor upsampling followed by a 3×3 convolution avoids the
    checkerboard artifacts that arise with ConvTranspose2d (Odena et al. 2016).

    Attributes:
        upsample: nn.Upsample with scale_factor=2, mode='nearest'.
        conv: 3×3 convolution applied after upsampling.
    """

    def __init__(self, channels: int) -> None:
        """Initialize the UpsampleBlock.

        Args:
            channels: Number of input and output channels (unchanged by upsampling).
        """
        super().__init__()

        self.upsample: nn.Upsample = nn.Upsample(
            scale_factor=2, mode="nearest"
        )
        self.conv: nn.Conv2d = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=True
        )

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply 2× upsampling followed by convolution.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, 2H, 2W).
        """
        x = self.upsample(x)
        x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# Decoder Stage (ResBlocks + Upsample)
# ---------------------------------------------------------------------------

class DecoderStage(nn.Module):
    """A single decoder stage: one or more ResBlocks followed by 2× upsampling.

    Each stage processes features at a fixed spatial resolution, then doubles
    the spatial resolution via UpsampleBlock.

    Attributes:
        res_blocks: Sequential residual blocks at the current resolution.
        upsample: 2× upsampling block (applied after all res_blocks).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 2,
    ) -> None:
        """Initialize a decoder stage.

        Args:
            in_channels: Number of input channels entering this stage.
            out_channels: Number of output channels leaving this stage
                (after the last ResBlock, before upsampling).
            num_res_blocks: Number of residual blocks in this stage.
                The first ResBlock handles the channel transition
                (in_channels → out_channels); subsequent blocks maintain
                out_channels throughout.
        """
        super().__init__()

        if num_res_blocks < 1:
            raise ValueError(
                f"num_res_blocks must be >= 1, got {num_res_blocks}."
            )

        # Build residual blocks: first block handles channel transition,
        # remaining blocks maintain out_channels.
        res_block_list: List[nn.Module] = []
        current_channels: int = in_channels
        for block_idx in range(num_res_blocks):
            target_channels: int = out_channels if block_idx == num_res_blocks - 1 else in_channels
            res_block_list.append(ResBlock(current_channels, target_channels))
            current_channels = target_channels

        self.res_blocks: nn.Sequential = nn.Sequential(*res_block_list)

        # 2× upsampling at the end of the stage.
        self.upsample: UpsampleBlock = UpsampleBlock(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual blocks then 2× upsampling.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape (B, out_channels, 2H, 2W).
        """
        x = self.res_blocks(x)
        x = self.upsample(x)
        return x


# ---------------------------------------------------------------------------
# Main Decoder
# ---------------------------------------------------------------------------

class VQGANDecoder(nn.Module):
    """VQ-GAN style decoder that reconstructs RGB images from latent feature maps.

    Upsamples from [B, latent_channels, H', W'] = [B, 768, 16, 16] to
    [B, 3, image_size, image_size] = [B, 3, 256, 256] through 4 stages of
    2× upsampling, each preceded by residual blocks.

    Full architecture:
        Input: [B, 768, 16, 16]
        initial_conv: Conv2d(768→512, 3×3)
        attention_block: AttentionBlock(512) at 16×16
        Stage 1: ResBlock(512→512) + ResBlock(512→256) + Upsample×2 → [B, 256, 32, 32]
        Stage 2: ResBlock(256→256) + ResBlock(256→128) + Upsample×2 → [B, 128, 64, 64]
        Stage 3: ResBlock(128→128) + ResBlock(128→64)  + Upsample×2 → [B, 64, 128, 128]
        Stage 4: ResBlock(64→64)                       + Upsample×2 → [B, 64, 256, 256]
        final_norm: GroupNorm(32, 64)
        final_act: SiLU()
        out_conv: Conv2d(64→3, 3×3) + tanh
        Output: [B, 3, 256, 256]

    Attributes:
        initial_conv: Initial 3×3 conv projecting latent_channels → 512.
        attention_block: Self-attention at the 16×16 bottleneck.
        layers: nn.ModuleList of 4 DecoderStage modules.
        final_norm: GroupNorm before the output conv.
        out_conv: Final 3×3 conv projecting 64 → 3 channels.
        latent_channels: Input channel dimension (768 from config).
        image_size: Output spatial resolution (256 from config).
    """

    # Channel schedule: latent_channels → stage_channels → ... → final_channels → 3
    # Derived from the upsampling factor analysis in the Logic Analysis.
    _INITIAL_CHANNELS: int = 512   # After initial_conv
    _STAGE_CHANNELS: List[int] = [512, 256, 128, 64]  # Output channels per stage
    _FINAL_CHANNELS: int = 64      # Channels entering out_conv
    _OUTPUT_CHANNELS: int = 3      # RGB output

    # Number of residual blocks per stage.
    # Stages at lower resolution (larger feature maps) use fewer blocks
    # to balance compute. Stage 1 (16→32) uses 2 blocks; others use 2.
    _NUM_RES_BLOCKS_PER_STAGE: List[int] = [2, 2, 2, 1]

    def __init__(
        self,
        latent_channels: int = 768,
        image_size: int = 256,
    ) -> None:
        """Initialize the VQGANDecoder.

        Args:
            latent_channels: Number of channels in the input latent feature map.
                From config.frvae.latent_channels = 768.
                Must be positive.
            image_size: Target output spatial resolution (height = width).
                From config.frvae.image_size = 256.
                Must be a power of 2 and at least 16.

        Raises:
            ValueError: If latent_channels <= 0 or image_size is not a
                positive power of 2 >= 16.
        """
        super().__init__()

        if latent_channels <= 0:
            raise ValueError(
                f"latent_channels must be positive, got {latent_channels}."
            )
        if image_size < 16 or (image_size & (image_size - 1)) != 0:
            raise ValueError(
                f"image_size must be a power of 2 and >= 16, got {image_size}. "
                "The decoder requires an integer number of 2× upsampling stages."
            )

        self.latent_channels: int = latent_channels
        self.image_size: int = image_size

        # Compute the number of upsampling stages required.
        # latent_spatial_size = 16 (from config), image_size = 256.
        # num_stages = log2(image_size / latent_spatial_size) = log2(256/16) = 4.
        latent_spatial_size: int = 16  # config.frvae.latent_spatial_size
        upsample_factor: int = image_size // latent_spatial_size
        num_stages: int = int(math.log2(upsample_factor))

        if 2 ** num_stages != upsample_factor:
            raise ValueError(
                f"image_size / latent_spatial_size = {image_size} / {latent_spatial_size} "
                f"= {upsample_factor} must be a power of 2. "
                f"Got non-integer log2: {math.log2(upsample_factor):.4f}."
            )

        if num_stages != len(self._STAGE_CHANNELS):
            raise ValueError(
                f"Number of upsampling stages ({num_stages}) does not match "
                f"len(_STAGE_CHANNELS)={len(self._STAGE_CHANNELS)}. "
                f"Adjust _STAGE_CHANNELS for image_size={image_size}."
            )

        # --- Initial projection: latent_channels → _INITIAL_CHANNELS ---
        # 3×3 conv to project from 768 to 512 while preserving spatial structure.
        self.initial_conv: nn.Conv2d = nn.Conv2d(
            latent_channels,
            self._INITIAL_CHANNELS,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        nn.init.kaiming_normal_(self.initial_conv.weight, nonlinearity="linear")
        nn.init.zeros_(self.initial_conv.bias)

        # --- Attention block at 16×16 bottleneck ---
        # Captures global spatial context before upsampling begins.
        self.attention_block: AttentionBlock = AttentionBlock(self._INITIAL_CHANNELS)

        # --- Decoder stages ---
        # Each stage: ResBlocks (channel transition) + 2× Upsample.
        # Stage input channels: [512, 256, 128, 64]
        # Stage output channels: [256, 128, 64, 64]
        # (Stage 1 input = _INITIAL_CHANNELS = 512)
        stage_list: List[nn.Module] = []
        stage_in_channels: int = self._INITIAL_CHANNELS  # 512

        for stage_idx in range(num_stages):
            stage_out_channels: int = self._STAGE_CHANNELS[stage_idx]
            num_res_blocks: int = self._NUM_RES_BLOCKS_PER_STAGE[stage_idx]

            stage = DecoderStage(
                in_channels=stage_in_channels,
                out_channels=stage_out_channels,
                num_res_blocks=num_res_blocks,
            )
            stage_list.append(stage)
            stage_in_channels = stage_out_channels  # Output feeds into next stage

        # nn.ModuleList as specified in the design interface.
        self.layers: nn.ModuleList = nn.ModuleList(stage_list)

        # --- Final normalization and output convolution ---
        # Applied after all upsampling stages, before the RGB projection.
        self.final_norm: nn.GroupNorm = _safe_group_norm(self._FINAL_CHANNELS)

        # out_conv: Conv2d(64, 3, 3×3) as specified in the design interface.
        self.out_conv: nn.Conv2d = nn.Conv2d(
            self._FINAL_CHANNELS,
            self._OUTPUT_CHANNELS,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        # Initialize out_conv with small weights so early outputs are near zero
        # (tanh(~0) ≈ 0), which is a stable starting point for image generation.
        nn.init.normal_(self.out_conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, f: Tensor) -> Tensor:
        """Decode a latent feature map into a full-resolution RGB image.

        Implements the full decoding pipeline:
            f [B, 768, 16, 16]
            → initial_conv → attention_block
            → 4 decoder stages (each: ResBlocks + 2× Upsample)
            → final_norm → SiLU → out_conv → tanh
            → x_hat [B, 3, 256, 256]

        The output is in the range [-1, 1], matching the input normalization
        specified in config.yaml (mean=std=0.5, i.e., [-1, 1] range).

        Args:
            f: Latent feature map of shape (B, latent_channels, H', W').
               For the default config: (B, 768, 16, 16).
               This is the composed quantized feature map f_tilde from
               ResidualQuantizer.decode_all(), computed as:
                   f_tilde = Σ_i T(v_i^q, H', W')

        Returns:
            Reconstructed image batch x_hat of shape (B, 3, image_size, image_size).
            For the default config: (B, 3, 256, 256).
            Values are in [-1, 1] (tanh output).

        Raises:
            RuntimeError: If the input spatial dimensions do not match the
                expected latent_spatial_size (16×16 for the default config).
        """
        # Validate input spatial dimensions.
        expected_spatial: int = 16  # config.frvae.latent_spatial_size
        if f.shape[-2] != expected_spatial or f.shape[-1] != expected_spatial:
            raise RuntimeError(
                f"Expected input spatial size {expected_spatial}×{expected_spatial}, "
                f"but got {f.shape[-2]}×{f.shape[-1]}. "
                f"Input shape: {tuple(f.shape)}. "
                "Ensure the ResidualQuantizer outputs f_tilde at the correct resolution."
            )

        # --- Initial projection: [B, 768, 16, 16] → [B, 512, 16, 16] ---
        x: Tensor = self.initial_conv(f)

        # --- Attention at 16×16 bottleneck: [B, 512, 16, 16] → [B, 512, 16, 16] ---
        x = self.attention_block(x)

        # --- Decoder stages: progressive 2× upsampling ---
        # Stage 1: [B, 512, 16, 16]  → [B, 256, 32, 32]
        # Stage 2: [B, 256, 32, 32]  → [B, 128, 64, 64]
        # Stage 3: [B, 128, 64, 64]  → [B, 64, 128, 128]
        # Stage 4: [B, 64, 128, 128] → [B, 64, 256, 256]
        stage: nn.Module
        for stage in self.layers:
            x = stage(x)

        # --- Final normalization and activation ---
        x = self.final_norm(x)   # GroupNorm(32, 64)
        x = F.silu(x)            # SiLU activation

        # --- Output projection: [B, 64, 256, 256] → [B, 3, 256, 256] ---
        x = self.out_conv(x)

        # --- Output activation: clamp to [-1, 1] ---
        # Matches the training pipeline normalization (config mean=std=0.5).
        x_hat: Tensor = torch.tanh(x)

        return x_hat

    def extra_repr(self) -> str:
        """Return a human-readable string with key decoder configuration.

        Returns:
            String describing the decoder's input/output dimensions.
        """
        return (
            f"latent_channels={self.latent_channels}, "
            f"image_size={self.image_size}×{self.image_size}, "
            f"channel_schedule={self._INITIAL_CHANNELS}→"
            f"{'→'.join(str(c) for c in self._STAGE_CHANNELS)}→"
            f"{self._OUTPUT_CHANNELS}"
        )
