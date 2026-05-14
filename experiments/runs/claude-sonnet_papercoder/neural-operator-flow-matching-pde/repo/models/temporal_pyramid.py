## models/temporal_pyramid.py
"""Latent temporal pyramid for efficient multi-scale PDE field representation.

Implements the latent temporal pyramid described in Section 3.3 of the paper:
"Bridging Neural Operator and Flow Matching for a Generative PDE Foundation Model".

The core idea is to represent the 4-frame training window at progressively
coarser spatial resolutions for older (more distant) frames, exploiting the
Markovian nature of PDE dynamics. This reduces the total token count fed to
the SiT Transformer from 4 × 256 = 1024 tokens (vanilla video diffusion) to
4 + 16 + 64 + 256 = 340 tokens, yielding a 15× efficiency gain in attention
complexity (paper Section 4.2).

Spatial resolution mapping (config: fmt.temporal_pyramid.downsample_factors):
    frame 0 (oldest):  Down(y_0, factor=8)  → (B, C, 2,  2 ) →   4 tokens
    frame 1:           Down(y_1, factor=4)  → (B, C, 4,  4 ) →  16 tokens
    frame 2:           Down(y_2, factor=2)  → (B, C, 8,  8 ) →  64 tokens
    frame 3 (newest):  y_3 (identity)       → (B, C, 16, 16) → 256 tokens
    ─────────────────────────────────────────────────────────────────────────
    Total:                                                       340 tokens

Efficiency gain vs. vanilla 4-frame video diffusion with 256 tokens/frame:
    η = (4 × 256)² / (4² + 16² + 64² + 256²)
      = 1,048,576 / 69,904
      ≈ 15  (paper Section 4.2)

Downsampling method: F.avg_pool2d (config: fmt.temporal_pyramid.downsample_method).

This module has zero learnable parameters — all operations are deterministic
average pooling. It inherits nn.Module for consistent device management and
integration with FMT's module hierarchy.

Integration:
    Called from FMT.forward() to build the pyramid from 4 noisy latents and
    flatten them into a (B, 340, C) token sequence for the patch embedding layer.
    Called from FMT.__init__() via get_token_counts() to set the positional
    embedding size (sum = 340).
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class TemporalPyramid(nn.Module):
    """Latent temporal pyramid for multi-scale PDE field representation.

    Applies spatial average pooling at different downsampling factors to the
    4 latent frames in the training window, producing a coarse-to-fine token
    sequence that reduces attention complexity by 15× compared to processing
    all frames at full resolution.

    The token layout in the flattened output is:
        [4 tokens from frame 0 at 2×2,
         16 tokens from frame 1 at 4×4,
         64 tokens from frame 2 at 8×8,
         256 tokens from frame 3 at 16×16]

    This ordering is fixed and must be consistent with the positional embedding
    layout in FMT (shared knowledge point 8 from the task specification).

    Attributes:
        latent_size: Spatial resolution of the full-resolution latent grid.
            From config: p2vae.latent_size = 16.
        downsample_factors: Per-frame downsampling factors applied to the
            spatial dimensions. From config:
            fmt.temporal_pyramid.downsample_factors = [8, 4, 2, 1].
            factors[i] is applied to frame i (0 = oldest, 3 = newest).
        output_sizes: Spatial resolution after downsampling per frame.
            Computed as [latent_size // f for f in downsample_factors].
            With defaults: [2, 4, 8, 16].
        token_counts: Number of tokens per frame after flattening spatial dims.
            Computed as [s * s for s in output_sizes].
            With defaults: [4, 16, 64, 256].
        num_frames: Number of frames in the training window (always 4).
        total_tokens: Total token count across all frames (always 340).
    """

    def __init__(
        self,
        latent_size: int = 16,
        downsample_factors: Optional[List[int]] = None,
    ) -> None:
        """Initialize the TemporalPyramid.

        Args:
            latent_size: Spatial resolution of the full-resolution latent grid
                (height = width = latent_size). From config:
                p2vae.latent_size = 16. Must be divisible by all factors in
                downsample_factors.
            downsample_factors: Per-frame spatial downsampling factors. Element
                i is the factor applied to frame i (0 = oldest, 3 = newest).
                From config: fmt.temporal_pyramid.downsample_factors = [8,4,2,1].
                Defaults to [8, 4, 2, 1] if None.
                - factor=8: 16×16 → 2×2 (4 tokens, oldest frame)
                - factor=4: 16×16 → 4×4 (16 tokens)
                - factor=2: 16×16 → 8×8 (64 tokens)
                - factor=1: 16×16 → 16×16 (256 tokens, newest frame, identity)

        Raises:
            ValueError: If downsample_factors does not have exactly 4 elements
                (matching data.trajectory_length = 4 from config).
            ValueError: If latent_size is not divisible by any factor in
                downsample_factors (would produce non-integer output sizes).
            ValueError: If any factor is less than 1.
        """
        super().__init__()

        # Apply default from config: fmt.temporal_pyramid.downsample_factors
        if downsample_factors is None:
            downsample_factors = [8, 4, 2, 1]

        # Validate number of frames matches data.trajectory_length = 4.
        if len(downsample_factors) != 4:
            raise ValueError(
                f"downsample_factors must have exactly 4 elements to match "
                f"data.trajectory_length=4 from config, "
                f"got {len(downsample_factors)}: {downsample_factors}."
            )

        # Validate all factors are positive integers.
        for i, factor in enumerate(downsample_factors):
            if factor < 1:
                raise ValueError(
                    f"All downsample factors must be >= 1, "
                    f"but factors[{i}] = {factor}."
                )

        # Validate latent_size is divisible by all factors.
        for i, factor in enumerate(downsample_factors):
            if latent_size % factor != 0:
                raise ValueError(
                    f"latent_size={latent_size} must be divisible by "
                    f"downsample_factors[{i}]={factor}. "
                    f"Got remainder {latent_size % factor}."
                )

        self.latent_size: int = latent_size
        self.downsample_factors: List[int] = list(downsample_factors)
        self.num_frames: int = 4

        # Precompute output spatial sizes per frame.
        # With defaults (latent_size=16, factors=[8,4,2,1]): [2, 4, 8, 16]
        self.output_sizes: List[int] = [
            latent_size // factor for factor in self.downsample_factors
        ]

        # Precompute token counts per frame (spatial_size^2).
        # With defaults: [4, 16, 64, 256]
        self.token_counts: List[int] = [
            size * size for size in self.output_sizes
        ]

        # Total tokens across all frames.
        # With defaults: 4 + 16 + 64 + 256 = 340
        self.total_tokens: int = sum(self.token_counts)

        logger.info(
            "TemporalPyramid initialized: latent_size=%d, "
            "downsample_factors=%s, output_sizes=%s, "
            "token_counts=%s, total_tokens=%d",
            self.latent_size,
            self.downsample_factors,
            self.output_sizes,
            self.token_counts,
            self.total_tokens,
        )

    def downsample(self, y: Tensor, factor: int) -> Tensor:
        """Spatially downsample a latent tensor by the given factor.

        Applies F.avg_pool2d with kernel_size=factor and stride=factor,
        reducing the spatial resolution from (H, W) to (H//factor, W//factor).
        For factor=1, returns the input unchanged (identity operation).

        The downsampling method is average pooling as specified in config:
        fmt.temporal_pyramid.downsample_method = "avg_pool2d".

        Args:
            y: Latent tensor of shape (B, C, H, W). Typically H=W=latent_size=16
                for full-resolution latents from P2VAE. The channel dimension C
                is preserved unchanged (only spatial dims are reduced).
            factor: Spatial downsampling factor. Must evenly divide both H and W.
                From config: fmt.temporal_pyramid.downsample_factors = [8,4,2,1].
                - factor=1: identity (no pooling), returns y unchanged
                - factor=2: (B,C,16,16) → (B,C,8,8)
                - factor=4: (B,C,16,16) → (B,C,4,4)
                - factor=8: (B,C,16,16) → (B,C,2,2)

        Returns:
            Downsampled tensor of shape (B, C, H//factor, W//factor).
            Same dtype and device as input y.

        Raises:
            ValueError: If factor < 1.
        """
        if factor < 1:
            raise ValueError(
                f"Downsampling factor must be >= 1, got {factor}."
            )

        # Identity path: no pooling needed for factor=1 (newest frame).
        # This avoids unnecessary computation and is cleaner than avg_pool2d
        # with kernel_size=1.
        if factor == 1:
            return y

        # Apply average pooling: reduces (B, C, H, W) → (B, C, H//factor, W//factor).
        # No padding needed since factor evenly divides latent_size by construction.
        return F.avg_pool2d(
            y,
            kernel_size=factor,
            stride=factor,
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
        )

    def build_pyramid(self, latents: List[Tensor]) -> List[Tensor]:
        """Apply per-frame downsampling to build the multi-scale pyramid.

        Takes 4 latent tensors (one per frame in the training window) and
        applies the corresponding downsampling factor to each, producing a
        list of tensors at progressively finer spatial resolutions.

        The frame ordering follows the paper's convention (Section 3.3):
            latents[0] = y_0 (oldest frame) → downsampled most aggressively
            latents[1] = y_1
            latents[2] = y_2
            latents[3] = y_3 (newest frame) → kept at full resolution

        This ordering is critical for consistency with the positional embedding
        layout in FMT (shared knowledge point 8): tokens are ordered as
        [4 from frame 0, 16 from frame 1, 64 from frame 2, 256 from frame 3].

        Args:
            latents: List of exactly 4 latent tensors, each of shape
                (B, C, latent_size, latent_size). These are the noisy
                interpolated latents y_{s,t_s}^{k_s} for s=0,1,2,3,
                constructed by FlowMarchingKernel.sample_interpolation
                in FMTTrainer.train_step().

        Returns:
            List of 4 downsampled tensors:
                pyramid[0]: (B, C, 2,  2 ) — frame 0 at 2×2 (factor=8)
                pyramid[1]: (B, C, 4,  4 ) — frame 1 at 4×4 (factor=4)
                pyramid[2]: (B, C, 8,  8 ) — frame 2 at 8×8 (factor=2)
                pyramid[3]: (B, C, 16, 16) — frame 3 at 16×16 (factor=1)
            All tensors share the same batch size B and channel count C
            as the inputs.

        Raises:
            ValueError: If len(latents) != 4.
        """
        if len(latents) != self.num_frames:
            raise ValueError(
                f"build_pyramid expects exactly {self.num_frames} latent tensors "
                f"(matching data.trajectory_length=4 from config), "
                f"got {len(latents)}."
            )

        pyramid: List[Tensor] = []
        for frame_idx in range(self.num_frames):
            factor: int = self.downsample_factors[frame_idx]
            downsampled: Tensor = self.downsample(latents[frame_idx], factor)
            pyramid.append(downsampled)

        return pyramid

    def flatten_pyramid(self, pyramid: List[Tensor]) -> Tensor:
        """Flatten and concatenate pyramid levels into a single token sequence.

        Converts each spatial map in the pyramid from (B, C, H, W) to a
        token sequence (B, H*W, C), then concatenates all levels along the
        token dimension to produce the full (B, 340, C) input for the SiT
        Transformer's patch embedding layer.

        The spatial-to-token reshape uses .permute(0, 2, 3, 1).reshape(B, H*W, C)
        which maps spatial positions in row-major (C-contiguous) order:
            position (h, w) → token index h * W + w

        This ordering is consistent with standard Vision Transformer conventions
        and must match the positional embedding layout in FMT.__init__.

        Args:
            pyramid: List of 4 tensors from build_pyramid:
                pyramid[0]: (B, C, 2,  2 ) →  4 tokens
                pyramid[1]: (B, C, 4,  4 ) → 16 tokens
                pyramid[2]: (B, C, 8,  8 ) → 64 tokens
                pyramid[3]: (B, C, 16, 16) → 256 tokens

        Returns:
            Concatenated token sequence of shape (B, 340, C) where:
                - 340 = 4 + 16 + 64 + 256 (config: fmt.temporal_pyramid.total_tokens)
                - C = latent_channels = 16 (config: p2vae.latent_channels)
            The token layout is:
                tokens[:, 0:4,   :] ← frame 0 tokens (2×2 spatial)
                tokens[:, 4:20,  :] ← frame 1 tokens (4×4 spatial)
                tokens[:, 20:84, :] ← frame 2 tokens (8×8 spatial)
                tokens[:, 84:,   :] ← frame 3 tokens (16×16 spatial)

        Raises:
            ValueError: If len(pyramid) != 4.
        """
        if len(pyramid) != self.num_frames:
            raise ValueError(
                f"flatten_pyramid expects exactly {self.num_frames} pyramid levels, "
                f"got {len(pyramid)}."
            )

        token_sequences: List[Tensor] = []

        for level_idx, level_tensor in enumerate(pyramid):
            b, c, h, w = level_tensor.shape

            # Validate token count matches precomputed expectation.
            expected_tokens: int = self.token_counts[level_idx]
            actual_tokens: int = h * w
            if actual_tokens != expected_tokens:
                raise ValueError(
                    f"Pyramid level {level_idx} has spatial size ({h}, {w}) "
                    f"giving {actual_tokens} tokens, but expected "
                    f"{expected_tokens} tokens (output_size={self.output_sizes[level_idx]})."
                )

            # Reshape (B, C, H, W) → (B, H*W, C) in row-major spatial order.
            # Step 1: permute to (B, H, W, C) — spatial dims before channel.
            # Step 2: reshape to (B, H*W, C) — flatten spatial dims.
            tokens: Tensor = level_tensor.permute(0, 2, 3, 1).reshape(b, h * w, c)
            token_sequences.append(tokens)

        # Concatenate all levels along the token dimension (dim=1).
        # Result: (B, 4+16+64+256, C) = (B, 340, C)
        flattened: Tensor = torch.cat(token_sequences, dim=1)

        return flattened

    def get_token_counts(self) -> List[int]:
        """Return the number of tokens contributed by each pyramid level.

        Used by FMT.__init__() to:
          1. Compute the total positional embedding size:
             sum(get_token_counts()) = 340 (config: fmt.temporal_pyramid.total_tokens)
          2. Construct level-specific positional embedding slices for
             indexing into the (1, 340, embed_dim) positional embedding tensor.

        Returns:
            List of 4 integers representing the token count per pyramid level:
                [4, 16, 64, 256]
            Corresponding to spatial resolutions [2×2, 4×4, 8×8, 16×16].
            From config: fmt.temporal_pyramid.token_counts = [4, 16, 64, 256].
        """
        return list(self.token_counts)

    def get_output_sizes(self) -> List[int]:
        """Return the spatial resolution at each pyramid level.

        Utility method for debugging and for FMT to compute per-level
        positional embedding shapes.

        Returns:
            List of 4 integers representing the spatial size (height = width)
            at each pyramid level: [2, 4, 8, 16].
            Computed as [latent_size // factor for factor in downsample_factors].
        """
        return list(self.output_sizes)

    def extra_repr(self) -> str:
        """Return a string representation of the module's configuration.

        Used by PyTorch's print(module) to display module details.

        Returns:
            String summarizing the key configuration parameters.
        """
        return (
            f"latent_size={self.latent_size}, "
            f"downsample_factors={self.downsample_factors}, "
            f"output_sizes={self.output_sizes}, "
            f"token_counts={self.token_counts}, "
            f"total_tokens={self.total_tokens}"
        )
