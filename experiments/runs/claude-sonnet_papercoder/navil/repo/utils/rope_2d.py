## utils/rope_2d.py
"""2D Rotary Position Embedding (RoPE2D) for the NaViL visual encoder.

The visual encoder uses bidirectional attention with 2D-RoPE to capture
global spatial relationships in image patches (Section 4.1 of the paper).

Design reference:
- Per-head dimension: dim=64 (1472/23 for NaViL-2B, 1792/28 for NaViL-9B)
- Applied to Q and K tensors in each VisualEncoderLayer
- Standalone utility with no internal project dependencies

2D-RoPE strategy:
- Split the per-head dimension in half: first dim//2 dims encode height
  position, last dim//2 dims encode width position.
- Each half uses standard 1D RoPE with its own position sequence.
- This allows the model to independently attend to row and column positions.

Frequency formula (from task spec):
    freqs = 1 / (base ** (arange(0, dim//2, 2) / (dim//2)))

With dim=64, half_dim=32:
    arange(0, 32, 2) = [0, 2, 4, ..., 30]  →  16 base frequency values
    Each frequency covers a pair of dimensions (standard RoPE pairing).
    repeat_interleave(2) expands to 32 values covering the full half_dim.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of x by splitting in half and negating/swapping.

    For a tensor of shape (..., dim):
      - x1 = x[..., :dim//2]
      - x2 = x[..., dim//2:]
      - returns cat([-x2, x1], dim=-1)

    This implements the rotation matrix in RoPE:
        [cos θ  -sin θ] [x1]   [x1 cos θ - x2 sin θ]
        [sin θ   cos θ] [x2] = [x1 sin θ + x2 cos θ]
    which equals ``x * cos + rotate_half(x) * sin``.

    Args:
        x: Tensor of shape (..., dim) where dim is even.

    Returns:
        Tensor of the same shape as x.
    """
    half: int = x.shape[-1] // 2
    x1: torch.Tensor = x[..., :half]   # (..., dim//2)
    x2: torch.Tensor = x[..., half:]   # (..., dim//2)
    return torch.cat([-x2, x1], dim=-1)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for visual tokens arranged in a patch grid.

    Splits the per-head dimension equally between height and width axes:
      - Dimensions [0 : dim//2]      encode the row (height) position.
      - Dimensions [dim//2 : dim]    encode the column (width) position.

    Each half uses independent 1D RoPE frequencies, allowing the model to
    attend to row and column positions separately within a single attention
    head.

    Angle cache:
        To avoid recomputing angles on every forward pass, the constructor
        pre-builds angle tensors of shape (max_height, half_dim) and
        (max_width, half_dim) and registers them as non-persistent buffers.
        ``build_freqs`` slices these caches for the actual grid size and
        computes cos/sin on the fly.

    Args:
        dim:        Per-head feature dimension. Must be divisible by 4.
                    Typical value: 64 (NaViL-2B: 1472/23≈64,
                    NaViL-9B: 1792/28=64).
        max_height: Maximum grid height in patches for the angle cache.
                    Should be set to ceil(max_image_H / patch_size).
        max_width:  Maximum grid width in patches for the angle cache.
                    Should be set to ceil(max_image_W / patch_size).
        base:       RoPE base frequency. Default 10000.0 (standard RoPE).

    Example::

        rope2d = RoPE2D(dim=64, max_height=256, max_width=256)
        # q shape: (B, num_heads, H*W, 64)
        q_rotated = rope2d(q, grid_hw=(H, W))
    """

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()

        if dim % 4 != 0:
            raise ValueError(
                f"dim must be divisible by 4 for 2D-RoPE (got dim={dim}). "
                "Each spatial axis uses dim//2 dimensions, and standard RoPE "
                "requires pairs within each half, so dim//2 must be even, "
                "i.e., dim must be divisible by 4."
            )

        self.dim: int = dim
        self.max_height: int = max_height
        self.max_width: int = max_width
        self.base: float = base

        # ------------------------------------------------------------------ #
        # Build base frequency vector                                          #
        # ------------------------------------------------------------------ #
        # Each spatial axis gets half_dim = dim//2 dimensions.
        # Within each half, RoPE pairs adjacent dimensions, so we need
        # half_dim//2 = dim//4 distinct frequency values per axis.
        #
        # Formula (task spec):
        #   freqs = 1 / (base ** (arange(0, half_dim, 2) / half_dim))
        #
        # Example with dim=64, half_dim=32:
        #   idx   = [0, 2, 4, ..., 30]          shape: (16,)
        #   freqs = 1 / (10000 ** (idx / 32))   shape: (16,)
        #
        # This matches θ_i = 1 / (base^(2i/d)) with d=half_dim.
        half_dim: int = dim // 2  # 32 for dim=64

        idx: torch.Tensor = torch.arange(0, half_dim, 2, dtype=torch.float32)
        # shape: (half_dim//2,)  e.g. (16,) for dim=64

        base_freqs: torch.Tensor = 1.0 / (base ** (idx / half_dim))
        # shape: (half_dim//2,)

        # Register as non-persistent buffer: moves with module but excluded
        # from state_dict (recomputed from hyperparameters on load).
        self.register_buffer("_base_freqs", base_freqs, persistent=False)

        # ------------------------------------------------------------------ #
        # Pre-build angle cache                                                #
        # ------------------------------------------------------------------ #
        # angles[p, i] = p * base_freqs[i]  for p in range(max_pos)
        # Then repeat_interleave(2) expands each frequency to cover its pair
        # of dimensions: [θ_0, θ_0, θ_1, θ_1, ..., θ_{k-1}, θ_{k-1}]
        # Final shape: (max_height, half_dim) and (max_width, half_dim)

        h_positions: torch.Tensor = torch.arange(
            max_height, dtype=torch.float32
        ).unsqueeze(1)  # (max_height, 1)

        w_positions: torch.Tensor = torch.arange(
            max_width, dtype=torch.float32
        ).unsqueeze(1)  # (max_width, 1)

        # Raw angles: (max_height, half_dim//2) and (max_width, half_dim//2)
        h_angles_raw: torch.Tensor = h_positions * base_freqs.unsqueeze(0)
        w_angles_raw: torch.Tensor = w_positions * base_freqs.unsqueeze(0)

        # Expand each frequency to cover a pair of dimensions via
        # repeat_interleave(2): shape becomes (max_height, half_dim)
        h_angles: torch.Tensor = h_angles_raw.repeat_interleave(2, dim=-1)
        # shape: (max_height, half_dim)

        w_angles: torch.Tensor = w_angles_raw.repeat_interleave(2, dim=-1)
        # shape: (max_width, half_dim)

        self.register_buffer("_h_angles", h_angles, persistent=False)
        self.register_buffer("_w_angles", w_angles, persistent=False)

    def build_freqs(
        self, height: int, width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build 2D position frequency tensors for a (height, width) patch grid.

        For each of the height*width spatial positions (h, w) in row-major
        order, constructs a dim-dimensional angle vector:
            angles[h*width + w] = cat([h_angles[h], w_angles[w]])

        Then returns cos and sin of these angles.

        Args:
            height: Grid height (patch rows). Must satisfy height <= max_height.
            width:  Grid width (patch columns). Must satisfy width <= max_width.

        Returns:
            cos_freqs: Float tensor of shape (height*width, dim).
            sin_freqs: Float tensor of shape (height*width, dim).

        Raises:
            ValueError: If height > max_height or width > max_width.
        """
        if height > self.max_height:
            raise ValueError(
                f"height={height} exceeds max_height={self.max_height}. "
                "Increase max_height when constructing RoPE2D."
            )
        if width > self.max_width:
            raise ValueError(
                f"width={width} exceeds max_width={self.max_width}. "
                "Increase max_width when constructing RoPE2D."
            )

        half_dim: int = self.dim // 2

        # Slice cached angle tensors to the required grid size
        h_angles: torch.Tensor = self._h_angles[:height]  # (height, half_dim)
        w_angles: torch.Tensor = self._w_angles[:width]   # (width,  half_dim)

        # Broadcast to (height, width, half_dim) for all (h, w) combinations.
        # h_angles: (height, 1, half_dim)  ×  w_angles: (1, width, half_dim)
        h_angles_2d: torch.Tensor = h_angles.unsqueeze(1).expand(
            height, width, half_dim
        )  # (height, width, half_dim)

        w_angles_2d: torch.Tensor = w_angles.unsqueeze(0).expand(
            height, width, half_dim
        )  # (height, width, half_dim)

        # Concatenate height and width angle vectors along the last dimension
        # to form the full per-head angle vector of size dim.
        angles_2d: torch.Tensor = torch.cat(
            [h_angles_2d, w_angles_2d], dim=-1
        )  # (height, width, dim)

        # Flatten spatial dimensions to token sequence order (row-major)
        angles_flat: torch.Tensor = angles_2d.reshape(height * width, self.dim)
        # shape: (height*width, dim)

        cos_freqs: torch.Tensor = torch.cos(angles_flat)  # (height*width, dim)
        sin_freqs: torch.Tensor = torch.sin(angles_flat)  # (height*width, dim)

        return cos_freqs, sin_freqs

    def apply_rotary(
        self,
        x: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 2D rotary position embeddings to a query or key tensor.

        Implements the standard RoPE formula:
            x_rotated = x * cos(freqs) + rotate_half(x) * sin(freqs)

        Args:
            x:          Query or key tensor of shape (B, num_heads, N, dim)
                        where N = height * width from the corresponding
                        ``build_freqs`` call.
            cos_freqs:  Cosine of position angles, shape (N, dim).
            sin_freqs:  Sine of position angles, shape (N, dim).

        Returns:
            Rotated tensor of the same shape as x.
        """
        # Broadcast freqs over batch and head dimensions:
        # (N, dim) → (1, 1, N, dim)
        cos: torch.Tensor = cos_freqs.unsqueeze(0).unsqueeze(0)
        sin: torch.Tensor = sin_freqs.unsqueeze(0).unsqueeze(0)

        return x * cos + rotate_half(x) * sin

    def forward(
        self,
        x: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """Apply 2D-RoPE to a query or key tensor given the spatial grid size.

        This is the primary entry point used by ``VisualEncoderLayer``.
        It calls ``build_freqs`` to obtain position-dependent cos/sin tensors
        and then calls ``apply_rotary`` to rotate x.

        Args:
            x:       Query or key tensor of shape (B, num_heads, N, dim)
                     where N = grid_hw[0] * grid_hw[1].
            grid_hw: Tuple (height, width) specifying the patch grid dimensions.
                     height * width must equal the sequence length N of x.

        Returns:
            Tensor of the same shape as x with 2D rotary embeddings applied.

        Example::

            rope2d = RoPE2D(dim=64, max_height=256, max_width=256)
            # q: (2, 23, 196, 64)  — batch=2, heads=23, 14*14=196 patches, dim=64
            q_rotated = rope2d(q, grid_hw=(14, 14))
        """
        height: int
        width: int
        height, width = grid_hw

        cos_freqs: torch.Tensor
        sin_freqs: torch.Tensor
        cos_freqs, sin_freqs = self.build_freqs(height, width)

        # Move frequency tensors to the same device and dtype as x.
        # The cached angle buffers are float32; x may be bfloat16 during
        # training. Casting here avoids dtype mismatch errors.
        cos_freqs = cos_freqs.to(dtype=x.dtype, device=x.device)
        sin_freqs = sin_freqs.to(dtype=x.dtype, device=x.device)

        return self.apply_rotary(x, cos_freqs, sin_freqs)
