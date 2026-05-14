## Code: models/positional_encoding.py

```python
## models/positional_encoding.py
"""Positional encoding components for SAM 2.

This module implements all positional encoding schemes used across the SAM 2
architecture:

1. PositionEmbeddingRandom — SAM-style random Fourier feature PE for the
   prompt encoder (identical to SAM's implementation per Section 4).

2. SinusoidalPositionalEncoding — Standard 1D sinusoidal absolute PE used
   as the base additive encoding in memory attention (Appendix D.1).

3. TemporalPositionalEncoding — Sinusoidal PE over recency-order indices,
   applied only to the N=6 recent frame memories in the memory bank. NOT
   applied to prompted frame memories (Section 4, config: num_recent_memories=6).

4. RoPE2D — 2D spatial Rotary Positional Embedding applied in memory
   attention self- and cross-attention layers. Object pointer tokens are
   explicitly excluded from rotation (Appendix D.1, Table 10).

Config references:
    model.use_rope_2d: true
    model.use_rpb: false
    model.num_recent_memories: 6
    model.memory_feature_dim: 64
    model.memory_attention_layers: 4

Paper references:
    Section 4: "we use 2d spatial Rotary Positional Embedding (RoPE)"
    Appendix D.1: "In addition to sinusoidal absolute positional embeddings,
        we use 2d spatial RoPE in self-attention and cross-attention layers.
        The object pointer tokens are excluded from RoPE."
    Section 4: "We embed temporal position information into the memories of
        N recent frames ... but not into those of prompted frames."
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. PositionEmbeddingRandom
# ---------------------------------------------------------------------------


class PositionEmbeddingRandom(nn.Module):
    """Random Fourier feature positional embedding for the SAM 2 prompt encoder.

    Identical to SAM's PositionEmbeddingRandom (Kirillov et al., 2023).
    Encodes 2D spatial positions using a fixed random Gaussian projection
    followed by sin/cos, producing a dense PE grid for image embeddings and
    sparse PE for point/box prompts.

    The random projection matrix is registered as a non-trainable buffer so
    it persists across checkpoints and moves with .to(device) calls.

    Args:
        num_pos_feats: Number of positional features per spatial dimension.
            The output dimension is num_pos_feats * 2 (sin + cos of each
            feature). Defaults to 64, producing 128-dim output.
        scale: Standard deviation of the Gaussian used to sample the random
            projection matrix. Defaults to 1.0 (SAM default).

    Example:
        pe = PositionEmbeddingRandom(num_pos_feats=64)
        # Dense grid PE for a 64x64 feature map
        grid_pe = pe.forward((64, 64))  # (64, 64, 128)
        # Sparse PE for 5 point prompts
        coords = torch.rand(5, 2)  # normalized [0,1] (x, y)
        point_pe = pe.forward_with_coords(coords, image_size=(1024, 1024))  # (5, 128)
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        if scale is None or scale <= 0.0:
            scale = 1.0

        self.num_pos_feats: int = num_pos_feats

        # Fixed random Gaussian projection matrix: shape (2, num_pos_feats)
        # The two input dimensions are normalized (x, y) coordinates in [0, 1].
        # Registered as a buffer: persists in state_dict, moves with .to(device),
        # but does NOT receive gradients.
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
        # Type annotation for IDE support (buffer is a Tensor at runtime)
        self.positional_encoding_gaussian_matrix: torch.Tensor

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply random Fourier feature encoding to normalized coordinates.

        Projects coordinates through the random Gaussian matrix and applies
        sin/cos to produce the final positional encoding.

        Args:
            coords: Coordinate tensor of shape (..., 2) with values in [0, 1].
                The last dimension contains (x, y) normalized coordinates.

        Returns:
            Positional encoding tensor of shape (..., num_pos_feats * 2).
            The output interleaves sin and cos features:
            [sin(proj_0), ..., sin(proj_{K-1}), cos(proj_0), ..., cos(proj_{K-1})]
        """
        # coords: (..., 2) in [0, 1]
        # Scale to [0, 2*pi] for the projection
        coords = 2.0 * coords - 1.0  # shift to [-1, 1]

        # Project: (..., 2) @ (2, num_pos_feats) → (..., num_pos_feats)
        # Cast matrix to same dtype as coords for bfloat16 compatibility
        proj = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)

        # Apply 2*pi scaling before sin/cos (standard Fourier feature convention)
        proj = 2.0 * math.pi * proj

        # Concatenate sin and cos: (..., num_pos_feats * 2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate a dense positional encoding grid for a spatial feature map.

        Creates a meshgrid of normalized (x, y) coordinates covering the
        entire feature map and encodes each position.

        Args:
            size: (H, W) spatial dimensions of the feature map.

        Returns:
            Dense PE grid of shape (H, W, num_pos_feats * 2).
            The encoding is in float32 regardless of model precision, as it
            is typically added to embeddings that are then cast.
        """
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device

        # Create normalized coordinate grids in [0, 1]
        # y_embed: (H,) normalized row coordinates
        # x_embed: (W,) normalized column coordinates
        y_embed = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
        x_embed = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w

        # Build 2D meshgrid: each of shape (H, W)
        y_grid, x_grid = torch.meshgrid(y_embed, x_embed, indexing="ij")

        # Stack to (H, W, 2) with (x, y) ordering (SAM convention)
        coords = torch.stack([x_grid, y_grid], dim=-1)  # (H, W, 2)

        # Apply random Fourier encoding: (H, W, num_pos_feats * 2)
        return self._pe_encoding(coords)

    def forward_with_coords(
        self,
        coords_input: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Encode sparse point coordinates for click and box prompts.

        Normalizes pixel-space coordinates to [0, 1] relative to the image
        size, then applies the random Fourier encoding.

        Args:
            coords_input: Pixel-space coordinates of shape (N, 2) or (B, N, 2)
                in (x, y) order (column, row). Values in [0, image_size].
            image_size: (H, W) of the input image for normalization.

        Returns:
            Positional encoding of shape (..., num_pos_feats * 2), where ...
            matches the leading dimensions of coords_input.
        """
        h, w = image_size
        # Normalize to [0, 1]: divide x by W, y by H
        coords = coords_input.clone().float()
        coords[..., 0] = coords[..., 0] / w  # x normalized by width
        coords[..., 1] = coords[..., 1] / h  # y normalized by height

        return self._pe_encoding(coords)

    def get_dense_pe(self, size: Tuple[int, int]) -> torch.Tensor:
        """Get dense PE for use as image_pe in the mask decoder.

        Returns the PE in the format expected by the mask decoder:
        (1, num_pos_feats * 2, H, W) — batch dim added, channels first.

        Args:
            size: (H, W) spatial dimensions of the feature map.

        Returns:
            Dense PE of shape (1, num_pos_feats * 2, H, W).
        """
        # forward returns (H, W, C) → permute to (C, H, W) → add batch dim
        pe = self.forward(size)  # (H, W, C)
        pe = pe.permute(2, 0, 1)  # (C, H, W)
        return pe.unsqueeze(0)  # (1, C, H, W)


# ---------------------------------------------------------------------------
# 2. SinusoidalPositionalEncoding
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Standard 1D sinusoidal absolute positional encoding.

    Used as the base additive PE in memory attention (Appendix D.1):
    "In addition to sinusoidal absolute positional embeddings, we use 2d
    spatial Rotary Positional Embedding (RoPE)."

    The sinusoidal PE is added to the flattened spatial token sequence before
    the memory attention transformer blocks. RoPE2D is applied separately
    within the attention computation.

    Frequencies follow the standard Transformer formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    The PE table is precomputed and registered as a non-trainable buffer.

    Args:
        d_model: Embedding dimension. Must be even.
        max_len: Maximum sequence length to precompute. Defaults to 10000,
            which covers any practical spatial sequence (e.g., 1024^2 / 16^2
            = 4096 tokens at stride-16 encoding).
        dropout: Optional dropout rate applied after adding PE. Defaults to 0.0.

    Example:
        pe = SinusoidalPositionalEncoding(d_model=256, max_len=4096)
        x = torch.randn(2, 1024, 256)  # (B, L, d_model)
        x_with_pe = pe(x)  # (B, L, d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        max_len: int = 10000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model: int = d_model
        self.max_len: int = max_len
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        # Precompute PE table of shape (max_len, d_model) in float32
        pe = self._build_pe_table(max_len, d_model)

        # Register as buffer: (1, max_len, d_model) for broadcasting over batch
        self.register_buffer("pe", pe.unsqueeze(0))
        self.pe: torch.Tensor  # type annotation for IDE

    @staticmethod
    def _build_pe_table(max_len: int, d_model: int) -> torch.Tensor:
        """Precompute the sinusoidal PE table.

        Args:
            max_len: Number of positions to precompute.
            d_model: Embedding dimension.

        Returns:
            PE table of shape (max_len, d_model), dtype float32.
        """
        # Position indices: (max_len, 1)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

        # Frequency divisors: (d_model // 2,)
        # div_term[i] = 10000^(2i / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # PE table: (max_len, d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims: cos

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding to input sequence.

        Args:
            x: Input tensor of shape (B, L, d_model) where L is the sequence
                length (e.g., H*W flattened spatial tokens).

        Returns:
            Tensor of shape (B, L, d_model) with PE added. The PE is cast to
            match the dtype of x for bfloat16 compatibility.

        Raises:
            ValueError: If L exceeds max_len.
        """
        seq_len = x.shape[1]
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}. "
                "Increase max_len in SinusoidalPositionalEncoding.__init__."
            )

        # self.pe: (1, max_len, d_model) → slice to (1, L, d_model)
        # Cast to x's dtype for bfloat16 / float16 compatibility
        pe_slice = self.pe[:, :seq_len, :].to(dtype=x.dtype)

        return self.dropout(x + pe_slice)


# ---------------------------------------------------------------------------
# 3. TemporalPositionalEncoding
# ---------------------------------------------------------------------------


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over recency-order indices for memory bank.

    Applied ONLY to the N=6 recent unprompted frame memories in the memory
    bank. NOT applied to prompted frame memories.

    From Section 4 of the paper: "We embed temporal position information into
    the memories of N recent frames, allowing the model to represent short-term
    object motion, but not into those of prompted frames, because the training
    signal from prompted frames is sparser and it is more difficult to
    generalize to the inference setting where prompted frames may come from a
    very different temporal range than seen during training."

    Config reference: model.num_recent_memories: 6

    The encoding uses recency order (0 = most recent, N-1 = oldest in the
    FIFO queue), NOT absolute video frame timestamps. This ensures
    generalization: during inference, absolute frame numbers are arbitrary,
    but the relative ordering within the N-frame window is always 0..N-1.

    Args:
        d_model: Embedding dimension of the memory features. Should match
            model.memory_feature_dim (64 per config).
        max_len: Maximum number of recent memories to support. Should be at
            least model.num_recent_memories (6 per config). Defaults to 64
            to provide headroom for ablations (Table 9c tests up to N=8).

    Example:
        tpe = TemporalPositionalEncoding(d_model=64, max_len=64)
        # Encode 6 recent memories (recency indices 0..5)
        indices = list(range(6))
        temporal_pe = tpe(indices)  # (6, 64)
        # Add to memory features before cross-attention
        memory_features += temporal_pe.unsqueeze(0)  # broadcast over batch
    """

    def __init__(
        self,
        d_model: int = 64,
        max_len: int = 64,
    ) -> None:
        super().__init__()

        self.d_model: int = d_model
        self.max_len: int = max_len

        # Precompute PE table of shape (max_len, d_model) in float32
        pe = SinusoidalPositionalEncoding._build_pe_table(max_len, d_model)

        # Register as buffer: (max_len, d_model)
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor  # type annotation for IDE

    def forward(self, frame_indices: List[int]) -> torch.Tensor:
        """Retrieve temporal PE vectors for a list of recency-order indices.

        Args:
            frame_indices: List of integer recency indices. Each index
                represents the recency position of a memory in the FIFO queue:
                0 = most recent frame, 1 = one step back, ..., N-1 = oldest.
                Length must be <= max_len.

        Returns:
            Temporal PE tensor of shape (len(frame_indices), d_model),
            dtype float32. The caller (MemoryBank.get_memory_for_attention)
            adds this to the spatial memory features before cross-attention.

        Raises:
            ValueError: If any index in frame_indices exceeds max_len - 1.
        """
        if not frame_indices:
            # Return empty tensor with correct feature dimension
            return torch.zeros(
                0, self.d_model,
                dtype=self.pe.dtype,
                device=self.pe.device,
            )

        max_idx = max(frame_indices)
        if max_idx >= self.max_len:
            raise ValueError(
                f"frame_index {max_idx} exceeds max_len {self.max_len} - 1. "
                "Increase max_len in TemporalPositionalEncoding.__init__."
            )

        # Index into the PE table: (len(frame_indices), d_model)
        indices_tensor = torch.tensor(
            frame_indices,
            dtype=torch.long,
            device=self.pe.device,
        )
        return self.pe[indices_tensor]  # (N, d_model)


# ---------------------------------------------------------------------------
# 4. RoPE2D
# ---------------------------------------------------------------------------


class RoPE2D(nn.Module):
    """2D spatial Rotary Positional Embedding for SAM 2 memory attention.

    Implements 2D-RoPE as described in Su et al. (2021) and Heo et al. (2024),
    applied in both self-attention and cross-attention layers of the memory
    attention module.

    From Appendix D.1: "we use 2d spatial Rotary Positional Embedding (RoPE)
    in self-attention and cross-attention layers. The object pointer tokens are
    excluded from RoPE as they do not have specific spatial correspondence."

    Config reference: model.use_rope_2d: true

    Mathematical foundation:
        Standard 1D RoPE rotates Q and K vectors by position-dependent angles
        before computing attention. For 2D spatial positions (h, w), we split
        the head dimension in half: the first half encodes height position,
        the second half encodes width position.

        For a token at position (row, col) with head dimension D:
        - Dimensions 0..D/2-1: rotated by row * theta_i
        - Dimensions D/2..D-1: rotated by col * theta_i

        where theta_i = 1 / (10000^(4i/D)) for i = 0..D/4-1.

        The rotation is applied as complex multiplication:
        [x_{2i}, x_{2i+1}] → [x_{2i}*cos - x_{2i+1}*sin, x_{2i}*sin + x_{2i+1}*cos]

    Object pointer exclusion:
        When the sequence contains both spatial tokens (H*W) and object pointer
        tokens (appended after spatial tokens), RoPE is applied only to the
        spatial portion. The pointer tokens are passed through unchanged.
        The spatial length `hw` is passed explicitly to rotate_queries_and_keys.

    Args:
        dim: Per-head attention dimension. Must be divisible by 4 (split into
            height and width halves, each further split into sin/cos pairs).
        max_seq_len: Maximum sequence length (H or W) to precompute. Defaults
            to 256, which covers 256x256 spatial maps at stride-4 encoding.

    Example:
        rope = RoPE2D(dim=64, max_seq_len=256)
        # Q and K of shape (B, num_heads, H*W, head_dim)
        q = torch.randn(2, 8, 64*64, 64)
        k = torch.randn(2, 8, 64*64, 64)
        q_rot, k_rot = rope.rotate_queries_and_keys(q, k, h=64, w=64)
    """

    def __init__(
        self,
        dim: int = 64,
        max_seq_len: int = 256,
    ) -> None:
        super().__init__()

        if dim % 4 != 0:
            raise ValueError(
                f"RoPE2D requires dim divisible by 4, got {dim}. "
                "The dimension is split into height/width halves, each "
                "further split into sin/cos pairs."
            )

        self.dim: int = dim
        self.max_seq_len: int = max_seq_len

        # Number of frequency bands per spatial dimension (height or width)
        # Each spatial dimension uses dim/4 frequency bands
        self.half_dim: int = dim // 2       # per-spatial-dimension size
        self.num_freqs: int = dim // 4      # frequency bands per dimension

        # Precompute frequency bands: theta_i = 1 / (10000^(4i/dim))
        # Shape: (num_freqs,)
        freqs = 1.0 / (
            10000.0 ** (
                torch.arange(0, self.num_freqs, dtype=torch.float32)
                * (4.0 / dim)
            )
        )
        self.register_buffer("freqs", freqs)
        self.freqs: torch.Tensor  # type annotation

        # Precompute sin/cos tables for positions 0..max_seq_len-1
        # Shape: (max_seq_len, num_freqs)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # Outer product: (max_seq_len, num_freqs)
        angles = torch.outer(positions, freqs)

        self.register_buffer("sin_table", torch.sin(angles))
        self.register_buffer("cos_table", torch.cos(angles))
        self.sin_table: torch.Tensor  # type annotation
        self.cos_table: torch.Tensor  # type annotation

    def _build_freqs(
        self,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build 2D sin/cos frequency grids for a spatial map of size (h, w).

        Constructs separate frequency grids for height and width positions,
        then concatenates them to form the full 2D RoPE encoding.

        Args:
            h: Height of the spatial feature map.
            w: Width of the spatial feature map.

        Returns:
            Tuple of (sin_2d, cos_2d), each of shape (h*w, dim//2).
            The first dim//4 dimensions encode height, the last dim//4
            dimensions encode width.

        Raises:
            ValueError: If h or w exceeds max_seq_len.
        """
        if h > self.max_seq_len:
            raise ValueError(
                f"Height {h} exceeds max_seq_len {self.max_seq_len}. "
                "Increase max_seq_len in RoPE2D.__init__."
            )
        if w > self.max_seq_len:
            raise ValueError(
                f"Width {w} exceeds max_seq_len {self.max_seq_len}. "
                "Increase max_seq_len in RoPE2D.__init__."
            )

        # Height frequencies: (h, num_freqs)
        sin_h = self.sin_table[:h]  # (h, num_freqs)
        cos_h = self.cos_table[:h]  # (h, num_freqs)

        # Width frequencies: (w, num_freqs)
        sin_w = self.sin_table[:w]  # (w, num_freqs)
        cos_w = self.cos_table[:w]  # (w, num_freqs)

        # Expand to 2D grids: (h, w, num_freqs) each
        # Height: repeat each row across all columns
        sin_h_2d = sin_h.unsqueeze(1).expand(h, w, self.num_freqs)  # (h, w, num_freqs)
        cos_h_2d = cos_h.unsqueeze(1).expand(h, w, self.num_freqs)  # (h, w, num_freqs)

        # Width: repeat each column across all rows
        sin_w_2d = sin_w.unsqueeze(0).expand(h, w, self.num_freqs)  # (h, w, num_freqs)
        cos_w_2d = cos_w.unsqueeze(0).expand(h, w, self.num_freqs)  # (h, w, num_freqs)

        # Flatten spatial dimensions: (h*w, num_freqs)
        sin_h_flat = sin_h_2d.reshape(h * w, self.num_freqs)
        cos_h_flat = cos_h_2d.reshape(h * w, self.num_freqs)
        sin_w_flat = sin_w_2d.reshape(h * w, self.num_freqs)
        cos_w_flat = cos_w_2d.reshape(h * w, self.num_freqs)

        # Concatenate height and width encodings: (h*w, dim//2)
        sin_2d = torch.cat([sin_h_flat, sin_w_flat], dim=-1)  # (h*w, half_dim)
        cos_2d = torch.cat([cos_h_flat, cos_w_flat], dim=-1)  # (h*w, half_dim)

        return sin_2d, cos_2d

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension by splitting into pairs and negating.

        For a vector [x0, x1, x2, x3, ...], produces [-x1, x0, -x3, x2, ...].
        This implements the rotation matrix:
            [cos  -sin] [x0]   [x0*cos - x1*sin]
            [sin   cos] [x1] = [x0*sin + x1*cos]

        The negation of the second element in each pair is precomputed here;
        the caller multiplies by sin and cos separately.

        Args:
            x: Tensor of shape (..., D) where D is even.

        Returns:
            Tensor of shape (..., D) with pairs rotated.
        """
        # Split into even and odd indexed dimensions
        x1 = x[..., 0::2]   # (..., D//2) — even indices
        x2 = x[..., 1::2]   # (..., D//2) — odd indices

        # Interleave [-x2, x1]: produces [-x1, x0, -x3, x2, ...]
        # Stack along new last dim then flatten: (..., D//2, 2) → (..., D)
        rotated = torch.stack([-x2, x1], dim=-1)
        return rotated.flatten(-2)  # (..., D)

    def _apply_rope(
        self,
        x: torch.Tensor,
        sin_2d: torch.Tensor,
        cos_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 2D RoPE rotation to a sequence of spatial tokens.

        Applies the rotation formula:
            x_rot = x * cos + rotate_half(x) * sin

        Only the first half_dim dimensions are rotated (the spatial encoding
        dimensions). The remaining dimensions (if any) are passed through
        unchanged. This handles the case where dim < head_dim.

        Args:
            x: Token tensor of shape (B, num_heads, L, head_dim) where
                L = h*w (spatial tokens only, no pointer tokens).
            sin_2d: Sin values of shape (L, half_dim).
            cos_2d: Cos values of shape (L, half_dim).

        Returns:
            Rotated tensor of shape (B, num_heads, L, head_dim).
        """
        B, num_heads, L, head_dim = x.shape

        # Cast sin/cos to match x's dtype for bfloat16 compatibility
        sin_vals = sin_2d.to(dtype=x.dtype)  # (L, half_dim)
        cos_vals = cos_2d.to(dtype=x.dtype)  # (L, half_dim)

        # Reshape for broadcasting: (1, 1, L, half_dim)
        sin_vals = sin_vals.unsqueeze(0).unsqueeze(0)
        cos_vals = cos_vals.unsqueeze(0).unsqueeze(0)

        if head_dim == self.half_dim:
            # All dimensions are spatial encoding dimensions — rotate all
            x_rot = x * cos_vals + self._rotate_half(x) * sin_vals
        elif head_dim > self.half_dim:
            # Only rotate the first half_dim dimensions; pass rest through
            x_spatial = x[..., :self.half_dim]
            x_rest = x[..., self.half_dim:]
            x_spatial_rot = (
                x_spatial * cos_vals + self._rotate_half(x_spatial) * sin_vals
            )
            x_rot = torch.cat([x_spatial_rot, x_rest], dim=-1)
        else:
            # head_dim < half_dim: rotate all available dimensions
            # Truncate sin/cos to head_dim
            sin_trunc = sin_vals[..., :head_dim]
            cos_trunc = cos_vals[..., :head_dim]
            x_rot = x * cos_trunc + self._rotate_half(x) * sin_trunc

        return x_rot

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Apply 2D RoPE to a spatial token sequence.

        Convenience method for applying RoPE to a single tensor. For the
        typical use case of rotating both Q and K, use rotate_queries_and_keys
        instead (more efficient: builds freqs once for both).

        Args:
            x: Token tensor of shape (B, num_heads, L, head_dim) where
                L = h*w. Must contain only spatial tokens (no pointer tokens).
            h: Height of the spatial feature map.
            w: Width of the spatial feature map.

        Returns:
            Rotated tensor of shape (B, num_heads, L, head_dim).

        Raises:
            ValueError: If L != h*w.
        """
        L =