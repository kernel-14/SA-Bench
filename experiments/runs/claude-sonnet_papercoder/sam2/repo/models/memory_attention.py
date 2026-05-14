## Code: models/memory_attention.py

```python
## models/memory_attention.py
"""Memory attention module for SAM 2.

This module conditions current frame features on past predictions stored in
the memory bank. It is the core temporal reasoning component of SAM 2's
streaming architecture.

Architecture (Section 4, Appendix D.1):
    L=4 stacked transformer blocks, each performing:
        1. Self-attention (with 2D-RoPE on spatial frame tokens)
        2. Cross-attention to spatial memories + object pointer tokens
           (2D-RoPE on spatial portions; object pointers excluded from RoPE)
        3. MLP (2-layer, 4x expansion, ReLU activation)

Key design decisions:
    - 2D-RoPE in both self- and cross-attention (Table 10 default)
    - No RPB → enables FlashAttention-2 (Appendix A.2.2)
    - Spatial memories projected from 64-dim to 256-dim per layer
    - Object pointer tokens excluded from RoPE (Appendix D.1)
    - Empty memory bank → cross-attention output is zero (image-only mode)

Config references:
    model.memory_attention_layers: 4
    model.memory_attention_self_attn: 4
    model.memory_attention_cross_attn: 4
    model.memory_feature_dim: 64
    model.fpn_out_channels: 256
    model.use_rope_2d: true
    model.use_flash_attention: true
    model.input_resolution: 1024

Paper references:
    Section 4: "We stack L transformer blocks, the first one taking the image
        encoding from the current frame as input. Each block performs
        self-attention, followed by cross-attention to memories of
        (prompted/unprompted) frames and object pointers, stored in a memory
        bank, followed by an MLP."
    Appendix D.1: "The object pointer tokens are excluded from RoPE as they
        do not have specific spatial correspondence."
    Appendix A.2.2: "Removing RPB also allows us to enable FlashAttention-2."
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import FlashAttention-2; fall back gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # type: ignore
    _FLASH_ATTN_AVAILABLE = True
    logger.info("FlashAttention-2 is available and will be used in MemoryAttention.")
except ImportError:
    _FLASH_ATTN_AVAILABLE = False
    logger.warning(
        "FlashAttention-2 not found. Falling back to standard PyTorch attention. "
        "Install flash-attn for full performance: pip install flash-attn==2.5.9.post1"
    )


# ---------------------------------------------------------------------------
# MemoryBankOutput dataclass (imported from memory_bank to avoid circular deps)
# We re-define a minimal version here to avoid circular imports.
# The actual MemoryBankOutput is defined in models/memory_bank.py.
# ---------------------------------------------------------------------------

@dataclass
class _MemoryBankOutputRef:
    """Reference type for type hints — actual class is in models/memory_bank.py.

    This avoids circular imports: memory_attention.py does not import from
    memory_bank.py. The actual MemoryBankOutput dataclass has the same fields.
    """
    spatial_memories: torch.Tensor   # [B, N_mem_tokens, memory_feature_dim=64]
    object_pointers: torch.Tensor    # [B, N_ptr, memory_feature_dim=64]
    temporal_embeddings: torch.Tensor  # [B, N_recent_tokens, memory_feature_dim=64]


# ---------------------------------------------------------------------------
# SAM2Config reference (forward declaration to avoid circular import)
# MemoryAttention.__init__ accepts a config object with these attributes.
# ---------------------------------------------------------------------------

class _SAM2ConfigProtocol:
    """Protocol for SAM2Config attributes used by MemoryAttention.

    MemoryAttention only reads these fields from the config. The actual
    SAM2Config dataclass is defined in models/sam2.py.
    """
    memory_attention_layers: int      # 4
    memory_attention_self_attn: int   # 4
    memory_attention_cross_attn: int  # 4
    memory_feature_dim: int           # 64
    fpn_out_channels: int             # 256
    use_rope_2d: bool                 # True
    use_flash_attention: bool         # True
    input_resolution: int             # 1024


# ---------------------------------------------------------------------------
# RoPE2D
# ---------------------------------------------------------------------------


class RoPE2D(nn.Module):
    """2D spatial Rotary Positional Embedding for SAM 2 memory attention.

    Applies position-dependent rotation to query and key vectors in attention,
    encoding 2D spatial position (row, column) of each token.

    From Appendix D.1: "we use 2d spatial Rotary Positional Embedding (RoPE)
    in self-attention and cross-attention layers. The object pointer tokens are
    excluded from RoPE as they do not have specific spatial correspondence."

    Implementation:
        - Split head_dim in half: first half encodes row (y), second half
          encodes column (x) position.
        - For each half, apply standard 1D RoPE with frequencies:
          theta_i = 1 / (10000^(2i / half_dim)) for i = 0..half_dim//2 - 1
        - Rotation: x_rot = x * cos(freq) + rotate_half(x) * sin(freq)

    Args:
        dim: Per-head attention dimension (d_model // num_heads = 256 // 4 = 64).
            Must be divisible by 4.
        max_seq_len: Maximum spatial sequence length (H or W) to support.
            Defaults to 256, covering 256×256 spatial maps at stride-4.
            At stride-16 with 1024 input: H=W=64, so 256 is sufficient.

    Example:
        rope = RoPE2D(dim=64, max_seq_len=256)
        x = torch.randn(2, 64*64, 64)  # [B, H*W, head_dim]
        x_rot = rope(x, h=64, w=64)    # [B, H*W, head_dim]
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
                "The dimension is split into row/col halves, each further "
                "split into sin/cos pairs."
            )

        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        self.half_dim: int = dim // 2       # per-spatial-dimension size
        self.quarter_dim: int = dim // 4    # frequency bands per dimension

        # Precompute frequency bands for row and column dimensions.
        # theta_i = 1 / (10000^(2i / half_dim)) for i = 0..quarter_dim-1
        # Shape: (quarter_dim,)
        freqs = 1.0 / (
            10000.0 ** (
                torch.arange(0, self.quarter_dim, dtype=torch.float32)
                * (2.0 / self.half_dim)
            )
        )
        self.register_buffer("freqs", freqs)
        self.freqs: torch.Tensor  # type annotation for IDE

    def _build_freqs(self, h: int, w: int) -> torch.Tensor:
        """Build 2D sin/cos frequency tensor for a spatial map of size (h, w).

        Constructs separate frequency grids for row and column positions,
        then concatenates them to form the full 2D RoPE encoding.

        The output encodes each of the h*w spatial positions as a vector of
        shape (dim,) where:
        - dims 0..half_dim-1: row position encoding (sin/cos interleaved)
        - dims half_dim..dim-1: column position encoding (sin/cos interleaved)

        Args:
            h: Height of the spatial feature map.
            w: Width of the spatial feature map.

        Returns:
            Complex-valued frequency tensor of shape (h*w, dim//2) where
            each element is a complex number exp(i * theta) representing
            the rotation angle. Stored as real tensor of shape (h*w, dim//2)
            with alternating real/imag parts.

            Actually returns (sin_vals, cos_vals) packed as a single tensor
            of shape (h*w, dim) for efficient application.

        Raises:
            ValueError: If h or w exceeds max_seq_len.
        """
        if h > self.max_seq_len:
            raise ValueError(
                f"Height {h} exceeds RoPE2D.max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len in RoPE2D.__init__."
            )
        if w > self.max_seq_len:
            raise ValueError(
                f"Width {w} exceeds RoPE2D.max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len in RoPE2D.__init__."
            )

        device = self.freqs.device
        dtype = torch.float32  # always compute in float32 for numerical stability

        # Row position angles: [h, quarter_dim]
        row_ids = torch.arange(h, device=device, dtype=dtype)
        row_angles = torch.outer(row_ids, self.freqs)  # [h, quarter_dim]

        # Column position angles: [w, quarter_dim]
        col_ids = torch.arange(w, device=device, dtype=dtype)
        col_angles = torch.outer(col_ids, self.freqs)  # [w, quarter_dim]

        # Expand to 2D grids: [h, w, quarter_dim]
        # Row: each row value repeated across all columns
        row_angles_2d = row_angles.unsqueeze(1).expand(h, w, self.quarter_dim)
        # Col: each column value repeated across all rows
        col_angles_2d = col_angles.unsqueeze(0).expand(h, w, self.quarter_dim)

        # Flatten spatial dimensions: [h*w, quarter_dim]
        row_angles_flat = row_angles_2d.reshape(h * w, self.quarter_dim)
        col_angles_flat = col_angles_2d.reshape(h * w, self.quarter_dim)

        # Interleave sin/cos for each dimension pair:
        # For row: [sin(r0), cos(r0), sin(r1), cos(r1), ...] → [h*w, half_dim]
        # For col: [sin(c0), cos(c0), sin(c1), cos(c1), ...] → [h*w, half_dim]
        sin_row = torch.sin(row_angles_flat)  # [h*w, quarter_dim]
        cos_row = torch.cos(row_angles_flat)  # [h*w, quarter_dim]
        sin_col = torch.sin(col_angles_flat)  # [h*w, quarter_dim]
        cos_col = torch.cos(col_angles_flat)  # [h*w, quarter_dim]

        # Stack and interleave: [h*w, quarter_dim, 2] → [h*w, half_dim]
        row_encoding = torch.stack([sin_row, cos_row], dim=-1).reshape(h * w, self.half_dim)
        col_encoding = torch.stack([sin_col, cos_col], dim=-1).reshape(h * w, self.half_dim)

        # Concatenate row and col encodings: [h*w, dim]
        # First half_dim dims = row encoding, second half_dim dims = col encoding
        freqs_2d = torch.cat([row_encoding, col_encoding], dim=-1)  # [h*w, dim]

        return freqs_2d

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs of dimensions by 90 degrees.

        For input [..., d0, d1, d2, d3, ...], produces [..., -d1, d0, -d3, d2, ...].
        This implements the rotation matrix for each (sin, cos) pair:
            [cos  -sin] [d0]   [d0*cos - d1*sin]
            [sin   cos] [d1] = [d0*sin + d1*cos]

        The negation of odd-indexed elements is precomputed here; the caller
        multiplies by sin and cos separately.

        Args:
            x: Tensor of shape (..., D) where D is even.

        Returns:
            Tensor of shape (..., D) with pairs rotated.
        """
        # Split into even (0, 2, 4, ...) and odd (1, 3, 5, ...) indexed dims
        x_even = x[..., 0::2]   # (..., D//2) — sin components
        x_odd = x[..., 1::2]    # (..., D//2) — cos components

        # Interleave [-x_odd, x_even]: produces [-d1, d0, -d3, d2, ...]
        rotated = torch.stack([-x_odd, x_even], dim=-1)
        return rotated.flatten(-2)  # (..., D)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Apply 2D RoPE rotation to a spatial token sequence.

        The rotation formula is:
            x_rot = x * cos_part + rotate_half(x) * sin_part

        where cos_part and sin_part are extracted from the interleaved
        (sin, cos) encoding built by _build_freqs.

        Args:
            x: Token tensor of shape (B, h*w, head_dim) or (B, h*w, d_model).
                Must contain only spatial tokens (no pointer tokens appended).
                The sequence length must equal h*w.
            h: Height of the spatial feature map.
            w: Width of the spatial feature map.

        Returns:
            Rotated tensor of same shape as x.

        Raises:
            ValueError: If x.shape[1] != h*w.
        """
        B, L, D = x.shape

        if L != h * w:
            raise ValueError(
                f"RoPE2D.forward: sequence length {L} != h*w={h*w}. "
                f"Got h={h}, w={w}. Ensure only spatial tokens are passed "
                "(no object pointer tokens)."
            )

        # Build 2D frequency grid: [h*w, dim]
        freqs_2d = self._build_freqs(h, w)  # [h*w, dim]

        # Cast to match x's dtype for bfloat16 compatibility
        freqs_2d = freqs_2d.to(dtype=x.dtype, device=x.device)

        # Extract sin and cos parts from interleaved encoding
        # freqs_2d has shape [h*w, dim] where pairs are (sin, cos)
        # sin_part: even indices (0, 2, 4, ...) → [h*w, dim//2]
        # cos_part: odd indices (1, 3, 5, ...) → [h*w, dim//2]
        sin_part = freqs_2d[..., 0::2]  # [h*w, dim//2]
        cos_part = freqs_2d[..., 1::2]  # [h*w, dim//2]

        # Interleave back to [h*w, dim] for element-wise multiplication
        # sin_full[..., 0::2] = sin_part, sin_full[..., 1::2] = sin_part
        # cos_full[..., 0::2] = cos_part, cos_full[..., 1::2] = cos_part
        sin_full = torch.stack([sin_part, sin_part], dim=-1).flatten(-2)  # [h*w, dim]
        cos_full = torch.stack([cos_part, cos_part], dim=-1).flatten(-2)  # [h*w, dim]

        # Truncate or pad to match x's last dimension D
        if D <= sin_full.shape[-1]:
            sin_full = sin_full[..., :D]
            cos_full = cos_full[..., :D]
        else:
            # D > dim: pad with zeros (no rotation for extra dims)
            pad_size = D - sin_full.shape[-1]
            sin_full = F.pad(sin_full, (0, pad_size), value=0.0)
            cos_full = F.pad(cos_full, (0, pad_size), value=1.0)

        # Add batch dimension for broadcasting: [1, h*w, D]
        sin_full = sin_full.unsqueeze(0)
        cos_full = cos_full.unsqueeze(0)

        # Apply rotation: x_rot = x * cos + rotate_half(x) * sin
        x_rot = x * cos_full + self._rotate_half(x) * sin_full

        return x_rot


# ---------------------------------------------------------------------------
# MemoryAttentionLayer
# ---------------------------------------------------------------------------


class MemoryAttentionLayer(nn.Module):
    """Single transformer block for SAM 2 memory attention.

    Each block performs:
        1. Self-attention on current frame tokens (with 2D-RoPE)
        2. Cross-attention to spatial memories + object pointer tokens
           (2D-RoPE on spatial portions; object pointers excluded)
        3. MLP (2-layer, 4x expansion, ReLU)

    All three sub-operations use pre-norm (LayerNorm before the operation)
    with residual connections, following the standard transformer convention.

    Memory projection:
        Spatial memories are stored at memory_feature_dim=64 but attention
        operates at d_model=256. A linear projection (mem_proj) is applied
        per-layer before cross-attention. Similarly for object pointers (ptr_proj).

    Empty memory handling:
        When both spatial_memories and object_pointers are empty (image-only
        mode), the cross-attention output is set to zero, and the residual
        connection passes the query through unchanged. This ensures SAM 2
        "behaves like SAM" on single images (Section 4).

    Args:
        d_model: Model dimension (frame embedding channels = fpn_out_channels = 256).
        nhead: Number of attention heads. Defaults to 4.
        memory_feature_dim: Dimension of stored memory features. Defaults to 64.
        use_rope: If True, apply 2D-RoPE in self- and cross-attention.
        use_flash_attn: If True, use FlashAttention-2 when available.
        mlp_ratio: MLP hidden dimension ratio. Defaults to 4.0.
        dropout: Dropout rate for attention and MLP. Defaults to 0.0.

    Example:
        layer = MemoryAttentionLayer(d_model=256, nhead=4, memory_feature_dim=64)
        query = torch.randn(2, 4096, 256)   # [B, H*W, d_model]
        mem_keys = torch.randn(2, 384, 64)  # [B, N_mem_tokens, 64]
        mem_vals = torch.randn(2, 384, 64)
        obj_ptrs = torch.randn(2, 16, 64)   # [B, N_ptr, 64]
        out = layer(query, mem_keys, mem_vals, obj_ptrs, h=64, w=64)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        memory_feature_dim: int = 64,
        use_rope: bool = True,
        use_flash_attn: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model: int = d_model
        self.nhead: int = nhead
        self.head_dim: int = d_model // nhead
        self.memory_feature_dim: int = memory_feature_dim
        self.use_rope: bool = use_rope
        self.use_flash_attn: bool = use_flash_attn and _FLASH_ATTN_AVAILABLE

        # ------------------------------------------------------------------
        # Self-attention (frame tokens attend to each other)
        # ------------------------------------------------------------------
        self.self_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # ------------------------------------------------------------------
        # Cross-attention (frame tokens attend to memory bank)
        # Keys and values come from projected memory features (d_model after proj)
        # ------------------------------------------------------------------
        self.cross_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            kdim=d_model,
            vdim=d_model,
        )

        # ------------------------------------------------------------------
        # MLP: 2-layer with ReLU, 4x expansion (standard transformer MLP)
        # ------------------------------------------------------------------
        mlp_hidden_dim: int = int(d_model * mlp_ratio)
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, d_model),
        )

        # ------------------------------------------------------------------
        # Layer norms: pre-norm convention
        # norm1: before self-attention
        # norm2: before cross-attention
        # norm3: before MLP
        # ------------------------------------------------------------------
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm3: nn.LayerNorm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # 2D-RoPE for spatial token rotation
        # Applied to Q and K in self-attention and cross-attention
        # Object pointer tokens are excluded (no spatial correspondence)
        # ------------------------------------------------------------------
        if use_rope:
            self.rope_2d: RoPE2D = RoPE2D(
                dim=self.head_dim,
                max_seq_len=256,  # covers 256×256 spatial maps at stride-4
            )
        else:
            self.rope_2d = None  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # Projection layers: memory_feature_dim (64) → d_model (256)
        # mem_proj: projects spatial memory features
        # ptr_proj: projects object pointer tokens
        # ------------------------------------------------------------------
        self.mem_proj: nn.Linear = nn.Linear(memory_feature_dim, d_model, bias=True)
        self.ptr_proj: nn.Linear = nn.Linear(memory_feature_dim, d_model, bias=True)

        # Dropout for residual connections
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _apply_rope_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D-RoPE to query and key tensors for self-attention.

        Reshapes from [B, L, d_model] to [B, L, nhead, head_dim], applies
        per-head rotation, then reshapes back.

        Args:
            q: Query tensor of shape [B, L, d_model].
            k: Key tensor of shape [B, L, d_model].
            h: Spatial height (L = h*w).
            w: Spatial width.

        Returns:
            Tuple of rotated (q, k), each of shape [B, L, d_model].
        """
        B, L, D = q.shape

        # Reshape to per-head: [B, L, nhead, head_dim]
        q_heads = q.view(B, L, self.nhead, self.head_dim)
        k_heads = k.view(B, L, self.nhead, self.head_dim)

        # Apply RoPE per head: process each head's spatial tokens
        # rope_2d.forward expects [B, L, head_dim] — process all heads together
        # by merging batch and head dims: [B*nhead, L, head_dim]
        q_merged = q_heads.permute(0, 2, 1, 3).reshape(B * self.nhead, L, self.head_dim)
        k_merged = k_heads.permute(0, 2, 1, 3).reshape(B * self.nhead, L, self.head_dim)

        q_rot = self.rope_2d(q_merged, h=h, w=w)  # [B*nhead, L, head_dim]
        k_rot = self.rope_2d(k_merged, h=h, w=w)  # [B*nhead, L, head_dim]

        # Reshape back: [B, L, d_model]
        q_rot = q_rot.reshape(B, self.nhead, L, self.head_dim).permute(0, 2, 1, 3)
        q_rot = q_rot.reshape(B, L, D)
        k_rot = k_rot.reshape(B, self.nhead, L, self.head_dim).permute(0, 2, 1, 3)
        k_rot = k_rot.reshape(B, L, D)

        return q_rot, k_rot

    def _apply_rope_to_cross_qk(
        self,
        q: torch.Tensor,
        mem_k: torch.Tensor,
        h: int,
        w: int,
        h_mem: int,
        w_mem: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 2D-RoPE to cross-attention Q and spatial memory K.

        Object pointer keys are NOT rotated (excluded from RoPE per Appendix D.1).
        This method only rotates the spatial memory portion of the keys.

        Args:
            q: Query tensor (frame tokens) of shape [B, L_q, d_model].
            mem_k: Spatial memory key tensor of shape [B, L_mem, d_model].
                This is the projected spatial memory portion only (no pointers).
            h: Query spatial height (L_q = h*w).
            w: Query spatial width.
            h_mem: Memory spatial height (L_mem = h_mem*w_mem).
            w_mem: Memory spatial width.

        Returns:
            Tuple of (q_rot, mem_k_rot), each with same shape as inputs.
        """
        B, L_q, D = q.shape
        B_m, L_mem, D_m = mem_k.shape

        # Rotate query
        q_heads = q.view(B, L_q, self.nhead, self.head_dim)
        q_merged = q_heads.permute(0, 2, 1, 3).reshape(B * self.nhead, L_q, self.head_dim)
        q_rot = self.rope_2d(q_merged, h=h, w=w)
        q_rot = q_rot.reshape(B, self.nhead, L_q, self.head_dim).permute(0, 2, 1, 3)
        q_rot = q_rot.reshape(B, L_q, D)

        # Rotate spatial memory keys
        mem_k_heads = mem_k.view(B_m, L_mem, self.nhead, self.head_dim)
        mem_k_merged = mem_k_heads.permute(0, 2, 1, 3).reshape(
            B_m * self.nhead, L_mem, self.head_dim
        )
        mem_k_rot = self.rope_2d(mem_k_merged, h=h_mem, w=w_mem)
        mem_k_rot = mem_k_rot.reshape(B_m, self.nhead, L_mem, self.head_dim).permute(
            0, 2, 1, 3
        )
        mem_k_rot = mem_k_rot.reshape(B_m, L_mem, D_m)

        return q_rot, mem_k_rot

    def _flash_self_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute self-attention using FlashAttention-2.

        FlashAttention-2 expects inputs in [B, L, nhead, head_dim] format.

        Args:
            q: Query of shape [B, L, d_model].
            k: Key of shape [B, L, d_model].
            v: Value of shape [B, L, d_model].

        Returns:
            Attention output of shape [B, L, d_model].
        """
        B, L, D = q.shape

        # Reshape to [B, L, nhead, head_dim]
        q_fa = q.view(B, L, self.nhead, self.head_dim)
        k_fa = k.view(B, L, self.nhead, self.head_dim)
        v_fa = v.view(B, L, self.nhead, self.head_dim)

        # FlashAttention-2 requires float16 or bfloat16
        orig_dtype = q_fa.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            q_fa = q_fa.to(torch.bfloat16)
            k_fa = k_fa.to(torch.bfloat16)
            v_fa = v_fa.to(torch.bfloat16)

        out = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=0.0, causal=False)

        # Reshape back: [B, L, d_model]
        out = out.reshape(B, L, D)

        if orig_dtype not in (torch.float16, torch.bfloat16):
            out = out.to(orig_dtype)

        return out

    def _flash_cross_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-attention using FlashAttention-2.

        Supports different sequence lengths for Q vs K/V (cross-attention).

        Args:
            q: Query of shape [B, L_q, d_model].
            k: Key of shape [B, L_k