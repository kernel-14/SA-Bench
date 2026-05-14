# models/visual_encoder.py

"""
Visual Encoder for the NaViL native multimodal model.

Implements a Vision Transformer (ViT) with bidirectional attention and 2D Rotary
Position Embeddings (2D-RoPE) as described in the NaViL paper (Sec. 3.1 and 5.1).
The encoder takes raw pixel images padded to multiples of 32 and outputs a
sequence of visual tokens ready for the connector stage.

The architecture faithfully follows the configuration driven by the
``config.yaml`` file, with parameters such as depth, width, number of heads, and
patch size directly exposed.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: RMS Normalization (used by Transformer blocks)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (as in the base LLM, InternLM2 / Qwen3).

    Args:
        hidden_size: size of the input dimension.
        eps: small constant for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight.to(x.dtype) * x.to(input_dtype)


# ---------------------------------------------------------------------------
# 2D Rotary Position Embedding (RoPE2D)
# ---------------------------------------------------------------------------

class RoPE2D(nn.Module):
    """
    Applies 2D Rotary Position Embeddings to query and key tensors.

    For a visual token located at grid position (row, col), the head dimension
    is split into two halves.  The first half receives RoPE based on the row
    index, and the second half receives RoPE based on the column index.

    Args:
        head_dim: dimension of each attention head (must be even).
        max_grid_size: maximum number of rows/columns the encoder may see.
            Precomputed cos/sin tables are built up to this value.
        theta: base frequency for RoPE (default 10000.0 as usual).
    """

    def __init__(
        self,
        head_dim: int,
        max_grid_size: int = 1024,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for 2D RoPE")
        self.head_dim = head_dim
        self.half_head_dim = head_dim // 2
        self.max_grid_size = max_grid_size

        # Build separate inverse frequency vectors for row and column halves.
        # Each half is responsible for head_dim // 4 frequencies (since we split
        # the half into two equal parts for row/col). A typical 1D RoPE for a
        # dimension d uses frequencies for indices 0..d/2-1.  We'll treat each
        # "half" as an independent RoPE.
        half_dim = self.half_head_dim // 2
        freq = 1.0 / (
            theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        )

        # Precompute cos/sin for all position values up to max_grid_size.
        pos = torch.arange(max_grid_size, dtype=torch.float32)
        # Reshape for broadcasting: [max_grid_size, 1] @ [1, half_dim] -> [max_grid_size, half_dim]
        pos_freq = pos.unsqueeze(1) * freq.unsqueeze(0)
        cos = pos_freq.cos().unsqueeze(0)  # [1, max_grid_size, half_dim]
        sin = pos_freq.sin().unsqueeze(0)

        # We need separate tables for rows and columns (identical values).
        self.register_buffer("row_cos", cos)
        self.register_buffer("row_sin", sin)
        self.register_buffer("col_cos", cos)
        self.register_buffer("col_sin", sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension (needed for RoPE)."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_embedding(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to a single half-dimension tensor."""
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D-RoPE to queries and keys.

        Args:
            q: Query tensor of shape (B, num_heads, L, head_dim)
            k: Key tensor of shape (B, num_heads, L, head_dim)
            row_ids: Integer tensor of shape (L,) giving the row index of each
                visual token in the 2D grid.
            col_ids: Integer tensor of shape (L,) giving the column index.

        Returns:
            Tuple of rotated query and key (same shapes).
        """
        # Split head dimension into two halves.
        q1, q2 = q[..., : self.half_head_dim], q[..., self.half_head_dim :]
        k1, k2 = k[..., : self.half_head_dim], k[..., self.half_head_dim :]

        # Fetch cos/sin for each token's row/col.
        # The precomputed tables are [1, max_grid_size, half_dim // 2].
        # We index using row_ids (L,) -> [B, 1, L, half_dim//2] after expansion.
        row_idx = row_ids.view(1, 1, -1, 1).expand(
            -1, -1, -1, self.half_head_dim // 2
        )
        col_idx = col_ids.view(1, 1, -1, 1).expand(
            -1, -1, -1, self.half_head_dim // 2
        )

        row_cos = self.row_cos.gather(2, row_idx)  # [1, 1, L, half_dim//2]
        row_sin = self.row_sin.gather(2, row_idx)
        col_cos = self.col_cos.gather(2, col_idx)
        col_sin = self.col_sin.gather(2, col_idx)

        # Apply row-based RoPE to the first half, col-based to the second half.
        q1_r = self._apply_rotary_embedding(q1, row_cos, row_sin)
        q2_r = self._apply_rotary_embedding(q2, col_cos, col_sin)
        k1_r = self._apply_rotary_embedding(k1, row_cos, row_sin)
        k2_r = self._apply_rotary_embedding(k2, col_cos, col_sin)

        q_rotated = torch.cat([q1_r, q2_r], dim=-1)
        k_rotated = torch.cat([k1_r, k2_r], dim=-1)

        return q_rotated, k_rotated


# ---------------------------------------------------------------------------
# Transformer Block used by the Visual Encoder
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    A single Transformer block with bidirectional self-attention and SwiGLU FFN,
    using 2D-RoPE for position encoding.

    Args:
        hidden_dim:  model width (e.g. 1472).
        num_heads:   number of attention heads.
        mlp_width:   intermediate size of the SwiGLU FFN.
        rope:        instance of ``RoPE2D`` to apply to queries and keys.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_width: int,
        rope: RoPE2D,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Layer norms
        self.attn_norm = RMSNorm(hidden_dim)
        self.ffn_norm = RMSNorm(hidden_dim)

        # Attention projections (no bias as typical for decoder-only LLMs)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # RoPE module (shared across blocks)
        self.rope = rope

        # SwiGLU FFN
        self.gate_proj = nn.Linear(hidden_dim, mlp_width, bias=False)
        self.up_proj = nn.Linear(hidden_dim, mlp_width, bias=False)
        self.down_proj = nn.Linear(mlp_width, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for a single visual encoder block.

        Args:
            x:       Input tensor of shape (B, L, hidden_dim).
            row_ids: 1D tensor of row indices (L,).
            col_ids: 1D tensor of column indices (L,).

        Returns:
            Output tensor after attention and FFN with residuals.
        """
        B, L, C = x.shape

        # --- Self-Attention (bidirectional) ---
        residual = x
        x_norm = self.attn_norm(x)

        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply 2D-RoPE
        q, k = self.rope(q, k, row_ids, col_ids)

        # Scaled dot-product attention (no causal mask)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=False
        )

        # Reassemble heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, C)
        attn_out = self.o_proj(attn_out)

        x = residual + attn_out

        # --- SwiGLU Feed-Forward Network ---
        residual = x
        x_norm = self.ffn_norm(x)

        gate = F.silu(self.gate_proj(x_norm))
        up = self.up_proj(x_norm)
        down = self.down_proj(gate * up)

        x = residual + down

        return x


# ---------------------------------------------------------------------------
# Patch Embedding (convolutional projection)
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Converts a raw image into a grid of patch embeddings.

    Equivalent to a ``Conv2d`` with kernel and stride equal to ``patch_size``
    and output channels equal to the model width.

    Args:
        patch_size: size of each square patch (default 16).
        in_channels: number of input channels (3 for RGB).
        embed_dim: output dimension (model width).
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 1472,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: image tensor of shape (B, 3, H, W).

        Returns:
            tensor of shape (B, embed_dim, H//patch_size, W//patch_size).
        """
        return self.proj(x)


# ---------------------------------------------------------------------------
# Full Visual Encoder
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """
    NaViL visual encoder: a stack of bidirectional Transformer blocks with
    2D‑RoPE, preceded by a patch embedding layer.

    The constructor is driven by the values from ``config.yaml`` (depth, width,
    patch_size, num_attention_heads, mlp_width).  All parameters have sensible
    defaults for the 2B variant.

    Args:
        depth:                   Number of Transformer blocks.
        width:                   Hidden dimension (model width).
        patch_size:              Patch size for the embedding convolution.
        mlp_width:               FFN intermediate dimension.  If None, defaults
                                 to 4 * width.
        num_heads:               Number of attention heads.  If None, defaults
                                 to width // 64 (common for ViT).
        rope_max_grid_size:      Maximum grid dimension for pre‑computed RoPE
                                 tables.  Choose large enough to cover all
                                 possible resolutions (e.g. 1024).
    """

    def __init__(
        self,
        depth: int = 24,              # from config: model.visual_encoder.depth
        width: int = 1472,            # model.visual_encoder.width
        patch_size: int = 16,         # model.visual_encoder.patch_size
        mlp_width: Optional[int] = None,  # model.visual_encoder.mlp_width
        num_heads: Optional[int] = None,   # model.visual_encoder.num_attention_heads
        rope_max_grid_size: int = 1024,
    ) -> None:
        super().__init__()
        # ---- Architectural hyperparameters ----
        self.width = width
        self.depth = depth
        self.patch_size = patch_size

        # Resolve optional parameters
        if mlp_width is None:
            mlp_width = 4 * width   # typical ViT expansion
        if num_heads is None:
            # Choose a divisor that makes head dim an integer; common is 64.
            head_dim_candidate = 64
            while width % head_dim_candidate != 0 and head_dim_candidate > 0:
                head_dim_candidate -= 1
            if width % head_dim_candidate != 0:
                raise ValueError(
                    f"width {width} is not divisible by any plausible head dim; "
                    "please supply num_heads explicitly."
                )
            num_heads = width // head_dim_candidate

        if width % num_heads != 0:
            raise ValueError(
                f"width ({width}) must be divisible by num_heads ({num_heads})."
            )

        head_dim = width // num_heads

        # ---- Sub‑modules ----
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=3,
            embed_dim=width,
        )

        # Shared 2D‑RoPE module (used by all blocks)
        self.rope = RoPE2D(
            head_dim=head_dim,
            max_grid_size=rope_max_grid_size,
        )

        # Transformer blocks (bidirectional)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=width,
                    num_heads=num_heads,
                    mlp_width=mlp_width,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the visual encoder.

        Args:
            x: A batch of images of shape ``(B, 3, H, W)`` where ``H`` and
               ``W`` are multiples of 32 (padding handled by preprocessing).

        Returns:
            A tensor of shape ``(B, L, width)`` containing the visual token
            sequence (``L = (H // patch_size) * (W // patch_size)``).
        """
        B, C, H, W = x.shape

        # Patch embedding
        patch_feats = self.patch_embed(x)  # (B, width, h_patches, w_patches)
        h_patches, w_patches = patch_feats.shape[2], patch_feats.shape[3]
        patch_tokens = patch_feats.flatten(2).transpose(1, 2)  # (B, L, width)

        # Compute grid positions (row, col) for each patch token.
        device = patch_tokens.device
        # row_ids and col_ids are 1D tensors of length L
        row_ids = (
            torch.arange(h_patches, device=device)
            .unsqueeze(1)
            .repeat(1, w_patches)
            .flatten()
        )
        col_ids = (
            torch.arange(w_patches, device=device)
            .unsqueeze(0)
            .repeat(h_patches, 1)
            .flatten()
        )

        # Process through each Transformer block
        x = patch_tokens
        for block in self.blocks:
            x = block(x, row_ids, col_ids)

        return x
