from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from layers import (
    AdaLayerNorm,
    CausalTemporalAttention,
    CrossAttention,
    FeedForward,
    MultiHeadAttention,
    PrefixEnhancedSpatialAttention,
    TimestepEmbedding,
)


# ---------------------------------------------------------------------------
# Causal Generation Block (Figure 3(c) in paper)
# ---------------------------------------------------------------------------

class CausalGenerationBlock(nn.Module):
    """
    One transformer block for Ca2-VDM with:
      1. Causal temporal attention
      2. Prefix-enhanced spatial attention
      3. (Optional) Visual-text cross attention
      4. Feed-forward network

    All sub-layers use adaLN conditioning on the timestep embedding.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: Optional[int],
        cond_dim: int,
        prefix_len: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # Causal temporal attention
        self.norm_temp = nn.LayerNorm(dim, eps=1e-6)
        self.temporal_attn = CausalTemporalAttention(dim, num_heads, dropout)
        self.adaLN_temp = AdaLayerNorm(dim, cond_dim)

        # Prefix-enhanced spatial attention
        self.norm_spatial = nn.LayerNorm(dim, eps=1e-6)
        self.spatial_attn = PrefixEnhancedSpatialAttention(dim, num_heads, prefix_len, dropout)
        self.adaLN_spatial = AdaLayerNorm(dim, cond_dim)

        # Visual-text cross attention (optional)
        self.use_cross_attn = context_dim is not None
        if self.use_cross_attn:
            self.norm_cross = nn.LayerNorm(dim, eps=1e-6)
            self.cross_attn = CrossAttention(dim, context_dim, num_heads, dropout)
            self.adaLN_cross = AdaLayerNorm(dim, cond_dim)

        # Feed-forward
        self.norm_ff = nn.LayerNorm(dim, eps=1e-6)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)
        self.adaLN_ff = AdaLayerNorm(dim, cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        prefix_frames: int,
        context: Optional[torch.Tensor] = None,
        temporal_cached_k: Optional[torch.Tensor] = None,
        temporal_cached_v: Optional[torch.Tensor] = None,
        spatial_cached_k: Optional[torch.Tensor] = None,
        spatial_cached_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: (B, L, HW, dim) — input hidden states
            cond: (B, cond_dim) — timestep conditioning
            prefix_frames: P — number of clean prefix frames
            context: (B, S, context_dim) — text context (optional)
            temporal_cached_k: (B*HW, P_k, dim) — temporal KV-cache keys
            temporal_cached_v: (B*HW, P_k, dim) — temporal KV-cache values
            spatial_cached_k: (B, P', HW, dim) — spatial KV-cache keys
            spatial_cached_v: (B, P', HW, dim) — spatial KV-cache values
        Returns:
            x: (B, L, HW, dim) — updated hidden states
            new_temp_k: (B*HW, L, dim) — new temporal keys (for cache writing)
            new_temp_v: (B*HW, L, dim) — new temporal values
            new_spatial_k: (B, l, HW, dim) — new spatial keys (for cache writing)
            new_spatial_v: (B, l, HW, dim) — new spatial values
        """
        B, L, HW, C = x.shape

        # ---- Causal Temporal Attention ----
        # Permute: treat HW as batch dimension, L as sequence
        x_temp = rearrange(x, "b l hw c -> (b hw) l c")
        # Apply adaLN: expand cond from (B, cond_dim) to (B*HW, cond_dim)
        cond_expanded = cond.unsqueeze(1).expand(-1, HW, -1).reshape(B * HW, -1)
        x_temp_norm = self.adaLN_temp(self.norm_temp(x_temp), cond_expanded)

        temp_out, new_temp_k, new_temp_v = self.temporal_attn(
            x_temp_norm, temporal_cached_k, temporal_cached_v
        )
        x_temp = x_temp + temp_out
        x = rearrange(x_temp, "(b hw) l c -> b l hw c", b=B, hw=HW)

        # ---- Prefix-Enhanced Spatial Attention ----
        x_spatial_norm = self._apply_adaLN_spatial(x, cond, L, HW, B)

        spatial_out, new_spatial_k, new_spatial_v = self.spatial_attn(
            x_spatial_norm, prefix_frames, spatial_cached_k, spatial_cached_v
        )
        x = x + spatial_out

        # ---- Visual-Text Cross Attention ----
        if self.use_cross_attn and context is not None:
            x_flat = rearrange(x, "b l hw c -> (b l) hw c")
            x_flat_norm = self.norm_cross(x_flat)
            ctx = context.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, -1, context.shape[-1])
            cross_out = self.cross_attn(x_flat_norm, ctx)
            x_flat = x_flat + cross_out
            x = rearrange(x_flat, "(b l) hw c -> b l hw c", b=B, l=L)

        # ---- Feed-Forward ----
        x_flat = rearrange(x, "b l hw c -> (b l) hw c")
        x_ff_norm = self.norm_ff(x_flat)
        ff_out = self.ff(x_ff_norm)
        x_flat = x_flat + ff_out
        x = rearrange(x_flat, "(b l) hw c -> b l hw c", b=B, l=L)

        return x, new_temp_k, new_temp_v, new_spatial_k, new_spatial_v

    def _apply_adaLN_spatial(
        self, x: torch.Tensor, cond: torch.Tensor, L: int, HW: int, B: int
    ) -> torch.Tensor:
        """Apply adaLN with cond broadcast over L and HW dims."""
        # cond: (B, cond_dim) -> shift/scale: (B, dim) -> broadcast to (B, L, HW, dim)
        shift, scale = self.adaLN_spatial.proj(cond).chunk(2, dim=-1)  # (B, dim) each
        shift = shift[:, None, None, :]  # (B, 1, 1, dim)
        scale = scale[:, None, None, :]
        return self.adaLN_spatial.norm(x) * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Bidirectional Generation Block (for OS-Fix and OS-Ext baselines)
# ---------------------------------------------------------------------------

class BidirectionalGenerationBlock(nn.Module):
    """
    Transformer block with bidirectional temporal attention (baseline).
    Used for OS-Fix and OS-Ext baselines.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: Optional[int],
        cond_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Bidirectional temporal attention
        self.norm_temp = nn.LayerNorm(dim, eps=1e-6)
        self.temporal_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.adaLN_temp = AdaLayerNorm(dim, cond_dim)

        # Spatial self-attention
        self.norm_spatial = nn.LayerNorm(dim, eps=1e-6)
        self.spatial_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.adaLN_spatial = AdaLayerNorm(dim, cond_dim)

        # Visual-text cross attention (optional)
        self.use_cross_attn = context_dim is not None
        if self.use_cross_attn:
            self.norm_cross = nn.LayerNorm(dim, eps=1e-6)
            self.cross_attn = CrossAttention(dim, context_dim, num_heads, dropout)

        # Feed-forward
        self.norm_ff = nn.LayerNorm(dim, eps=1e-6)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, HW, dim)
            cond: (B, cond_dim)
            context: (B, S, context_dim)
        Returns:
            x: (B, L, HW, dim)
        """
        B, L, HW, C = x.shape

        # Temporal attention (bidirectional)
        x_temp = rearrange(x, "b l hw c -> (b hw) l c")
        cond_exp = cond.unsqueeze(1).expand(-1, HW, -1).reshape(B * HW, -1)
        x_temp_norm = self.adaLN_temp(self.norm_temp(x_temp), cond_exp)
        x_temp = x_temp + self.temporal_attn(x_temp_norm, x_temp_norm, x_temp_norm)
        x = rearrange(x_temp, "(b hw) l c -> b l hw c", b=B, hw=HW)

        # Spatial attention
        x_flat = rearrange(x, "b l hw c -> (b l) hw c")
        cond_exp2 = cond.unsqueeze(1).expand(-1, L, -1).reshape(B * L, -1)
        x_flat_norm = self.adaLN_spatial(self.norm_spatial(x_flat), cond_exp2)
        x_flat = x_flat + self.spatial_attn(x_flat_norm, x_flat_norm, x_flat_norm)
        x = rearrange(x_flat, "(b l) hw c -> b l hw c", b=B, l=L)

        # Cross attention
        if self.use_cross_attn and context is not None:
            x_flat = rearrange(x, "b l hw c -> (b l) hw c")
            ctx = context.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, -1, context.shape[-1])
            x_flat = x_flat + self.cross_attn(self.norm_cross(x_flat), ctx)
            x = rearrange(x_flat, "(b l) hw c -> b l hw c", b=B, l=L)

        # Feed-forward
        x_flat = rearrange(x, "b l hw c -> (b l) hw c")
        x_flat = x_flat + self.ff(self.norm_ff(x_flat))
        x = rearrange(x_flat, "(b l) hw c -> b l hw c", b=B, l=L)

        return x


# ---------------------------------------------------------------------------
# KV-Cache Queue for Temporal Attention
# ---------------------------------------------------------------------------

class TemporalKVCacheQueue:
    """
    Queue structure for temporal KV-cache (Section 3.3 in paper).

    Stores clean temporal keys and values for each layer.
    When P_k reaches P_max, the oldest chunk is dequeued.
    """

    def __init__(self, num_layers: int, max_frames: int):
        self.num_layers = num_layers
        self.max_frames = max_frames
        # Each entry: list of (K, V) tensors per layer
        self.cache_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self.cache_v: List[Optional[torch.Tensor]] = [None] * num_layers

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.cache_k[layer_idx], self.cache_v[layer_idx]

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """
        Append new chunk KV to cache and dequeue oldest if exceeds max_frames.

        Args:
            layer_idx: which transformer layer
            new_k: (B*HW, l, dim) — new chunk keys
            new_v: (B*HW, l, dim) — new chunk values
        """
        if self.cache_k[layer_idx] is None:
            self.cache_k[layer_idx] = new_k
            self.cache_v[layer_idx] = new_v
        else:
            self.cache_k[layer_idx] = torch.cat([self.cache_k[layer_idx], new_k], dim=1)
            self.cache_v[layer_idx] = torch.cat([self.cache_v[layer_idx], new_v], dim=1)

        # Dequeue oldest frames if exceeds max_frames
        cur_len = self.cache_k[layer_idx].shape[1]
        if cur_len > self.max_frames:
            excess = cur_len - self.max_frames
            self.cache_k[layer_idx] = self.cache_k[layer_idx][:, excess:]
            self.cache_v[layer_idx] = self.cache_v[layer_idx][:, excess:]

    def reset(self) -> None:
        self.cache_k = [None] * self.num_layers
        self.cache_v = [None] * self.num_layers

    @property
    def current_length(self) -> int:
        if self.cache_k[0] is None:
            return 0
        return self.cache_k[0].shape[1]


class SpatialKVCache:
    """
    Spatial KV-cache for prefix-enhanced spatial attention (Section 3.3).

    Only stores the most recent chunk's spatial KV (overwritten each AR step).
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.cache_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self.cache_v: List[Optional[torch.Tensor]] = [None] * num_layers

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.cache_k[layer_idx], self.cache_v[layer_idx]

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Overwrite spatial cache with most recent chunk."""
        self.cache_k[layer_idx] = new_k
        self.cache_v[layer_idx] = new_v

    def reset(self) -> None:
        self.cache_k = [None] * self.num_layers
        self.cache_v = [None] * self.num_layers
