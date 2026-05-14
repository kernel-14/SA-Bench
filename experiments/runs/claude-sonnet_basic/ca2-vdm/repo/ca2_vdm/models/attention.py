"""
Attention modules for Ca2-VDM.

Implements:
  - CausalTemporalAttention: Causal (lower-triangular masked) temporal attention
    that enables KV-cache reuse across autoregression steps.
  - PrefixEnhancedSpatialAttention: Spatial attention that concatenates a
    sub-prefix of clean frames to enhance guidance from conditional frames.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def get_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build a lower-triangular causal mask of shape (seq_len, seq_len).
    M[i, j] = -inf if i < j else 0.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # upper triangle = -inf, rest = 0
    return mask


class CausalTemporalAttention(nn.Module):
    """
    Causal temporal attention (Section 3.2 of Ca2-VDM).

    Each frame can only attend to its preceding frames (causal mask).
    During autoregressive inference, the KV-cache of clean prefix frames
    is precomputed and reused across all denoising timesteps.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, H, L, D/H)"""
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, H, L, D/H)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, L, D/H) -> (B, L, D)"""
        B, H, L, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, L, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for causal temporal attention.

        During training:
          x has shape (B*H*W, L, C) where L = P + l (prefix + denoising target).
          A causal mask is applied so each frame only attends to preceding frames.

        During inference (denoising stage):
          x has shape (B*H*W, l, C) — only the denoising target frames.
          kv_cache contains (K_0^{0:P_k}, V_0^{0:P_k}) from clean prefix frames.
          The cache is concatenated with the current noisy KVs before attention.

        Args:
            x: Input tensor of shape (B_spatial, L_query, C).
            kv_cache: Optional tuple (K_cache, V_cache) of shape
                      (B_spatial, L_cache, C) each, representing clean prefix KVs.
            return_kv: If True, also return the (K, V) of the current input
                       (used during cache writing stage).

        Returns:
            out: Attention output of shape (B_spatial, L_query, C).
            kv: Optional tuple (K, V) if return_kv=True.
        """
        B, L_q, C = x.shape
        device = x.device

        Q = self.q_proj(x)  # (B, L_q, C)
        K = self.k_proj(x)  # (B, L_q, C)
        V = self.v_proj(x)  # (B, L_q, C)

        # Concatenate cached KVs from clean prefix
        if kv_cache is not None:
            K_cache, V_cache = kv_cache  # (B, L_cache, C)
            K_full = torch.cat([K_cache, K], dim=1)  # (B, L_cache + L_q, C)
            V_full = torch.cat([V_cache, V], dim=1)
        else:
            K_full = K
            V_full = V

        L_kv = K_full.shape[1]

        # Split into heads
        Q_h = self._split_heads(Q)       # (B, H, L_q, Dh)
        K_h = self._split_heads(K_full)  # (B, H, L_kv, Dh)
        V_h = self._split_heads(V_full)  # (B, H, L_kv, Dh)

        # Attention scores
        attn = torch.matmul(Q_h, K_h.transpose(-2, -1)) * self.scale  # (B, H, L_q, L_kv)

        # Apply causal mask
        # During training: full causal mask over L_kv = L_q (no cache)
        # During inference: queries are the denoising target (L_q = l),
        #   keys/values include cache (L_kv = L_cache + l).
        #   The cache frames are all "past" so no masking needed for them;
        #   we only need causal masking within the L_q query frames.
        if kv_cache is not None:
            # Cache frames are all in the past — only mask within query frames
            L_cache = K_cache.shape[1]
            # Build mask: queries attend freely to cache, causally to query frames
            mask = torch.zeros(L_q, L_kv, device=device)
            # Upper triangle of the query-to-query block
            query_block_mask = torch.full((L_q, L_q), float("-inf"), device=device)
            query_block_mask = torch.triu(query_block_mask, diagonal=1)
            mask[:, L_cache:] = query_block_mask
        else:
            # Full causal mask
            mask = get_causal_mask(L_q, device)  # (L_q, L_q)

        attn = attn + mask.unsqueeze(0).unsqueeze(0)  # broadcast over B, H
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V_h)  # (B, H, L_q, Dh)
        out = self._merge_heads(out)   # (B, L_q, C)
        out = self.out_proj(out)

        if return_kv:
            return out, (K, V)
        return out, None


class PrefixEnhancedSpatialAttention(nn.Module):
    """
    Prefix-enhanced spatial attention (Section 3.2 of Ca2-VDM, Eq. 4).

    For each frame in the denoising target, the key and value are enhanced
    by concatenating a sub-prefix of P' clean frames along the spatial dimension.
    This provides stronger guidance from the conditional frames.

    During training:
      - For frames i >= P (denoising target): K(i) = W_K [h_0^{P-P'}, ..., h_0^{P-1}, h_t^i]
      - For frames i < P (clean prefix): K(i) = W_K [h_0^i, ..., h_0^i] (self-repeat P' times)

    During inference (denoising stage):
      - The spatial KV-cache stores the clean spatial KVs of the most recent chunk.
      - These are concatenated to the denoising target's spatial features.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        prefix_len: P', number of prefix frames to concatenate (default 3).
        dropout: Dropout probability.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        prefix_len: int = 3,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.prefix_len = prefix_len
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, H, N, D/H)"""
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, N, D/H) -> (B, N, D)"""
        B, H, N, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        prefix_frames: Optional[torch.Tensor] = None,
        return_kv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for prefix-enhanced spatial attention.

        Args:
            x: Input of shape (L, B, H*W, C) or (L, HW, C) depending on usage.
               Here we expect (L, HW, C) where L is the number of frames,
               HW is the spatial dimension (treated as batch), C is channels.
               Actually we process frame-by-frame: x is (B_frames, HW, C).
            prefix_frames: Clean prefix frames of shape (P', HW, C) for enhancement.
                           If None, self-attention only (used for prefix frames themselves).
            return_kv: If True, return (K, V) for caching.

        Returns:
            out: Shape (B_frames, HW, C).
            kv: Optional (K, V) each of shape (B_frames, HW*(1+P'), C) if return_kv.
        """
        B, HW, C = x.shape

        Q = self.q_proj(x)  # (B, HW, C)

        if prefix_frames is not None:
            # prefix_frames: (P', HW, C)
            # Concatenate prefix spatially: [prefix_frames, x] along HW dim
            # For each frame in x, K and V are computed from [prefix, x_i]
            # prefix_frames is broadcast to match batch B
            P_prime = prefix_frames.shape[0]
            # Expand prefix to batch: (P', HW, C) -> (B, P'*HW, C)
            prefix_expanded = prefix_frames.unsqueeze(0).expand(B, -1, -1, -1)
            prefix_expanded = prefix_expanded.reshape(B, P_prime * HW, C)
            # Concatenate: (B, (P'+1)*HW, C)
            kv_input = torch.cat([prefix_expanded, x], dim=1)
        else:
            # Self-repeat for clean prefix frames (self-attention with self-repeat)
            P_prime = self.prefix_len
            x_repeated = x.unsqueeze(1).expand(-1, P_prime, -1, -1)
            x_repeated = x_repeated.reshape(B, P_prime * HW, C)
            kv_input = torch.cat([x_repeated, x], dim=1)

        K = self.k_proj(kv_input)  # (B, (P'+1)*HW, C)
        V = self.v_proj(kv_input)

        # Split heads
        Q_h = self._split_heads(Q)  # (B, H, HW, Dh)
        K_h = self._split_heads(K)  # (B, H, (P'+1)*HW, Dh)
        V_h = self._split_heads(V)

        # Attention: (B, H, HW, (P'+1)*HW)
        attn = torch.matmul(Q_h, K_h.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V_h)  # (B, H, HW, Dh)
        out = self._merge_heads(out)   # (B, HW, C)
        out = self.out_proj(out)

        if return_kv:
            return out, (K, V)
        return out, None
