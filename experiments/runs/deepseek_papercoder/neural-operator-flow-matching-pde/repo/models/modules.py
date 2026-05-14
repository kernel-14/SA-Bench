## models/modules.py
"""
Core building blocks for the P2VAE and FMT architectures.

This module provides:
- RMSNorm: Root Mean Square Layer Normalisation
- SwiGLUFeedForward: Swish-Gated Linear Unit feed‑forward network
- MultiHeadSelfAttention: FlashAttention‑backed multi‑head self‑attention
- CrossAttnPool: Compresses a token sequence into a single vector via cross‑attention
- AdaLNZero: Zero‑initialised adaptive layer norm modulation for SiT conditioning
- TransformerBlock: A single transformer block for the SiT backbone, combining
  multi‑head self‑attention, SwiGLU MLP, RMSNorm and AdaLN‑Zero modulation.

All classes are implemented from scratch using PyTorch.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Optional FlashAttention import
# -----------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    HAVE_FLASH = True
except ImportError:
    _flash_attn_func = None
    HAVE_FLASH = False

# -----------------------------------------------------------------------------
# Helper classes
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (used in Llama‑2)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise along the last dimension."""
        dtype = torch.float32  # compute in float32
        x_float = x.to(dtype)
        rms = torch.rsqrt(torch.mean(x_float.pow(2), dim=-1, keepdim=True) + self.eps)
        x_norm = x_float * rms
        return (x_norm * self.weight).to(x.dtype)


class SwiGLUFeedForward(nn.Module):
    """Feed‑forward network with SwiGLU activation (Llama‑2 style)."""

    def __init__(self, dim: int, hidden_multiplier: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * hidden_multiplier)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: w3(SiLU(w1(x)) * w2(x))."""
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        return self.dropout(self.w3(gate * up))


class MultiHeadSelfAttention(nn.Module):
    """Multi‑head self‑attention with optional FlashAttention v2."""

    def __init__(
        self,
        dim: int,
        heads: int,
        use_flash_attn: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        self.use_flash = use_flash_attn and HAVE_FLASH

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, L, dim)
        Returns:
            output of same shape.
        """
        B, L, _ = x.shape

        # Compute Q, K, V and reshape for multi‑head attention
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash:
            # FlashAttention expects (B, L, heads, head_dim)
            q = q.transpose(1, 2)  # (B, L, heads, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = _flash_attn_func(q, k, v, causal=False)
            out = out.transpose(1, 2)  # back to (B, heads, L, head_dim)
        else:
            # PyTorch SDPA with optional flash backend
            out = F.scaled_dot_product_attention(
                q, k, v,
                scale=self.scale,
                enable_flash=True,
            )

        # Merge heads -> (B, L, dim)
        out = out.transpose(1, 2).contiguous().reshape(B, L, self.dim)
        return self.proj(out)


# -----------------------------------------------------------------------------
# Core building blocks for the FMT and VAE
# -----------------------------------------------------------------------------

class CrossAttnPool(nn.Module):
    """
    Compresses a sequence of tokens into a single embedding via cross-attention.

    A learnable query attends over all spatial tokens, and the attended result
    is projected to the desired output dimension. Used inside DiffusionForcing
    to feed the GRU with a summary of the current latent frame.

    Args:
        token_dim: Feature dimension of input tokens.
        num_heads: Number of attention heads.
        output_dim: Dimension of the pooled output vector (default = token_dim).
        query_dim: Internal dimension for query/key (default = token_dim).
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int = 8,
        output_dim: Optional[int] = None,
        query_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        if query_dim is None:
            query_dim = token_dim
        if output_dim is None:
            output_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Learnable query token
        self.query = nn.Parameter(torch.randn(1, 1, query_dim) * 0.02)

        # Projections for keys and values
        self.key_proj = nn.Linear(token_dim, query_dim, bias=False)
        self.value_proj = nn.Linear(token_dim, token_dim, bias=False)

        # Output projection (optional if dims differ)
        if output_dim != token_dim:
            self.out_proj = nn.Linear(token_dim, output_dim, bias=False)
        else:
            self.out_proj = nn.Identity()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: shape (B, L, token_dim)
        Returns:
            pooled: shape (B, output_dim)
        """
        B, L, _ = tokens.shape

        # Prepare query, key, value
        q = self.query.expand(B, -1, -1)          # (B, 1, query_dim)
        k = self.key_proj(tokens)                # (B, L, query_dim)
        v = self.value_proj(tokens)              # (B, L, token_dim)

        # Reshape for multi‑head attention: (B, heads, seq, head_dim)
        q = q.reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Use scaled dot‑product attention (with flash if available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            enable_flash=True,
        )
        # attn_out: (B, heads, 1, head_dim) -> (B, 1, token_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, 1, self.token_dim)
        out = self.out_proj(attn_out.squeeze(1))  # (B, output_dim)
        return out


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalisation with Zero‑initialisation (AdaLN‑Zero).

    Outputs scale, shift, and gate tensors used to modulate a transformer sub‑layer.
    The final linear layer is initialised to zero so that the modulation has no
    effect at the beginning of training.

    Args:
        dim: Feature dimension of the modulated sub‑layer.
        cond_dim: Dimension of the conditioning vector.
    """

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        hidden_dim = dim * 4  # standard SiT multiplier
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim * 3, bias=True),
        )

        # Zero‑initialise the final linear layer (weight and bias)
        nn.init.constant_(self.mlp[-1].weight, 0.0)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

        # Initialise the first layer normally
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[0].bias, 0.0)

    def get_ada_params(
        self, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            c: Conditioning tensor of shape (..., cond_dim).
        Returns:
            scale, shift, gate: each of shape (..., dim).
        """
        out = self.mlp(c)  # (..., 3*dim)
        scale, shift, gate = torch.chunk(out, 3, dim=-1)
        return scale, shift, gate


class TransformerBlock(nn.Module):
    """
    A single transformer block for the SiT backbone.

    It combines:
    - RMSNorm + MultiHeadSelfAttention + AdaLN‑Zero modulation
    - RMSNorm + SwiGLUFeedForward + AdaLN‑Zero modulation

    The conditioning vector `c` can be of shape (B, L, cond_dim) for per‑token
    modulation, or (B, cond_dim) which will be broadcast to all tokens.
    The block expects per‑token conditions to handle multi‑frame and multi‑scale
    sequences in the FMT temporal pyramid.

    Args:
        dim: Embedding dimension.
        heads: Number of attention heads.
        cond_dim: Dimension of the conditioning vector.
        use_flash_attn: Whether to use FlashAttention‑2 (if available).
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        cond_dim: int,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads

        # AdaLN-Zero for attention and feed‑forward sub‑layers
        self.ada_ln_attn = AdaLNZero(dim, cond_dim)
        self.ada_ln_ff = AdaLNZero(dim, cond_dim)

        # Layer norms
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        # Attention and feed‑forward
        self.attn = MultiHeadSelfAttention(dim, heads, use_flash_attn=use_flash_attn)
        self.ff = SwiGLUFeedForward(dim, hidden_multiplier=4, dropout=0.0)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Token sequence of shape (B, L, dim).
            c: Conditioning vector. Shape can be:
                - (B, cond_dim): same condition for all tokens (broadcast).
                - (B, L, cond_dim): per‑token condition (used in FMT).
        Returns:
            Updated token sequence of shape (B, L, dim).
        """
        B, L, _ = x.shape

        # Ensure c has per‑token shape (B, L, cond_dim)
        if c.dim() == 2:
            c = c.unsqueeze(1).expand(-1, L, -1)

        # Flatten to (B*L, cond_dim) for the AdaLN MLP, then reshape results
        c_flat = c.reshape(B * L, -1)

        # ---------- Attention sub‑layer ----------
        scale_a, shift_a, gate_a = self.ada_ln_attn.get_ada_params(c_flat)
        scale_a = scale_a.reshape(B, L, self.dim)
        shift_a = shift_a.reshape(B, L, self.dim)
        gate_a  = gate_a.reshape(B, L, self.dim)

        x_norm = self.norm1(x)
        x_mod = x_norm * (1.0 + scale_a) + shift_a
        attn_out = self.attn(x_mod)
        x = x + gate_a * attn_out

        # ---------- Feed‑forward sub‑layer ----------
        scale_f, shift_f, gate_f = self.ada_ln_ff.get_ada_params(c_flat)
        scale_f = scale_f.reshape(B, L, self.dim)
        shift_f = shift_f.reshape(B, L, self.dim)
        gate_f  = gate_f.reshape(B, L, self.dim)

        x_norm = self.norm2(x)
        x_mod = x_norm * (1.0 + scale_f) + shift_f
        ff_out = self.ff(x_mod)
        x = x + gate_f * ff_out

        return x
