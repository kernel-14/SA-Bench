"""Attention mechanisms for MM-DiT.

Implements:
- Full sequence attention (bidirectional)
- Blockwise causal attention for autoregressive video generation
- Joint text-image/video attention (MM-DiT style)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


def build_causal_mask(
    num_frames: int,
    tokens_per_frame: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build blockwise causal attention mask.

    Each frame can attend to all tokens in previous frames and itself,
    but not to future frames. This implements the blockwise causal attention
    described in the paper for autoregressive video generation.

    Returns:
        mask: (total_tokens, total_tokens) boolean mask where True = masked (not attended)
    """
    total = num_frames * tokens_per_frame
    mask = torch.ones(total, total, device=device, dtype=torch.bool)

    for i in range(num_frames):
        for j in range(i + 1):
            # Frame i can attend to frame j (j <= i)
            row_start = i * tokens_per_frame
            row_end = (i + 1) * tokens_per_frame
            col_start = j * tokens_per_frame
            col_end = (j + 1) * tokens_per_frame
            mask[row_start:row_end, col_start:col_end] = False

    return mask


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional QK normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if provided (for temporal dimension)
        if rope_freqs is not None:
            q, k = apply_rope(q, k, rope_freqs)

        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if FLASH_ATTN_AVAILABLE and mask is None and x.dtype in (torch.float16, torch.bfloat16):
            # Use flash attention for efficiency
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()
            out = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=0.0, causal=False)
            out = out.reshape(B, N, C)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, N, C)

        return self.proj(out)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
) -> tuple:
    """Apply rotary position embeddings to q and k."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    cos = freqs[..., 0].unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)
    sin = freqs[..., 1].unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class JointAttention(nn.Module):
    """Joint attention over text and visual tokens (MM-DiT style).

    Text and visual tokens are concatenated and processed jointly,
    allowing bidirectional information flow between modalities.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Visual stream projections
        self.vis_qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.vis_proj = nn.Linear(dim, dim)

        # Text stream projections
        self.txt_qkv = nn.Linear(context_dim, 3 * dim, bias=True)
        self.txt_proj = nn.Linear(dim, context_dim)

        if qk_norm:
            self.vis_q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
            self.vis_k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
            self.txt_q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
            self.txt_k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        else:
            self.vis_q_norm = self.vis_k_norm = nn.Identity()
            self.txt_q_norm = self.txt_k_norm = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        vis_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        vis_mask: Optional[torch.Tensor] = None,
        rope_freqs: Optional[torch.Tensor] = None,
    ) -> tuple:
        B, Nv, _ = vis_tokens.shape
        B, Nt, _ = txt_tokens.shape

        # Project visual tokens
        vis_qkv = self.vis_qkv(vis_tokens).reshape(B, Nv, 3, self.num_heads, self.head_dim)
        vis_q, vis_k, vis_v = vis_qkv.unbind(2)
        vis_q = self.vis_q_norm(vis_q)
        vis_k = self.vis_k_norm(vis_k)

        # Apply RoPE to visual tokens
        if rope_freqs is not None:
            vis_q, vis_k = apply_rope(vis_q, vis_k, rope_freqs)

        # Project text tokens
        txt_qkv = self.txt_qkv(txt_tokens).reshape(B, Nt, 3, self.num_heads, self.head_dim)
        txt_q, txt_k, txt_v = txt_qkv.unbind(2)
        txt_q = self.txt_q_norm(txt_q)
        txt_k = self.txt_k_norm(txt_k)

        # Concatenate text and visual for joint attention
        q = torch.cat([txt_q, vis_q], dim=1).transpose(1, 2)  # (B, H, Nt+Nv, D)
        k = torch.cat([txt_k, vis_k], dim=1).transpose(1, 2)
        v = torch.cat([txt_v, vis_v], dim=1).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask to visual portion only
        if vis_mask is not None:
            # Extend mask: text tokens can attend to everything, visual tokens follow causal mask
            full_mask = torch.zeros(
                B, 1, Nt + Nv, Nt + Nv,
                device=vis_tokens.device,
                dtype=torch.bool,
            )
            full_mask[:, :, Nt:, Nt:] = vis_mask.unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(full_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2)  # (B, Nt+Nv, H*D)
        out = out.reshape(B, Nt + Nv, -1)

        txt_out = self.txt_proj(out[:, :Nt])
        vis_out = self.vis_proj(out[:, Nt:])

        return vis_out, txt_out
