"""Fundamental building blocks for NaViL.

Layers: RMSNorm, 1D/2D Rotary Position Embedding, MultiHeadAttention,
SwiGLU FeedForward, PatchEmbedding.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(dtype)


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex-valued rotary frequency embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_2d(
    dim: int,
    max_height: int,
    max_width: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute 2D rotary frequency embeddings for visual encoder.

    Splits head_dim equally between height and width axes.
    """
    half_dim = dim // 2
    h_dim = half_dim // 2
    w_dim = half_dim - h_dim

    freqs_h = 1.0 / (theta ** (torch.arange(0, h_dim, 2).float() / h_dim))
    freqs_w = 1.0 / (theta ** (torch.arange(0, w_dim, 2).float() / w_dim))

    h_pos = torch.arange(max_height)
    w_pos = torch.arange(max_width)

    freqs_h = torch.outer(h_pos, freqs_h)
    freqs_w = torch.outer(w_pos, freqs_w)

    freqs_cis_h = torch.polar(torch.ones_like(freqs_h), freqs_h)
    freqs_cis_w = torch.polar(torch.ones_like(freqs_w), freqs_w)

    return freqs_cis_h, freqs_cis_w


def apply_rotary_emb_1d(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply 1D rotary embeddings to queries or keys.

    Args:
        x: [batch, seq_len, n_heads, head_dim]
        freqs_cis: [seq_len, head_dim] complex tensor
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: x.shape[1]].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(3).to(x.dtype)


def apply_rotary_emb_2d(
    x: torch.Tensor,
    freqs_cis_h: torch.Tensor,
    freqs_cis_w: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Apply 2D rotary embeddings for visual encoder.

    Args:
        x: [batch, seq_len, n_heads, head_dim]
        freqs_cis_h: [max_h, h_dim] complex
        freqs_cis_w: [max_w, w_dim] complex
        height, width: spatial dimensions of the feature map
    """
    batch, seq_len, n_heads, head_dim = x.shape
    half = head_dim // 2
    h_dim = half // 2
    w_dim = half - h_dim

    x_h = x[..., :h_dim]
    x_w = x[..., h_dim : h_dim + w_dim]
    x_rest = x[..., h_dim + w_dim :]

    x_h = x_h.reshape(batch, height, width, n_heads, h_dim)
    x_w = x_w.reshape(batch, height, width, n_heads, w_dim)

    x_h_complex = torch.view_as_complex(x_h.float().reshape(*x_h.shape[:-1], -1, 2))
    x_w_complex = torch.view_as_complex(x_w.float().reshape(*x_w.shape[:-1], -1, 2))

    freqs_cis_h = freqs_cis_h[:height].unsqueeze(1).unsqueeze(0).unsqueeze(0)
    freqs_cis_w = freqs_cis_w[:width].unsqueeze(0).unsqueeze(0).unsqueeze(0)

    x_h_rot = x_h_complex * freqs_cis_h
    x_w_rot = x_w_complex * freqs_cis_w

    x_h_out = torch.view_as_real(x_h_rot).flatten(-2).reshape(batch, seq_len, n_heads, h_dim)
    x_w_out = torch.view_as_real(x_w_rot).flatten(-2).reshape(batch, seq_len, n_heads, w_dim)

    return torch.cat([x_h_out, x_w_out, x_rest], dim=-1).to(x.dtype)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_type: str = "1d",
    ):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rope = use_rope
        self.rope_type = rope_type

        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.W_O = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        freqs_cis_h: Optional[torch.Tensor] = None,
        freqs_cis_w: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.W_K(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.W_V(x).view(batch, seq_len, self.n_heads, self.head_dim)

        if self.use_rope and freqs_cis is not None:
            if self.rope_type == "1d":
                q = apply_rotary_emb_1d(q, freqs_cis)
                k = apply_rotary_emb_1d(k, freqs_cis)
            elif self.rope_type == "2d":
                assert height is not None and width is not None
                q = apply_rotary_emb_2d(q, freqs_cis_h, freqs_cis_w, height, width)
                k = apply_rotary_emb_2d(k, freqs_cis_h, freqs_cis_w, height, width)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.W_O(out)


class ModalityMultiHeadAttention(nn.Module):
    """Modality-specific multi-head attention for MoE.

    Two sets of Q/K/V/O projection weights: one for visual tokens, one for text tokens.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_type: str = "1d",
    ):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rope = use_rope
        self.rope_type = rope_type

        self.W_Q_vis = nn.Linear(dim, dim, bias=False)
        self.W_K_vis = nn.Linear(dim, dim, bias=False)
        self.W_V_vis = nn.Linear(dim, dim, bias=False)
        self.W_O_vis = nn.Linear(dim, dim, bias=False)

        self.W_Q_txt = nn.Linear(dim, dim, bias=False)
        self.W_K_txt = nn.Linear(dim, dim, bias=False)
        self.W_V_txt = nn.Linear(dim, dim, bias=False)
        self.W_O_txt = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        freqs_cis_h: Optional[torch.Tensor] = None,
        freqs_cis_w: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with modality-specific projections.

        Args:
            x: [batch, seq_len, dim]
            modality_mask: [batch, seq_len] bool tensor, True = visual, False = text
        """
        batch, seq_len, _ = x.shape
        device = x.device

        mask_vis = modality_mask.unsqueeze(-1)
        mask_txt = (~modality_mask).unsqueeze(-1)

        q_vis = self.W_Q_vis(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k_vis = self.W_K_vis(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v_vis = self.W_V_vis(x).view(batch, seq_len, self.n_heads, self.head_dim)

        q_txt = self.W_Q_txt(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k_txt = self.W_K_txt(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v_txt = self.W_V_txt(x).view(batch, seq_len, self.n_heads, self.head_dim)

        q = q_vis * mask_vis.unsqueeze(2) + q_txt * mask_txt.unsqueeze(2)
        k = k_vis * mask_vis.unsqueeze(2) + k_txt * mask_txt.unsqueeze(2)
        v = v_vis * mask_vis.unsqueeze(2) + v_txt * mask_txt.unsqueeze(2)

        if self.use_rope and freqs_cis is not None:
            if self.rope_type == "1d":
                q = apply_rotary_emb_1d(q, freqs_cis)
                k = apply_rotary_emb_1d(k, freqs_cis)
            elif self.rope_type == "2d":
                assert height is not None and width is not None
                q = apply_rotary_emb_2d(q, freqs_cis_h, freqs_cis_w, height, width)
                k = apply_rotary_emb_2d(k, freqs_cis_h, freqs_cis_w, height, width)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)

        out_vis = self.W_O_vis(out)
        out_txt = self.W_O_txt(out)

        return out_vis * mask_vis + out_txt * mask_txt


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network with SiLU activation."""

    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.W_gate = nn.Linear(dim, mlp_dim, bias=False)
        self.W_up = nn.Linear(dim, mlp_dim, bias=False)
        self.W_down = nn.Linear(mlp_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.W_gate(x))
        up = self.W_up(x)
        return self.dropout(self.W_down(gate * up))


class ModalitySwiGLUFFN(nn.Module):
    """Modality-specific SwiGLU FFN for MoE.

    Two sets of gate/up/down weights: visual and text.
    """

    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.W_gate_vis = nn.Linear(dim, mlp_dim, bias=False)
        self.W_up_vis = nn.Linear(dim, mlp_dim, bias=False)
        self.W_down_vis = nn.Linear(mlp_dim, dim, bias=False)

        self.W_gate_txt = nn.Linear(dim, mlp_dim, bias=False)
        self.W_up_txt = nn.Linear(dim, mlp_dim, bias=False)
        self.W_down_txt = nn.Linear(mlp_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        mask_vis = modality_mask.unsqueeze(-1)
        mask_txt = (~modality_mask).unsqueeze(-1)

        gate_vis = F.silu(self.W_gate_vis(x))
        gate_txt = F.silu(self.W_gate_txt(x))
        gate = gate_vis * mask_vis + gate_txt * mask_txt

        up_vis = self.W_up_vis(x)
        up_txt = self.W_up_txt(x)
        up = up_vis * mask_vis + up_txt * mask_txt

        hidden = gate * up

        down_vis = self.W_down_vis(hidden)
        down_txt = self.W_down_txt(hidden)
        return self.dropout(down_vis * mask_vis + down_txt * mask_txt)


class PatchEmbedding(nn.Module):
    """Patch embedding layer for images.

    Converts an image to patch embeddings with stride (patch_size).
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 1472):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Convert image to patch tokens.

        Args:
            x: [batch, channels, height, width]

        Returns:
            tokens: [batch, num_patches, embed_dim]
            h, w: number of patches in height and width
        """
        x = self.proj(x)
        batch, embed_dim, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


def create_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create causal attention mask."""
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).to(dtype)


def create_bidirectional_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create bidirectional (no masking) attention mask."""
    return torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
