from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    """Two-layer MLP with configurable activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), activation()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        if sigmoid_output:
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head attention with optional 2d-RoPE."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert embedding_dim % num_heads == 0

        internal_dim = embedding_dim // downsample_rate
        self.q_proj = nn.Linear(embedding_dim, internal_dim)
        self.k_proj = nn.Linear(embedding_dim, internal_dim)
        self.v_proj = nn.Linear(embedding_dim, internal_dim)
        self.out_proj = nn.Linear(internal_dim, embedding_dim)
        self.internal_dim = internal_dim
        self.dropout = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        B, N, C = x.shape
        x = x.reshape(B, N, num_heads, C // num_heads)
        return x.transpose(1, 2)  # (B, num_heads, N, head_dim)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        B, H, N, D = x.shape
        return x.transpose(1, 2).reshape(B, N, H * D)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        scale = math.sqrt(q.shape[-1])
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)
        out = self._recombine_heads(out)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# 2D Rotary Positional Embedding (RoPE)
# Used in memory attention self- and cross-attention layers.
# Reference: Su et al. (2021) RoFormer; Heo et al. (2024) 2d-RoPE for ViT.
# ---------------------------------------------------------------------------

def build_2d_sincos_position_embedding(
    h: int, w: int, embed_dim: int, temperature: float = 10000.0
) -> Tensor:
    """Build 2D sin-cos positional embedding of shape (h*w, embed_dim)."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos"
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    dim_t = torch.arange(embed_dim // 4, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / (embed_dim // 4))

    pos_x = grid_x.flatten()[:, None] / dim_t[None, :]
    pos_y = grid_y.flatten()[:, None] / dim_t[None, :]

    pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)
    pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)
    return torch.cat([pos_x, pos_y], dim=1)  # (h*w, embed_dim)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding to x. x: (..., seq_len, head_dim)."""
    return x * cos + rotate_half(x) * sin


class RoPE2D(nn.Module):
    """2D Rotary Positional Embedding for spatial feature maps."""

    def __init__(self, head_dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.temperature = temperature

    def _build_freqs(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        half = self.head_dim // 2
        dim_t = torch.arange(half // 2, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * dim_t / half)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij",
        )
        pos_x = grid_x.flatten()[:, None] / dim_t[None, :]  # (h*w, half//2)
        pos_y = grid_y.flatten()[:, None] / dim_t[None, :]

        emb_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)  # (h*w, half)
        emb_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)
        emb = torch.cat([emb_x, emb_y], dim=-1)  # (h*w, head_dim)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin

    def forward(self, q: Tensor, k: Tensor, h: int, w: int) -> Tuple[Tensor, Tensor]:
        """
        q, k: (B, num_heads, h*w, head_dim)
        Returns rotated q and k.
        """
        cos, sin = self._build_freqs(h, w, q.device, q.dtype)
        cos = cos[None, None, :, :]  # (1, 1, h*w, head_dim)
        sin = sin[None, None, :, :]
        return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (absolute, for memory attention)
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    """Sinusoidal 2D positional embedding for spatial feature maps."""

    def __init__(self, num_pos_feats: int = 64, temperature: float = 10000.0, normalize: bool = True) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) → returns (B, num_pos_feats*2, H, W)."""
        B, _, H, W = x.shape
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """Learned 2D positional embedding."""

    def __init__(self, num_pos_feats: int = 128) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape
        i = torch.arange(W, device=x.device)
        j = torch.arange(H, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),
            y_emb.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos


# ---------------------------------------------------------------------------
# Drop path (stochastic depth)
# ---------------------------------------------------------------------------

def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x / keep_prob * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Window partition / unpartition helpers (for windowed attention in Hiera)
# ---------------------------------------------------------------------------

def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    """
    Partition feature map into non-overlapping windows.
    x: (B, H, W, C)
    Returns: (B*num_windows, window_size, window_size, C), (H, W)
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> Tensor:
    """
    Reverse window_partition.
    windows: (B*num_windows, window_size, window_size, C)
    Returns: (B, H, W, C)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x
