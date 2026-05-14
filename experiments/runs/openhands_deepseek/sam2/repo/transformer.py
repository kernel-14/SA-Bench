"""Transformer utilities for SAM 2: attention, RoPE, two-way blocks, MLP."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """MLP with GELU activation."""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RoPE2D(nn.Module):
    """2D Rotary Positional Embedding (Su et al., 2021; Heo et al., 2024).

    Used in memory attention self-attention and cross-attention layers.
    Object pointer tokens are excluded from RoPE since they have no spatial correspondence.
    """
    def __init__(self, head_dim: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.max_h = max_h
        self.max_w = max_w
        self._init_rope()

    def _init_rope(self):
        half_dim = self.head_dim // 2
        freqs = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))

        # Separate frequency bands for H and W
        self.register_buffer("freqs_h", freqs[:self.head_dim // 4])
        self.register_buffer("freqs_w", freqs[:self.head_dim // 4])

    def compute_rope_embeddings(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings for RoPE."""
        pos_h = torch.arange(h, device=device).float()
        pos_w = torch.arange(w, device=device).float()

        # H dimension
        theta_h = torch.outer(pos_h, self.freqs_h.to(device))
        cos_h = torch.cos(theta_h)
        sin_h = torch.sin(theta_h)

        # W dimension
        theta_w = torch.outer(pos_w, self.freqs_w.to(device))
        cos_w = torch.cos(theta_w)
        sin_w = torch.sin(theta_w)

        cos = torch.cat([cos_h.unsqueeze(1).expand(-1, w, -1), cos_w.unsqueeze(0).expand(h, -1, -1)], dim=-1)
        sin = torch.cat([sin_h.unsqueeze(1).expand(-1, w, -1), sin_w.unsqueeze(0).expand(h, -1, -1)], dim=-1)

        cos = cos.reshape(h * w, -1)
        sin = sin.reshape(h * w, -1)
        return cos, sin

    def rotate_queries_or_keys(self, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to queries or keys."""
        half_dim = self.head_dim // 2
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]

        t_rot = t.float()
        t_rot_reshape = t_rot.reshape(-1, t_rot.shape[-2], t_rot.shape[-1])

        t1 = t_rot_reshape[..., :half_dim]
        t2 = t_rot_reshape[..., half_dim:]
        rotated = torch.cat([t1 * cos.unsqueeze(0) - t2 * sin.unsqueeze(0),
                            t1 * sin.unsqueeze(0) + t2 * cos.unsqueeze(0)], dim=-1)
        return rotated.to(t.dtype)


class MultiheadAttention(nn.Module):
    """Standard multi-head attention with optional RoPE."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 use_rope: bool = False, is_cross_attn: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.is_cross_attn = is_cross_attn

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rope = RoPE2D(self.head_dim)

    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        return x.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                rope_h: int = 0, rope_w: int = 0,
                exclude_rope_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Nq, _ = query.shape
        B, Nk, _ = key.shape

        q = self._reshape_for_attention(self.q_proj(query))
        k = self._reshape_for_attention(self.k_proj(key))
        v = self._reshape_for_attention(self.v_proj(value))

        if self.use_rope and rope_h > 0 and rope_w > 0:
            cos, sin = self.rope.compute_rope_embeddings(rope_h, rope_w, q.device)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

            q_rope = self.rope.rotate_queries_or_keys(
                q.permute(0, 2, 1, 3).reshape(B, Nq, -1), cos, sin)
            q_rope = q_rope.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            k_rope = self.rope.rotate_queries_or_keys(
                k.permute(0, 2, 1, 3).reshape(B, Nk, -1), cos, sin)
            k_rope = k_rope.reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            if exclude_rope_indices is not None:
                q = torch.where(exclude_rope_indices, q, q_rope)
                k = torch.where(exclude_rope_indices, k, k_rope)
            else:
                q, k = q_rope, k_rope

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, Nq, self.d_model)
        out = self.out_proj(out)
        return out


class TwoWayTransformerBlock(nn.Module):
    """Two-way transformer block from SAM: cross-attends between prompt tokens and image tokens.

    Each block performs:
    1. Self-attention on prompt tokens
    2. Cross-attention from prompt to image
    3. MLP on prompt tokens
    4. Cross-attention from image to prompt
    5. MLP on image tokens
    """
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_attn_token_to_image = MultiheadAttention(d_model, num_heads, dropout, is_cross_attn=True)
        self.mlp_token = Mlp(d_model, int(d_model * mlp_ratio), drop=dropout)

        self.cross_attn_image_to_token = MultiheadAttention(d_model, num_heads, dropout, is_cross_attn=True)
        self.mlp_image = Mlp(d_model, int(d_model * mlp_ratio), drop=dropout)

        self.norm1_token = nn.LayerNorm(d_model)
        self.norm2_token = nn.LayerNorm(d_model)
        self.norm3_token = nn.LayerNorm(d_model)
        self.norm1_image = nn.LayerNorm(d_model)
        self.norm2_image = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, image_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention on query tokens
        q = queries + self.self_attn(self.norm1_token(queries), self.norm1_token(queries), self.norm1_token(queries))
        # Cross-attention: tokens attend to image
        q = q + self.cross_attn_token_to_image(self.norm2_token(q), image_tokens, image_tokens)
        q = q + self.mlp_token(self.norm3_token(q))

        # Cross-attention: image attends to tokens
        img = image_tokens + self.cross_attn_image_to_token(self.norm1_image(image_tokens), q, q)
        img = img + self.mlp_image(self.norm2_image(img))

        return q, img


class TwoWayTransformer(nn.Module):
    """Stack of two-way transformer blocks."""
    def __init__(self, depth: int, d_model: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayTransformerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.final_token_norm = nn.LayerNorm(d_model)
        self.final_image_norm = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, image_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            queries, image_tokens = layer(queries, image_tokens)
        queries = self.final_token_norm(queries)
        image_tokens = self.final_image_norm(image_tokens)
        return queries, image_tokens


class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention and cross-attention to memory."""
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, use_rope: bool = True,
                 use_cross_attn: bool = True):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout, use_rope=use_rope)
        self.norm1 = nn.LayerNorm(d_model)

        if use_cross_attn:
            self.cross_attn = MultiheadAttention(d_model, num_heads, dropout, use_rope=use_rope, is_cross_attn=True)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.cross_attn = None
            self.norm2 = None

        self.mlp = Mlp(d_model, int(d_model * mlp_ratio), drop=dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None,
                rope_h: int = 0, rope_w: int = 0) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                               rope_h=rope_h, rope_w=rope_w)
        if self.cross_attn is not None and memory is not None:
            x = x + self.cross_attn(self.norm2(x), memory, memory,
                                    rope_h=rope_h, rope_w=rope_w)
        x = x + self.mlp(self.norm3(x))
        return x


def get_sinusoidal_pos_embed(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal absolute positional embedding."""
    position = torch.arange(seq_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
