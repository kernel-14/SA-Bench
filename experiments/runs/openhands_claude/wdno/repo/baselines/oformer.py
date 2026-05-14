"""
Operator Transformer (OFormer) baseline.

Reference: Li et al. (2023), "Transformer for Partial Differential Equations' Operator Learning"

Hyperparameters from Table 30 (1D) and Table 35 (2D):
  1D:
    - Encoder: SpatialEncoder2D, input_channels=2, token_dim=39, encoded_dim=256, heads=4, depth=6
    - Decoder: PointWiseDecoder2DSimple, out_channels=1, scale=0.5, res=120
    - Training: batch=32, iterations=500000, lr=1e-4

  2D:
    - Encoder: SpatialTemporalEncoder2D, input_channels=3, token_dim=49, encoded_dim=192, heads=1, depth=5
    - Decoder: PointWiseDecoder2D, out_channels=1, token_dim=96, propagate_forward=True
    - Training: batch=8, iterations=1000000, lr=1e-4
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        if context is not None:
            kv = self.to_qkv(context).chunk(3, dim=-1)
            _, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), kv)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.norm1(x), self.norm2(context))
        x = x + self.mlp(self.norm3(x))
        return x


class OFormer1D(nn.Module):
    """
    OFormer for 1D PDE data.

    Hyperparameters (Table 30):
      Encoder: input_channels=2, token_dim=39, encoded_dim=256, heads=4, depth=6
      Decoder: out_channels=1, scale=0.5, res=120
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        token_dim: int = 39,
        encoded_dim: int = 256,
        heads: int = 4,
        depth: int = 6,
        dropout: float = 0.05,
    ):
        super().__init__()
        # Encoder: embed spatial tokens
        self.input_proj = nn.Linear(in_channels, token_dim)
        self.pos_emb = nn.Linear(2, token_dim)  # (t, x) coordinates
        self.encoder = nn.Sequential(*[
            TransformerBlock(token_dim, heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.encoder_proj = nn.Linear(token_dim, encoded_dim)

        # Decoder: cross-attention to query points
        self.query_proj = nn.Linear(2, encoded_dim)  # query (t, x) coordinates
        self.decoder = nn.ModuleList([
            CrossAttentionBlock(encoded_dim, heads)
            for _ in range(2)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(encoded_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_channels),
        )

    def forward(self, x: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input [B, C_in, T, X] (condition: u0 + f)
            coords: query coordinates [B, T_out*X_out, 2] (optional)
        Returns:
            output [B, C_out, T, X]
        """
        B, C, T, X = x.shape

        # Flatten spatial-temporal tokens
        x_flat = rearrange(x, "b c t x -> b (t x) c")  # [B, T*X, C]

        # Create coordinate grid
        t_grid = torch.linspace(0, 1, T, device=x.device)
        x_grid = torch.linspace(0, 1, X, device=x.device)
        T_g, X_g = torch.meshgrid(t_grid, x_grid, indexing="ij")
        coords_grid = torch.stack([T_g.flatten(), X_g.flatten()], dim=-1)  # [T*X, 2]
        coords_grid = coords_grid.unsqueeze(0).expand(B, -1, -1)  # [B, T*X, 2]

        # Encode
        tokens = self.input_proj(x_flat) + self.pos_emb(coords_grid)
        tokens = self.encoder(tokens)
        tokens = self.encoder_proj(tokens)  # [B, T*X, encoded_dim]

        # Decode at query points (same as input for simulation)
        query = self.query_proj(coords_grid)  # [B, T*X, encoded_dim]
        for block in self.decoder:
            query = block(query, tokens)

        out = self.output_proj(query)  # [B, T*X, C_out]
        return rearrange(out, "b (t x) c -> b c t x", t=T, x=X)


class OFormer2D(nn.Module):
    """
    OFormer for 2D PDE data.

    Hyperparameters (Table 35):
      Encoder: input_channels=3, token_dim=49, encoded_dim=192, heads=1, depth=5
      Decoder: out_channels=1, token_dim=96
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        token_dim: int = 49,
        encoded_dim: int = 192,
        heads: int = 1,
        depth: int = 5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, token_dim)
        self.pos_emb = nn.Linear(3, token_dim)  # (t, h, w) coordinates
        self.encoder = nn.Sequential(*[
            TransformerBlock(token_dim, heads)
            for _ in range(depth)
        ])
        self.encoder_proj = nn.Linear(token_dim, encoded_dim)

        decoder_dim = encoded_dim // 2
        self.query_proj = nn.Linear(3, decoder_dim)
        self.decoder = nn.ModuleList([
            CrossAttentionBlock(decoder_dim, heads)
            for _ in range(2)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(decoder_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, T, H, W] → [B, C_out, T, H, W]"""
        B, C, T, H, W = x.shape

        # Subsample for efficiency (process patches)
        x_flat = rearrange(x, "b c t h w -> b (t h w) c")

        t_g = torch.linspace(0, 1, T, device=x.device)
        h_g = torch.linspace(0, 1, H, device=x.device)
        w_g = torch.linspace(0, 1, W, device=x.device)
        T_g, H_g, W_g = torch.meshgrid(t_g, h_g, w_g, indexing="ij")
        coords = torch.stack([T_g.flatten(), H_g.flatten(), W_g.flatten()], dim=-1)
        coords = coords.unsqueeze(0).expand(B, -1, -1)

        tokens = self.input_proj(x_flat) + self.pos_emb(coords)
        tokens = self.encoder(tokens)
        tokens = self.encoder_proj(tokens)

        query = self.query_proj(coords[:, :, :3])
        for block in self.decoder:
            query = block(query, tokens)

        out = self.output_proj(query)
        return rearrange(out, "b (t h w) c -> b c t h w", t=T, h=H, w=W)
