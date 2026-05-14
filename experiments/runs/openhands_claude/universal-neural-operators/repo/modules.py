from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    CodomainAttention,
    CrossAttentionBlock,
    FNOBlock1d,
    FNOBlock2d,
    LocalAttentionBlock,
    MambaBlock,
    SelfAttentionBlock,
    SpectralConv1d,
    SpectralConv2d,
    SwinBlock,
)


# ---------------------------------------------------------------------------
# Adapter layers: lifting (L) and projection (P)
# These are the problem-specific adapters described in Section 3.
# During fine-tuning only these layers are trained; the backbone is frozen.
# ---------------------------------------------------------------------------

class LiftingLayer(nn.Module):
    """
    Point-wise lifting: maps n_in input functions → hidden_dim.
    L(a) = σ(A_L · a + b_L)
    """

    def __init__(self, n_in: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_in, *spatial)
        Returns:
            (B, hidden_dim, *spatial)
        """
        spatial = x.shape[2:]
        # Permute channels to last for Linear
        x = x.permute(0, *range(2, 2 + len(spatial)), 1)  # (B, *spatial, n_in)
        x = self.net(x)
        x = x.permute(0, -1, *range(1, 1 + len(spatial)))  # (B, hidden_dim, *spatial)
        return x


class ProjectionLayer(nn.Module):
    """
    Point-wise projection: maps hidden_dim → n_out output functions.
    P(v) = A_P · v + b_P
    """

    def __init__(self, hidden_dim: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, hidden_dim, *spatial)
        Returns:
            (B, n_out, *spatial)
        """
        spatial = x.shape[2:]
        x = x.permute(0, *range(2, 2 + len(spatial)), 1)
        x = self.net(x)
        x = x.permute(0, -1, *range(1, 1 + len(spatial)))
        return x


# ---------------------------------------------------------------------------
# FNO backbone: stack of FNO integral-operator blocks
# θ_F = {A_t, b_t, θ_{k,t} : t = 1, ..., n_layers}
# ---------------------------------------------------------------------------

class FNOBackbone1d(nn.Module):
    """Stack of 1-D FNO blocks forming the shared backbone."""

    def __init__(self, hidden_dim: int, n_layers: int, modes: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            FNOBlock1d(hidden_dim, modes) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class FNOBackbone2d(nn.Module):
    """Stack of 2-D FNO blocks forming the shared backbone."""

    def __init__(self, hidden_dim: int, n_layers: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            FNOBlock2d(hidden_dim, modes1, modes2) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Post-lifting Mamba backbone (MambaFNO)
# Inserts M_φ between lifting and FNO blocks (Section 3, eq. 2)
# ---------------------------------------------------------------------------

class MambaFNOBackbone1d(nn.Module):
    """Mamba SSM module followed by FNO blocks (1-D)."""

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        modes: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.mamba = MambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(hidden_dim, modes) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mamba(x)
        for block in self.fno_blocks:
            x = block(x)
        return x


class MambaFNOBackbone2d(nn.Module):
    """Mamba SSM module followed by FNO blocks (2-D)."""

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        modes1: int,
        modes2: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.mamba = MambaBlock(hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(hidden_dim, modes1, modes2) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mamba(x)
        for block in self.fno_blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Post-lifting local attention backbone (LocalAttnFNO)
# ---------------------------------------------------------------------------

class LocalAttnFNOBackbone1d(nn.Module):
    """Local attention module followed by FNO blocks (1-D)."""

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        modes: int,
        num_heads: int = 4,
        window_size: int = 16,
    ) -> None:
        super().__init__()
        self.local_attn = LocalAttentionBlock(hidden_dim, num_heads=num_heads, window_size=window_size)
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(hidden_dim, modes) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_attn(x)
        for block in self.fno_blocks:
            x = block(x)
        return x


class LocalAttnFNOBackbone2d(nn.Module):
    """Local attention module followed by FNO blocks (2-D)."""

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        modes1: int,
        modes2: int,
        num_heads: int = 4,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.local_attn = LocalAttentionBlock(hidden_dim, num_heads=num_heads, window_size=window_size)
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(hidden_dim, modes1, modes2) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_attn(x)
        for block in self.fno_blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Perceiver IO backbone
# Architecture: cross-attn(input→latent) → self-attn(latent) → cross-attn(latent→output)
# Keys/values for encoder cross-attn come from FNO-mapped inputs (Section 3)
# ---------------------------------------------------------------------------

class PerceiverIOBackbone(nn.Module):
    """
    Perceiver IO-based neural operator backbone.

    Encoder: latent queries attend to FNO-processed input (K=FNO_K(X), V=FNO_V(X), Q=L)
    Processor: self-attention on latent
    Decoder: input queries attend to latent (Q=X_proj, K/V from latent)
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        n_latents: int,
        n_self_attn_layers: int,
        modes1: int,
        modes2: int,
        num_heads: int = 4,
        spatial_dim: int = 2,
    ) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.latent_dim = latent_dim

        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(1, n_latents, latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)

        # FNO mappings for keys and values in encoder cross-attention
        if spatial_dim == 1:
            self.fno_k = SpectralConv1d(hidden_dim, latent_dim, modes1)
            self.fno_v = SpectralConv1d(hidden_dim, latent_dim, modes1)
        else:
            self.fno_k = SpectralConv2d(hidden_dim, latent_dim, modes1, modes2)
            self.fno_v = SpectralConv2d(hidden_dim, latent_dim, modes1, modes2)

        # Encoder: cross-attention from input to latent
        self.encoder_cross_attn = CrossAttentionBlock(latent_dim, latent_dim, num_heads=num_heads)

        # Processor: self-attention on latent
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionBlock(latent_dim, num_heads=num_heads)
            for _ in range(n_self_attn_layers)
        ])

        # Decoder: cross-attention from latent to output positions
        # Q from input (projected), K/V from latent
        self.input_to_query = nn.Linear(hidden_dim, latent_dim)
        self.decoder_cross_attn = CrossAttentionBlock(latent_dim, latent_dim, num_heads=num_heads)
        self.out_proj = nn.Linear(latent_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, hidden_dim, *spatial)
        Returns:
            (B, hidden_dim, *spatial)
        """
        B, C, *spatial = x.shape
        L = 1
        for s in spatial:
            L *= s

        # FNO-based keys and values: (B, latent_dim, *spatial) → (B, L, latent_dim)
        K = self.fno_k(x).view(B, self.latent_dim, L).permute(0, 2, 1)
        V = self.fno_v(x).view(B, self.latent_dim, L).permute(0, 2, 1)

        # Latent queries: (B, n_latents, latent_dim)
        latents = self.latents.expand(B, -1, -1)

        # Encoder cross-attention: Q=latents, K/V from FNO(input)
        latents = self.encoder_cross_attn(latents, K)

        # Processor self-attention
        for layer in self.self_attn_layers:
            latents = layer(latents)

        # Decoder: Q from input positions, K/V from latent
        x_flat = x.view(B, C, L).permute(0, 2, 1)  # (B, L, C)
        queries = self.input_to_query(x_flat)  # (B, L, latent_dim)
        out = self.decoder_cross_attn(queries, latents)  # (B, L, latent_dim)
        out = self.out_proj(out)  # (B, L, hidden_dim)

        return out.permute(0, 2, 1).view(B, C, *spatial)


# ---------------------------------------------------------------------------
# CoDA-NO backbone (codomain attention + FNO blocks)
# ---------------------------------------------------------------------------

class CodaNOBackbone(nn.Module):
    """
    CoDA-NO backbone: alternating codomain attention and FNO blocks.
    Codomain attention computes similarity between features (channels),
    not between spatial samples.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        modes1: int,
        modes2: int,
        num_heads: int = 4,
        spatial_dim: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "coda_attn": CodomainAttention(
                    hidden_dim, modes1, modes2, num_heads=num_heads, spatial_dim=spatial_dim
                ),
                "fno": FNOBlock2d(hidden_dim, modes1, modes2) if spatial_dim == 2
                       else FNOBlock1d(hidden_dim, modes1),
            }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer["coda_attn"](x)
            x = layer["fno"](x)
        return x


# ---------------------------------------------------------------------------
# Swin-v2 backbone
# Hierarchical vision transformer with shifted windows
# ---------------------------------------------------------------------------

class SwinBackbone(nn.Module):
    """
    Swin-v2 style backbone for 2-D spatial fields.
    Alternates W-MSA and SW-MSA blocks.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        window_size: int = 8,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(hidden_dim, window_size, num_heads, shift=(i % 2 == 1))
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, hidden_dim, H, W)
        Returns:
            (B, hidden_dim, H, W)
        """
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).view(B, H * W, C)
        for block in self.blocks:
            tokens = block(tokens, H, W)
        tokens = self.norm(tokens)
        return tokens.view(B, H, W, C).permute(0, 3, 1, 2)
