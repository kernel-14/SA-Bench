"""
Building blocks for the SongUNet (NCSN++) architecture from EDM (Karras et al., 2022).
"""
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for noise levels."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half_dim, dtype=torch.float32, device=t.device)
            / half_dim
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def fourier_embedding(inputs: torch.Tensor, num_freqs: int, max_freq: float = 1.0) -> torch.Tensor:
    """Fourier embedding for noise levels."""
    freq = torch.linspace(0, max_freq, num_freqs, device=inputs.device)
    angle = inputs[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
    return emb


class LinearBlock(nn.Module):
    """Linear layer with optional activation."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "silu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class Conv2d(nn.Module):
    """Weight-normalized 2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="linear")
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample layer with a conv."""

    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsample layer with a conv."""

    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            x = self.conv(x)
        else:
            x = self.pool(x)
        return x


class GroupNorm(nn.Module):
    """Group normalization with optional SiLU."""

    def __init__(self, num_groups: int, num_channels: int, apply_act: bool = True):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=min(num_groups, num_channels),
            num_channels=num_channels,
            eps=1e-6,
        )
        self.apply_act = apply_act
        if apply_act:
            self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gn(x)
        if self.apply_act:
            x = self.act(x)
        return x


class SelfAttention(nn.Module):
    """Multi-head self-attention for 2D feature maps."""

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        self.qkv = Conv2d(channels, channels * 3, kernel_size=1, padding=0, bias=False)
        self.proj = Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(2, 3).reshape(B, C, H, W)
        out = self.proj(out)
        return out


class ResBlock(nn.Module):
    """Residual block with optional attention, used in SongUNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_groups: int = 32,
        use_attention: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        self.norm1 = GroupNorm(num_groups, in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.embed_proj = nn.Linear(embed_dim, out_channels)

        if in_channels != out_channels:
            self.skip = Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.skip = nn.Identity()

        if use_attention:
            self.attn = SelfAttention(out_channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.conv1(h)

        emb_out = self.embed_proj(emb)[:, :, None, None]
        h = h + emb_out

        h = self.norm2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        h = h + self.skip(x)

        if self.use_attention:
            h = h + self.attn(h)

        return h


def get_timestep_embedding(
    t: torch.Tensor,
    embed_dim: int,
    embedding_type: str = "positional",
    max_period: int = 10000,
) -> torch.Tensor:
    """Get noise level embedding."""
    if embedding_type == "positional":
        emb = PositionalEmbedding(embed_dim, max_period=max_period)
        return emb(t)
    elif embedding_type == "fourier":
        return fourier_embedding(t, embed_dim // 2)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
