"""Basic neural network layers used throughout the model."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class AdaLayerNorm(nn.Module):
    """Adaptive layer norm conditioned on timestep and class embeddings."""

    def __init__(self, dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(condition_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.linear(F.silu(condition)).chunk(6, dim=-1)
        )
        x = self.norm(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero(nn.Module):
    """AdaLN-Zero variant used in DiT."""

    def __init__(self, dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(condition_dim, 2 * dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.linear(F.silu(condition)).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinusoidalPositionEmbedding(nn.Module):
    """2D sinusoidal position embedding for spatial dimensions."""

    def __init__(self, dim: int, max_resolution: int = 256):
        super().__init__()
        self.dim = dim
        self.max_resolution = max_resolution

    def forward(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        assert self.dim % 4 == 0, "dim must be divisible by 4 for 2D sinusoidal PE"
        half_dim = self.dim // 4

        # Frequency bands
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Height and width position indices
        y_pos = torch.arange(height, device=device).float()
        x_pos = torch.arange(width, device=device).float()

        # Outer product
        y_emb = y_pos.unsqueeze(1) * emb.unsqueeze(0)  # (H, half_dim)
        x_emb = x_pos.unsqueeze(1) * emb.unsqueeze(0)  # (W, half_dim)

        # Sin/cos
        y_emb = torch.cat([y_emb.sin(), y_emb.cos()], dim=-1)  # (H, dim//2)
        x_emb = torch.cat([x_emb.sin(), x_emb.cos()], dim=-1)  # (W, dim//2)

        # Combine: broadcast over H and W
        y_emb = y_emb.unsqueeze(1).expand(-1, width, -1)   # (H, W, dim//2)
        x_emb = x_emb.unsqueeze(0).expand(height, -1, -1)  # (H, W, dim//2)

        pos_emb = torch.cat([y_emb, x_emb], dim=-1)  # (H, W, dim)
        return pos_emb.reshape(height * width, self.dim)


class RotaryEmbedding1D(nn.Module):
    """1D Rotary Position Embedding for temporal dimension."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._build_cache(max_seq_len, dim)

    def _build_cache(self, seq_len: int, dim: int):
        theta = 1.0 / (
            self.base ** (torch.arange(0, dim, 2).float() / dim)
        )
        positions = torch.arange(seq_len).float()
        freqs = torch.outer(positions, theta)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by MLP projection."""

    def __init__(self, dim: int, out_dim: Optional[int] = None, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        out_dim = out_dim or dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _sinusoidal_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=timesteps.device).float()
            / half
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self._sinusoidal_embedding(timesteps)
        return self.mlp(emb)


class CausalConv3d(nn.Module):
    """3D causal convolution: each frame only attends to preceding frames."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: int = 0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        kt, kh, kw = kernel_size
        # Temporal causal padding: pad (kt-1) on the left, 0 on the right
        self.temporal_pad = kt - 1
        self.spatial_pad = (kh // 2, kw // 2)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        # Causal temporal padding
        x = F.pad(x, (
            self.spatial_pad[1], self.spatial_pad[1],
            self.spatial_pad[0], self.spatial_pad[0],
            self.temporal_pad, 0,
        ))
        return self.conv(x)


class ResBlock3D(nn.Module):
    """3D residual block with causal convolutions."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = CausalConv3d(channels, channels, (3, 3, 3))
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = CausalConv3d(channels, channels, (3, 3, 3))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + residual


class Downsample3D(nn.Module):
    """Spatial and temporal downsampling."""

    def __init__(self, channels: int, spatial_factor: int = 2, temporal_factor: int = 2):
        super().__init__()
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        stride = (temporal_factor, spatial_factor, spatial_factor)
        self.conv = CausalConv3d(channels, channels, (3, 3, 3), stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """Spatial and temporal upsampling."""

    def __init__(self, channels: int, spatial_factor: int = 2, temporal_factor: int = 2):
        super().__init__()
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        self.conv = CausalConv3d(channels, channels, (3, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = F.interpolate(
            x,
            size=(T * self.temporal_factor, H * self.spatial_factor, W * self.spatial_factor),
            mode="nearest",
        )
        return self.conv(x)


class AttentionPool(nn.Module):
    """Attention pooling for global context."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.query_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(self.query_token.expand(B, -1, -1))
        kv = self.kv(x).chunk(2, dim=-1)
        k, v = kv

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out).squeeze(1)
