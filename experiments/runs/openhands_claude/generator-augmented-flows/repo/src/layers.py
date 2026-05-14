import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(timesteps, self.dim, self.max_period)


class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        args = timesteps[:, None].float() * self.freqs[None] * 2 * math.pi
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-5):
        super().__init__(num_groups=min(num_groups, num_channels), num_channels=num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(x.dtype)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float = 0.0,
        resample: Optional[str] = None,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resample = resample

        self.norm1 = GroupNorm32(in_channels, num_groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.emb_proj = Linear(emb_channels, out_channels * 2)

        self.norm2 = GroupNorm32(out_channels, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

        if resample == "up":
            self.resample_op = Upsample(in_channels)
            self.resample_skip = Upsample(in_channels)
        elif resample == "down":
            self.resample_op = Downsample(in_channels)
            self.resample_skip = Downsample(in_channels)
        else:
            self.resample_op = nn.Identity()
            self.resample_skip = nn.Identity()

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.resample_op(h)
        h = self.conv1(h)

        emb_out = self.emb_proj(F.silu(emb))[:, :, None, None]
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        skip = self.resample_skip(x)
        skip = self.skip_conv(skip)
        return h + skip


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1, num_groups: int = 32):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = GroupNorm32(channels, num_groups)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (h d) x y -> b h (x y) d", h=self.num_heads)
        k = rearrange(k, "b (h d) x y -> b h (x y) d", h=self.num_heads)
        v = rearrange(v, "b (h d) x y -> b h (x y) d", h=self.num_heads)

        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.einsum("bhid,bhjd->bhij", q, k) * scale, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=H, y=W)
        out = self.proj_out(out)
        return x + out


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
