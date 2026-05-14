"""
2D-convolution U-Net for 1D PDE data.

Input data has shape [batch, C_in, T, X] where T and X are the time and
spatial dimensions of the wavelet-transformed coefficients.

Architecture (from Table 18/19):
  - Initial dimension: 128
  - Downsampling/Upsampling layers: 4
  - Convolution kernel size: 3
  - Dimension multiplier: [1, 2, 4, 8]
  - ResNet block groups: 8
  - Attention hidden dimension: 32
  - Attention heads: 4

The model takes a noisy input x_k and a conditioning signal cond,
concatenated along the channel dimension, plus a diffusion timestep k.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim: int, dim_out: Optional[int] = None) -> nn.Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out or dim, kernel_size=3, padding=1),
    )


def Downsample(dim: int, dim_out: Optional[int] = None) -> nn.Module:
    return nn.Conv2d(dim, dim_out or dim, kernel_size=4, stride=2, padding=1)


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


# ---------------------------------------------------------------------------
# ResNet block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: Optional[int] = None, groups: int = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * (q.shape[-2] ** -0.5)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet1D(nn.Module):
    """
    2D-convolution U-Net for 1D PDE data in wavelet domain.

    The input is the concatenation of:
      - noisy wavelet coefficients of the target (x_k): [batch, C_x, T', X']
      - wavelet coefficients of the condition (cond): [batch, C_cond, T', X']

    Total input channels = C_x + C_cond.

    Args:
        in_channels: number of channels of the noisy target (C_x)
        cond_channels: number of channels of the condition (C_cond)
        init_dim: initial projection dimension (default 128)
        dim_mults: channel multipliers per resolution level
        resnet_block_groups: groups for GroupNorm in ResNet blocks
        attn_heads: number of attention heads
        attn_dim_head: dimension per attention head
    """

    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        init_dim: int = 128,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        resnet_block_groups: int = 8,
        attn_heads: int = 4,
        attn_dim_head: int = 32,
    ):
        super().__init__()

        total_in = in_channels + cond_channels
        self.init_conv = nn.Conv2d(total_in, init_dim, kernel_size=3, padding=1)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embedding
        time_dim = init_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(init_dim),
            nn.Linear(init_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=attn_heads, dim_head=attn_dim_head))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

        # Decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                Residual(PreNorm(dim_out, LinearAttention(dim_out, heads=attn_heads, dim_head=attn_dim_head))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(init_dim, in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: noisy wavelet coefficients [batch, C_x, T', X']
            t: diffusion timestep [batch]
            cond: conditioning wavelet coefficients [batch, C_cond, T', X']
                  If None, unconditional forward pass (classifier-free guidance)
        Returns:
            predicted noise [batch, C_x, T', X']
        """
        if cond is not None:
            x_in = torch.cat([x, cond], dim=1)
        else:
            # Unconditional: zero out condition
            x_in = torch.cat([x, torch.zeros_like(x[:, :x.shape[1]])], dim=1)
            # Pad to correct cond channels if needed
            # This is handled by the caller passing zeros

        h = self.init_conv(x_in)
        r = h.clone()

        t_emb = self.time_mlp(t)

        skips = []
        for block1, block2, attn, downsample in self.downs:
            h = block1(h, t_emb)
            skips.append(h)
            h = block2(h, t_emb)
            h = attn(h)
            skips.append(h)
            h = downsample(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for block1, block2, attn, upsample in self.ups:
            h = torch.cat([h, skips.pop()], dim=1)
            h = block1(h, t_emb)
            h = torch.cat([h, skips.pop()], dim=1)
            h = block2(h, t_emb)
            h = attn(h)
            h = upsample(h)

        h = torch.cat([h, r], dim=1)
        h = self.final_res_block(h, t_emb)
        return self.final_conv(h)
