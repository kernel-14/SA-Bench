import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    """
    VQGAN-style encoder. Downsamples image to feature map.
    With ch_mult=[1,1,2,2,4] and 4 downsampling steps, 256->16 (factor 16).
    """

    def __init__(
        self,
        in_channels: int = 3,
        z_channels: int = 256,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        in_ch = ch
        self.down = nn.ModuleList()
        curr_res = 256
        for i_level in range(self.num_resolutions):
            out_ch = ch * ch_mult[i_level]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_ch, out_ch, dropout))
                if curr_res in attn_resolutions:
                    attns.append(SelfAttention2d(out_ch))
                else:
                    attns.append(nn.Identity())
                in_ch = out_ch
            downsample = Downsample(in_ch) if i_level < self.num_resolutions - 1 else nn.Identity()
            self.down.append(nn.ModuleDict({"blocks": blocks, "attns": attns, "downsample": downsample}))
            if i_level < self.num_resolutions - 1:
                curr_res //= 2

        # Middle
        self.mid_block1 = ResidualBlock(in_ch, in_ch, dropout)
        self.mid_attn = SelfAttention2d(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, dropout)

        self.norm_out = nn.GroupNorm(32, in_ch)
        self.conv_out = nn.Conv2d(in_ch, z_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for level in self.down:
            for block, attn in zip(level["blocks"], level["attns"]):
                h = block(h)
                h = attn(h)
            h = level["downsample"](h)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    """
    VQGAN-style decoder. Upsamples feature map back to image.
    """

    def __init__(
        self,
        out_channels: int = 3,
        z_channels: int = 256,
        ch: int = 128,
        ch_mult: Tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        in_ch = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, in_ch, 3, padding=1)

        # Middle
        self.mid_block1 = ResidualBlock(in_ch, in_ch, dropout)
        self.mid_attn = SelfAttention2d(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, dropout)

        curr_res = 16
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            out_ch = ch * ch_mult[i_level]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResidualBlock(in_ch, out_ch, dropout))
                if curr_res in attn_resolutions:
                    attns.append(SelfAttention2d(out_ch))
                else:
                    attns.append(nn.Identity())
                in_ch = out_ch
            upsample = Upsample(in_ch) if i_level > 0 else nn.Identity()
            self.up.append(nn.ModuleDict({"blocks": blocks, "attns": attns, "upsample": upsample}))
            if i_level > 0:
                curr_res *= 2

        self.norm_out = nn.GroupNorm(32, in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        for level in self.up:
            for block, attn in zip(level["blocks"], level["attns"]):
                h = block(h)
                h = attn(h)
            h = level["upsample"](h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


class AdaLN(nn.Module):
    """Adaptive Layer Normalization for class conditioning."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(cond)  # (B, 2*dim)
        scale, shift = scale_shift.chunk(2, dim=-1)
        if x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
