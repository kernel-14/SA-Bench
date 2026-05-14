import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    PositionalEmbedding,
    FourierEmbedding,
    Linear,
    GroupNorm32,
    ResnetBlock,
    AttentionBlock,
    Upsample,
    Downsample,
)


class SongUNet(nn.Module):
    """
    NCSNpp / SongUNet architecture from Karras et al. (2022) EDM implementation.
    Used as the backbone F_θ for consistency models.

    The network takes a noisy image x and noise level sigma as input and outputs
    a denoised prediction. The time embedding uses log(sigma/4) * 0.25 as the
    noise conditioning signal (c_noise from EDM).
    """

    def __init__(
        self,
        img_resolution: int,
        in_channels: int,
        out_channels: int,
        model_channels: int = 128,
        channel_mult: Union[List[int], tuple] = (1, 2, 2),
        num_blocks: Union[int, List[int]] = 3,
        attn_resolutions: Union[List[int], tuple] = (),
        dropout: float = 0.0,
        embedding_type: str = "positional",
        num_heads: int = 1,
        num_groups: int = 32,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels

        emb_dim = model_channels * 4

        if embedding_type == "positional":
            self.time_embed = nn.Sequential(
                PositionalEmbedding(model_channels),
                Linear(model_channels, emb_dim),
                nn.SiLU(),
                Linear(emb_dim, emb_dim),
            )
        elif embedding_type == "fourier":
            self.time_embed = nn.Sequential(
                FourierEmbedding(model_channels),
                Linear(model_channels, emb_dim),
                nn.SiLU(),
                Linear(emb_dim, emb_dim),
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        channel_mult = list(channel_mult)
        num_levels = len(channel_mult)

        if isinstance(num_blocks, int):
            num_blocks_list = [num_blocks] * num_levels
        else:
            num_blocks_list = list(num_blocks)

        channels = [model_channels * m for m in channel_mult]

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Build encoder
        # At each level: num_blocks ResNet blocks (+ optional attention), then downsample
        # Each ResNet block output is saved as a skip connection
        self.enc_blocks = nn.ModuleList()
        self.enc_downsamples = nn.ModuleList()
        self._enc_skip_channels = []  # track skip channel sizes for decoder

        current_res = img_resolution
        in_ch = channels[0]

        for level in range(num_levels):
            out_ch = channels[level]
            n_blocks = num_blocks_list[level]
            level_mods = nn.ModuleList()

            for b in range(n_blocks):
                level_mods.append(
                    ResnetBlock(in_ch, out_ch, emb_dim, dropout=dropout, num_groups=num_groups)
                )
                self._enc_skip_channels.append(out_ch)
                in_ch = out_ch
                if current_res in attn_resolutions:
                    level_mods.append(AttentionBlock(out_ch, num_heads=num_heads, num_groups=num_groups))

            self.enc_blocks.append(level_mods)

            if level < num_levels - 1:
                self.enc_downsamples.append(Downsample(out_ch))
                current_res //= 2
            else:
                self.enc_downsamples.append(nn.Identity())

        # Middle blocks
        mid_ch = channels[-1]
        self.mid_block1 = ResnetBlock(mid_ch, mid_ch, emb_dim, dropout=dropout, num_groups=num_groups)
        self.mid_attn = AttentionBlock(mid_ch, num_heads=num_heads, num_groups=num_groups)
        self.mid_block2 = ResnetBlock(mid_ch, mid_ch, emb_dim, dropout=dropout, num_groups=num_groups)

        # Build decoder
        # At each level (reversed): upsample (if not first), then num_blocks ResNet blocks
        # Each block concatenates with the corresponding encoder skip
        self.dec_blocks = nn.ModuleList()
        self.dec_upsamples = nn.ModuleList()

        skip_ch_iter = list(reversed(self._enc_skip_channels))
        in_ch = mid_ch

        for level in reversed(range(num_levels)):
            out_ch = channels[level]
            n_blocks = num_blocks_list[level]
            level_mods = nn.ModuleList()

            for b in range(n_blocks):
                skip_ch = skip_ch_iter.pop(0)
                level_mods.append(
                    ResnetBlock(in_ch + skip_ch, out_ch, emb_dim, dropout=dropout, num_groups=num_groups)
                )
                in_ch = out_ch
                if current_res in attn_resolutions:
                    level_mods.append(AttentionBlock(out_ch, num_heads=num_heads, num_groups=num_groups))

            self.dec_blocks.append(level_mods)

            if level > 0:
                self.dec_upsamples.append(Upsample(out_ch))
                current_res *= 2
            else:
                self.dec_upsamples.append(nn.Identity())

        # Output projection
        self.output_norm = GroupNorm32(channels[0], num_groups)
        self.output_proj = nn.Conv2d(channels[0], out_channels, 3, padding=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # c_noise conditioning: 0.25 * log(sigma)
        c_noise = 0.25 * torch.log(sigma.float())
        emb = self.time_embed(c_noise)

        h = self.input_proj(x)

        # Encoder: collect skip connections
        skips = []
        for level_mods, downsample in zip(self.enc_blocks, self.enc_downsamples):
            for mod in level_mods:
                if isinstance(mod, ResnetBlock):
                    h = mod(h, emb)
                    skips.append(h)
                else:
                    h = mod(h)
            h = downsample(h)

        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Decoder: pop skip connections in reverse order
        for level_mods, upsample in zip(self.dec_blocks, self.dec_upsamples):
            for mod in level_mods:
                if isinstance(mod, ResnetBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = mod(h, emb)
                else:
                    h = mod(h)
            h = upsample(h)

        h = F.silu(self.output_norm(h))
        h = self.output_proj(h)
        return h
