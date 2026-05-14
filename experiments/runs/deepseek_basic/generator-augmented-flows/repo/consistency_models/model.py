"""
Consistency Model implementation with skip-connection parameterization.

Based on:
- Song et al. (2023) "Consistency Models"
- Song and Dhariwal (2024) "Improved Techniques for Training Consistency Models"
- Karras et al. (2022) EDM framework

The model uses the parametrization from Equation (3):
    f_θ(x_t, σ_t) = c_skip(σ_t) * x_t + c_out(σ_t) * F_θ(x_t, σ_t)
"""

import torch
import torch.nn as nn
import numpy as np


def model_parameterization(
    x: torch.Tensor,
    sigma: torch.Tensor,
    f_theta_out: torch.Tensor,
    sigma_data: float = 0.5,
    sigma_min: float = 0.002,
) -> torch.Tensor:
    """Apply skip-connection parameterization from Equation (3)."""
    eps = 1e-8
    c_skip = sigma_data ** 2 / (sigma_data ** 2 + (sigma - sigma_min) ** 2 + eps)
    c_out = sigma_data * (sigma - sigma_min) / torch.sqrt(sigma_data ** 2 + sigma ** 2 + eps)
    while c_skip.dim() < x.dim():
        c_skip = c_skip.unsqueeze(-1)
        c_out = c_out.unsqueeze(-1)
    return c_skip * x + c_out * f_theta_out


class ConsistencyModel(nn.Module):
    """A consistency model f_θ that learns the output map of the PF-ODE."""
    def __init__(self, network: nn.Module, sigma_data: float = 0.5, sigma_min: float = 0.002):
        super().__init__()
        self.network = network
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min

    def forward(self, x: torch.Tensor, sigma: torch.Tensor):
        f_out = self.network(x, sigma)
        return model_parameterization(x, sigma, f_out, self.sigma_data, self.sigma_min)


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_positions: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_positions = max_positions

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        device = sigma.device
        half_dim = self.embedding_dim // 2
        emb = torch.log(torch.tensor(self.max_positions, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = sigma.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        scale = C ** -0.5
        attn = torch.bmm(q, k) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        out = self.proj(out)
        return x + out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels, dropout=0.0,
                 use_attention=False, resample=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resample = resample
        self.use_attention = use_attention

        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.attn = SelfAttention(out_channels) if use_attention else None

        if resample == "up":
            self.resample_op = nn.Upsample(scale_factor=2, mode="nearest")
        elif resample == "down":
            self.resample_op = nn.AvgPool2d(2)
        else:
            self.resample_op = None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.resample == "up":
            x = self.resample_op(x)
        elif self.resample == "down":
            x = self.resample_op(x)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        emb_out = self.emb_proj(emb)
        while emb_out.dim() < h.dim():
            emb_out = emb_out.unsqueeze(-1)
        h = h + emb_out

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        h = h + self.skip(x)

        if self.attn is not None:
            h = self.attn(h)

        return h


class SongUNet(nn.Module):
    """
    SongUNet architecture from EDM (Karras et al., 2022).
    Clean-room UNet implementation for consistency model training.
    Uses standard UNet skip connections: each resolution level output
    is concatenated with the corresponding decoder input.
    """

    def __init__(
        self,
        img_resolution: int,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        channel_mult: list = None,
        num_blocks: int = 3,
        attn_resolutions: list = None,
        dropout: float = 0.0,
        embedding_type: str = "positional",
    ):
        super().__init__()

        if channel_mult is None:
            channel_mult = [1, 2, 2, 2]
        if attn_resolutions is None:
            attn_resolutions = []

        self.num_resolutions = len(channel_mult)

        # Embedding
        emb_dim = 4 * model_channels
        self.map_noise = PositionalEmbedding(emb_dim)

        # Input
        self.input_layer = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        def get_nb(level):
            if isinstance(num_blocks, (list, tuple)):
                return num_blocks[level]
            return num_blocks

        # --- Build encoder ---
        # Each resolution level: nb residual blocks, then optional downsampling
        # We save the output of EACH RESOLUTION LEVEL (after all blocks, before downsampling)
        # as skip connections for the decoder.
        self.encoder_blocks = nn.ModuleList()  # blocks within each level
        self.encoder_downs = nn.ModuleList()    # downsampling layers
        self.skip_channels = []                  # channels of skips

        cur_res = img_resolution
        cur_ch = model_channels
        self.skip_channels.append(cur_ch)  # input-level skip

        for level in range(self.num_resolutions):
            block_ch = model_channels * channel_mult[level]
            nb = get_nb(level)
            level_blocks = nn.ModuleList()
            for _ in range(nb):
                use_attn = cur_res in attn_resolutions
                in_ch = cur_ch
                level_blocks.append(ResBlock(in_ch, block_ch, emb_dim, dropout, use_attn))
                cur_ch = block_ch
            self.encoder_blocks.append(level_blocks)

            if level < self.num_resolutions - 1:
                self.encoder_downs.append(ResBlock(cur_ch, cur_ch, emb_dim, dropout, False, resample="down"))
                self.skip_channels.append(cur_ch)
                cur_res //= 2
            else:
                self.encoder_downs.append(None)
                self.skip_channels.append(cur_ch)

        # --- Bottleneck ---
        self.bottleneck = nn.ModuleList()
        bottleneck_ch = model_channels * channel_mult[-1]
        nb_bn = get_nb(self.num_resolutions - 1)
        for _ in range(nb_bn):
            in_ch = cur_ch if not self.bottleneck else bottleneck_ch
            self.bottleneck.append(ResBlock(in_ch, bottleneck_ch, emb_dim, dropout,
                                            cur_res in attn_resolutions))
            cur_ch = bottleneck_ch

        # --- Build decoder ---
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()

        for level in reversed(range(self.num_resolutions)):
            block_ch = model_channels * channel_mult[level]
            nb = get_nb(level)

            # Skip from encoder at this level
            skip_ch = self.skip_channels[level + 1]

            level_blocks = nn.ModuleList()
            for block_idx in range(nb + 1):
                if block_idx == 0:
                    in_ch = cur_ch + skip_ch
                else:
                    in_ch = cur_ch

                if block_idx == nb and level > 0:
                    level_blocks.append(ResBlock(in_ch, block_ch, emb_dim, dropout, False, resample="up"))
                else:
                    level_blocks.append(ResBlock(in_ch, block_ch, emb_dim, dropout, False))
                cur_ch = block_ch

            self.decoder_blocks.append(level_blocks)

        # Output
        self.output_norm = nn.GroupNorm(min(32, cur_ch), cur_ch)
        self.output_act = nn.SiLU()
        self.output_layer = nn.Conv2d(cur_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.dim() > 1:
            sigma = sigma.flatten()
        emb = self.map_noise(sigma)

        h = self.input_layer(x)

        # Encoder: save skip at each resolution level
        skips = [h]  # input skip

        for level in range(self.num_resolutions):
            for block in self.encoder_blocks[level]:
                h = block(h, emb)
            skips.append(h)  # Save skip BEFORE downsampling
            if self.encoder_downs[level] is not None:
                h = self.encoder_downs[level](h, emb)

        # Bottleneck
        for block in self.bottleneck:
            h = block(h, emb)

        # Decoder: pop skips in reverse order
        for idx, level in enumerate(reversed(range(self.num_resolutions))):
            skip = skips.pop()
            for block_idx, block in enumerate(self.decoder_blocks[idx]):
                if block_idx == 0:
                    h = block(torch.cat([h, skip], dim=1), emb)
                else:
                    h = block(h, emb)

        h = self.output_norm(h)
        h = self.output_act(h)
        h = self.output_layer(h)
        return h
