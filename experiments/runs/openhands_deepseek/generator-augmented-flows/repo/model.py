"""
Consistency Model with SongUNet (NCSN++) backbone.
Based on improved consistency training (iCT) from Song & Dhariwal (2024)
with Generator-Augmented Flows (Issenhuth et al.).

The model uses the consistency parameterization:
    f_theta(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(x, sigma)
where:
    c_skip(sigma) = sigma_d^2 / (sigma_d^2 + sigma^2)
    c_out(sigma) = sigma_d * sigma / sqrt(sigma_d^2 + sigma^2)
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import (
    Conv2d,
    Downsample,
    GroupNorm,
    ResBlock,
    SelfAttention,
    Upsample,
    get_timestep_embedding,
)


class SongUNet(nn.Module):
    """
    SongUNet (NCSN++) architecture from EDM (Karras et al., 2022).
    Used as the backbone F_theta for the consistency model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_blocks: Tuple[int, ...] = (3,),
        channel_mult: Tuple[int, ...] = (1, 2, 2),
        attn_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        embedding_type: str = "positional",
        num_groups: int = 32,
        img_resolution: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.channel_mult = channel_mult
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.embedding_type = embedding_type
        self.num_groups = num_groups
        self.img_resolution = img_resolution

        embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.input_conv = Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Compute channel sizes for each resolution level
        num_levels = len(num_blocks)
        self.channels_list = [model_channels * m for m in channel_mult]

        # Downsampling path
        self.encoder_blocks = nn.ModuleList()
        encoder_channels = [model_channels]
        ch = model_channels
        for level in range(num_levels):
            out_ch = self.channels_list[level]
            for block_idx in range(num_blocks[level]):
                in_ch = ch if block_idx == 0 else out_ch
                use_attn = img_resolution // (2 ** level) in attn_resolutions
                self.encoder_blocks.append(
                    ResBlock(in_ch, out_ch, embed_dim, num_groups, use_attn, dropout)
                )
                encoder_channels.append(out_ch)
                ch = out_ch
            if level < num_levels - 1:
                self.encoder_blocks.append(Downsample(ch))
                encoder_channels.append(ch)
                img_resolution //= 2

        # Middle block
        mid_ch = self.channels_list[-1]
        mid_attn = img_resolution in attn_resolutions
        self.mid_block1 = ResBlock(mid_ch, mid_ch, embed_dim, num_groups, mid_attn, dropout)
        self.mid_attn = SelfAttention(mid_ch) if mid_attn else nn.Identity()
        self.mid_block2 = ResBlock(mid_ch, mid_ch, embed_dim, num_groups, use_attention=False, dropout=dropout)

        # Upsampling path
        self.decoder_blocks = nn.ModuleList()
        for level in reversed(range(num_levels)):
            out_ch = self.channels_list[level]
            for block_idx in range(num_blocks[level] + 1):
                skip_ch = encoder_channels.pop()
                in_ch = ch + skip_ch if block_idx == 0 else out_ch
                use_attn = img_resolution // (2 ** level) in attn_resolutions
                self.decoder_blocks.append(
                    ResBlock(in_ch, out_ch, embed_dim, num_groups, use_attn, dropout)
                )
                ch = out_ch
            if level > 0:
                self.decoder_blocks.append(Upsample(ch))
                img_resolution *= 2

        self.out_norm = GroupNorm(num_groups, ch, apply_act=True)
        self.out_conv = Conv2d(ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.conv.weight)
        if self.out_conv.conv.bias is not None:
            nn.init.zeros_(self.out_conv.conv.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        emb = get_timestep_embedding(
            sigma.log(), self.model_channels, self.embedding_type
        )
        emb = self.time_embed(emb)

        hs = []
        h = self.input_conv(x)
        hs.append(h)

        # Encoder
        idx = 0
        for level in range(len(self.num_blocks)):
            for _ in range(self.num_blocks[level]):
                h = self.encoder_blocks[idx](h, emb)
                hs.append(h)
                idx += 1
            if level < len(self.num_blocks) - 1:
                h = self.encoder_blocks[idx](h)
                hs.append(h)
                idx += 1

        # Middle
        h = self.mid_block1(h, emb)
        if isinstance(self.mid_attn, nn.Identity):
            pass
        else:
            h = h + self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Decoder
        idx = 0
        for level in reversed(range(len(self.num_blocks))):
            for _ in range(self.num_blocks[level] + 1):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[idx](h, emb)
                idx += 1
            if level > 0:
                h = self.decoder_blocks[idx](h)
                idx += 1

        h = self.out_norm(h)
        h = self.out_conv(h)
        return h


class ConsistencyModel(nn.Module):
    """
    Consistency model with the parameterization:
        f_theta(x, sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(x, sigma)

    Boundary condition: f_theta(x, sigma_0) = x (approximately)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_blocks: Tuple[int, ...] = (3,),
        channel_mult: Tuple[int, ...] = (1, 2, 2),
        attn_resolutions: Tuple[int, ...] = (),
        dropout: float = 0.0,
        embedding_type: str = "positional",
        sigma_data: float = 0.5,
        img_resolution: int = 32,
    ):
        super().__init__()
        self.sigma_data = sigma_data

        self.net = SongUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            embedding_type=embedding_type,
            img_resolution=img_resolution,
        )

    def _c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        c_skip(sigma) = sigma_d^2 / (sigma_d^2 + (sigma - sigma_0)^2)
        From Eq. (3) in the paper. When sigma = sigma_0, c_skip = 1 exactly.
        """
        sigma_0 = 0.002  # minimal noise level
        return (self.sigma_data ** 2) / (self.sigma_data ** 2 + (sigma - sigma_0) ** 2)

    def _c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        c_out(sigma) = sigma_d * (sigma - sigma_0) / sqrt(sigma_d^2 + sigma^2)
        From Eq. (3) in the paper. When sigma = sigma_0, c_out = 0 exactly.
        """
        sigma_0 = 0.002
        return self.sigma_data * (sigma - sigma_0) / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            sigma: Noise level tensor of shape (B,) or (B, 1, 1, 1)
        Returns:
            Denoised output of shape (B, C, H, W)
        """
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)

        c_skip = self._c_skip(sigma)
        c_out = self._c_out(sigma)

        # Reshape sigma for the network: the network expects per-sample sigma
        sigma_net = sigma.view(-1)

        f_x = self.net(x, sigma_net)

        return c_skip * x + c_out * f_x

    def sample(
        self,
        shape: Tuple[int, ...],
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        steps: int = 1,
        rho: float = 7.0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        One-step or multi-step generation following the consistency model sampling.
        For one-step: sample noise at sigma_max and denoise directly.
        For multi-step: alternate denoising and adding noise.
        """
        if steps == 1:
            z = torch.randn(shape, device=device) * sigma_max
            return self(z, torch.full((shape[0],), sigma_max, device=device))

        # Multi-step sampling
        sigma_steps = self._get_sigma_steps(sigma_min, sigma_max, steps, rho, device)
        x = torch.randn(shape, device=device) * sigma_max

        for i in range(steps - 1):
            x = self(x, torch.full((shape[0],), sigma_steps[i], device=device))
            if i < steps - 2:
                noise = torch.randn(shape, device=device)
                x = x + noise * sigma_steps[i + 1]

        x = self(x, torch.full((shape[0],), sigma_min, device=device))
        return x

    def _get_sigma_steps(
        self, sigma_min: float, sigma_max: float, steps: int, rho: float, device: torch.device
    ) -> torch.Tensor:
        """EDM-style sigma schedule descending from sigma_max to sigma_min."""
        rho_inv = 1.0 / rho
        steps_tensor = torch.linspace(0, 1, steps, device=device)
        sigmas = (sigma_max ** rho_inv + steps_tensor * (sigma_min ** rho_inv - sigma_max ** rho_inv)) ** rho
        return sigmas.flip(0)


class EMAHelper:
    """Exponential Moving Average helper for model parameters."""

    def __init__(self, model: nn.Module, mu: float = 0.9999):
        self.mu = mu
        self.shadow = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_val = (1.0 - self.mu) * param.data + self.mu * self.shadow[name]
                self.shadow[name] = new_val.clone()

    def apply_to(self, model: nn.Module):
        """Copy EMA parameters back to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def store(self, model: nn.Module):
        """Store current model parameters."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()

    def restore(self, model: nn.Module):
        """Restore stored model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
