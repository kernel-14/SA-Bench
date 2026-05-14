"""
Neural network model components for Flow Matching fine-tuning.

Architecture: Latent Diffusion / Flow Matching with U-Net backbone.
- Pre-trained autoencoder (VAE) for 512x512 -> 64x64 latent space
- Text-conditional U-Net with cross-attention (similar to Stable Diffusion)
- Text encoder: CLIP ViT-H-14

The paper uses:
- 512x512 resolution images
- Latent space: 64x64 x 4 channels (8x downsampling)
- U-Net with channel_mult=(1,2,4,4), model_channels=320
- 40 timesteps for fine-tuning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple

from noise_schedules import FlowMatchingSchedule


# ---------------------------------------------------------------------------
# Attention components
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(dim=1)
        # [B, heads, head_dim, HW] -> [B, heads, HW, head_dim]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class CrossAttention(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.LayerNorm(channels)
        self.norm_context = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(context_dim, channels, bias=False)
        self.to_v = nn.Linear(context_dim, channels, bias=False)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dims
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        h = self.norm(x_flat)
        ctx = self.norm_context(context)

        q = self.to_q(h).reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(ctx).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(ctx).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out


class SpatialTransformer(nn.Module):
    """Spatial transformer block with self-attention and cross-attention."""

    def __init__(self, channels: int, context_dim: int, num_heads: int = 8,
                 depth: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                SelfAttention(channels, num_heads),
                CrossAttention(channels, context_dim, num_heads),
                nn.Sequential(
                    nn.LayerNorm(channels),
                    nn.Linear(channels, channels * 4),
                    nn.GELU(),
                    nn.Linear(channels * 4, channels),
                ),
            ])
            for _ in range(depth)
        ])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        h = self.norm(x)
        h = self.proj_in(h)
        for self_attn, cross_attn, ff in self.layers:
            h = self_attn(h)
            if context is not None:
                h = cross_attn(h, context)
            # Feed-forward
            h_flat = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
            h_flat = h_flat + ff(h_flat)
            h = h_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj_out(h)
        return residual + h


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    def __init__(self, model_channels: int):
        super().__init__()
        time_embed_dim = model_channels * 4
        self.sinusoidal = SinusoidalTimeEmbedding(model_channels)
        self.mlp = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


# ---------------------------------------------------------------------------
# ResNet block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Conv2d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Time conditioning via scale-shift
        t_out = self.time_proj(F.silu(t_emb))[:, :, None, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# U-Net for Flow Matching
# ---------------------------------------------------------------------------

class UNetFlowMatching(nn.Module):
    """
    U-Net architecture for Flow Matching / Latent Diffusion.

    Architecture matches the paper's setup:
    - model_channels=320, channel_mult=(1,2,4,4)
    - num_res_blocks=2 per resolution
    - Spatial transformer with cross-attention at each resolution
    - Input: latent [B, 4, 64, 64] + time t + text context
    - Output: velocity field [B, 4, 64, 64]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (4, 2, 1),
        num_heads: int = 8,
        context_dim: int = 768,
        transformer_depth: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4

        self.time_embed = TimeEmbedding(model_channels)

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = model_channels
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, time_embed_dim, dropout)])
                if ds in attention_resolutions:
                    block.append(SpatialTransformer(out_ch, context_dim, num_heads, transformer_depth))
                self.down_blocks.append(block)
                input_block_chans.append(out_ch)
                ch = out_ch

            if level < len(channel_mult) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)
                ds *= 2
            else:
                self.down_samples.append(None)

        # Middle block
        self.mid_res1 = ResBlock(ch, ch, time_embed_dim, dropout)
        self.mid_attn = SpatialTransformer(ch, context_dim, num_heads, transformer_depth)
        self.mid_res2 = ResBlock(ch, ch, time_embed_dim, dropout)

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                block = nn.ModuleList([ResBlock(ch + skip_ch, out_ch, time_embed_dim, dropout)])
                if ds in attention_resolutions:
                    block.append(SpatialTransformer(out_ch, context_dim, num_heads, transformer_depth))
                self.up_blocks.append(block)
                ch = out_ch

            if level > 0:
                self.up_samples.append(
                    nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                  nn.Conv2d(ch, ch, 3, padding=1))
                )
                ds //= 2
            else:
                self.up_samples.append(None)

        # Output projection
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_proj = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Latent input [B, 4, H, W]
            t: Timestep [B] in [0, 1]
            context: Text embeddings [B, seq_len, context_dim]

        Returns:
            Velocity field [B, 4, H, W]
        """
        t_emb = self.time_embed(t)
        h = self.input_proj(x)

        # Encoder
        skips = [h]
        down_idx = 0
        for i, block in enumerate(self.down_blocks):
            res_block = block[0]
            h = res_block(h, t_emb)
            if len(block) > 1:
                h = block[1](h, context)
            skips.append(h)

            # Check if we need to downsample after this group
            # Downsample after every num_res_blocks blocks at each level
            if (i + 1) % (len(self.down_blocks) // len(self.down_samples)) == 0:
                if down_idx < len(self.down_samples) and self.down_samples[down_idx] is not None:
                    h = self.down_samples[down_idx](h)
                    skips.append(h)
                down_idx += 1

        # Middle
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_res2(h, t_emb)

        # Decoder
        up_idx = 0
        for i, block in enumerate(self.up_blocks):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            res_block = block[0]
            h = res_block(h, t_emb)
            if len(block) > 1:
                h = block[1](h, context)

            if (i + 1) % (len(self.up_blocks) // len(self.up_samples)) == 0:
                if up_idx < len(self.up_samples) and self.up_samples[up_idx] is not None:
                    h = self.up_samples[up_idx](h)
                up_idx += 1

        return self.out_proj(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Flow Matching model wrapper
# ---------------------------------------------------------------------------

class FlowMatchingModel(nn.Module):
    """
    Complete Flow Matching model with VAE encoder/decoder and text encoder.

    Wraps the U-Net velocity field with:
    - VAE for latent encoding/decoding
    - CLIP text encoder for conditioning
    - Classifier-free guidance support
    """

    def __init__(
        self,
        unet: UNetFlowMatching,
        vae: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        schedule: Optional[FlowMatchingSchedule] = None,
        vae_scale_factor: float = 0.18215,
    ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.schedule = schedule or FlowMatchingSchedule()
        self.vae_scale_factor = vae_scale_factor

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space using VAE."""
        if self.vae is None:
            return x
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
        return latent * self.vae_scale_factor

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image using VAE."""
        if self.vae is None:
            return z
        z = z / self.vae_scale_factor
        with torch.no_grad():
            image = self.vae.decode(z).sample
        return image

    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts using CLIP text encoder."""
        if self.text_encoder is None:
            raise ValueError("Text encoder not provided")
        return self.text_encoder(prompts)

    def velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity field v(x, t, text)."""
        return self.unet(x, t, text_embeddings)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.velocity(x, t, text_embeddings)

    @torch.no_grad()
    def generate(
        self,
        text_embeddings: torch.Tensor,
        null_embeddings: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        sigma_type: str = "zero",
        batch_size: int = 1,
        latent_size: int = 64,
        latent_channels: int = 4,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate images from text embeddings.

        Args:
            text_embeddings: [B, seq_len, context_dim]
            null_embeddings: Unconditional embeddings for CFG
            cfg_scale: Classifier-free guidance scale w
            sigma_type: "zero" (ODE) or "memoryless" (SDE)
            batch_size: Number of images to generate
            latent_size: Spatial size of latent
            latent_channels: Number of latent channels
            device: Target device
            seed: Random seed

        Returns:
            Generated images [B, 3, H, W] in [-1, 1]
        """
        if device is None:
            device = next(self.parameters()).device

        if seed is not None:
            torch.manual_seed(seed)

        x0 = torch.randn(batch_size, latent_channels, latent_size, latent_size, device=device)

        def velocity_fn(x, t, _):
            v_cond = self.velocity(x, t, text_embeddings)
            if cfg_scale > 0.0 and null_embeddings is not None:
                v_uncond = self.velocity(x, t, null_embeddings)
                return (1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond
            return v_cond

        if sigma_type == "zero":
            from sde_utils import sample_fm_ode
            z = sample_fm_ode(velocity_fn, x0, self.schedule)
        elif sigma_type == "memoryless":
            from sde_utils import sample_fm_sde_memoryless
            z = sample_fm_sde_memoryless(velocity_fn, x0, self.schedule)
        else:
            raise ValueError(f"Unknown sigma_type: {sigma_type}")

        return self.decode_latent(z)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_unet(config) -> UNetFlowMatching:
    """Build U-Net from config."""
    return UNetFlowMatching(
        in_channels=config.latent_channels,
        out_channels=config.latent_channels,
        model_channels=config.model_channels,
        channel_mult=config.channel_mult,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        num_heads=config.num_heads,
        context_dim=config.context_dim,
        transformer_depth=config.transformer_depth,
    )


def load_pretrained_model(
    checkpoint_path: str,
    config,
    device: torch.device,
) -> FlowMatchingModel:
    """Load pre-trained Flow Matching model from checkpoint."""
    unet = build_unet(config.model)
    schedule = FlowMatchingSchedule(
        num_timesteps=config.training.num_timesteps,
        sigma_offset_h=config.noise_schedule.sigma_offset_h,
    )
    model = FlowMatchingModel(unet=unet, schedule=schedule)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.unet.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.unet.load_state_dict(checkpoint["state_dict"])
    else:
        model.unet.load_state_dict(checkpoint)

    return model.to(device)
