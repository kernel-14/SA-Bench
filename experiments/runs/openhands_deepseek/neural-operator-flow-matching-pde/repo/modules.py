"""Neural network modules for the generative PDE foundation model.

Components:
- P2VAE: Pretrained Physics Variational Autoencoder (SD-VAE architecture)
- SiT Backbone: Scalable Interpolant Transformer with AdaLN-Zero
- GRU Diffusion Forcing: recurrent latent state for conditional flow marching
- Temporal Pyramids: coarse-to-fine spatial tokenization
- FMT: Flow Marching Transformer (full model)
- Flow Marching utilities: interpolation kernel, loss, euler sampler
"""

import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper Layers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama-2 style)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation (Llama-2 style)."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding."""
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


# ---------------------------------------------------------------------------
# P2VAE: SD-VAE Architecture
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with GroupNorm."""
    def __init__(self, channels: int, out_channels: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_channels = out_channels or channels
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SpatialAttention(nn.Module):
    """Spatial self-attention for 2D feature maps."""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.reshape(B, C, H, W)
        return self.proj(out)


class Downsample(nn.Module):
    """Strided convolution downsampling."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsampling + convolution."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder stage: ResBlocks + optional attention + downsampling."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        use_attention: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        ch = in_channels
        for i in range(num_res_blocks):
            self.res_blocks.append(ResBlock(ch, out_channels if i == num_res_blocks - 1 else ch, dropout))
            ch = out_channels
        self.attention = SpatialAttention(ch) if use_attention else nn.Identity()
        self.downsample = Downsample(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x)
        x = self.attention(x)
        x = self.downsample(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder stage: Upsampling + ResBlocks + optional attention."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        use_attention: bool,
        dropout: float = 0.0,
        skip_channels: int = 0,
    ):
        super().__init__()
        self.upsample = Upsample(in_channels)
        self.res_blocks = nn.ModuleList()
        ch = in_channels + skip_channels
        for i in range(num_res_blocks):
            self.res_blocks.append(ResBlock(ch, out_channels if i == num_res_blocks - 1 else ch, dropout))
            ch = out_channels if i == num_res_blocks - 1 else ch
        self.attention = SpatialAttention(ch) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        for block in self.res_blocks:
            x = block(x)
        x = self.attention(x)
        return x


class P2VAEEncoder(nn.Module):
    """Encoder for P2VAE."""
    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        z_channels: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_levels = len(channel_mult)
        channels_list = [base_dim * m for m in channel_mult]
        resolution = 128  # fixed input size

        self.conv_in = nn.Conv2d(in_channels, channels_list[0], 3, padding=1)

        self.blocks = nn.ModuleList()
        for i in range(self.num_levels):
            in_ch = channels_list[i]
            out_ch = channels_list[min(i + 1, self.num_levels - 1)]
            use_attn = resolution in attention_resolutions
            self.blocks.append(
                EncoderBlock(in_ch, out_ch, num_res_blocks, use_attn, dropout)
            )
            resolution //= 2

        self.mid_res_blocks = nn.ModuleList([
            ResBlock(channels_list[-1], channels_list[-1], dropout),
            ResBlock(channels_list[-1], channels_list[-1], dropout),
        ])
        self.mid_attention = SpatialAttention(channels_list[-1])

        self.norm_out = nn.GroupNorm(32, channels_list[-1])
        self.conv_mu = nn.Conv2d(channels_list[-1], z_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(channels_list[-1], z_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_in(x)
        skips = []
        for block in self.blocks:
            skips.append(x)
            x = block(x)

        for block in self.mid_res_blocks:
            x = block(x)
        x = self.mid_attention(x)

        x = self.norm_out(x)
        x = F.silu(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return mu, logvar


class P2VAEDecoder(nn.Module):
    """Decoder for P2VAE."""
    def __init__(
        self,
        out_channels: int = 3,
        base_dim: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        z_channels: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_levels = len(channel_mult)
        channels_list = [base_dim * m for m in channel_mult]
        resolution = 16  # latent size

        self.conv_in = nn.Conv2d(z_channels, channels_list[-1], 3, padding=1)

        self.mid_res_blocks = nn.ModuleList([
            ResBlock(channels_list[-1], channels_list[-1], dropout),
            ResBlock(channels_list[-1], channels_list[-1], dropout),
        ])
        self.mid_attention = SpatialAttention(channels_list[-1])

        self.blocks = nn.ModuleList()
        for i in reversed(range(self.num_levels)):
            in_ch = channels_list[min(i + 1, self.num_levels - 1)]
            out_ch = channels_list[i]
            use_attn = resolution in attention_resolutions
            self.blocks.append(
                DecoderBlock(in_ch, out_ch, num_res_blocks, use_attn, dropout)
            )
            resolution *= 2

        self.norm_out = nn.GroupNorm(32, channels_list[0])
        self.conv_out = nn.Conv2d(channels_list[0], out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(z)

        for block in self.mid_res_blocks:
            x = block(x)
        x = self.mid_attention(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


class P2VAE(nn.Module):
    """Pretrained Physics Variational Autoencoder.

    Compresses c3p128 physical fields to c16p16 latent grids (12x compression).
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_dim: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        z_channels: int = 16,
        dropout: float = 0.0,
        kl_weight: float = 1e-3,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.encoder = P2VAEEncoder(
            in_channels, base_dim, channel_mult, num_res_blocks,
            attention_resolutions, z_channels, dropout,
        )
        self.decoder = P2VAEDecoder(
            in_channels, base_dim, channel_mult, num_res_blocks,
            attention_resolutions, z_channels, dropout,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        recon_loss = 0.5 * F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, z, recon_loss, self.kl_weight * kl_loss


# ---------------------------------------------------------------------------
# SiT Backbone with AdaLN-Zero
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    """Adaptive Layer Norm with zero-initialized modulation."""
    def __init__(self, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.shift_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, hidden_dim),
        )
        self.scale_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, hidden_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, hidden_dim),
        )
        nn.init.zeros_(self.shift_mlp[1].weight)
        nn.init.zeros_(self.shift_mlp[1].bias)
        nn.init.zeros_(self.scale_mlp[1].weight)
        nn.init.zeros_(self.scale_mlp[1].bias)
        nn.init.zeros_(self.gate_mlp[1].weight)
        nn.init.zeros_(self.gate_mlp[1].bias)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.shift_mlp(c), self.scale_mlp(c), self.gate_mlp(c)


class SiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, cond_dim: int = None):
        super().__init__()
        cond_dim = cond_dim or hidden_dim
        self.norm1 = RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
        self.ada_ln1 = AdaLNZero(hidden_dim, cond_dim)

        self.norm2 = RMSNorm(hidden_dim)
        self.mlp = SwiGLU(hidden_dim, int(hidden_dim * mlp_ratio))
        self.ada_ln2 = AdaLNZero(hidden_dim, cond_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn = self.ada_ln1(c)
        shift_mlp, scale_mlp, gate_mlp = self.ada_ln2(c)

        h = modulate(self.norm1(x), shift_attn, scale_attn)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_attn.unsqueeze(1) * h

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class FinalLayer(nn.Module):
    """Final projection layer with AdaLN-Zero."""
    def __init__(self, hidden_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.ada_ln = AdaLNZero(hidden_dim, cond_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.ada_ln(c)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return gate.unsqueeze(1) * x


class SiT(nn.Module):
    """Scalable Interpolant Transformer backbone.

    Processes token sequences with AdaLN-Zero conditioning on time + hidden state.
    """
    def __init__(
        self,
        seq_len: int,
        patch_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        cond_dim: int = 512,
        out_dim: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        self.t_embedder = TimestepEmbedder(embed_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.blocks = nn.ModuleList([
            SiTBlock(embed_dim, num_heads, mlp_ratio, cond_dim)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(embed_dim, out_dim, cond_dim)

    def forward(
        self, patches: torch.Tensor, t: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        B = patches.shape[0]
        x = self.patch_proj(patches) + self.pos_embed[:, :patches.shape[1]]

        t_embed = self.t_embedder(t)
        c = self.t_proj(t_embed) + h.unsqueeze(1).expand(B, patches.shape[1], -1)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return x


# ---------------------------------------------------------------------------
# GRU Diffusion Forcing
# ---------------------------------------------------------------------------

class GRUDiffusionForcing(nn.Module):
    """GRU-based recurrent state for diffusion forcing.

    Maintains latent state h that is updated at each physical timestep.
    The current noisy state x_t^k is compressed via cross-attention to a
    single token before GRU update.
    """
    def __init__(self, hidden_dim: int, latent_channels: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_channels = latent_channels

        self.compress_norm = nn.LayerNorm(latent_channels)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(latent_channels, hidden_dim)
        self.value_proj = nn.Linear(latent_channels, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)

        self.t_proj = nn.Linear(hidden_dim, hidden_dim)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def compress_state(
        self, x_tk: torch.Tensor, h_prev: torch.Tensor, t_embed: torch.Tensor
    ) -> torch.Tensor:
        """Compress spatial latent x_t^k to a single token via cross-attention."""
        B, C, H, W = x_tk.shape
        x_flat = x_tk.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
        x_flat = self.compress_norm(x_flat)

        q = self.query_proj(h_prev + t_embed).unsqueeze(1)  # [B, 1, D]
        k = self.key_proj(x_flat)
        v = self.value_proj(x_flat)

        attn = F.scaled_dot_product_attention(q, k, v)
        return self.attn_out(attn.squeeze(1))  # [B, D]

    def forward(
        self, h_prev: torch.Tensor, x_tk: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        B = x_tk.shape[0]
        t_embed = self.t_proj(TimestepEmbedder(self.hidden_dim).to(x_tk.device)(t))
        compressed = self.compress_state(x_tk, h_prev, t_embed)
        h_new = self.gru(compressed, h_prev)
        return self.out_norm(h_new)


# ---------------------------------------------------------------------------
# Temporal Pyramids
# ---------------------------------------------------------------------------

def spatial_downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample latent grid spatially by factor using average pooling.

    Args:
        x: [B, C, H, W] latent tensor
        factor: downsampling factor (must divide H and W)
    Returns:
        [B, C, H//factor, W//factor]
    """
    if factor == 1:
        return x
    return F.avg_pool2d(x, kernel_size=factor, stride=factor)


def build_temporal_pyramid(
    y_frames: torch.Tensor, pyramid_ratios: Tuple[int, ...] = (8, 4, 2, 1)
) -> List[torch.Tensor]:
    """Build temporal pyramid: downsample each frame by its ratio.

    Args:
        y_frames: [B, T, C, H, W] latent frames (T=4)
        pyramid_ratios: downsampling factors per frame
    Returns:
        List of [B, N_i, C] token tensors (one per frame)
    """
    result = []
    for i, ratio in enumerate(pyramid_ratios):
        down = spatial_downsample(y_frames[:, i], ratio)
        B, C, H, W = down.shape
        tokens = down.reshape(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
        result.append(tokens)
    return result


# ---------------------------------------------------------------------------
# FMT: Flow Marching Transformer (Full Model)
# ---------------------------------------------------------------------------

class FMT(nn.Module):
    """Flow Marching Transformer.

    Processes 4 consecutive latent frames with temporal pyramids,
    GRU-based diffusion forcing, and SiT backbone to predict
    flow marching velocities.
    """
    def __init__(
        self,
        latent_channels: int = 16,
        latent_spatial_size: int = 16,
        embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        pyramid_ratios: Tuple[int, ...] = (8, 4, 2, 1),
        rnn_hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_spatial_size = latent_spatial_size
        self.embed_dim = embed_dim
        self.pyramid_ratios = pyramid_ratios
        self.num_frames = len(pyramid_ratios)

        total_tokens = sum(
            (latent_spatial_size // r) ** 2 for r in pyramid_ratios
        )

        self.sit = SiT(
            seq_len=total_tokens,
            patch_dim=latent_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            cond_dim=embed_dim,
            out_dim=latent_channels,
        )

        self.gru = GRUDiffusionForcing(rnn_hidden_dim, latent_channels)
        self.h_proj = nn.Sequential(
            nn.Linear(rnn_hidden_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.frame_embed = nn.Parameter(
            torch.randn(1, self.num_frames, embed_dim) * 0.02
        )
        self.frame_pos_offsets = self._compute_frame_offsets()

    def _compute_frame_offsets(self) -> List[int]:
        """Compute token offset for each frame in the concatenated sequence."""
        offsets = [0]
        for r in self.pyramid_ratios[:-1]:
            offsets.append(offsets[-1] + (self.latent_spatial_size // r) ** 2)
        return offsets

    def _add_frame_conditioning(
        self, all_tokens: torch.Tensor, frame_indices: List[int]
    ) -> torch.Tensor:
        """Add frame embedding to each token based on which frame it belongs to."""
        B = all_tokens.shape[0]
        for i, offset in enumerate(self.frame_pos_offsets):
            n_tokens = (self.latent_spatial_size // self.pyramid_ratios[i]) ** 2
            emb = self.frame_embed[:, i:i+1, :].expand(B, n_tokens, -1)
            all_tokens[:, offset:offset + n_tokens] = all_tokens[:, offset:offset + n_tokens] + emb
        return all_tokens

    def forward(
        self,
        y_noisy: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict flow marching velocities for all frames.

        Args:
            y_noisy: [B, T, C, H, W] noisy latent frames (T=4)
            t: [B, T] flow time for each frame
            h: [B, D] latent state from previous physical timestep (h_{s-1})
              used as conditioning for all frames

        Returns:
            velocities: [B, total_tokens, C] predicted velocity patches
            h_new: [B, D] updated GRU state after processing all frames
        """
        B = y_noisy.shape[0]
        T = y_noisy.shape[1]

        # Update GRU state: process each frame sequentially
        h_list = []
        h_curr = h
        for s in range(T):
            h_curr = self.gru(h_curr, y_noisy[:, s], t[:, s])
            h_list.append(h_curr)
        h_new = h_list[-1]

        # Build temporal pyramid for all frames
        pyramid_tokens = build_temporal_pyramid(y_noisy, self.pyramid_ratios)
        all_tokens = torch.cat(pyramid_tokens, dim=1)  # [B, total_tokens, C]
        all_tokens = self._add_frame_conditioning(all_tokens, list(range(T)))

        # Conditioning: use h (input condition, not the updated one)
        # Following Eq. 12: condition on h_{s-1}
        h_cond_proj = self.h_proj(h)

        # Use average t across frames for the SiT input
        t_avg = t.mean(dim=1)  # [B]

        velocities = self.sit(all_tokens, t_avg, h_cond_proj)
        velocities = velocities.reshape(B, all_tokens.shape[1], self.latent_channels)

        return velocities, h_new


# ---------------------------------------------------------------------------
# Flow Marching Utilities
# ---------------------------------------------------------------------------

def sample_x_t_k(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """Flow marching interpolation kernel.

    x_t^k = t*x_1 + k*(1-t)*x_0 + (1-t)*(1-k)*z, z ~ N(0, I)

    Args:
        x0: [B, ...] current state
        x1: [B, ...] next state
        t: [B, ...] or scalar flow time in [0, 1]
        k: bridge parameter: 1 = deterministic, 0 = stochastic

    Returns:
        x_t_k: interpolated noisy state
    """
    z = torch.randn_like(x0)
    mu = t * x1 + k * (1 - t) * x0
    sigma = (1 - t) * (1 - k)
    return mu + sigma * z


def flow_marching_loss_fn(
    velocity_pred: torch.Tensor,
    x_t_k: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Stabilized flow marching loss with (1-t) preconditioning.

    L = 0.5 * E[||(1-t) * g - (x1 - x_t^k)||^2]

    Args:
        velocity_pred: [B, ...] predicted velocity field
        x_t_k: [B, ...] current noisy state
        x1: [B, ...] target clean state
        t: [B, ...] or scalar time

    Returns:
        scalar loss
    """
    target = x1 - x_t_k
    weight = (1 - t).clamp(min=1e-3)
    diff = weight * velocity_pred - target
    return 0.5 * (diff ** 2).mean()


def euler_sampler_fmt(
    fmt: FMT,
    y_history: torch.Tensor,
    h_cond: torch.Tensor,
    k_values: torch.Tensor,
    num_steps: int = 100,
    dt: float = 0.01,
) -> torch.Tensor:
    """Euler ODE sampler for FMT: simultaneously propagates 4 frames from t=0 to t=1.

    The FMT sees all 4 frames (with specified k values) and predicts velocities
    for all frames at each step. Only the last frame (index 3) is typically the
    unknown future state.

    For deterministic prediction (k=[1,1,1,1]): all frames stay at their clean values.
    For generation (k=[1,1,1,k3<1]): frame 3 is noised, others are clean.

    Args:
        fmt: Flow Marching Transformer
        y_history: [B, 4, C, H, W] latent frames (clean for prediction)
        h_cond: [B, D] GRU conditioning state (from frames 0,1,2)
        k_values: [4] bridge parameters per frame
        num_steps: number of Euler steps
        dt: step size

    Returns:
        y_pred: [B, 4, C, H, W] predicted latents at t=1
    """
    B, T, C, H, W = y_history.shape
    device = y_history.device

    # Set up time vector: all frames start at t=0
    t_cur = torch.zeros(B, T, device=device)

    # k_values for noise level
    k_tensor = k_values.to(device).view(1, T, 1, 1, 1)

    # Build initial noisy states: y_t^k at t=0
    y_tk = y_history.clone()
    for i in range(T):
        ki = k_values[i].item()
        if ki < 1.0:
            z = torch.randn_like(y_history[:, i])
            # At t=0: y_t^k = k*y + (1-k)*z
            y_tk[:, i] = ki * y_history[:, i] + (1 - ki) * z

    for step in range(num_steps):
        if t_cur.max() >= 1.0 - 1e-6:
            break

        # Predict velocities for all frames
        velocities, _ = fmt(y_tk, t_cur, h_cond)
        velocity_frames = rearrange_velocities_to_frames(
            velocities, fmt.pyramid_ratios, fmt.latent_spatial_size, C
        )  # [B, 4, C, H, W]

        # Euler step: dy/dt = g(y, t, h)
        for i in range(T):
            weight = (1 - t_cur[:, i:i+1].view(B, 1, 1, 1)).clamp(min=1e-6)
            y_tk[:, i] = y_tk[:, i] + velocity_frames[:, i] * dt / weight

        t_cur = t_cur + dt

    return y_tk


def rearrange_velocities_to_frames(
    velocities: torch.Tensor,
    pyramid_ratios: Tuple[int, ...],
    latent_spatial_size: int,
    latent_channels: int,
) -> torch.Tensor:
    """Rearrange flattened velocity tokens back to per-frame spatial grid.

    Args:
        velocities: [B, total_tokens, C] flattened velocity predictions
        pyramid_ratios: downsampling ratios (e.g., (8,4,2,1))
        latent_spatial_size: full resolution H=W value
        latent_channels: C

    Returns:
        frames: [B, 4, C, H, W] Full-resolution velocity fields (upsampled from pyramids)
    """
    B = velocities.shape[0]
    frames = []
    offset = 0
    for ratio in pyramid_ratios:
        hw = latent_spatial_size // ratio
        n = hw * hw
        vel_s = velocities[:, offset:offset + n]  # [B, N, C]
        vel_s = vel_s.permute(0, 2, 1).reshape(B, latent_channels, hw, hw)
        if ratio > 1:
            vel_s = F.interpolate(vel_s, size=(latent_spatial_size, latent_spatial_size), mode="bilinear", align_corners=False)
        frames.append(vel_s)
        offset += n
    return torch.stack(frames, dim=1)  # [B, 4, C, H, W]


def autoregressive_predict(
    p2vae: nn.Module,
    fmt: FMT,
    x_history: torch.Tensor,
    num_steps: int = 100,
    dt: float = 0.01,
    device: str = "cuda",
) -> torch.Tensor:
    """Predict next state from clean history using FMT flow marching.

    Args:
        p2vae: Pretrained VAE for encoding/decoding
        fmt: Flow Marching Transformer
        x_history: [4, C, H, W] clean past states
        num_steps: Euler ODE steps
        dt: step size

    Returns:
        x_next: [1, C, H, W] predicted next state
    """
    x_in = x_history.unsqueeze(0).to(device)  # [1, 4, C, H, W]
    B = 1

    with torch.no_grad():
        mu_list = []
        for s in range(4):
            mu_s, _ = p2vae.encode(x_in[:, s])
            mu_list.append(mu_s)
        y_history = torch.stack(mu_list, dim=1)  # [1, 4, C_lat, H_lat, W_lat]

    # Build GRU condition from clean history (frames 0,1,2) at t=0, k=1
    h = torch.zeros(B, fmt.gru.hidden_dim, device=device)
    for s in range(3):
        h = fmt.gru(h, y_history[:, s], torch.zeros(B, device=device))

    # Deterministic prediction: k=[1,1,1,1]
    k_values = torch.tensor([1.0, 1.0, 1.0, 1.0])

    y_pred = euler_sampler_fmt(fmt, y_history, h, k_values, num_steps, dt)

    with torch.no_grad():
        x_next = p2vae.decode(y_pred[:, 3])  # last frame is the predicted next state

    return x_next


def autoregressive_rollout(
    p2vae: nn.Module,
    fmt: FMT,
    x_init: torch.Tensor,
    num_rollout_steps: int = 10,
    num_sampling_steps: int = 100,
    dt: float = 0.01,
    device: str = "cuda",
) -> torch.Tensor:
    """Long-term autoregressive rollout.

    Args:
        p2vae: P2VAE model
        fmt: FMT model
        x_init: [4, C, H, W] initial 4 frames
        num_rollout_steps: number of future steps to predict
        num_sampling_steps: Euler ODE steps per prediction
        dt: step size

    Returns:
        predictions: [num_rollout_steps, C, H, W] predicted future states
    """
    window = [x_init[i].cpu() for i in range(4)]
    predictions = []

    for _ in range(num_rollout_steps):
        x_hist = torch.stack(window[-4:])
        x_next = autoregressive_predict(p2vae, fmt, x_hist, num_sampling_steps, dt, device)
        x_next_cpu = x_next.squeeze(0).cpu()
        predictions.append(x_next_cpu)
        window.append(x_next_cpu)

    return torch.stack(predictions)


def generate_ensemble(
    p2vae: nn.Module,
    fmt: FMT,
    x_history: torch.Tensor,
    k3: float = 0.5,
    ensemble_size: int = 32,
    num_steps: int = 100,
    dt: float = 0.01,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate ensemble of next states by varying k3 noise.

    Args:
        p2vae: P2VAE model
        fmt: FMT model
        x_history: [4, C, H, W] clean past states
        k3: bridge parameter for frame 3
        ensemble_size: number of ensemble members
        num_steps: Euler ODE steps
        dt: step size

    Returns:
        ensemble: [ensemble_size, C, H, W] generated states
    """
    x_in = x_history.unsqueeze(0).to(device)

    with torch.no_grad():
        mu_list = []
        for s in range(4):
            mu_s, _ = p2vae.encode(x_in[:, s])
            mu_list.append(mu_s)
        y_history = torch.stack(mu_list, dim=1)

    # GRU condition from clean history
    h = torch.zeros(1, fmt.gru.hidden_dim, device=device)
    for s in range(3):
        h = fmt.gru(h, y_history[:, s], torch.zeros(1, device=device))

    k_values = torch.tensor([1.0, 1.0, 1.0, k3])

    ensemble = []
    for _ in range(ensemble_size):
        y_pred = euler_sampler_fmt(fmt, y_history, h, k_values, num_steps, dt)
        with torch.no_grad():
            x_pred = p2vae.decode(y_pred[:, 3])
        ensemble.append(x_pred)

    return torch.cat(ensemble, dim=0)


# ---------------------------------------------------------------------------
# Legacy simplified interface
# ---------------------------------------------------------------------------

def euler_sampler_single(
    fmt: FMT,
    x_t: torch.Tensor,
    t_start: float,
    h: torch.Tensor,
    num_steps: int = 100,
    dt: float = 0.01,
) -> torch.Tensor:
    """Simplified Euler ODE sampler for single-frame flow marching.

    Pads the single frame into a 4-frame sequence where frames 0-2 are
    zeroed and only frame 3 receives the actual noisy state. This is
    compatible with FMT's multi-frame interface.

    Args:
        fmt: FMT model
        x_t: [B, C, H, W] initial noisy state
        t_start: initial time
        h: [B, D] conditioning hidden state
        num_steps: number of Euler steps
        dt: step size

    Returns:
        x_1: [B, C, H, W] predicted clean state
    """
    B, C, H, W = x_t.shape
    device = x_t.device

    y_noisy = torch.zeros(B, 4, C, H, W, device=device)
    y_noisy[:, 3] = x_t

    t = torch.zeros(B, 4, device=device)
    t[:, 3] = t_start

    for step in range(num_steps):
        if t.max() >= 1.0:
            break
        velocities, _ = fmt(y_noisy, t, h)
        frame_vels = rearrange_velocities_to_frames(
            velocities, fmt.pyramid_ratios, H, C
        )
        y_noisy = y_noisy + frame_vels * dt
        t = t + dt
        t = t.clamp(max=1.0)

    return y_noisy[:, 3]
