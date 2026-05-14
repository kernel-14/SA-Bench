from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layers import (
    AdaLNZero,
    AttentionBlock2D,
    Downsample2D,
    GroupNorm32,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    ResBlock,
    RMSNorm,
    SpatialDownsample,
    SpatialUpsample,
    SwiGLU,
    TimestepEmbedding,
    Upsample2D,
    get_2d_sincos_pos_embed,
)


# ---------------------------------------------------------------------------
# P2VAE encoder
# ---------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    """SD-VAE style encoder: 3×128×128 → 32×16×16 (mean + logvar, 16 ch each).

    Architecture:
        Conv → [ResBlocks + optional Attn + Downsample] × n_stages → ResBlocks → Conv
    """

    def __init__(
        self,
        in_channels: int,
        base_dim: int,
        channel_mult: Tuple[int, ...],
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        z_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        ch = base_dim
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        curr_res = 128
        in_ch = ch
        for i, mult in enumerate(channel_mult):
            out_ch = base_dim * mult
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
                if curr_res in attn_resolutions:
                    block.append(AttentionBlock2D(in_ch))
            self.down_blocks.append(block)
            if i < len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample2D(in_ch)]))
                curr_res //= 2

        # Middle
        mid_ch = base_dim * channel_mult[-1]
        self.mid_res1 = ResBlock(mid_ch, mid_ch, dropout)
        self.mid_attn = AttentionBlock2D(mid_ch)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, dropout)

        self.norm_out = GroupNorm32(mid_ch)
        self.conv_out = nn.Conv2d(mid_ch, 2 * z_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.down_blocks:
            for layer in block:
                h = layer(h)
        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h  # (B, 2*z_channels, H_lat, W_lat)


# ---------------------------------------------------------------------------
# P2VAE decoder
# ---------------------------------------------------------------------------

class VAEDecoder(nn.Module):
    """SD-VAE style decoder: 16×16×16 → 3×128×128."""

    def __init__(
        self,
        out_channels: int,
        base_dim: int,
        channel_mult: Tuple[int, ...],
        num_res_blocks: int,
        attn_resolutions: Tuple[int, ...],
        z_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        rev_mult = list(reversed(channel_mult))
        mid_ch = base_dim * channel_mult[-1]

        self.conv_in = nn.Conv2d(z_channels, mid_ch, 3, padding=1)

        self.mid_res1 = ResBlock(mid_ch, mid_ch, dropout)
        self.mid_attn = AttentionBlock2D(mid_ch)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, dropout)

        self.up_blocks = nn.ModuleList()
        in_ch = mid_ch
        curr_res = 16
        for i, mult in enumerate(rev_mult):
            out_ch = base_dim * mult
            block = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                block.append(ResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
                if curr_res in attn_resolutions:
                    block.append(AttentionBlock2D(in_ch))
            self.up_blocks.append(block)
            if i < len(rev_mult) - 1:
                self.up_blocks.append(nn.ModuleList([Upsample2D(in_ch)]))
                curr_res *= 2

        self.norm_out = GroupNorm32(in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)
        for block in self.up_blocks:
            for layer in block:
                h = layer(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return h


# ---------------------------------------------------------------------------
# SiT block (Scalable Interpolant Transformer)
# ---------------------------------------------------------------------------

class SiTBlock(nn.Module):
    """Single SiT block: AdaLN-Zero + MHSA + AdaLN-Zero + SwiGLU FFN.

    Conditioning vector c is projected to (shift1, scale1, gate1, shift2, scale2, gate2)
    via AdaLN-Zero (zero-initialized final linear).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout, use_flash)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, mlp_ratio, dropout)
        self.adaLN = AdaLNZero(dim, cond_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: (B, cond_dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c)
        # Broadcast over token dimension
        shift1 = shift1.unsqueeze(1)
        scale1 = scale1.unsqueeze(1)
        gate1 = gate1.unsqueeze(1)
        shift2 = shift2.unsqueeze(1)
        scale2 = scale2.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)

        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h)
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.ffn(h)
        return x


# ---------------------------------------------------------------------------
# Latent temporal pyramid: patch embedding + positional encoding
# ---------------------------------------------------------------------------

class PyramidPatchEmbed(nn.Module):
    """Embed one pyramid level: downsample latent → flatten → linear projection.

    Args:
        latent_channels: channels in the latent (16)
        latent_spatial:  full-resolution latent size (16)
        factor:          spatial downsampling factor (1, 2, 4, or 8)
        embed_dim:       output token dimension
    """

    def __init__(
        self,
        latent_channels: int,
        latent_spatial: int,
        factor: int,
        embed_dim: int,
    ):
        super().__init__()
        self.factor = factor
        self.grid_size = latent_spatial // factor  # tokens per side
        self.n_tokens = self.grid_size ** 2
        token_dim = latent_channels  # each spatial position is one token

        self.downsample = SpatialDownsample(factor)
        self.proj = nn.Linear(token_dim, embed_dim)

        # Fixed 2D sinusoidal positional embedding
        pos = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        self.register_buffer("pos_embed", pos.unsqueeze(0))  # (1, n_tokens, embed_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, C, H, W) latent
        y_down = self.downsample(y)                          # (B, C, H/f, W/f)
        tokens = rearrange(y_down, "b c h w -> b (h w) c")  # (B, n_tokens, C)
        tokens = self.proj(tokens)                           # (B, n_tokens, embed_dim)
        return tokens + self.pos_embed


class PyramidOutputHead(nn.Module):
    """Project output tokens back to latent velocity at full resolution.

    For frames with factor > 1, upsample the predicted velocity map.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_channels: int,
        latent_spatial: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        self.grid_size = latent_spatial // factor
        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial

        self.norm = RMSNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, latent_channels)
        self.upsample = SpatialUpsample(factor)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, n_tokens, embed_dim)
        B = tokens.shape[0]
        h = self.proj(self.norm(tokens))                     # (B, n_tokens, C)
        h = rearrange(
            h, "b (h w) c -> b c h w",
            h=self.grid_size, w=self.grid_size,
        )                                                    # (B, C, H/f, W/f)
        h = self.upsample(h)                                 # (B, C, H, W)
        return h


# ---------------------------------------------------------------------------
# Diffusion forcing: GRU + cross-attention state compressor
# ---------------------------------------------------------------------------

class DiffusionForcingGRU(nn.Module):
    """Causal latent state tracker for conditional flow marching.

    At each physical timestep s:
      1. Compress the noisy latent y_{s,t}^k to a single token via cross-attention.
      2. Update hidden state h_s = GRU(h_{s-1}, compressed_token).

    The hidden state h_{s-1} is used as the condition for predicting the
    velocity at step s.
    """

    def __init__(
        self,
        embed_dim: int,
        latent_channels: int,
        latent_spatial: int,
        n_cross_attn_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        n_tokens = latent_spatial ** 2

        # Project latent tokens to embed_dim for cross-attention
        self.token_proj = nn.Linear(latent_channels, embed_dim)

        # Learnable query token for cross-attention compression
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.cross_attn = MultiHeadCrossAttention(
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=n_cross_attn_heads,
        )

        self.gru = nn.GRUCell(embed_dim, embed_dim)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.embed_dim, device=device)

    def forward(
        self,
        h_prev: torch.Tensor,
        y_noisy: torch.Tensor,
    ) -> torch.Tensor:
        """Update hidden state given previous hidden state and noisy latent.

        Args:
            h_prev:   (B, embed_dim) previous hidden state
            y_noisy:  (B, C, H, W) noisy latent at current step

        Returns:
            h_new: (B, embed_dim) updated hidden state
        """
        B = y_noisy.shape[0]
        # Flatten spatial dims and project to embed_dim
        tokens = rearrange(y_noisy, "b c h w -> b (h w) c")  # (B, H*W, C)
        tokens = self.token_proj(tokens)                       # (B, H*W, embed_dim)

        # Compress to single token via cross-attention
        query = self.query_token.expand(B, -1, -1)             # (B, 1, embed_dim)
        compressed = self.cross_attn(query, tokens)            # (B, 1, embed_dim)
        compressed = compressed.squeeze(1)                     # (B, embed_dim)

        h_new = self.gru(compressed, h_prev)
        return h_new


# ---------------------------------------------------------------------------
# Condition MLP: combine timestep embedding and GRU hidden state
# ---------------------------------------------------------------------------

class ConditionMLP(nn.Module):
    """Fuse timestep embedding and GRU hidden state into a single condition vector."""

    def __init__(self, time_embed_dim: int, gru_dim: int, out_dim: int):
        super().__init__()
        self.time_embed = TimestepEmbedding(time_embed_dim, out_dim)
        self.h_proj = nn.Linear(gru_dim, out_dim)
        self.norm = RMSNorm(out_dim)

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # t: (B,), h: (B, gru_dim)
        t_emb = self.time_embed(t)   # (B, out_dim)
        h_emb = self.h_proj(h)       # (B, out_dim)
        return self.norm(t_emb + h_emb)
