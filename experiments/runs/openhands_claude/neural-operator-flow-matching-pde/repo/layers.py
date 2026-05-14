import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama-2 style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Feed-forward networks
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Llama-2 style).

    Hidden dim is set to 2/3 * (mlp_ratio * dim) rounded to nearest multiple
    of 256 for hardware efficiency, matching common practice.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(2 * mlp_ratio * dim / 3)
        # Round to multiple of 256
        hidden = ((hidden + 255) // 256) * 256
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional FlashAttention v2.

    Falls back to scaled dot-product attention when flash_attn is unavailable.
    head_dim is fixed at 64 per the paper.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.use_flash = use_flash

        self._flash_available = False
        if use_flash:
            try:
                from flash_attn import flash_attn_qkvpacked_func  # noqa: F401
                self._flash_available = True
            except ImportError:
                pass

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if self._flash_available and mask is None and x.is_cuda:
            from flash_attn import flash_attn_qkvpacked_func
            qkv_fa = qkv.to(torch.bfloat16)
            out = flash_attn_qkvpacked_func(
                qkv_fa,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            out = out.to(x.dtype).reshape(B, N, C)
        else:
            q, k, v = qkv.unbind(2)  # each (B, N, H, D)
            q = q.transpose(1, 2)    # (B, H, N, D)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )
            out = out.transpose(1, 2).reshape(B, N, C)

        return self.proj(out)


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention for compressing a sequence to a query set."""

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert query_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(query_dim, query_dim, bias=False)
        self.k = nn.Linear(kv_dim, query_dim, bias=False)
        self.v = nn.Linear(kv_dim, query_dim, bias=False)
        self.proj = nn.Linear(query_dim, query_dim, bias=False)
        self.dropout = dropout

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Nq, _ = query.shape
        B, Nkv, _ = context.shape

        q = self.q(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.proj(out)


# ---------------------------------------------------------------------------
# AdaLN-Zero conditioning (DiT / SiT style)
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    """Adaptive Layer Norm Zero modulation.

    Projects a condition vector to (shift, scale, gate) × 2 (for attn + ffn).
    The final linear is zero-initialized so the block starts as identity.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor):
        # c: (B, cond_dim)
        out = self.linear(self.silu(c))  # (B, 6*dim)
        return out.chunk(6, dim=-1)      # 6 × (B, dim)


# ---------------------------------------------------------------------------
# Timestep / noise-level sinusoidal embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by two-layer MLP."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) values in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Spatial downsampling / upsampling for temporal pyramids
# ---------------------------------------------------------------------------

class SpatialDownsample(nn.Module):
    """Average-pool a 2D latent grid by an integer factor."""

    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        if self.factor == 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.factor, stride=self.factor)


class SpatialUpsample(nn.Module):
    """Bilinear upsample a 2D latent grid by an integer factor."""

    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1:
            return x
        return F.interpolate(x, scale_factor=self.factor, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 2D sinusoidal positional embedding
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings.

    Returns:
        pos_embed: (grid_size*grid_size, embed_dim)
    """
    assert embed_dim % 4 == 0
    half = embed_dim // 2
    quarter = embed_dim // 4

    omega = torch.arange(quarter, dtype=torch.float32) / quarter
    omega = 1.0 / (10000 ** omega)

    grid = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(grid, grid, indexing="ij")
    grid_y = grid_y.reshape(-1)  # (H*W,)
    grid_x = grid_x.reshape(-1)

    emb_y = torch.outer(grid_y, omega)  # (H*W, quarter)
    emb_x = torch.outer(grid_x, omega)

    pos_embed = torch.cat([
        torch.sin(emb_y), torch.cos(emb_y),
        torch.sin(emb_x), torch.cos(emb_x),
    ], dim=-1)  # (H*W, embed_dim)
    return pos_embed


# ---------------------------------------------------------------------------
# VAE-specific layers
# ---------------------------------------------------------------------------

class GroupNorm32(nn.GroupNorm):
    """GroupNorm with 32 groups, cast to float32 for stability."""

    def __init__(self, channels: int):
        super().__init__(num_groups=32, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class ResBlock(nn.Module):
    """Residual block used in the VAE encoder/decoder (SD-VAE style)."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = GroupNorm32(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = GroupNorm32(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock2D(nn.Module):
    """Single-head self-attention block for VAE (SD-VAE style)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = GroupNorm32(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W).permute(0, 1, 3, 2)
        q, k, v = qkv.unbind(1)  # each (B, H*W, C)
        scale = C ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample2D(nn.Module):
    """Strided convolution downsampling (factor 2)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    """Nearest-neighbor upsample + conv (factor 2)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
