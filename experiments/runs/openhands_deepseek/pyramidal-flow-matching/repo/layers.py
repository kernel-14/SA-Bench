"""Basic building blocks for MM-DiT: RoPE, attention, MLP, transformer layers."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import einops


class RotaryPositionalEmbedding(nn.Module):
    """1D Rotary Position Embedding (RoPE) for temporal dimension."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, offset: int = 0, device: torch.device = None) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype) + offset
        freqs = torch.outer(t, self.to(device).inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb), torch.sin(emb)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal position encoding for spatial dimensions."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        half_dim = self.dim // 4
        emb_h = math.log(self.max_period) / (half_dim - 1)
        emb_h = torch.exp(torch.arange(half_dim, device=device) * -emb_h)
        y = torch.arange(h, device=device).unsqueeze(1) * emb_h.unsqueeze(0)
        emb_h = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)

        emb_w = math.log(self.max_period) / (half_dim - 1)
        emb_w = torch.exp(torch.arange(half_dim, device=device) * -emb_w)
        x = torch.arange(w, device=device).unsqueeze(1) * emb_w.unsqueeze(0)
        emb_w = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        emb = torch.cat([
            emb_h.unsqueeze(1).expand(-1, w, -1),
            emb_w.unsqueeze(0).expand(h, -1, -1)
        ], dim=-1)
        return emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


class SelfAttention(nn.Module):
    """Self-attention with RoPE, QK normalization, and optional causal masking."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 32,
        head_dim: int = 64,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(dim, self.inner_dim, bias=True)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=True)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=True)
        self.o_proj = nn.Linear(self.inner_dim, dim, bias=True)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.rope = RotaryPositionalEmbedding(head_dim // 2, theta=rope_theta)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        temporal_pos_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if temporal_pos_ids is not None:
            max_id = temporal_pos_ids.max().item() + 1
            cos, sin = self.rope(max_id, device=x.device)
            gather_cos = cos[temporal_pos_ids].view(B, N, 1, self.head_dim // 4 * 2)
            gather_sin = sin[temporal_pos_ids].view(B, N, 1, self.head_dim // 4 * 2)

            q_t = q[..., : self.head_dim // 2]
            q_s = q[..., self.head_dim // 2 :]
            k_t = k[..., : self.head_dim // 2]
            k_s = k[..., self.head_dim // 2 :]

            cos_half = gather_cos[..., : self.head_dim // 4]
            sin_half = gather_sin[..., : self.head_dim // 4]

            def rotate_half_rope(x):
                x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
                return torch.cat([-x2, x1], dim=-1)

            q_t_rot = (q_t * cos_half) + (rotate_half_rope(q_t) * sin_half)
            k_t_rot = (k_t * cos_half) + (rotate_half_rope(k_t) * sin_half)

            q = torch.cat([q_t_rot, q_s], dim=-1)
            k = torch.cat([k_t_rot, k_s], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal_mask is not None:
            attn = attn + causal_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)
        out = self.o_proj(out)
        return out


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 32,
        head_dim: int = 64,
        dropout: float = 0.0,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(dim, self.inner_dim, bias=True)
        self.k_proj = nn.Linear(context_dim, self.inner_dim, bias=True)
        self.v_proj = nn.Linear(context_dim, self.inner_dim, bias=True)
        self.o_proj = nn.Linear(self.inner_dim, dim, bias=True)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        T = context.shape[1]

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(B, T, self.num_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.inner_dim)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(dim, int(dim * mult), bias=True)
        self.up = nn.Linear(dim, int(dim * mult), bias=True)
        self.down = nn.Linear(int(dim * mult), dim, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate(x))
        up = self.up(x)
        return self.down(self.dropout(gate * up))


class MMDiTBlock(nn.Module):
    """Multi-Modal DiT block with self-attention and cross-attention."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 32,
        head_dim: int = 64,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            qk_norm=qk_norm,
            rope_theta=rope_theta,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        self.ff = SwiGLUFFN(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        temporal_pos_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.norm1(x),
            causal_mask=causal_mask,
            temporal_pos_ids=temporal_pos_ids,
        )
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.ff(self.norm3(x))
        return x


class PatchEmbed(nn.Module):
    """2D patch embedding for spatial dimensions."""

    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchUnembed(nn.Module):
    """Reverse patch embedding back to 2D."""

    def __init__(self, patch_size: int, out_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        B, N, C = x.shape
        out = self.proj(x)
        out = out.view(B, h, w, self.patch_size, self.patch_size, -1)
        out = einops.rearrange(out, "b h w p1 p2 c -> b c (h p1) (w p2)")
        return out


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        half = dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)
