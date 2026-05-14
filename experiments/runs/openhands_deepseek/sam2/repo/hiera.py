"""Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles.

Based on Ryali et al. (2023) and Bolya et al. (2023).
Uses MAE pre-training, windowed attention with global attention in selected layers,
and absolute position embeddings (no relative positional bias, as per SAM 2 ablation).
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HieraConfig


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> torch.Tensor:
    """Generate 2D sine-cosine positional embedding."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=-1)
    grid = grid.reshape(-1, 2)

    pos_embed = _get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Generate 1D sine-cosine positional embedding from grid."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)

    out_h = torch.einsum("hw,d->hwd", pos[:, 0], omega)
    out_w = torch.einsum("hw,d->hwd", pos[:, 1], omega)

    emb = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)], dim=-1)
    return emb


class Mlp(nn.Module):
    """MLP with GELU activation."""
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention."""
    def __init__(self, dim: int, num_heads: int, window_size: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, window_size: Optional[int] = None) -> torch.Tensor:
        B, N, C = x.shape
        ws = window_size or self.window_size

        H = W = int(math.sqrt(N))
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class GlobalAttention(nn.Module):
    """Global multi-head self-attention for cross-window communication."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class HieraBlock(nn.Module):
    """Hiera transformer block with windowed attention and optional global attention."""
    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, drop_path: float = 0.0, use_global_attn: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        if use_global_attn:
            self.attn = GlobalAttention(dim, num_heads, dropout)
        else:
            self.attn = WindowAttention(dim, num_heads, window_size, dropout)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, drop=dropout)

        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Patch embedding for Hiera, using a convolution."""
    def __init__(self, kernel_size: int = 7, stride: int = 4, padding: int = 3,
                 in_chans: int = 3, embed_dim: int = 112):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # B, H', W', C
        x = self.norm(x)
        x = x.reshape(B, -1, x.shape[-1])
        return x


class HieraStage(nn.Module):
    """A Hiera stage: patch merging + series of blocks."""
    def __init__(self, dim_in: int, dim_out: int, depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 drop_path: float = 0.0, global_att_blocks: List[int] = None,
                 is_first_stage: bool = False):
        super().__init__()
        global_att_blocks = global_att_blocks or []

        # Patch merging (downsample) for all but first stage
        if not is_first_stage:
            self.downsample = nn.Sequential(
                nn.LayerNorm(dim_in, eps=1e-6),
                nn.Linear(dim_in, 4 * dim_out),
                nn.LayerNorm(4 * dim_out, eps=1e-6),
                nn.Linear(4 * dim_out, dim_out),
            )
        else:
            self.downsample = None

        # Blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = HieraBlock(
                dim=dim_out, num_heads=num_heads, window_size=window_size,
                mlp_ratio=mlp_ratio, dropout=dropout, drop_path=drop_path,
                use_global_attn=(i in global_att_blocks),
            )
            self.blocks.append(block)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.downsample is not None:
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.reshape(B, H, W, C)

            # 2x2 pooling
            x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5)
            x = x.reshape(B, (H // 2) * (W // 2), 4 * C)

            x = self.downsample(x)

        for blk in self.blocks:
            x = blk(x)
        return x


class Hiera(nn.Module):
    """Hiera: Hierarchical Vision Transformer."""
    def __init__(self, config: HieraConfig):
        super().__init__()
        self.config = config
        self.num_stages = len(config.depths)

        embed_dims = []
        current_dim = config.embed_dim
        for i in range(self.num_stages):
            embed_dims.append(current_dim)
            current_dim *= 2

        # Patch embedding
        self.patch_embed = PatchEmbed(
            kernel_size=config.patch_kernel[0], stride=config.patch_stride[0],
            padding=config.patch_padding, in_chans=3, embed_dim=config.embed_dim,
        )

        # Positional embeddings for each stage (stored as ModuleList of Parameter)
        for i in range(self.num_stages):
            grid_size = config.image_size // (config.strides[i])
            dim = embed_dims[i]
            pos = get_2d_sincos_pos_embed(dim, grid_size)
            self.register_buffer(f"pos_embed_{i}", pos, persistent=True)

        # Build stages
        dp_rates = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()
        dp_idx = 0

        self.stages = nn.ModuleList()
        for i, depth in enumerate(config.depths):
            is_first = (i == 0)
            dim_in = config.embed_dim if i == 0 else embed_dims[i - 1]
            dim_out = embed_dims[i]

            # Convert global_att_blocks to indices within this stage
            stage_start = sum(config.depths[:i])
            stage_global = [b - stage_start for b in config.global_att_blocks
                          if stage_start <= b < stage_start + depth]

            stage = HieraStage(
                dim_in=dim_in, dim_out=dim_out, depth=depth,
                num_heads=config.num_heads[i], window_size=config.window_size,
                mlp_ratio=config.mlp_ratio, dropout=config.dropout,
                drop_path=max(dp_rates[dp_idx:dp_idx + depth]) if dp_rates else 0.0,
                global_att_blocks=stage_global,
                is_first_stage=is_first,
            )
            self.stages.append(stage)
            dp_idx += depth

        self.embed_dims = embed_dims
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

        # Special init for patch_embed
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning features from all stages.

        Returns:
            List of stage features [stage1, stage2, stage3, stage4]
            each of shape (B, N_i, C_i) where N_i = (H / stride_i) * (W / stride_i)
        """
        outputs = []

        x = self.patch_embed(x)  # Stage 0 embedding

        for i, stage in enumerate(self.stages):
            pos_embed = getattr(self, f"pos_embed_{i}")
            x = x + pos_embed.to(x.dtype)
            x = stage(x)
            outputs.append(x)

        return outputs


def create_hiera(config: HieraConfig) -> Hiera:
    """Factory function to create a Hiera model."""
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    return Hiera(config)
