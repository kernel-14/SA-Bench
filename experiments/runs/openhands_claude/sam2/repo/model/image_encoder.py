"""
Hiera image encoder for SAM 2.

Architecture (Ryali et al., 2023; Bolya et al., 2023):
- Hierarchical Vision Transformer with 4 stages.
- Windowed attention in most layers; global attention in a subset of layers.
- Absolute positional embeddings (no RPB); interpolated to span across windows.
- MAE pre-training initialization.
- FPN head fuses Stage 3 (stride 16) and Stage 4 (stride 32) features.
- Stage 1 (stride 4) and Stage 2 (stride 8) features passed as skip connections
  to the mask decoder.

Encoder variants (Table 12):
  T:  global attn blocks at positions 5, 7, 9
  S:  global attn blocks at positions 7, 10, 13
  B+: global attn blocks at positions 12, 16, 20
  L:  global attn blocks at positions 23, 33, 43
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import DropPath, LayerNorm2d, window_partition, window_unpartition


# ---------------------------------------------------------------------------
# Hiera block components
# ---------------------------------------------------------------------------

class HieraAttention(nn.Module):
    """Multi-head attention for Hiera, supporting windowed and global modes."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.attn_drop > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_drop)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HieraBlock(nn.Module):
    """Single Hiera transformer block with optional windowed attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        window_size: int = 0,  # 0 = global attention
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = HieraAttention(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, H, W, C)
        shortcut = x
        x = self.norm1(x)
        H, W = x.shape[1], x.shape[2]

        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)
            x = x.view(-1, self.window_size * self.window_size, x.shape[-1])
            x = self.attn(x)
            x = x.view(-1, self.window_size, self.window_size, shortcut.shape[-1])
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = x.view(x.shape[0], H * W, x.shape[-1])
            x = self.attn(x)
            x = x.view(x.shape[0], H, W, shortcut.shape[-1])

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HieraDownsampleBlock(nn.Module):
    """Pooling block that downsamples spatial resolution between Hiera stages."""

    def __init__(self, in_dim: int, out_dim: int, stride: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, H, W, C)
        x = self.norm(x)
        B, H, W, C = x.shape
        # Pool by taking every stride-th element
        x = x[:, ::self.stride, ::self.stride, :]
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# Hiera encoder
# ---------------------------------------------------------------------------

# Stage configurations: (num_blocks, embed_dim, num_heads, window_size)
HIERA_CONFIGS = {
    "T": {
        "embed_dim": 96,
        "num_heads": [1, 2, 4, 8],
        "depths": [2, 3, 16, 3],
        "window_size": 8,
        "global_attn_blocks": [5, 7, 9],
        "drop_path_rate": 0.1,
    },
    "S": {
        "embed_dim": 96,
        "num_heads": [1, 2, 4, 8],
        "depths": [2, 3, 16, 3],
        "window_size": 8,
        "global_attn_blocks": [7, 10, 13],
        "drop_path_rate": 0.1,
    },
    "B+": {
        "embed_dim": 112,
        "num_heads": [2, 4, 8, 16],
        "depths": [2, 3, 16, 3],
        "window_size": 8,
        "global_attn_blocks": [12, 16, 20],
        "drop_path_rate": 0.2,
    },
    "L": {
        "embed_dim": 144,
        "num_heads": [2, 4, 8, 16],
        "depths": [2, 6, 36, 4],
        "window_size": 8,
        "global_attn_blocks": [23, 33, 43],
        "drop_path_rate": 0.3,
    },
}


class HieraPatchEmbed(nn.Module):
    """Patch embedding: conv with stride 4 (stride-4 features = Stage 1 input)."""

    def __init__(self, in_chans: int = 3, embed_dim: int = 96, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # (B, C, H/4, W/4)
        x = x.permute(0, 2, 3, 1)  # (B, H/4, W/4, C)
        x = self.norm(x)
        return x


class HieraImageEncoder(nn.Module):
    """
    Hiera hierarchical image encoder for SAM 2.

    Produces 4 feature maps at strides 4, 8, 16, 32.
    FPN fuses stride-16 and stride-32 → frame embedding for memory attention.
    Stride-4 and stride-8 features are skip connections for the mask decoder.
    """

    def __init__(
        self,
        img_size: int = 1024,
        in_chans: int = 3,
        variant: str = "B+",
        out_chans: int = 256,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        cfg = HIERA_CONFIGS[variant]
        embed_dim: int = cfg["embed_dim"]
        depths: List[int] = cfg["depths"]
        num_heads: List[int] = cfg["num_heads"]
        window_size: int = cfg["window_size"]
        global_attn_blocks: List[int] = cfg["global_attn_blocks"]
        drop_path_rate: float = cfg["drop_path_rate"]

        self.img_size = img_size
        self.patch_embed = HieraPatchEmbed(in_chans, embed_dim, patch_size=4)

        # Absolute positional embedding (interpolated to span windows)
        num_patches = (img_size // 4) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size // 4, img_size // 4, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Build stages
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        # Stage channel dims: embed_dim * [1, 2, 4, 8]
        stage_dims = [embed_dim * (2 ** i) for i in range(4)]
        # Hiera uses a different channel progression; follow paper's head counts
        # For T/S: 96, 192, 384, 768; for B+: 112, 224, 448, 896; for L: 144, 288, 576, 1152
        stage_dims = [embed_dim * (2 ** i) for i in range(4)]

        block_idx = 0
        for stage_i, (depth, n_heads) in enumerate(zip(depths, num_heads)):
            dim = stage_dims[stage_i]
            blocks = nn.ModuleList()
            for _ in range(depth):
                # Determine if this block uses global attention
                is_global = block_idx in global_attn_blocks
                ws = 0 if is_global else window_size
                blocks.append(HieraBlock(
                    dim=dim,
                    num_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[block_idx],
                    window_size=ws,
                ))
                block_idx += 1
            self.stages.append(blocks)

            # Downsample between stages (not after last stage)
            if stage_i < 3:
                self.downsample_layers.append(
                    HieraDownsampleBlock(dim, stage_dims[stage_i + 1], stride=2)
                )

        # FPN: fuse stride-16 (stage 3) and stride-32 (stage 4) → out_chans
        self.fpn_lateral3 = nn.Conv2d(stage_dims[2], out_chans, kernel_size=1)
        self.fpn_lateral4 = nn.Conv2d(stage_dims[3], out_chans, kernel_size=1)
        self.fpn_output3 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)

        # Projection layers for skip connections (stride 4 and 8)
        self.skip_proj1 = nn.Sequential(
            nn.Conv2d(stage_dims[0], out_chans // 4, kernel_size=1),
            LayerNorm2d(out_chans // 4),
        )
        self.skip_proj2 = nn.Sequential(
            nn.Conv2d(stage_dims[1], out_chans // 2, kernel_size=1),
            LayerNorm2d(out_chans // 2),
        )

        self.out_chans = out_chans
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _interpolate_pos_embed(self, x: Tensor) -> Tensor:
        """Interpolate positional embedding to match current feature map size."""
        H, W = x.shape[1], x.shape[2]
        if self.pos_embed.shape[1] == H and self.pos_embed.shape[2] == W:
            return self.pos_embed
        pos = self.pos_embed.permute(0, 3, 1, 2)  # (1, C, H0, W0)
        pos = F.interpolate(pos, size=(H, W), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1)  # (1, H, W, C)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Returns:
            frame_embedding: (B, out_chans, H/16, W/16) — used by memory attention
            skip_features: list of [(B, out_chans//4, H/4, W/4),
                                     (B, out_chans//2, H/8, W/8)]
        """
        x = self.patch_embed(x)  # (B, H/4, W/4, C)
        x = x + self._interpolate_pos_embed(x)

        stage_outputs: List[Tensor] = []
        for stage_i, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x)
            stage_outputs.append(x)  # (B, H/stride, W/stride, C)
            if stage_i < len(self.downsample_layers):
                x = self.downsample_layers[stage_i](x)

        # stage_outputs: [stride4, stride8, stride16, stride32]
        s1, s2, s3, s4 = [o.permute(0, 3, 1, 2) for o in stage_outputs]

        # FPN: top-down pathway
        p4 = self.fpn_lateral4(s4)
        p3 = self.fpn_lateral3(s3)
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3 = self.fpn_output3(p3)  # (B, out_chans, H/16, W/16)

        # Skip connections for mask decoder
        skip1 = self.skip_proj1(s1)  # (B, out_chans//4, H/4, W/4)
        skip2 = self.skip_proj2(s2)  # (B, out_chans//2, H/8, W/8)

        return p3, [skip1, skip2]


def build_image_encoder(variant: str = "B+", img_size: int = 1024, out_chans: int = 256) -> HieraImageEncoder:
    return HieraImageEncoder(img_size=img_size, variant=variant, out_chans=out_chans)
