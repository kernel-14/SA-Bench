"""
Hiera image encoder for SAM 2.

Based on:
- Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles (Ryali et al., 2023)
- Window attention improvements (Bolya et al., 2023)
- MAE pre-training (He et al., 2022)

The encoder uses a hierarchical structure with 4 stages, producing multi-scale features.
Stages 1 and 2 produce stride 4 and 8 features used as skip connections in the mask decoder.
Stages 3 and 4 produce stride 16 and 32 features fused via FPN for the image embedding.
"""

import math
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition feature map into non-overlapping windows."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """Reverse window partitioning."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class PatchEmbed(nn.Module):
    """Image to patch embedding with 4x4 patches."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (7, 7),
        stride: Tuple[int, int] = (4, 4),
        padding: Tuple[int, int] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class MLP(nn.Module):
    """MLP block used in transformer layers."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HieraAttention(nn.Module):
    """Multi-head attention for Hiera, supporting windowed and global attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.window_size = window_size
        self.use_rel_pos = use_rel_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        # Window partition for local attention
        if self.window_size > 0:
            x_windows, (Hp, Wp) = window_partition(x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        else:
            x_windows = x.view(B, H * W, C)

        Bw, N, _ = x_windows.shape
        qkv = self.qkv(x_windows).reshape(Bw, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_windows = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        x_windows = self.proj(x_windows)

        # Reverse window partition
        if self.window_size > 0:
            x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
            x = window_unpartition(x_windows, self.window_size, (Hp, Wp), (H, W))
        else:
            x = x_windows.view(B, H, W, C)

        return x


class HieraBlock(nn.Module):
    """Hiera transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        window_size: int = 0,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HieraAttention(
            dim=dim,
            num_heads=num_heads,
            use_rel_pos=use_rel_pos,
            window_size=window_size,
            input_size=input_size,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class HieraDownsample(nn.Module):
    """Downsampling layer between Hiera stages using strided convolution."""

    def __init__(self, in_dim: int, out_dim: int, stride: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B H W C
        x = self.norm(x)
        # Downsample spatially
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B C H W
        # Use average pooling for downsampling
        x = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.proj(x)
        return x


class FPN(nn.Module):
    """Feature Pyramid Network to fuse multi-scale features from Hiera stages 3 and 4."""

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        features: list of [B, C, H, W] tensors from coarse to fine
        Returns fused feature at the finest scale.
        """
        # Process from coarsest to finest
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )

        # Apply output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        # Return the finest scale output
        return outputs[0]


class HieraImageEncoder(nn.Module):
    """
    Hiera hierarchical image encoder for SAM 2.

    Architecture:
    - Stage 1: stride 4, window attention
    - Stage 2: stride 8, window attention
    - Stage 3: stride 16, mix of window and global attention
    - Stage 4: stride 32, mix of window and global attention

    Stages 3 and 4 features are fused via FPN to produce image embeddings.
    Stages 1 and 2 features are used as skip connections in the mask decoder.
    """

    def __init__(
        self,
        img_size: int = 1024,
        in_chans: int = 3,
        embed_dim: int = 96,
        num_heads: int = 1,
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # B+ config
        global_attn_indexes: Tuple[int, ...] = (12, 16, 20),  # B+ config
        window_size: int = 8,
        out_chans: int = 256,
        drop_path_rate: float = 0.2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim

        # Patch embedding: 4x4 patches
        self.patch_embed = PatchEmbed(
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Absolute positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // 4, img_size // 4, embed_dim)
        )

        # Stage dimensions (each stage doubles channels)
        stage_dims = [embed_dim * (2 ** i) for i in range(len(stages))]

        # Build stages
        total_depth = sum(stages)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        block_idx = 0
        for stage_idx, (num_blocks, dim) in enumerate(zip(stages, stage_dims)):
            stage_blocks = nn.ModuleList()
            for i in range(num_blocks):
                global_idx = block_idx + i
                is_global = global_idx in global_attn_indexes
                stage_blocks.append(
                    HieraBlock(
                        dim=dim,
                        num_heads=num_heads * (2 ** stage_idx),
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[global_idx],
                        window_size=0 if is_global else window_size,
                    )
                )
            self.stages.append(stage_blocks)
            block_idx += num_blocks

            # Add downsampling between stages (not after last stage)
            if stage_idx < len(stages) - 1:
                self.downsamples.append(
                    HieraDownsample(dim, stage_dims[stage_idx + 1])
                )

        # FPN to fuse stage 3 and 4 features (stride 16 and 32)
        self.fpn = FPN(
            in_channels=[stage_dims[2], stage_dims[3]],
            out_channels=out_chans,
        )

        # Project stage 1 and 2 features for skip connections
        self.skip_proj1 = nn.Conv2d(stage_dims[0], out_chans // 4, kernel_size=1)
        self.skip_proj2 = nn.Conv2d(stage_dims[1], out_chans // 2, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            image_embedding: [B, out_chans, H/16, W/16] - main embedding for memory attention
            skip_features: list of [B, C, H/4, W/4] and [B, C, H/8, W/8] for mask decoder
        """
        # Patch embedding
        x = self.patch_embed(x)  # B H/4 W/4 C

        # Add positional embedding (interpolate if needed)
        if x.shape[1:3] != self.pos_embed.shape[1:3]:
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2),
                size=x.shape[1:3],
                mode='bicubic',
                align_corners=False,
            ).permute(0, 2, 3, 1)
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed

        stage_outputs = []

        for stage_idx, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)
            stage_outputs.append(x)

            if stage_idx < len(self.downsamples):
                x = self.downsamples[stage_idx](x)

        # stage_outputs: [stride4, stride8, stride16, stride32]
        # Convert to B C H W format
        feats = [s.permute(0, 3, 1, 2) for s in stage_outputs]

        # FPN fusion of stride 16 and 32 features
        # Pass from coarsest (stride32) to finest (stride16)
        image_embedding = self.fpn([feats[3], feats[2]])  # B out_chans H/16 W/16

        # Skip connections for mask decoder
        skip1 = self.skip_proj1(feats[0])  # B out_chans/4 H/4 W/4
        skip2 = self.skip_proj2(feats[1])  # B out_chans/2 H/8 W/8

        return image_embedding, [skip1, skip2]


# Model size configurations
HIERA_CONFIGS = {
    "tiny": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": (1, 2, 7, 2),
        "global_attn_indexes": (5, 7, 9),
        "drop_path_rate": 0.1,
    },
    "small": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": (1, 2, 11, 2),
        "global_attn_indexes": (7, 10, 13),
        "drop_path_rate": 0.1,
    },
    "base_plus": {
        "embed_dim": 112,
        "num_heads": 2,
        "stages": (2, 3, 16, 3),
        "global_attn_indexes": (12, 16, 20),
        "drop_path_rate": 0.2,
    },
    "large": {
        "embed_dim": 144,
        "num_heads": 2,
        "stages": (2, 6, 36, 4),
        "global_attn_indexes": (23, 33, 43),
        "drop_path_rate": 0.3,
    },
}


def build_hiera_encoder(model_size: str = "base_plus", img_size: int = 1024, out_chans: int = 256) -> HieraImageEncoder:
    """Build a Hiera image encoder of the specified size."""
    config = HIERA_CONFIGS[model_size]
    return HieraImageEncoder(
        img_size=img_size,
        out_chans=out_chans,
        **config,
    )
