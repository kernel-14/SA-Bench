"""
Image Encoder for SAM 2.

Based on Hiera (hierarchical vision transformer) pre-trained with MAE.
Uses a Feature Pyramid Network (FPN) to fuse stride 16 and 32 features
from stages 3 and 4 to produce image embeddings for each frame.

Stride 4 and 8 features from stages 1 and 2 bypass memory attention
and are added to upsampling layers in the mask decoder for high-resolution details.

Key design choices (from Appendix D.1):
- Windowed absolute positional embeddings (following Bolya et al. 2023)
- No relative positional encoding (RPB removed from all image encoder layers)
- Global attention only in a subset of layers (see Table 12)
- Interpolated global positional embedding to span across windows

Image encoder sizes: T, S, B+, L
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


# ---------------------------------------------------------------------------
# Hiera Configuration
# ---------------------------------------------------------------------------

def get_hiera_config(encoder_size: str):
    """Return Hiera configuration for different encoder sizes.

    From Table 12:
      T:  global attn blocks at layers 5-7-9
      S:  global attn blocks at layers 7-10-13
      B+: global attn blocks at layers 12-16-20
      L:  global attn blocks at layers 23-33-43
    """
    configs = {
        "tiny": {
            "embed_dim": 96,
            "num_heads": 1,
            "stages": [1, 2, 7, 2],
            "stage_depths": [1, 2, 7, 2],
            "global_att_blocks": [5, 7, 9],
            "window_size": 14,
            "drop_path_rate": 0.1,
        },
        "small": {
            "embed_dim": 96,
            "num_heads": 1,
            "stages": [1, 2, 11, 2],
            "stage_depths": [1, 2, 11, 2],
            "global_att_blocks": [7, 10, 13],
            "window_size": 14,
            "drop_path_rate": 0.1,
        },
        "base_plus": {
            "embed_dim": 112,
            "num_heads": 2,
            "stages": [2, 3, 16, 3],
            "stage_depths": [2, 3, 16, 3],
            "global_att_blocks": [12, 16, 20],
            "window_size": 14,
            "drop_path_rate": 0.2,
        },
        "large": {
            "embed_dim": 144,
            "num_heads": 2,
            "stages": [2, 6, 36, 4],
            "stage_depths": [2, 6, 36, 4],
            "global_att_blocks": [23, 33, 43],
            "window_size": 14,
            "drop_path_rate": 0.3,
        },
    }
    return configs[encoder_size]


# ---------------------------------------------------------------------------
# Windowed Attention
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, window_size: int):
    """Partition input into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding) - 2D
# ---------------------------------------------------------------------------

class RoPE2D(nn.Module):
    """2D Rotary Position Embedding as used in memory attention (Appendix A.2.2)."""

    def __init__(self, dim: int, max_size: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_size = max_size
        # Precompute frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Apply 2D RoPE to input tensor.
        Args:
            x: [B, N, C] input tokens
            h, w: height and width of feature map
        """
        B, N, C = x.shape
        assert C == self.dim

        # Generate position indices
        pos_h = torch.arange(h, device=x.device).float()
        pos_w = torch.arange(w, device=x.device).float()

        # Mesh grid
        pos_h, pos_w = torch.meshgrid(pos_h, pos_w, indexing="ij")
        pos_h = pos_h.reshape(-1)
        pos_w = pos_w.reshape(-1)

        # Compute sin/cos for h and w
        freqs_h = pos_h[:, None] * self.inv_freq[None, :]
        freqs_w = pos_w[:, None] * self.inv_freq[None, :]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        # Apply rotation
        x_reshaped = x.reshape(B, h * w, C // 2, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        x_out = torch.zeros_like(x_reshaped)
        x_out[..., 0] = x1 * cos - x2 * sin
        x_out[..., 1] = x1 * sin + x2 * cos
        return x_out.reshape(B, N, C)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Transformer block with option for windowed or global attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_global_attn: bool = False,
        window_size: int = 14,
        use_rope: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_global_attn = use_global_attn
        self.window_size = window_size
        self.use_rope = use_rope

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        if use_rope:
            self.rope = RoPE2D(head_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope:
            # Apply RoPE to q and k
            q = self.rope(q.reshape(B * self.num_heads, N, -1), h, w).reshape_as(q)
            k = self.rope(k.reshape(B * self.num_heads, N, -1), h, w).reshape_as(k)

        if not self.use_global_attn:
            # Windowed attention: partition into windows
            q = q.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, C // self.num_heads, h, w)
            k = k.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, C // self.num_heads, h, w)
            v = v.permute(0, 1, 3, 4, 2).reshape(B, self.num_heads, C // self.num_heads, h, w)

            # Pad to multiples of window_size
            pad_h = (self.window_size - h % self.window_size) % self.window_size
            pad_w = (self.window_size - w % self.window_size) % self.window_size
            if pad_h > 0 or pad_w > 0:
                q = F.pad(q, (0, pad_w, 0, pad_h))
                k = F.pad(k, (0, pad_w, 0, pad_h))
                v = F.pad(v, (0, pad_w, 0, pad_h))

            hp, wp = h + pad_h, w + pad_w
            q = q.reshape(B, self.num_heads, C // self.num_heads, hp // self.window_size, self.window_size,
                          wp // self.window_size, self.window_size)
            q = q.permute(0, 3, 5, 1, 4, 6, 2).reshape(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads)
            k = k.reshape(B, self.num_heads, C // self.num_heads, hp // self.window_size, self.window_size,
                          wp // self.window_size, self.window_size)
            k = k.permute(0, 3, 5, 1, 4, 6, 2).reshape(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads)
            v = v.reshape(B, self.num_heads, C // self.num_heads, hp // self.window_size, self.window_size,
                          wp // self.window_size, self.window_size)
            v = v.permute(0, 3, 5, 1, 4, 6, 2).reshape(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads)

            # Attention
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // self.num_heads))
            attn = F.softmax(attn, dim=-1)
            x_attn = (attn @ v)

            # Reverse window partition
            x_attn = x_attn.reshape(B, hp // self.window_size, wp // self.window_size, self.num_heads,
                                     self.window_size, self.window_size, C // self.num_heads)
            x_attn = x_attn.permute(0, 3, 5, 1, 6, 2, 4).reshape(B, self.num_heads, C // self.num_heads, hp, wp)
            if pad_h > 0 or pad_w > 0:
                x_attn = x_attn[:, :, :, :h, :w]
            x_attn = x_attn.reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 3, 1, 2).reshape(B, N, C)
        else:
            # Global attention
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // self.num_heads))
            attn = F.softmax(attn, dim=-1)
            x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x_attn = self.proj(x_attn)
        x = shortcut + self.drop_path(x_attn)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Patch Embedding and Merging
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Patch embedding for Hiera image encoder."""
    def __init__(self, in_chans: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, C, H/4, W/4]
        return x


class PatchMerge(nn.Module):
    """Merge patches at the start of each Hiera stage, doubling channels."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


# ---------------------------------------------------------------------------
# Hiera Stage
# ---------------------------------------------------------------------------

class HieraStage(nn.Module):
    """One stage of the Hiera encoder with transformer blocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        global_att_blocks: List[int],
        window_size: int = 14,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_global = (i + 1) in global_att_blocks
            self.blocks.append(
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    use_global_attn=use_global,
                    window_size=window_size,
                    drop_path=drop_path,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        for block in self.blocks:
            x = block(x, H, W)
        x = x.reshape(B, H, W, C)
        return x


# ---------------------------------------------------------------------------
# FPN Neck
# ---------------------------------------------------------------------------

class FPN(nn.Module):
    """Feature Pyramid Network to fuse stride 16 (stage 3) and stride 32 (stage 4) features."""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, kernel_size=1)
            for in_c in in_channels
        ])
        # Output convolutions
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # features ordered from highest resolution to lowest
        laterals = [
            lateral_conv(feat)
            for feat, lateral_conv in zip(features, self.lateral_convs)
        ]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            target_size = laterals[i].shape[-2:]
            laterals[i] = laterals[i] + F.interpolate(laterals[i + 1], size=target_size, mode="nearest")

        # Output convolutions
        outputs = [
            output_conv(lateral)
            for lateral, output_conv in zip(laterals, self.output_convs)
        ]
        return outputs


# ---------------------------------------------------------------------------
# Full Hiera Image Encoder
# ---------------------------------------------------------------------------

class HieraImageEncoder(nn.Module):
    """
    Hiera-based image encoder for SAM 2.

    Produces:
    - image_embedding: FPN-fused features from stages 3+4 (for memory attention and mask decoder)
    - high_res_features: stride 4 and stride 8 features (skip connections for mask decoder)
    """

    def __init__(
        self,
        encoder_size: str = "base_plus",
        img_size: int = 1024,
        out_channels: int = 256,
        use_abs_pos: bool = True,
    ):
        super().__init__()
        config = get_hiera_config(encoder_size)
        embed_dim = config["embed_dim"]
        num_heads = config["num_heads"]
        stage_depths = config["stage_depths"]
        global_att_blocks = config["global_att_blocks"]
        window_size = config["window_size"]
        drop_path_rate = config["drop_path_rate"]

        self.encoder_size = encoder_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        # Stage channel dimensions (double at each stage after stage 1)
        stage_dims = [embed_dim, embed_dim, embed_dim * 2, embed_dim * 4]

        # Patch embedding
        self.patch_embed = PatchEmbed(in_chans=3, embed_dim=embed_dim, patch_size=4)

        # Stage 1 (stride 4)
        self.patch_merge1 = nn.Identity()  # No merging before stage 1
        self.stage1 = HieraStage(
            dim=stage_dims[0],
            depth=stage_depths[0],
            num_heads=num_heads,
            global_att_blocks=global_att_blocks,
            window_size=window_size,
        )

        # Stage 2 (stride 8)
        self.patch_merge2 = PatchMerge(stage_dims[0], stage_dims[1])
        self.stage2 = HieraStage(
            dim=stage_dims[1],
            depth=stage_depths[1],
            num_heads=num_heads,
            global_att_blocks=global_att_blocks,
            window_size=window_size,
        )

        # Stage 3 (stride 16)
        self.patch_merge3 = PatchMerge(stage_dims[1], stage_dims[2])
        self.stage3 = HieraStage(
            dim=stage_dims[2],
            depth=stage_depths[2],
            num_heads=num_heads,
            global_att_blocks=global_att_blocks,
            window_size=window_size,
        )

        # Stage 4 (stride 32)
        self.patch_merge4 = PatchMerge(stage_dims[2], stage_dims[3])
        self.stage4 = HieraStage(
            dim=stage_dims[3],
            depth=stage_depths[3],
            num_heads=num_heads,
            global_att_blocks=global_att_blocks,
            window_size=window_size,
        )

        # FPN to fuse stages 3 and 4
        self.fpn = FPN(
            in_channels=[stage_dims[2], stage_dims[3]],
            out_channels=out_channels,
        )

        # Projections for high-res skip features
        self.highres_proj = nn.ModuleDict({
            "stride_4": nn.Conv2d(stage_dims[0], out_channels, kernel_size=1),
            "stride_8": nn.Conv2d(stage_dims[1], out_channels, kernel_size=1),
        })

        # Absolute positional embeddings (windowed, interpolated globally)
        if use_abs_pos:
            max_h = img_size // 4  # stride 4
            max_w = img_size // 4
            self.pos_embed = nn.Parameter(torch.zeros(1, max_h, max_w, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: input images [B, 3, H, W]

        Returns:
            image_embedding: [B, C, H/16, W/16] fused features for memory attention and mask decoder
            high_res_features: list of [B, C, H/4, W/4] and [B, C, H/8, W/8] for mask decoder skip connections
        """
        B, C, H, W = x.shape

        # Patch embed
        x = self.patch_embed(x)  # [B, C, H/4, W/4]

        # Add positional embedding
        if self.pos_embed is not None:
            _, _, ph, pw = self.pos_embed.shape
            # Interpolate if needed
            if (ph, pw) != (x.shape[2], x.shape[3]):
                pos = F.interpolate(
                    self.pos_embed.permute(0, 3, 1, 2),
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                ).permute(0, 2, 3, 1)
            else:
                pos = self.pos_embed
            x = x + pos.permute(0, 3, 1, 2)

        # Stage 1: stride 4
        x = x.permute(0, 2, 3, 1)  # [B, H/4, W/4, C]
        feat1 = self.stage1(x)  # [B, H/4, W/4, C]
        skip4 = feat1.permute(0, 3, 1, 2)  # [B, C, H/4, W/4]
        skip4_proj = self.highres_proj["stride_4"](skip4)

        # Stage 2: stride 8
        feat2 = self.stage2(feat1)
        skip8 = feat2.permute(0, 3, 1, 2)  # [B, C, H/8, W/8]
        skip8_proj = self.highres_proj["stride_8"](skip8)

        # Stage 3: stride 16
        feat3_in = feat2.permute(0, 3, 1, 2)  # [B, C, H/8, W/8]
        feat3_in = self.patch_merge3(feat3_in)  # [B, 2C, H/16, W/16]
        feat3 = feat3_in.permute(0, 2, 3, 1)  # [B, H/16, W/16, 2C]
        feat3 = self.stage3(feat3)
        stage3_feat = feat3.permute(0, 3, 1, 2)  # [B, 2C, H/16, W/16]

        # Stage 4: stride 32
        feat4_in = self.patch_merge4(stage3_feat)  # [B, 4C, H/32, W/32]
        feat4 = feat4_in.permute(0, 2, 3, 1)
        feat4 = self.stage4(feat4)
        stage4_feat = feat4.permute(0, 3, 1, 2)  # [B, 4C, H/32, W/32]

        # FPN fusion of stages 3 and 4
        fpn_features = self.fpn([stage3_feat, stage4_feat])
        image_embedding = fpn_features[0]  # [B, out_channels, H/16, W/16]

        return image_embedding, [skip4_proj, skip8_proj]
