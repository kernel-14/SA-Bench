"""Image encoder for SAM 2: wraps Hiera with FPN to produce multi-scale features.

- FPN fuses stride 16 (Stage 3) and stride 32 (Stage 4) features -> image embeddings
- Stride 4 (Stage 1) and stride 8 (Stage 2) features are passed through skip connections
  to the mask decoder for high-resolution detail.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HieraConfig
from hiera import Hiera, create_hiera


class FPN(nn.Module):
    """Feature Pyramid Network to fuse multi-scale Hiera features."""
    def __init__(self, in_dims: List[int], out_dim: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_dim in in_dims:
            lateral = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
            output = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)
            self.lateral_convs.append(lateral)
            self.output_convs.append(output)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, features: List[torch.Tensor], feature_spatial_sizes: List[Tuple[int, int]]) -> torch.Tensor:
        """Fuse features from different stages.

        Args:
            features: List of feature maps in spatial format [B, C, H_i, W_i]
            feature_spatial_sizes: List of (H_i, W_i) for each feature

        Returns:
            Fused feature as flat tokens [B, N, C]
        """
        resized_features = []
        for feat, lateral_conv, output_conv in zip(features, self.lateral_convs, self.output_convs):
            lat = lateral_conv(feat)
            out = output_conv(lat)
            resized_features.append(out)

        # Top-down fusion (upsample smaller features and add)
        fused = resized_features[-1]
        for feat in reversed(resized_features[:-1]):
            fused = F.interpolate(fused, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            fused = fused + feat

        # Final output conv
        fused = self.output_convs[-1](fused)

        # Reshape to token format
        B, C, H, W = fused.shape
        fused = fused.permute(0, 2, 3, 1).reshape(B, H * W, C)
        fused = self.norm(fused)
        return fused


class ImageEncoder(nn.Module):
    """SAM 2 image encoder: Hiera backbone + FPN + skip connections.

    Outputs:
        - image_embeddings: FPN-fused features from stages 3-4 [B, N_16, 256]
        - high_res_features: list of stride 4 and 8 features for mask decoder skip connections
    """
    def __init__(self, config: HieraConfig):
        super().__init__()
        self.config = config
        self.hiera = create_hiera(config)

        # FPN to fuse stage 3 (stride 16) and stage 4 (stride 32) features
        stage3_dim = self.hiera.embed_dims[2]  # stride 16
        stage4_dim = self.hiera.embed_dims[3]  # stride 32
        self.fpn = FPN(
            in_dims=[stage3_dim, stage4_dim],
            out_dim=config.fpn_output_dim,
        )

        self.fpn_output_dim = config.fpn_output_dim

        # Projections for high-res skip connections (stage 1 stride 4, stage 2 stride 8)
        self.high_res_proj = nn.ModuleList([
            nn.Conv2d(self.hiera.embed_dims[0], config.fpn_output_dim, kernel_size=1, bias=False),
            nn.Conv2d(self.hiera.embed_dims[1], config.fpn_output_dim, kernel_size=1, bias=False),
        ])

        self.image_size = config.image_size

    def _to_spatial(self, x: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        """Convert flat tokens to spatial format."""
        B, N, C = x.shape
        H, W = spatial_size
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            image_embeddings: [B, N_16, 256] FPN-fused features
            high_res_features: List of [B, C, H/4, W/4] and [B, C, H/8, W/8]
        """
        B, _, H, W = x.shape

        # Hiera forward: get features from all 4 stages
        stage_features = self.hiera(x)

        strides = [4, 8, 16, 32]
        spatial_sizes = [(H // s, W // s) for s in strides]
        stage_spatial = []

        for feat, (sh, sw) in zip(stage_features, spatial_sizes):
            stage_spatial.append(self._to_spatial(feat, (sh, sw)))

        # High-res skip features (stages 1 and 2, strides 4 and 8)
        high_res_features = [
            self.high_res_proj[0](stage_spatial[0]),  # stride 4
            self.high_res_proj[1](stage_spatial[1]),  # stride 8
        ]

        # FPN fuse stages 3 and 4 (strides 16 and 32)
        fpn_features = [stage_spatial[2], stage_spatial[3]]
        fpn_sizes = [spatial_sizes[2], spatial_sizes[3]]
        image_embeddings = self.fpn(fpn_features, fpn_sizes)

        return image_embeddings, high_res_features
