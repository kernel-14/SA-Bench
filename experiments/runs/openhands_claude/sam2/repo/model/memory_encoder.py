"""
Memory encoder for SAM 2.

Generates a spatial memory feature from the current frame's predicted mask
and the unconditioned image embedding from the Hiera encoder.

Architecture (Section 4, Appendix D.1):
  1. Downsample the predicted mask with a convolutional module.
  2. Sum element-wise with the unconditioned frame embedding.
  3. Apply light-weight convolutional layers to fuse the information.
  4. Project to memory_dim (64) for storage in the memory bank.

The memory encoder reuses the image embeddings from the Hiera encoder
(not a separate image encoder), allowing memory features to benefit from
the strong image representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import LayerNorm2d


class MaskDownsampler(nn.Module):
    """
    Downsamples a binary mask to match the spatial resolution of the image embedding.
    Input:  (B, 1, H_img, W_img)
    Output: (B, embed_dim, H_emb, W_emb)  where H_emb = H_img / 16
    """

    def __init__(self, embed_dim: int = 256, kernel_size: int = 4, stride: int = 4) -> None:
        super().__init__()
        # Downsample by factor of 16 (4 × 4) using two stride-4 convolutions
        # or a single stride-16 convolution; we use a cascade for better gradients.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, kernel_size=kernel_size, stride=stride, padding=0),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=kernel_size, stride=stride, padding=0),
            LayerNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, mask: Tensor) -> Tensor:
        return self.encoder(mask)


class MemoryEncoder(nn.Module):
    """
    Encodes a frame's predicted mask + image embedding into a memory feature.

    Steps:
      1. Downsample mask → (B, embed_dim, H_emb, W_emb)
      2. Element-wise sum with unconditioned image embedding
      3. Light-weight conv fusion → project to memory_dim
    """

    def __init__(
        self,
        embed_dim: int = 256,
        memory_dim: int = 64,
        num_fuse_layers: int = 2,
    ) -> None:
        super().__init__()
        self.mask_downsampler = MaskDownsampler(embed_dim=embed_dim)

        # Light-weight conv layers to fuse mask + image embedding
        fuse_layers: list[nn.Module] = []
        for i in range(num_fuse_layers):
            in_ch = embed_dim if i == 0 else embed_dim
            fuse_layers += [
                nn.Conv2d(in_ch, embed_dim, kernel_size=3, padding=1),
                LayerNorm2d(embed_dim),
                nn.GELU(),
            ]
        self.fuse = nn.Sequential(*fuse_layers)

        # Project to memory_dim for compact storage
        self.proj = nn.Conv2d(embed_dim, memory_dim, kernel_size=1)

        # Learned occlusion embedding added to memory features of occluded frames
        self.occlusion_embed = nn.Parameter(torch.zeros(1, memory_dim, 1, 1))
        nn.init.trunc_normal_(self.occlusion_embed, std=0.02)

    def forward(
        self,
        image_embedding: Tensor,
        mask_logits: Tensor,
        is_occluded: bool = False,
    ) -> Tensor:
        """
        Args:
            image_embedding: (B, embed_dim, H_emb, W_emb) — unconditioned Hiera output
            mask_logits:     (B, 1, H_img, W_img) — predicted mask logits (before sigmoid)
            is_occluded:     if True, add learned occlusion embedding to memory

        Returns:
            memory: (B, memory_dim, H_emb, W_emb)
        """
        # Convert logits to soft mask in [0, 1]
        mask = torch.sigmoid(mask_logits)

        # Resize mask to match image embedding spatial size
        H_emb, W_emb = image_embedding.shape[-2:]
        mask_resized = F.interpolate(mask, size=(H_emb * 4, W_emb * 4), mode="bilinear", align_corners=False)

        # Downsample mask to embedding resolution
        mask_emb = self.mask_downsampler(mask_resized)

        # Resize to exactly match image_embedding if needed
        if mask_emb.shape[-2:] != image_embedding.shape[-2:]:
            mask_emb = F.interpolate(mask_emb, size=image_embedding.shape[-2:], mode="bilinear", align_corners=False)

        # Fuse
        fused = image_embedding + mask_emb
        fused = self.fuse(fused)
        memory = self.proj(fused)

        if is_occluded:
            memory = memory + self.occlusion_embed

        return memory
