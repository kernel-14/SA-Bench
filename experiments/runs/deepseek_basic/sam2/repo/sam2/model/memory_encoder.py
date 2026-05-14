"""
Memory Encoder for SAM 2.

The memory encoder generates a memory by:
1. Downsampling the output mask using a convolutional module
2. Summing it element-wise with the unconditioned frame embedding from the image encoder
3. Applying light-weight convolutional layers to fuse the information

(Appendix D.1):
- Reuses the image embeddings produced by the Hiera encoder (no separate image encoder)
- Fuses predicted mask information with image embeddings
- Memory features projected to dimension 64 for storage in memory bank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MemoryEncoder(nn.Module):
    """
    Memory encoder: transforms mask predictions and image embeddings into memory features.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        memory_dim: int = 64,
        mask_downsample_kernel: int = 4,
    ):
        """
        Args:
            embed_dim: input channel dimension (from image encoder)
            memory_dim: output memory channel dimension (64 by default)
            mask_downsample_kernel: kernel size for mask downsampling
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim

        # Mask downsampling convolution
        # Convolves the predicted mask to match the image embedding spatial size
        self.mask_downsample = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, kernel_size=mask_downsample_kernel, stride=mask_downsample_kernel, padding=0),
            nn.LayerNorm([embed_dim // 2, 64, 64]),  # assuming 256x256 -> 64x64 features
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
        )

        # Light-weight convolutional layers to fuse mask + image embedding
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([embed_dim, 64, 64]),  # H=W=64 at stride 16 for 1024 input
            nn.GELU(),
            nn.Conv2d(embed_dim, memory_dim, kernel_size=1),
        )

    def forward(
        self,
        mask_pred: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mask_pred: [B, 1, H_img, W_img] predicted mask at original image resolution
            image_embedding: [B, C, H, W] unconditioned image embedding from encoder

        Returns:
            memory: [B, memory_dim, H, W] memory features
        """
        B, C, H, W = image_embedding.shape

        # Downsample mask to match image embedding spatial size
        mask_down = self.mask_downsample(mask_pred)  # [B, C, H, W]

        # Element-wise sum with image embedding
        fused = mask_down + image_embedding  # [B, C, H, W]

        # Light-weight conv fusion and project to memory_dim
        memory = self.fusion(fused)  # [B, memory_dim, H, W]

        return memory
