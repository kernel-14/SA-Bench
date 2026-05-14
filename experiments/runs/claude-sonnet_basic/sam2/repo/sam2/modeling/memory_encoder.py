"""
Memory encoder for SAM 2.

The memory encoder generates memory features from the current frame's prediction.
It fuses the predicted mask with the image encoder's frame embedding using
convolutional layers, producing compact memory representations for the memory bank.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D feature maps (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskDownsampler(nn.Module):
    """
    Downsamples the predicted mask to match the image embedding spatial size.
    Uses strided convolutions to reduce spatial resolution.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        total_stride: int = 16,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        num_layers = int(torch.log2(torch.tensor(total_stride // stride)).item()) + 1
        # Build a series of strided convolutions to downsample the mask
        layers = []
        in_ch = 1
        out_ch = embed_dim // (2 ** (num_layers - 1))
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                LayerNorm2d(out_ch),
                activation if i < num_layers - 1 else nn.Identity(),
            ])
            in_ch = out_ch
            out_ch = min(out_ch * 2, embed_dim)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MemoryEncoder(nn.Module):
    """
    Memory encoder for SAM 2.

    Generates memory features by:
    1. Downsampling the predicted mask to match image embedding size
    2. Element-wise summing with the image encoder's frame embedding
    3. Applying lightweight convolutional layers to fuse the information

    The memory features are projected to a lower-dimensional space (64-dim)
    for efficient storage in the memory bank.
    """

    def __init__(
        self,
        out_dim: int = 64,
        in_dim: int = 256,
        mask_downsampler: Optional[nn.Module] = None,
        fuser: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.out_dim = out_dim

        if mask_downsampler is None:
            # Default mask downsampler: 4 strided convolutions to go from full res to stride 16
            self.mask_downsampler = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
                LayerNorm2d(4),
                nn.GELU(),
                nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
                LayerNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
                LayerNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, in_dim, kernel_size=3, stride=2, padding=1),
            )
        else:
            self.mask_downsampler = mask_downsampler

        if fuser is None:
            # Lightweight convolutional fusion layers
            self.fuser = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
                LayerNorm2d(in_dim),
                nn.GELU(),
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            )
        else:
            self.fuser = fuser

        # Project to lower-dimensional memory features
        self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

        # Learned occlusion embedding added to memory features of occluded frames
        self.occlusion_embed = nn.Parameter(torch.zeros(1, out_dim, 1, 1))

    def forward(
        self,
        current_vision_feats: torch.Tensor,
        pred_masks: torch.Tensor,
        is_mask_from_pts: bool = False,
        is_occluded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate memory features from current frame prediction.

        Args:
            current_vision_feats: [B, C, H, W] image encoder features (stride 16)
            pred_masks: [B, 1, H_full, W_full] predicted mask at full resolution
            is_mask_from_pts: whether the mask was generated from point prompts
            is_occluded: [B] boolean tensor indicating if object is occluded

        Returns:
            memory: [B, out_dim, H, W] memory features at stride 16 resolution
        """
        # Downsample mask to match image embedding spatial size
        mask_feat = self.mask_downsampler(pred_masks)

        # Ensure spatial sizes match
        if mask_feat.shape[-2:] != current_vision_feats.shape[-2:]:
            mask_feat = F.interpolate(
                mask_feat,
                size=current_vision_feats.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        # Element-wise sum of mask features and image features
        fused = current_vision_feats + mask_feat

        # Apply fusion layers
        fused = self.fuser(fused)

        # Project to memory dimension
        memory = self.out_proj(fused)

        # Add occlusion embedding for occluded frames
        if is_occluded is not None:
            # is_occluded: [B] boolean
            occ_mask = is_occluded.float().view(-1, 1, 1, 1)
            memory = memory + occ_mask * self.occlusion_embed

        return memory
