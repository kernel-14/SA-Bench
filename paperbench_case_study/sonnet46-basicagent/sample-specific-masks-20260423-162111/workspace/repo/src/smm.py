"""
SMM: Sample-specific Multi-channel Masks for Visual Reprogramming.

This module implements the full SMM framework including:
- Input transformation f_in(x; delta, phi) = r(x) + delta * f_mask(r(x) | phi)
- Baseline methods: Pad, Narrow, Medium, Full watermarks
- Ablation variants: Only-delta, Only-f_mask, Single-channel f_mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mask_generator import MaskGeneratorWithInterpolation


class SMMReprogramming(nn.Module):
    """
    Full SMM framework for visual reprogramming.

    f_in(x_i | phi, delta) = r(x_i) + delta * f_mask(r(x_i) | phi)

    where:
    - r(x_i) is the resized image (bilinear upsampling to img_size)
    - delta is a shared learnable noise pattern (initialized to zeros)
    - f_mask is the sample-specific mask generator (CNN + patch-wise interpolation)

    Args:
        model_name: 'ResNet18', 'ResNet50', or 'ViT_B32'
        patch_size: size of patches for interpolation (default: 8 = 2^3)
    """
    def __init__(self, model_name='ResNet18', patch_size=8):
        super().__init__()

        if model_name == 'ViT_B32':
            self.img_size = 384
            cnn_type = 'ViT_B32'
        else:
            self.img_size = 224
            cnn_type = 'ResNet'

        self.model_name = model_name
        self.patch_size = patch_size

        # Learnable noise pattern delta, initialized to zeros (shape: 3 x H x W)
        self.delta = nn.Parameter(
            torch.zeros(3, self.img_size, self.img_size), requires_grad=True
        )

        # Sample-specific mask generator
        self.f_mask = MaskGeneratorWithInterpolation(
            model_name=cnn_type,
            patch_size=patch_size,
            img_size=self.img_size
        )

    def forward(self, x):
        """
        Apply SMM input transformation.

        Args:
            x: input images (B, 3, H, W) - already resized to img_size

        Returns:
            reprogrammed images (B, 3, H, W)
        """
        # Generate sample-specific masks
        mask = self.f_mask(x)  # (B, 3, H, W)

        # Apply: r(x) + delta * f_mask(r(x))
        # delta is broadcast over batch dimension
        reprogrammed = x + self.delta.unsqueeze(0) * mask
        return reprogrammed

    def get_delta(self):
        return self.delta

    def get_mask_params(self):
        return self.f_mask.parameters()


class SharedPatternReprogramming(nn.Module):
    """
    Baseline: Shared pattern VR (Full watermark).
    f_in(x_i) = r(x_i) + delta (M is all-ones, no f_mask)

    This is the "Only delta" ablation variant and the "Full" baseline.
    """
    def __init__(self, model_name='ResNet18'):
        super().__init__()
        if model_name == 'ViT_B32':
            self.img_size = 384
        else:
            self.img_size = 224

        # Learnable noise pattern, initialized to zeros
        self.delta = nn.Parameter(
            torch.zeros(3, self.img_size, self.img_size), requires_grad=True
        )

    def forward(self, x):
        return x + self.delta.unsqueeze(0)


class MaskedWatermarkReprogramming(nn.Module):
    """
    Baseline: Watermark with a fixed binary shared mask.
    f_in(x_i) = r(x_i) + M * delta

    Supports: Narrow (width=28), Medium (width=56), Full (all ones)
    """
    def __init__(self, model_name='ResNet18', mask_type='full'):
        super().__init__()
        if model_name == 'ViT_B32':
            self.img_size = 384
        else:
            self.img_size = 224

        self.mask_type = mask_type

        # Learnable noise pattern
        self.delta = nn.Parameter(
            torch.zeros(3, self.img_size, self.img_size), requires_grad=True
        )

        # Create fixed binary mask
        mask = self._create_mask(mask_type)
        self.register_buffer('mask', mask)

    def _create_mask(self, mask_type):
        H = W = self.img_size
        mask = torch.zeros(1, 3, H, W)

        if mask_type == 'full':
            mask.fill_(1.0)
        elif mask_type == 'narrow':
            # Width = 28 (1/8 of 224)
            width = H // 8
            mask[:, :, :width, :] = 1.0
            mask[:, :, H-width:, :] = 1.0
            mask[:, :, :, :width] = 1.0
            mask[:, :, :, W-width:] = 1.0
        elif mask_type == 'medium':
            # Width = 56 (1/4 of 224)
            width = H // 4
            mask[:, :, :width, :] = 1.0
            mask[:, :, H-width:, :] = 1.0
            mask[:, :, :, :width] = 1.0
            mask[:, :, :, W-width:] = 1.0
        elif mask_type == 'pad':
            # Padding-based: zeros in center (where target image is placed)
            # The center region is the target image location
            # For padding-based, the noise is around the image
            # Assuming target image is centered at 224x224 with some padding
            # The exact size depends on the target image size
            # For simplicity, use a border of width 32
            width = 32
            mask[:, :, :width, :] = 1.0
            mask[:, :, H-width:, :] = 1.0
            mask[:, :, :, :width] = 1.0
            mask[:, :, :, W-width:] = 1.0
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

        return mask

    def forward(self, x):
        return x + self.mask * self.delta.unsqueeze(0)


class SampleSpecificPatternReprogramming(nn.Module):
    """
    Ablation: Sample-specific pattern without shared delta.
    f_in(x_i) = r(x_i) + f_mask(r(x_i))
    (No shared delta, only f_mask)
    """
    def __init__(self, model_name='ResNet18', patch_size=8):
        super().__init__()
        if model_name == 'ViT_B32':
            self.img_size = 384
            cnn_type = 'ViT_B32'
        else:
            self.img_size = 224
            cnn_type = 'ResNet'

        self.f_mask = MaskGeneratorWithInterpolation(
            model_name=cnn_type,
            patch_size=patch_size,
            img_size=self.img_size
        )

    def forward(self, x):
        mask = self.f_mask(x)
        return x + mask


class SingleChannelSMMReprogramming(nn.Module):
    """
    Ablation: Single-channel version of SMM.
    f_in(x_i) = r(x_i) + delta * f_mask_s(r(x_i))
    where f_mask_s averages the penultimate-layer output to produce a single channel.
    """
    def __init__(self, model_name='ResNet18', patch_size=8):
        super().__init__()
        if model_name == 'ViT_B32':
            self.img_size = 384
            cnn_type = 'ViT_B32'
        else:
            self.img_size = 224
            cnn_type = 'ResNet'

        self.delta = nn.Parameter(
            torch.zeros(3, self.img_size, self.img_size), requires_grad=True
        )

        # Use the same CNN but average the output to single channel
        self.f_mask = MaskGeneratorWithInterpolation(
            model_name=cnn_type,
            patch_size=patch_size,
            img_size=self.img_size
        )

    def forward(self, x):
        mask_3ch = self.f_mask(x)  # (B, 3, H, W)
        # Average to single channel and broadcast to 3 channels
        mask_1ch = mask_3ch.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        mask = mask_1ch.expand_as(x)  # (B, 3, H, W)
        return x + self.delta.unsqueeze(0) * mask


class PaddingReprogramming(nn.Module):
    """
    Padding-based reprogramming (Chen et al., 2023).
    Centers the target image and adds noise pattern around it.
    The noise is only in the padding region (M_i=1 for padding, M_i=0 for image).
    """
    def __init__(self, model_name='ResNet18', target_img_size=32):
        super().__init__()
        if model_name == 'ViT_B32':
            self.img_size = 384
        else:
            self.img_size = 224

        self.target_img_size = target_img_size

        # Learnable noise pattern (only in padding region)
        self.delta = nn.Parameter(
            torch.zeros(3, self.img_size, self.img_size), requires_grad=True
        )

        # Create mask: 0 where image is placed, 1 in padding
        mask = torch.ones(1, 3, self.img_size, self.img_size)
        H = W = self.img_size
        # Center crop region
        pad_h = (H - target_img_size) // 2
        pad_w = (W - target_img_size) // 2
        if pad_h > 0 and pad_w > 0:
            mask[:, :, pad_h:pad_h+target_img_size, pad_w:pad_w+target_img_size] = 0.0
        self.register_buffer('mask', mask)

    def forward(self, x):
        """
        x: already resized to img_size (the target image is upsampled/padded)
        """
        return x + self.mask * self.delta.unsqueeze(0)
