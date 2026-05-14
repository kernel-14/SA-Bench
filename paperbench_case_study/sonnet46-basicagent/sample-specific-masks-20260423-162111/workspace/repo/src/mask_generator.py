"""
Mask Generator Module for SMM (Sample-specific Multi-channel Masks).

Architecture:
- 5-layer CNN for ResNet (224x224 input)
- 6-layer CNN for ViT-B32 (384x384 input)

Both use 3x3 conv layers (padding=1, stride=1) and 2x2 MaxPool layers.
The number of MaxPool layers l=3 in both architectures, giving output size H/8 x W/8.
The final layer outputs 3 channels (three-channel mask).

Parameter counts (from Table 4):
- ResNet (5-layer): 26,499 parameters
- ViT-B32 (6-layer): 102,339 parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGenerator5Layer(nn.Module):
    """
    5-layer CNN mask generator for ResNet (224x224 input).
    Architecture (Figure 8):
      Conv(3->16) -> MaxPool -> Conv(16->32) -> MaxPool -> Conv(32->64) -> MaxPool
      -> Conv(64->32) -> Conv(32->3)
    3 MaxPool layers => output size: H/8 x W/8
    Total params: ~26,499
    """
    def __init__(self):
        super().__init__()
        # All conv layers: kernel=3, padding=1, stride=1 => preserves spatial size
        # MaxPool: kernel=2, stride=2 => halves spatial size
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class MaskGenerator6Layer(nn.Module):
    """
    6-layer CNN mask generator for ViT-B32 (384x384 input).
    Architecture (Figure 9):
      Conv(3->16) -> MaxPool -> Conv(16->32) -> MaxPool -> Conv(32->64) -> MaxPool
      -> Conv(64->64) -> Conv(64->32) -> Conv(32->3)
    3 MaxPool layers => output size: H/8 x W/8
    Total params: ~102,339
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(32, 3, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x


def patch_wise_interpolation(mask, patch_size):
    """
    Patch-wise interpolation: upscale mask from H/patch_size x W/patch_size
    back to H x W by repeating each pixel value in a patch_size x patch_size block.

    This avoids floating-point interpolation and backpropagation through the
    upsampling step, as described in Section 3.3.

    Args:
        mask: Tensor of shape (B, C, H', W') where H'=H//patch_size, W'=W//patch_size
        patch_size: int, the size of each patch (2^l)

    Returns:
        Upscaled mask of shape (B, C, H'*patch_size, W'*patch_size)
    """
    if patch_size == 1:
        return mask
    # Use repeat_interleave to expand each pixel to patch_size x patch_size
    # This is equivalent to nearest-neighbor upsampling but without gradient flow
    with torch.no_grad():
        upscaled = mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    # Detach to avoid backpropagation through the interpolation step
    return upscaled.detach()


class MaskGeneratorWithInterpolation(nn.Module):
    """
    Full mask generator module combining CNN and patch-wise interpolation.

    Args:
        model_name: 'ResNet' (uses 5-layer CNN) or 'ViT_B32' (uses 6-layer CNN)
        patch_size: 2^l where l is the number of MaxPool layers (default: 8 = 2^3)
        img_size: target output size (H, W) of the mask
    """
    def __init__(self, model_name='ResNet', patch_size=8, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        if model_name == 'ViT_B32':
            self.cnn = MaskGenerator6Layer()
        else:
            self.cnn = MaskGenerator5Layer()

    def forward(self, x):
        """
        Args:
            x: input image tensor (B, 3, H, W)
        Returns:
            mask: (B, 3, H, W) - same spatial size as input
        """
        # Generate small mask via CNN
        small_mask = self.cnn(x)  # (B, 3, H/patch_size, W/patch_size)

        # Upscale using patch-wise interpolation
        if self.patch_size > 1:
            mask = patch_wise_interpolation(small_mask, self.patch_size)
        else:
            mask = small_mask

        # Handle non-divisible cases: crop or pad to match input size
        H, W = x.shape[2], x.shape[3]
        if mask.shape[2] != H or mask.shape[3] != W:
            mask = mask[:, :, :H, :W]

        return mask


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Verify parameter counts match Table 4
    gen5 = MaskGenerator5Layer()
    gen6 = MaskGenerator6Layer()
    print(f"5-layer CNN params: {count_parameters(gen5):,}")  # Expected: ~26,499
    print(f"6-layer CNN params: {count_parameters(gen6):,}")  # Expected: ~102,339

    # Test forward pass
    x_resnet = torch.randn(2, 3, 224, 224)
    x_vit = torch.randn(2, 3, 384, 384)

    gen_resnet = MaskGeneratorWithInterpolation(model_name='ResNet', patch_size=8, img_size=224)
    gen_vit = MaskGeneratorWithInterpolation(model_name='ViT_B32', patch_size=8, img_size=384)

    mask_resnet = gen_resnet(x_resnet)
    mask_vit = gen_vit(x_vit)

    print(f"ResNet mask shape: {mask_resnet.shape}")  # Expected: (2, 3, 224, 224)
    print(f"ViT mask shape: {mask_vit.shape}")        # Expected: (2, 3, 384, 384)
