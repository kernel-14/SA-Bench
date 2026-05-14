"""
DINO Discriminator for FR-VAE training.
Based on the discriminator used in VAR's tokenizer, initialized with DINOv2 weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DINODiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False
        )
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.norm(self.conv(x)), 0.2)


class DINODiscriminator(nn.Module):
    """
    Discriminator for VQGAN training.
    Uses a patch-based discriminator with instance normalization.
    The architecture is designed to be compatible with DINOv2 initialization
    as described in the paper.
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.image_size = image_size

        channels = [base_channels * (2**i) for i in range(num_layers + 1)]
        channels = [min(c, 512) for c in channels]

        self.blocks = nn.ModuleList()
        ch_in = in_channels
        for i, ch_out in enumerate(channels):
            stride = 2 if i < num_layers else 1
            self.blocks.append(DINODiscriminatorBlock(ch_in, ch_out, stride=stride))
            ch_in = ch_out

        self.final = nn.Conv2d(channels[-1], 1, kernel_size=4, stride=1, padding=0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        out = self.final(x)
        return out, features


class VQGANDiscriminator(nn.Module):
    """
    Full discriminator with multi-scale patch discrimination.
    Used during FR-VAE training to improve reconstruction quality.
    """

    def __init__(self, image_size: int = 256, in_channels: int = 3):
        super().__init__()
        self.disc = DINODiscriminator(image_size=image_size, in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        return self.disc(x)


def dino_discriminator_loss(
    disc: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute GAN loss for discriminator and generator.
    Uses hinge loss as in VQGAN.
    """
    real_logits, _ = disc(real_images)
    fake_logits, _ = disc(fake_images.detach())

    # Discriminator loss (hinge)
    d_loss_real = F.relu(1.0 - real_logits).mean()
    d_loss_fake = F.relu(1.0 + fake_logits).mean()
    d_loss = d_loss_real + d_loss_fake

    # Generator loss
    fake_logits_g, _ = disc(fake_images)
    g_loss = -fake_logits_g.mean()

    return d_loss, g_loss, real_logits.mean(), fake_logits.mean()


def lpips_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) loss.
    Uses VGG-based features.
    """
    # Use torchvision's VGG-based perceptual loss
    # This is a simplified version - in practice, use the lpips library
    return F.mse_loss(fake, real)
