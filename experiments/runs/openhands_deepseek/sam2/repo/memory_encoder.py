"""Memory encoder for SAM 2.

Generates a memory by:
1. Downsampling the output mask using a convolutional module
2. Summing it element-wise with the unconditioned frame embedding from the image encoder
3. Light-weight convolutional layers to fuse the information

The resulting memory feature is stored in the memory bank for cross-attention in future frames.
"""

import torch
import torch.nn as nn

from config import MemoryEncoderConfig


class MaskDownSampler(nn.Module):
    """Downsample the output mask to match spatial dimensions of image encoder features."""
    def __init__(self, kernel_size: int = 7, stride: int = 4, padding: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GELU(),
            nn.Conv2d(64, 256, kernel_size=1),
        )

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Args: masks [B, 1, H, W]. Returns: [B, 256, H/s, W/s]."""
        return self.conv(masks)


class Fuser(nn.Module):
    """Fuse mask features with image encoder features using light-weight convolutions."""
    def __init__(self, input_dim: int = 256, output_dim: int = 64,
                 num_conv_layers: int = 2, kernel_size: int = 3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_conv_layers):
            out_dim = output_dim if i == num_conv_layers - 1 else input_dim
            layers.extend([
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(8, out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MemoryEncoder(nn.Module):
    """Memory encoder for SAM 2.

    Produces memory features by combining the predicted mask with image encoder features.

    The memory encoder reuses the image embeddings from the Hiera encoder rather than
    using a separate image encoder. This allows memory features to benefit from the
    strong Hiera representations, especially as the image encoder is scaled up.
    """
    def __init__(self, config: MemoryEncoderConfig):
        super().__init__()
        self.config = config
        self.mask_downsampler = MaskDownSampler(
            kernel_size=config.mask_downsample_kernel,
            stride=config.mask_downsample_stride,
            padding=config.mask_downsample_kernel // 2,
        )
        self.fuser = Fuser(
            input_dim=config.encoder_dim,
            output_dim=config.memory_dim,
            num_conv_layers=config.conv_layers,
            kernel_size=config.kernel_size,
        )
        self.memory_dim = config.memory_dim

        # Projection to match image encoder output dim to encoder_dim (256)
        self.image_proj = nn.Conv2d(config.encoder_dim, config.encoder_dim, kernel_size=1)

    def forward(self, image_embeddings: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Generate memory features.

        Args:
            image_embeddings: [B, H*W, C] unconditioned frame embeddings from image encoder
            masks: [B, 1, H_img, W_img] predicted mask for the current frame

        Returns:
            memory_features: [B, memory_dim, H_mem, W_mem] spatial memory features
        """
        B, N, C = image_embeddings.shape

        # Convert flat token image embeddings to spatial format
        H_enc = W_enc = int(N ** 0.5)
        img_spatial = image_embeddings.reshape(B, H_enc, W_enc, C).permute(0, 3, 1, 2)

        # Project image embeddings to encoder_dim
        img_spatial = self.image_proj(img_spatial)

        # Downsample mask
        mask_feats = self.mask_downsampler(masks)

        # Resize mask features to match image embeddings spatial size
        mask_feats = nn.functional.interpolate(
            mask_feats, size=(H_enc, W_enc), mode="bilinear", align_corners=False
        )

        # Element-wise sum: mask features + image features
        fused = mask_feats + img_spatial

        # Light-weight convolutions to fuse
        memory = self.fuser(fused)

        return memory
