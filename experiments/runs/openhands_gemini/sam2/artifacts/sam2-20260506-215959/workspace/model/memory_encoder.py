
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MemoryEncoder(nn.Module):
    """
    Memory encoder module for SAM 2.
    Generates spatial memory features from the predicted mask and image embeddings.
    """
    def __init__(
        self,
        embed_dim: int,
        mask_in_chans: int = 1,
        output_channels: int = 64, # Memory channels (Config.MEMORY_CHANNELS)
        kernel_size: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_channels = output_channels

        self.conv_mask = nn.Sequential(
            nn.Conv2d(mask_in_chans, embed_dim // 4, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2),
        )

        self.fusion_convs = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, output_channels, kernel_size=kernel_size, padding=kernel_size // 2),
        )

    def forward(
        self,
        image_embedding: torch.Tensor, # (B, C, H_embed, W_embed) from ImageEncoder
        predicted_mask: torch.Tensor, # (B, 1, H_orig, W_orig) predicted mask
        target_size: Tuple[int, int], # H_embed, W_embed
    ) -> torch.Tensor:
        """
        Args:
            image_embedding (torch.Tensor): The image embedding from the image encoder.
            predicted_mask (torch.Tensor): The predicted mask from the mask decoder.
            target_size (Tuple[int, int]): The desired spatial size for the memory features (H_embed, W_embed).

        Returns:
            torch.Tensor: Spatial memory features (B, output_channels, H_embed, W_embed).
        """
        # Downsample predicted mask to image embedding size
        # Predicted mask is B, N_masks, H_orig, W_orig. We need B, 1, H_orig, W_orig.
        # Assuming for memory encoding, we take the best mask (or a specific mask index).
        # For simplicity, let's assume predicted_mask is already (B, 1, H_orig, W_orig)
        # If it comes from MaskDecoder (B, N_masks, H, W), we need to select one.
        # For now, we assume a single mask is passed here.
        
        # Resize mask to the same spatial dimensions as the image embedding
        resized_mask = F.interpolate(
            predicted_mask,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # Embed mask
        mask_embedding = self.conv_mask(resized_mask)

        # Sum element-wise with image embedding
        fused_embedding = image_embedding + mask_embedding

        # Apply fusion convolutions
        memory_features = self.fusion_convs(fused_embedding)

        return memory_features

