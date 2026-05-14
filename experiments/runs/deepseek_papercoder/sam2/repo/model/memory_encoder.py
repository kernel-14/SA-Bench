# model/memory_encoder.py
"""
Memory Encoder for SAM 2.

Generates compact spatial memory features that are stored in the
memory bank and later attended to by subsequent frames.  It fuses the
predicted segmentation mask of the current frame with the unconditioned
image embedding produced by the Hiera image encoder (via FPN).

The design follows the paper's description (Section 4 and Appendix D.1):
  1. Downsample the mask (stride 16) to match the spatial resolution of the image embedding.
  2. Sum the downsampled mask features with the unconditioned image embedding.
  3. Apply lightweight convolutional layers to fuse the information and project
     the channel dimension to a compact size (default 64).

Typical usage inside SAM2Model.forward:
    memory_embed = memory_encoder(pred_mask, unconditioned_image_embed)
    memory_bank.add_memory(memory_embed, ...)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MemoryEncoder(nn.Module):
    """
    Memory encoder that transforms a binary / soft mask and the corresponding
    image embedding into a spatial memory feature map.

    Args:
        in_ch: Number of input feature channels (must match the channel dimension
            of the unconditioned image embedding). Default from config is 256.
        out_ch: Number of output feature channels for the memory (compact size).
            Default from config is 64.

    The forward method expects:
        mask:          (B, 1, 1024, 1024) prediction for the current frame.
        image_embed:   (B, in_ch, 64, 64) unconditioned image embedding.

    Returns:
        memory:        (B, out_ch, 64, 64) ready to be stored in the memory bank.
    """

    def __init__(self, in_ch: int = 256, out_ch: int = 64) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Mask downsample block: 1024x1024 -> 64x64, 1 channel -> in_ch
        # Four strided convs (each stride 2) progressively halve the spatial size
        # while increasing the number of channels.
        # ------------------------------------------------------------------
        self.mask_downsample = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, in_ch, kernel_size=3, stride=2, padding=1),
        )

        # ------------------------------------------------------------------
        # Fusion block: mix downsampled mask features with image embedding,
        # then reduce channel count and further process.
        # Two convolutional layers keep the spatial resolution unchanged (64x64).
        # ------------------------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

        # Initialise weights using standard conv initialisation
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Kaiming uniform initialisation for convolutional layers."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self, mask: torch.Tensor, image_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            mask: Predicted mask of the current frame, shape (B, 1, 1024, 1024).
            image_embed: Unconditioned image embedding from the Hiera encoder + FPN,
                         shape (B, in_ch, 64, 64).

        Returns:
            memory: Compact spatial memory features of shape (B, out_ch, 64, 64).
        """
        # 1. Downsample mask to align with image embedding spatial size and channels
        mask_feat = self.mask_downsample(mask)          # (B, in_ch, 64, 64)

        # 2. Element‑wise sum with the unconditioned image embedding
        fused = mask_feat + image_embed                 # (B, in_ch, 64, 64)

        # 3. Lightweight fusion conv layers
        memory = self.fusion(fused)                     # (B, out_ch, 64, 64)

        return memory


# ---------------------------------------------------------------------------
# Quick smoke test (runs only when executing this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy inputs matching the default configuration
    encoder = MemoryEncoder(in_ch=256, out_ch=64)
    B = 2
    mask = torch.randn(B, 1, 1024, 1024)
    image_embed = torch.randn(B, 256, 64, 64)
    memory = encoder(mask, image_embed)
    print(f"input mask: {mask.shape}")
    print(f"input embed: {image_embed.shape}")
    print(f"output memory: {memory.shape}")   # Expected: (2, 64, 64, 64)

    # Check that the output can be used as intended
    assert memory.shape[0] == B
    assert memory.shape[1] == 64
    assert memory.shape[2:] == (64, 64), "Memory spatial size mismatch"
    print("All tests passed.")
