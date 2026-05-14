"""
models/discriminator.py

DINO‑based conditional discriminator for the FR‑VAE tokenizer training.
Uses a frozen DINOv2 backbone to extract multi‑scale features from both real and
reconstructed images, concatenates them, and passes the result through a small
convolutional head to obtain pixel‑wise realism scores.

Architecture follows the VQ‑GAN + DINO discriminator pattern described in the NFIG paper.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.vision_transformer import VisionTransformer


# ----------------------------------------------------------------------
# DINOv2 input normalisation constants (standard ImageNet statistics)
# ----------------------------------------------------------------------
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoDiscriminator(nn.Module):
    """
    Conditional patch discriminator that exploits frozen DINOv2 features.

    Args:
        dino_model: A pretrained Vision Transformer model (e.g. dinov2_vitb14).
        input_dim: Number of input image channels (3). Ignored internally;
            kept for compatibility with the project's design specification.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(
        self,
        dino_model: VisionTransformer,
        input_dim: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # ---- Freeze the DINO backbone ----
        self.dino = dino_model
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()

        # ---- Select intermediate transformer blocks for multi‑scale features ----
        # For vitb14 (depth=12) we pick blocks 3, 6, 9, 11 (0‑based).
        self.feature_indices = [3, 6, 9, 11]
        self.dino_dim = self.dino.hidden_dim   # 768 for vit‑base

        # ---- Register forward hooks to capture block outputs ----
        self._real_features: Dict[str, torch.Tensor] = {}
        self._fake_features: Dict[str, torch.Tensor] = {}
        self._hook_handles = []

        for idx in self.feature_indices:
            block = self.dino.blocks[idx]
            handle = block.register_forward_hook(self._make_hook(f"layer_{idx}"))
            self._hook_handles.append(handle)

        # ---- Convolutional head ----
        total_in_ch = len(self.feature_indices) * (2 * self.dino_dim)   # real + fake per layer
        hidden_ch = 256

        self.head = nn.Sequential(
            nn.Conv2d(total_in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )

    # ------------------------------------------------------------------
    # Hook helper
    # ------------------------------------------------------------------
    def _make_hook(self, name: str):
        """
        Return a forward hook that stores the block's output in a dictionary
        with key `name`.

        Args:
            name: Unique identifier for the current layer.

        Returns:
            callable: forward hook function.
        """
        def hook(module, input, output):
            # `output` is the hidden states tensor after the block: (B, N+1, D)
            self._hook_outputs[name] = output
        return hook

    # ------------------------------------------------------------------
    # Feature extraction pipeline (per image)
    # ------------------------------------------------------------------
    def _preprocess_for_dino(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert images from the tokenizer's range [-1, 1] to DINOv2 input
        normalisation.

        Args:
            x: Image batch of shape (B, 3, H, W) in [-1, 1].

        Returns:
            Normalised tensor ready for DINOv2.
        """
        x_01 = (x + 1.0) / 2.0                      # range [0, 1]
        x_norm = (x_01 - _DINO_MEAN.to(x.device)) / _DINO_STD.to(x.device)
        return x_norm

    def _extract_dino_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Run the DINOv2 model on a batch and collect intermediate patch features
        from the selected transformer blocks.

        Args:
            x: A batch of images in range [-1, 1] with shape (B, 3, H, W).

        Returns:
            List of tensors, each of shape (B, D, Hp, Wp) where Hp, Wp are
            the spatial dimensions of the patch grid.
        """
        # Prepare fresh storage for hook outputs
        self._hook_outputs: Dict[str, torch.Tensor] = {}

        # Normalise images for DINOv2
        dino_input = self._preprocess_for_dino(x)

        # Forward pass through the frozen DINO model (no_grad to save memory)
        with torch.no_grad():
            _ = self.dino(dino_input)

        # Gather patch features and reshape to 2D spatial grid
        patch_size = self.dino.patch_embed.patch_size[0]   # e.g. 14 for vitb14
        B, _, H, W = x.shape
        Hp = H // patch_size
        Wp = W // patch_size

        features = []
        for idx in self.feature_indices:
            out = self._hook_outputs[f"layer_{idx}"]      # (B, N+1, D)
            patch_tokens = out[:, 1:, :]                  # discard class token -> (B, N, D)
            N = patch_tokens.size(1)

            # Verify that the number of patches matches the expected grid
            if Hp * Wp != N:
                # Fallback: create square root if possible (should always hold)
                side = int(N**0.5)
                if side * side == N:
                    Hp, Wp = side, side
                else:
                    raise RuntimeError(
                        f"Non‑square patch sequence with length {N} encountered."
                    )
            spatial = patch_tokens.permute(0, 2, 1).reshape(B, self.dino_dim, Hp, Wp)
            features.append(spatial)

        return features

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, reconst: torch.Tensor) -> torch.Tensor:
        """
        Produce pixel‑wise realism logits for a pair (real, reconstruction).

        Args:
            image: Batch of real images, shape (B, 3, H, W) in [-1, 1].
            reconst: Batch of reconstructed images, same shape.

        Returns:
            Logit map of shape (B, 1, Hp, Wp) where Hp, Wp are the DINO
            patch grid dimensions.
        """
        # Extract multi‑scale features from both images
        real_feats = self._extract_dino_features(image)    # list of (B, D, Hp, Wp)
        fake_feats = self._extract_dino_features(reconst)

        # Concatenate real + fake for each layer, then concatenate all layers
        layer_cats = []
        for fr, ff in zip(real_feats, fake_feats):
            layer_cats.append(torch.cat([fr, ff], dim=1))   # (B, 2D, Hp, Wp)

        combined = torch.cat(layer_cats, dim=1)            # (B, Σ(2D), Hp, Wp)

        # Apply the small convolutional head
        logits = self.head(combined)                       # (B, 1, Hp, Wp)
        return logits

    # ------------------------------------------------------------------
    # Cleanup helpers (optional)
    # ------------------------------------------------------------------
    def remove_hooks(self) -> None:
        """Deregister all forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
