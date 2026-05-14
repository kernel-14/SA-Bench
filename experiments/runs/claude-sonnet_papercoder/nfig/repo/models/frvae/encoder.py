## models/frvae/encoder.py
"""VQ-GAN encoder for the FR-VAE, built on a pretrained DINOv2-base backbone.

The encoder maps an input image x ∈ R^(B×3×256×256) to a spatial feature map
f ∈ R^(B×768×16×16), which is then passed to the FrequencyDecomposer.

Key design decisions (see Logic Analysis):
  - DINOv2-base uses patch_size=14; for 256×256 input, 256/14 ≈ 18.28 patches.
    To obtain exactly 16×16 spatial tokens (matching latent_spatial_size=16 and
    scale_factors[-1]=16), the input is resized to 224×224 internally before
    encoding (224/14 = 16 exactly).
  - The training pipeline normalizes images to [-1, 1] (mean=std=0.5 per config),
    but DINOv2 expects ImageNet normalization. Re-normalization is applied inside
    forward() via registered buffers (device-safe, non-trainable).
  - A learned projection layer (nn.Linear 768→latent_channels) is included to
    allow fine-tuning of the feature space even when dimensions already match.
  - The backbone is fine-tuned during FR-VAE training (not frozen), consistent
    with "initialized with pretrained weights from DINOv2-base" in Section 4.1.
  - During NFIG Transformer training, the entire FR-VAE (including this encoder)
    is frozen by the NFIGTrainer.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "timm is required for the DINOv2 encoder. "
        "Install it with: pip install timm>=0.6.14"
    ) from exc


class VQGANEncoder(nn.Module):
    """DINOv2-based encoder that produces spatial latent feature maps.

    Wraps a pretrained DINOv2-base ViT backbone (vit_base_patch14_dinov2)
    and reshapes its patch token outputs into a 2D spatial feature map
    suitable for frequency decomposition and residual quantization.

    The encoder performs the following steps in forward():
        1. Re-normalize from [-1, 1] (training pipeline) to DINOv2 ImageNet stats.
        2. Resize input from 256×256 to 224×224 (to get exactly 16×16 patches).
        3. Extract patch tokens via backbone.forward_features().
        4. Remove the CLS token; keep the 256 patch tokens.
        5. Project patch tokens via self.proj (Linear 768 → latent_channels).
        6. Reshape to spatial map (B, latent_channels, 16, 16).

    Attributes:
        backbone: Pretrained DINOv2-base ViT backbone (nn.Module).
        proj: Linear projection from backbone output dim to latent_channels.
        latent_channels: Output channel dimension (768 per config).
        spatial_size: Output spatial resolution per side (16 per config).
        patch_size: Backbone patch size (14 for DINOv2-base).
    """

    # DINOv2 ImageNet normalization constants (fixed, non-learnable).
    # Source: https://github.com/facebookresearch/dinov2
    _DINO_MEAN: tuple = (0.485, 0.456, 0.406)
    _DINO_STD: tuple = (0.229, 0.224, 0.225)

    # Internal resolution that yields exactly 16×16 patches with patch_size=14.
    # 224 / 14 = 16 (exact integer).
    _INTERNAL_RESOLUTION: int = 224

    # Expected spatial output size (matches config.frvae.latent_spatial_size = 16
    # and config.frvae.scale_factors[-1] = 16).
    _EXPECTED_SPATIAL_SIZE: int = 16

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        latent_channels: int = 768,
        pretrained: bool = True,
    ) -> None:
        """Initialize the VQGANEncoder.

        Args:
            model_name: timm model identifier for the DINOv2 backbone.
                From config.frvae.encoder_model = "vit_base_patch14_dinov2".
                Must be a ViT model with a patch_embed attribute.
            latent_channels: Output channel dimension of the encoder.
                From config.frvae.latent_channels = 768.
                Must equal config.frvae.codebook_dim = 768.
            pretrained: Whether to load pretrained DINOv2 weights.
                From config.frvae.pretrained_encoder = True.

        Raises:
            ValueError: If the backbone's embed_dim does not match expectations
                or if the computed spatial size does not equal _EXPECTED_SPATIAL_SIZE.
            RuntimeError: If timm cannot load the specified model.
        """
        super().__init__()

        self.latent_channels: int = latent_channels

        # --- Load DINOv2 backbone via timm ---
        # num_classes=0: removes the classification head, returns features only.
        # pretrained=pretrained: loads ImageNet-pretrained weights when True.
        self.backbone: nn.Module = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head; we use forward_features()
        )

        # Retrieve backbone output dimension (embed_dim).
        # DINOv2-base: embed_dim = 768.
        backbone_dim: int = self._get_backbone_embed_dim()

        # Retrieve patch size from the backbone's patch embedding layer.
        # DINOv2-base: patch_size = 14.
        self.patch_size: int = self._get_patch_size()

        # Compute the spatial output size for the internal resolution.
        # For DINOv2-base: 224 / 14 = 16.
        self.spatial_size: int = self._INTERNAL_RESOLUTION // self.patch_size

        # Validate that the spatial size matches the expected latent_spatial_size.
        if self.spatial_size != self._EXPECTED_SPATIAL_SIZE:
            raise ValueError(
                f"Computed spatial_size={self.spatial_size} does not match "
                f"expected _EXPECTED_SPATIAL_SIZE={self._EXPECTED_SPATIAL_SIZE}. "
                f"For model_name='{model_name}' with patch_size={self.patch_size}, "
                f"the internal resolution {self._INTERNAL_RESOLUTION} yields "
                f"{self._INTERNAL_RESOLUTION}/{self.patch_size}={self.spatial_size} "
                f"patches per side. Adjust _INTERNAL_RESOLUTION or model_name."
            )

        # --- Projection layer ---
        # Maps backbone_dim → latent_channels. When both are 768, this is an
        # identity-equivalent linear layer that still allows learned adaptation.
        # bias=True follows standard practice for feature projection layers.
        self.proj: nn.Linear = nn.Linear(backbone_dim, latent_channels, bias=True)

        # Initialize projection as near-identity when dimensions match.
        if backbone_dim == latent_channels:
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        else:
            # Xavier uniform initialization for dimension-changing projections.
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        # --- Re-normalization buffers ---
        # The training pipeline normalizes images to [-1, 1] with mean=std=0.5.
        # DINOv2 expects ImageNet normalization: mean=[0.485,0.456,0.406],
        # std=[0.229,0.224,0.225].
        #
        # Conversion formula (applied in forward()):
        #   x_01 = x_neg1_1 * 0.5 + 0.5          # [-1,1] → [0,1]
        #   x_dino = (x_01 - dino_mean) / dino_std
        #
        # Registered as buffers so they move with .to(device) / .cuda() calls
        # and are saved/loaded with state_dict(), but are NOT trainable parameters.
        dino_mean: Tensor = torch.tensor(
            self._DINO_MEAN, dtype=torch.float32
        ).view(1, 3, 1, 1)  # (1, 3, 1, 1) for broadcasting over (B, 3, H, W)

        dino_std: Tensor = torch.tensor(
            self._DINO_STD, dtype=torch.float32
        ).view(1, 3, 1, 1)  # (1, 3, 1, 1) for broadcasting over (B, 3, H, W)

        self.register_buffer("dino_mean", dino_mean, persistent=True)
        self.register_buffer("dino_std", dino_std, persistent=True)

    def _get_backbone_embed_dim(self) -> int:
        """Retrieve the embedding dimension from the backbone.

        Tries multiple attribute names used by different timm ViT variants.

        Returns:
            Integer embedding dimension (768 for DINOv2-base).

        Raises:
            AttributeError: If no known embed_dim attribute is found.
        """
        # timm ViT models expose embed_dim directly.
        if hasattr(self.backbone, "embed_dim"):
            return int(self.backbone.embed_dim)
        # Some timm models use num_features (equivalent to embed_dim for ViTs).
        if hasattr(self.backbone, "num_features"):
            return int(self.backbone.num_features)
        raise AttributeError(
            f"Cannot determine embed_dim from backbone of type "
            f"'{type(self.backbone).__name__}'. "
            "Expected 'embed_dim' or 'num_features' attribute."
        )

    def _get_patch_size(self) -> int:
        """Retrieve the patch size from the backbone's patch embedding.

        Returns:
            Integer patch size (14 for DINOv2-base).

        Raises:
            AttributeError: If the patch embedding structure is not recognized.
        """
        # Standard timm ViT: backbone.patch_embed.patch_size is a tuple (H, W).
        if hasattr(self.backbone, "patch_embed"):
            patch_embed = self.backbone.patch_embed
            if hasattr(patch_embed, "patch_size"):
                ps = patch_embed.patch_size
                # patch_size can be an int or a tuple (H, W).
                if isinstance(ps, (tuple, list)):
                    return int(ps[0])
                return int(ps)
            # Some timm versions store it as proj.kernel_size.
            if hasattr(patch_embed, "proj") and hasattr(patch_embed.proj, "kernel_size"):
                ks = patch_embed.proj.kernel_size
                if isinstance(ks, (tuple, list)):
                    return int(ks[0])
                return int(ks)
        raise AttributeError(
            f"Cannot determine patch_size from backbone of type "
            f"'{type(self.backbone).__name__}'. "
            "Expected backbone.patch_embed.patch_size or "
            "backbone.patch_embed.proj.kernel_size."
        )

    def _renormalize(self, x: Tensor) -> Tensor:
        """Convert images from [-1, 1] (training pipeline) to DINOv2 stats.

        The training pipeline (config.data.mean=0.5, config.data.std=0.5)
        normalizes pixel values to [-1, 1]. DINOv2 was pretrained with
        ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).

        Conversion:
            x_01   = x * 0.5 + 0.5          # [-1, 1] → [0, 1]
            x_dino = (x_01 - dino_mean) / dino_std

        Args:
            x: Input tensor of shape (B, 3, H, W) with values in [-1, 1].

        Returns:
            Re-normalized tensor of shape (B, 3, H, W) in DINOv2's expected range.
        """
        # Step 1: [-1, 1] → [0, 1]
        x_01: Tensor = x * 0.5 + 0.5

        # Step 2: [0, 1] → DINOv2 normalized range.
        # self.dino_mean and self.dino_std are registered buffers of shape (1,3,1,1).
        x_dino: Tensor = (x_01 - self.dino_mean) / self.dino_std  # type: ignore[operator]

        return x_dino

    def forward(self, x: Tensor) -> Tensor:
        """Encode an input image batch into a spatial latent feature map.

        Implements the full encoding pipeline:
            x (B,3,256,256) → re-normalize → resize to 224×224
            → DINOv2 patch tokens (B,256,768) → project → reshape
            → f (B,768,16,16)

        Args:
            x: Input image batch of shape (B, 3, H, W) with values in [-1, 1].
                Typically H = W = 256 (config.data.image_size = 256).
                The encoder handles other input sizes via internal resizing,
                but 256×256 is the expected and tested resolution.

        Returns:
            Spatial feature map f of shape (B, latent_channels, spatial_size, spatial_size).
            For the default config: (B, 768, 16, 16).
            Values are in an unconstrained real range (not normalized).
        """
        batch_size: int = x.shape[0]

        # --- Step 1: Re-normalize from [-1, 1] to DINOv2 ImageNet stats ---
        x_dino: Tensor = self._renormalize(x)

        # --- Step 2: Resize to 224×224 for exact 16×16 patch grid ---
        # Only resize if the input is not already at the internal resolution.
        # Using bilinear interpolation with align_corners=False (standard for
        # feature extraction; matches torchvision.transforms.Resize behavior).
        if x_dino.shape[-2] != self._INTERNAL_RESOLUTION or \
                x_dino.shape[-1] != self._INTERNAL_RESOLUTION:
            x_resized: Tensor = F.interpolate(
                x_dino,
                size=(self._INTERNAL_RESOLUTION, self._INTERNAL_RESOLUTION),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        else:
            x_resized = x_dino

        # --- Step 3: Extract patch tokens via DINOv2 backbone ---
        # forward_features() returns the full token sequence including CLS token.
        # Shape: (B, 1 + spatial_size^2, embed_dim) = (B, 257, 768) for DINOv2-base.
        features: Tensor = self.backbone.forward_features(x_resized)

        # Handle different timm output formats.
        # Some timm versions return a dict; others return a tensor directly.
        if isinstance(features, dict):
            # timm >= 0.9 may return {'x_norm_clstoken': ..., 'x_norm_patchtokens': ...}
            if "x_norm_patchtokens" in features:
                # Directly get patch tokens without CLS.
                patch_tokens: Tensor = features["x_norm_patchtokens"]
                # Shape: (B, spatial_size^2, embed_dim) = (B, 256, 768)
            elif "x_prenorm" in features:
                all_tokens: Tensor = features["x_prenorm"]
                patch_tokens = all_tokens[:, 1:, :]  # Remove CLS token
            else:
                # Fallback: try to get the first tensor value.
                all_tokens = next(iter(features.values()))
                if all_tokens.dim() == 3:
                    patch_tokens = all_tokens[:, 1:, :]
                else:
                    raise RuntimeError(
                        f"Unexpected backbone output format: {list(features.keys())}"
                    )
        else:
            # Standard timm output: tensor of shape (B, 1 + N_patches, embed_dim).
            # Index 0 is the CLS token; indices 1: are patch tokens.
            all_tokens: Tensor = features  # (B, 257, 768)
            patch_tokens = all_tokens[:, 1:, :]  # (B, 256, 768)

        # Validate patch token count.
        expected_n_patches: int = self.spatial_size * self.spatial_size  # 256
        actual_n_patches: int = patch_tokens.shape[1]
        if actual_n_patches != expected_n_patches:
            raise RuntimeError(
                f"Expected {expected_n_patches} patch tokens "
                f"(spatial_size={self.spatial_size}²), "
                f"but got {actual_n_patches}. "
                f"Input shape after resize: {x_resized.shape}. "
                f"Check that the backbone patch_size={self.patch_size} and "
                f"internal resolution={self._INTERNAL_RESOLUTION} are consistent."
            )

        # --- Step 4: Project patch tokens ---
        # Shape: (B, 256, 768) → (B, 256, latent_channels)
        patch_tokens_projected: Tensor = self.proj(patch_tokens)

        # --- Step 5: Reshape to 2D spatial feature map ---
        # (B, 256, latent_channels) → (B, latent_channels, 16, 16)
        # permute: (B, N, C) → (B, C, N)
        # reshape: (B, C, N) → (B, C, H', W')
        f: Tensor = (
            patch_tokens_projected
            .permute(0, 2, 1)  # (B, latent_channels, 256)
            .reshape(batch_size, self.latent_channels, self.spatial_size, self.spatial_size)
        )

        return f

    def get_backbone_parameters(self):
        """Return an iterator over backbone parameters for differential LR.

        During FR-VAE training, the backbone is typically trained with a lower
        learning rate than the rest of the model (e.g., 0.1× the base LR).
        This method enables the trainer to set up parameter groups.

        Returns:
            Iterator over backbone nn.Parameter objects.
        """
        return self.backbone.parameters()

    def get_head_parameters(self):
        """Return an iterator over non-backbone parameters (proj layer).

        These parameters are trained at the full FR-VAE learning rate.

        Returns:
            Iterator over proj nn.Parameter objects.
        """
        return self.proj.parameters()

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (no gradient updates).

        Called by NFIGTrainer when the FR-VAE is used as a frozen tokenizer
        during NFIG Transformer training.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (enable gradient updates).

        Called to resume fine-tuning after a frozen warm-up phase.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def extra_repr(self) -> str:
        """Return a string with key encoder configuration for repr().

        Returns:
            Human-readable string describing the encoder configuration.
        """
        return (
            f"latent_channels={self.latent_channels}, "
            f"spatial_size={self.spatial_size}×{self.spatial_size}, "
            f"patch_size={self.patch_size}, "
            f"internal_resolution={self._INTERNAL_RESOLUTION}"
        )
