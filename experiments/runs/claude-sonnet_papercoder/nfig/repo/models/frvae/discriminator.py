## models/frvae/discriminator.py
"""DINO-based discriminator for adversarial training of the FR-VAE.

The DINODiscriminator uses a pretrained DINOv2-base backbone (frozen) to
extract semantically rich patch-level features, then applies a lightweight
learned head to produce spatial discriminator logits.

This design follows VAR's tokenizer discriminator (referenced in paper
Section 4.1: "we integrate the DINO discriminator from VAR's tokenizer").
The ablation study (Table 5, row 3) shows this component reduces rFID from
1.40 to 0.85 — the single largest improvement in the tokenizer pipeline.

Architecture:
    Input x ∈ R^(B, 3, 256, 256) in [-1, 1]
    → re-normalize to DINOv2 ImageNet stats
    → DINOv2-base backbone (frozen) → patch tokens [B, num_patches, 768]
    → head: Linear(768→256) → LeakyReLU(0.2) → Linear(256→1)
    → logits [B, num_patches]

The GAN loss weight of 0.5 (config.frvae.gan_loss_weight = 0.5, from paper
Appendix B.1) is applied in NFIGLosses, not here. This class only produces
raw logits for consumption by NFIGLosses.gan_discriminator_loss() and
NFIGLosses.gan_generator_loss().

Config values used:
    config.frvae.use_dino_discriminator = True
    config.frvae.gan_loss_weight = 0.5  (used in NFIGLosses, not here)
"""

import torch
import torch.nn as nn
from torch import Tensor

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "timm is required for the DINODiscriminator. "
        "Install it with: pip install timm>=0.6.14"
    ) from exc


class DINODiscriminator(nn.Module):
    """Patch-level discriminator built on a frozen DINOv2-base backbone.

    Distinguishes real ImageNet images from FR-VAE reconstructions by
    producing per-patch logits. The frozen DINOv2 backbone provides
    semantically rich, spatially aware features without requiring
    pixel-level reconstruction training.

    The backbone is always kept in eval() mode (frozen weights, no
    BatchNorm/Dropout effects). Only the lightweight head is trained
    by optimizer_d in FRVAETrainer.

    Attributes:
        backbone: Frozen pretrained DINOv2-base ViT (nn.Module).
        head: Trainable MLP head mapping patch features to logits (nn.Sequential).
        dino_mean: Registered buffer for DINOv2 ImageNet mean (1, 3, 1, 1).
        dino_std: Registered buffer for DINOv2 ImageNet std (1, 3, 1, 1).
    """

    # DINOv2 ImageNet normalization constants (fixed, non-learnable).
    # Source: https://github.com/facebookresearch/dinov2
    _DINO_MEAN: tuple = (0.485, 0.456, 0.406)
    _DINO_STD: tuple = (0.229, 0.224, 0.225)

    # DINOv2-base hidden dimension (embed_dim = 768).
    _BACKBONE_DIM: int = 768

    # Head intermediate dimension: provides capacity while staying lightweight.
    _HEAD_HIDDEN_DIM: int = 256

    # LeakyReLU negative slope: standard for GAN discriminator heads.
    _LEAKY_RELU_SLOPE: float = 0.2

    # timm model identifier for DINOv2-base (patch_size=14, embed_dim=768).
    _MODEL_NAME: str = "vit_base_patch14_dinov2"

    def __init__(self, pretrained: bool = True) -> None:
        """Initialize the DINODiscriminator.

        Loads the DINOv2-base backbone via timm, freezes all backbone
        parameters, and builds the trainable classification head.

        Args:
            pretrained: Whether to load pretrained DINOv2 weights via timm.
                Should be True in all production runs (config:
                use_dino_discriminator = True implies pretrained features).
                Set to False only for unit testing without internet access.
        """
        super().__init__()

        # --- Load DINOv2-base backbone ---
        # num_classes=0: removes the classification head; we use forward_features().
        self.backbone: nn.Module = timm.create_model(
            self._MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
        )

        # --- Freeze all backbone parameters ---
        # The backbone provides fixed semantic features throughout FR-VAE training.
        # Only the head is updated by optimizer_d. This:
        #   1. Preserves rich DINO pretraining
        #   2. Reduces memory overhead (no backbone gradients)
        #   3. Maintains a stable feature space for consistent discrimination
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Force backbone into eval mode immediately after construction.
        # This is also enforced in train() to prevent accidental mode switches.
        self.backbone.eval()

        # --- Trainable head: patch features → discriminator logits ---
        # Maps [B, num_patches, 768] → [B, num_patches, 1] via a 2-layer MLP.
        # No final activation — raw logits for hinge/BCE loss in NFIGLosses.
        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(self._BACKBONE_DIM, self._HEAD_HIDDEN_DIM, bias=True),
            nn.LeakyReLU(negative_slope=self._LEAKY_RELU_SLOPE, inplace=True),
            nn.Linear(self._HEAD_HIDDEN_DIM, 1, bias=True),
        )
        self._init_head_weights()

        # --- Re-normalization buffers ---
        # Training pipeline normalizes images to [-1, 1] (config mean=std=0.5).
        # DINOv2 expects ImageNet normalization.
        # Registered as buffers: move with .to(device)/.cuda(), saved in state_dict,
        # but NOT trainable parameters.
        dino_mean: Tensor = torch.tensor(
            self._DINO_MEAN, dtype=torch.float32
        ).view(1, 3, 1, 1)  # (1, 3, 1, 1) for broadcasting over (B, 3, H, W)

        dino_std: Tensor = torch.tensor(
            self._DINO_STD, dtype=torch.float32
        ).view(1, 3, 1, 1)  # (1, 3, 1, 1) for broadcasting over (B, 3, H, W)

        self.register_buffer("dino_mean", dino_mean, persistent=True)
        self.register_buffer("dino_std", dino_std, persistent=True)

    def _init_head_weights(self) -> None:
        """Initialize head weights for stable GAN training.

        The first linear layer uses Kaiming normal initialization (appropriate
        for LeakyReLU activations). The final linear layer uses a small normal
        initialization so early discriminator outputs are near zero, avoiding
        saturated gradients at the start of training.
        """
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=self._LEAKY_RELU_SLOPE,
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Override the final layer with small normal init for training stability.
        final_linear: nn.Linear = self.head[-1]  # type: ignore[index]
        nn.init.normal_(final_linear.weight, mean=0.0, std=0.02)
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)

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

    def _extract_patch_tokens(self, x_dino: Tensor) -> Tensor:
        """Extract patch-level features from the frozen DINOv2 backbone.

        Handles both dict and tensor return formats from different timm versions.
        The backbone is always called without gradient computation since it is
        frozen (requires_grad=False on all parameters).

        Args:
            x_dino: DINOv2-normalized image tensor of shape (B, 3, H, W).

        Returns:
            Patch token features of shape (B, num_patches, backbone_dim).
            For DINOv2-base with 256×256 input: (B, ~324, 768).
            (256/14 ≈ 18.28 → 18×18 = 324 patches for non-resized 256×256 input)

        Raises:
            RuntimeError: If the backbone output format is not recognized.
        """
        # No gradient through the frozen backbone.
        with torch.no_grad():
            features = self.backbone.forward_features(x_dino)

        # Handle different timm output formats across versions.
        if isinstance(features, dict):
            # timm >= 0.9 DINOv2 returns a dict with normalized patch tokens.
            if "x_norm_patchtokens" in features:
                # Shape: (B, num_patches, 768) — patch tokens without CLS.
                patch_tokens: Tensor = features["x_norm_patchtokens"]
            elif "x_prenorm" in features:
                # Pre-norm tokens: skip CLS token at index 0.
                all_tokens: Tensor = features["x_prenorm"]
                patch_tokens = all_tokens[:, 1:, :]
            else:
                # Fallback: try the first tensor value in the dict.
                first_value = next(iter(features.values()))
                if isinstance(first_value, Tensor) and first_value.dim() == 3:
                    # Assume [B, 1+num_patches, dim] format; skip CLS.
                    patch_tokens = first_value[:, 1:, :]
                else:
                    raise RuntimeError(
                        f"Unrecognized backbone output dict keys: {list(features.keys())}. "
                        "Expected 'x_norm_patchtokens' or 'x_prenorm'."
                    )
        elif isinstance(features, Tensor):
            # Older timm versions return a tensor of shape (B, 1+num_patches, dim).
            # Index 0 is the CLS token; indices 1: are patch tokens.
            if features.dim() == 3:
                patch_tokens = features[:, 1:, :]
            elif features.dim() == 2:
                # Some models return (B, dim) — global feature only.
                # Unsqueeze to (B, 1, dim) for consistent downstream processing.
                patch_tokens = features.unsqueeze(1)
            else:
                raise RuntimeError(
                    f"Unexpected backbone output tensor shape: {features.shape}. "
                    "Expected 3D tensor (B, 1+num_patches, dim) or 2D (B, dim)."
                )
        else:
            raise RuntimeError(
                f"Backbone forward_features() returned unexpected type: "
                f"{type(features).__name__}. Expected dict or Tensor."
            )

        return patch_tokens

    def forward(self, x: Tensor) -> Tensor:
        """Compute discriminator logits for a batch of images.

        Implements the full discrimination pipeline:
            x [B, 3, 256, 256] in [-1, 1]
            → re-normalize to DINOv2 stats
            → frozen backbone → patch tokens [B, num_patches, 768]
            → head (Linear → LeakyReLU → Linear) → [B, num_patches, 1]
            → squeeze → logits [B, num_patches]

        The output logits are consumed by NFIGLosses:
            - gan_discriminator_loss(real_logits, fake_logits): called with
              real_logits = discriminator(x) and
              fake_logits = discriminator(x_hat.detach())
            - gan_generator_loss(fake_logits): called with
              fake_logits = discriminator(x_hat) (no detach)

        Args:
            x: Image batch of shape (B, 3, H, W) with values in [-1, 1].
               Accepts both real images (from DataLoader) and reconstructed
               images (from FR-VAE decoder). Typically H = W = 256.

        Returns:
            Spatial discriminator logits of shape (B, num_patches).
            Each value is a raw (un-activated) logit for one image patch.
            Higher values indicate the discriminator judges the patch as real.
            NFIGLosses averages over the patch dimension when computing loss.
        """
        # --- Step 1: Re-normalize from [-1, 1] to DINOv2 ImageNet stats ---
        x_dino: Tensor = self._renormalize(x)

        # --- Step 2: Extract patch tokens via frozen DINOv2 backbone ---
        # Shape: (B, num_patches, 768)
        patch_tokens: Tensor = self._extract_patch_tokens(x_dino)

        # --- Step 3: Apply trainable head to each patch independently ---
        # The head is applied to the last dimension (feature dim = 768).
        # Input:  (B, num_patches, 768)
        # Output: (B, num_patches, 1)
        logits: Tensor = self.head(patch_tokens)

        # --- Step 4: Squeeze the trailing singleton dimension ---
        # (B, num_patches, 1) → (B, num_patches)
        logits = logits.squeeze(-1)

        return logits

    def train(self, mode: bool = True) -> "DINODiscriminator":
        """Override train() to always keep the backbone in eval mode.

        The backbone is frozen and must remain in eval() mode regardless of
        whether the discriminator as a whole is in training or eval mode.
        This prevents any BatchNorm or Dropout layers in the backbone from
        switching to training behavior, which would destabilize the fixed
        feature space.

        Only the head switches between train/eval modes normally.

        Args:
            mode: If True, sets the module to training mode. If False, sets
                it to evaluation mode. The backbone always stays in eval().

        Returns:
            self (for method chaining, consistent with nn.Module.train()).
        """
        # Apply the requested mode to the full module (including head).
        super().train(mode)

        # Override: always force backbone back to eval mode.
        # This undoes any train() call that super().train(mode) applied to backbone.
        self.backbone.eval()

        return self

    def extra_repr(self) -> str:
        """Return a human-readable string with key discriminator configuration.

        Returns:
            String describing the discriminator's backbone and head architecture.
        """
        # Count trainable parameters in the head.
        head_params: int = sum(
            p.numel() for p in self.head.parameters() if p.requires_grad
        )
        # Count total backbone parameters (all frozen).
        backbone_params: int = sum(p.numel() for p in self.backbone.parameters())

        return (
            f"backbone='{self._MODEL_NAME}' (frozen, {backbone_params:,} params), "
            f"head=Linear({self._BACKBONE_DIM}→{self._HEAD_HIDDEN_DIM})"
            f"→LeakyReLU({self._LEAKY_RELU_SLOPE})→Linear({self._HEAD_HIDDEN_DIM}→1) "
            f"({head_params:,} trainable params)"
        )
