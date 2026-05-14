## models/vit_wrapper.py
"""ViT-B/16 backbone wrapper for the PEFT Visual Recognition reproduction study.

This module provides the ViTWrapper class, which loads and manages the
Vision Transformer (ViT-B/16) backbone pretrained on ImageNet-21K via the
`timm` library. It serves as the foundational backbone manager for all
VTAB-1K, many-shot, and robustness experiments.

Key responsibilities:
- Load ViT-B/16 (ImageNet-21K) with configurable drop_path_rate
- Expose the backbone and individual Transformer blocks for PEFT injection
- Provide freeze/unfreeze utilities for parameter management
- Expose trainable parameter access for optimizer construction
- Hold architecture constants for param cap enforcement

Paper reference: "We employ the ViT-B/16 pre-trained on ImageNet-21K as the
backbone." (Section 3, Appendix A.1)

Config reference: config.yaml -> backbones.imagenet21k_vit

Typical usage:
    vit_wrapper = ViTWrapper(
        model_name="vit_base_patch16_224_in21k",
        pretrained=True,
        drop_path_rate=0.1,
    )
    backbone = vit_wrapper.get_backbone()
    layer_0 = vit_wrapper.get_layer(0)
    vit_wrapper.freeze_backbone()
    trainable = vit_wrapper.get_trainable_params()
    n_params = vit_wrapper.count_trainable_params()
"""

import logging
from typing import List

import timm
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture constants from config.yaml: backbones.imagenet21k_vit
# ---------------------------------------------------------------------------

# Approximate total parameter count for ViT-B/16 (~86M).
# Used by PEFTFactory.check_param_cap() to compute the absolute maximum
# of 0.015 * 86_000_000 = 1_290_000 trainable parameters.
# config.yaml: backbones.imagenet21k_vit.total_params: 86_000_000
TOTAL_PARAMS: int = 86_000_000

# Embedding dimension D for ViT-B/16.
# config.yaml: backbones.imagenet21k_vit.embed_dim: 768
VIT_EMBED_DIM: int = 768

# Number of Transformer layers M for ViT-B/16.
# config.yaml: backbones.imagenet21k_vit.num_layers: 12
VIT_NUM_LAYERS: int = 12

# Spatial patch size P for ViT-B/16.
# config.yaml: backbones.imagenet21k_vit.patch_size: 16
VIT_PATCH_SIZE: int = 16

# Number of attention heads for ViT-B/16.
# config.yaml: backbones.imagenet21k_vit.num_heads: 12
VIT_NUM_HEADS: int = 12

# Default model name matching config.yaml: backbones.imagenet21k_vit.name
_DEFAULT_MODEL_NAME: str = "vit_base_patch16_224_in21k"


class ViTWrapper:
    """Wrapper for the ViT-B/16 backbone pretrained on ImageNet-21K.

    Loads the backbone via `timm.create_model` with `num_classes=0` to
    obtain a pure feature extractor (no classification head). The head is
    created separately in `PEFTFactory` as a randomly initialized
    `nn.Linear(embed_dim, num_classes)` per downstream dataset.

    The `drop_path_rate` parameter is passed directly to `timm.create_model`,
    enabling stochastic depth regularization. The paper identifies this as
    critically important: "we find the drop path rate particularly important.
    Ignoring it (i.e., setting it to 0) significantly degrades the performance,
    potentially due to over-fitting." (Section 3)

    After construction, the backbone is NOT frozen — `freeze_backbone()` is
    called explicitly by `PEFTFactory.build()` before applying PEFT
    modifications. This separation allows `PEFTFactory` to control the freeze
    state based on the method (e.g., full FT never calls `freeze_backbone()`).

    Attributes:
        backbone: The timm VisionTransformer instance with num_classes=0.
            Returns CLS token features of shape (B, embed_dim) from forward().
        embed_dim: Token embedding dimension D = 768.
        num_layers: Number of Transformer blocks M = 12.
        patch_size: Spatial patch size P = 16.
        num_heads: Number of attention heads = 12.
        model_name: The timm model identifier string.
        drop_path_rate: Stochastic depth rate applied to all blocks.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
    ) -> None:
        """Loads the ViT-B/16 backbone from timm.

        Args:
            model_name: timm model identifier. Default: 'vit_base_patch16_224_in21k'
                (config.yaml: backbones.imagenet21k_vit.name).
                This maps to ViT-B/16 pretrained on ImageNet-21K (~14M images).
            pretrained: If True, downloads and loads ImageNet-21K pretrained
                weights from timm's model hub. Default: True
                (config.yaml: backbones.imagenet21k_vit.pretrained: true).
            drop_path_rate: Stochastic depth drop rate applied linearly across
                the 12 Transformer blocks. Default: 0.0 (no drop path).
                The paper's search grid is {0.0, 0.1}
                (config.yaml: vtab.hyperparam_search.drop_path_rate).
                Setting this to 0.1 is a key finding of the paper.

        Raises:
            RuntimeError: If timm fails to load the model (e.g., network
                unavailable for pretrained weights download).
            ValueError: If model_name is not a valid timm model identifier.
        """
        self.model_name: str = model_name
        self.drop_path_rate: float = drop_path_rate

        # Architecture constants from config.yaml: backbones.imagenet21k_vit
        self.embed_dim: int = VIT_EMBED_DIM
        self.num_layers: int = VIT_NUM_LAYERS
        self.patch_size: int = VIT_PATCH_SIZE
        self.num_heads: int = VIT_NUM_HEADS

        _logger.info(
            "Loading ViT backbone: model_name='%s', pretrained=%s, "
            "drop_path_rate=%.2f",
            model_name,
            pretrained,
            drop_path_rate,
        )

        # ------------------------------------------------------------------
        # Load the backbone via timm.
        # num_classes=0: removes the classification head, returning the
        # CLS token feature (shape: B×768) from forward().
        # drop_path_rate: passed directly to timm's ViT constructor, which
        # distributes it linearly across the 12 blocks.
        # ------------------------------------------------------------------
        try:
            self.backbone: nn.Module = timm.create_model(
                model_name,
                pretrained=pretrained,
                drop_path_rate=drop_path_rate,
                num_classes=0,  # Feature extractor only; head created in PEFTFactory
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Failed to load timm model '{model_name}' "
                f"(pretrained={pretrained}, drop_path_rate={drop_path_rate}): {exc}\n"
                "Ensure timm is installed and the model name is valid. "
                "For pretrained=True, an internet connection is required on first run."
            ) from exc

        # ------------------------------------------------------------------
        # Validate that the loaded model has the expected architecture.
        # This guards against accidentally loading a different ViT variant.
        # ------------------------------------------------------------------
        self._validate_architecture()

        _logger.info(
            "ViT backbone loaded successfully: embed_dim=%d, num_layers=%d, "
            "patch_size=%d, num_heads=%d, total_params=%d",
            self.embed_dim,
            self.num_layers,
            self.patch_size,
            self.num_heads,
            sum(p.numel() for p in self.backbone.parameters()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_backbone(self) -> nn.Module:
        """Returns the timm VisionTransformer backbone module.

        The returned module is the raw timm VisionTransformer with
        num_classes=0. Its forward() method returns CLS token features
        of shape (B, embed_dim) = (B, 768).

        This method is called by PEFTFactory.build(), which deepcopies the
        returned backbone for each hyperparameter trial to avoid state sharing:
            backbone_copy = copy.deepcopy(vit_wrapper.get_backbone())

        Returns:
            The timm VisionTransformer nn.Module instance.
        """
        return self.backbone

    def get_layer(self, idx: int) -> nn.Module:
        """Returns the Transformer block at the given index.

        The timm VisionTransformer stores its 12 Transformer blocks in
        self.backbone.blocks (a nn.Sequential). Each block is a
        timm.models.vision_transformer.Block instance with the structure:

            Block
            ├── norm1 (LayerNorm: 768)       — h₂ = norm1(h₁)
            ├── attn (Attention)              — h₃..h₅
            │   ├── qkv (Linear: 768→2304)   — fused Q, K, V projections
            │   ├── attn_drop (Dropout)
            │   └── proj (Linear: 768→768)   — FC_attn output projection
            ├── drop_path (DropPath/Identity) — stochastic depth
            ├── norm2 (LayerNorm: 768)        — h₆ = norm2(h₅)
            └── mlp (Mlp)                     — h₇..h₉
                ├── fc1 (Linear: 768→3072)
                ├── act (GELU)
                ├── drop (Dropout)
                └── fc2 (Linear: 3072→768)

        PEFT methods use this to inject modules at specific positions:
        - VPT: wraps block forward to prepend prompt tokens
        - Adapters: wraps block forward to inject adapter modules
        - LoRA: replaces block.attn.qkv with LoRALinear
        - FacT: registers forward hooks on weight matrices
        - Selective: iterates block parameters for selective unfreezing

        Args:
            idx: Block index in [0, num_layers - 1]. Block 0 is the first
                Transformer layer (closest to the patch embedding).

        Returns:
            The timm Block nn.Module at position idx.

        Raises:
            IndexError: If idx is outside [0, num_layers - 1].
        """
        if not (0 <= idx < self.num_layers):
            raise IndexError(
                f"Layer index {idx} is out of range. "
                f"ViT-B/16 has {self.num_layers} layers (indices 0 to {self.num_layers - 1})."
            )
        return self.backbone.blocks[idx]

    def freeze_backbone(self) -> None:
        """Freezes all backbone parameters by setting requires_grad=False.

        This is the default state for all PEFT methods — the backbone is
        frozen and only PEFT-specific parameters are trained. Called by
        PEFTFactory.build() before applying PEFT modifications.

        After calling this method, get_trainable_params() returns an empty
        list until PEFT modules are applied (which selectively re-enable
        requires_grad on specific parameters).

        Note: For full fine-tuning (method='full'), PEFTFactory calls
        unfreeze_backbone() instead of freeze_backbone().
        """
        frozen_count: int = 0
        for param in self.backbone.parameters():
            param.requires_grad = False
            frozen_count += 1

        _logger.info(
            "Backbone frozen: %d parameter tensors set to requires_grad=False.",
            frozen_count,
        )

    def unfreeze_backbone(self) -> None:
        """Unfreezes all backbone parameters by setting requires_grad=True.

        Used for full fine-tuning (method='full') where all backbone
        parameters are trained end-to-end. Called by PEFTFactory.build()
        when method='full'.

        After calling this method, get_trainable_params() returns all
        backbone parameters.
        """
        unfrozen_count: int = 0
        for param in self.backbone.parameters():
            param.requires_grad = True
            unfrozen_count += 1

        _logger.info(
            "Backbone unfrozen: %d parameter tensors set to requires_grad=True.",
            unfrozen_count,
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Returns all backbone parameters with requires_grad=True.

        This list is passed to the AdamW optimizer in Trainer._build_optimizer().
        Must be called AFTER PEFT modifications are applied (i.e., after
        PEFTFactory.build() completes), not before.

        Call sequence:
            1. vit_wrapper = ViTWrapper(...)
            2. backbone = copy.deepcopy(vit_wrapper.get_backbone())
            3. Apply PEFT modifications (selectively re-enable requires_grad)
            4. peft_model = PEFTModel(backbone, head, peft_module, method)
            5. trainable = peft_model.get_trainable_params()  # includes PEFT params

        Note: PEFTModel.get_trainable_params() is the primary method used
        by Trainer. This method on ViTWrapper is a utility for inspecting
        backbone-only trainable parameters.

        Returns:
            List of nn.Parameter objects with requires_grad=True from the
            backbone. Empty list if freeze_backbone() was called and no
            PEFT modifications have re-enabled any parameters.
        """
        return [
            param
            for param in self.backbone.parameters()
            if param.requires_grad
        ]

    def count_trainable_params(self) -> int:
        """Returns the total number of trainable backbone parameters.

        Sums numel() for all parameters with requires_grad=True. Used by
        PEFTFactory.check_param_cap() to enforce the ≤1.5% constraint:

            cap = 0.015 * TOTAL_PARAMS = 0.015 * 86_000_000 = 1_290_000

        Paper: "We set a cap for PEFT size ≤ 1.5% of ViT-B/16."
        Config: config.yaml -> peft_param_cap.ratio: 0.015

        Returns:
            Integer count of trainable backbone parameters. Returns 0 if
            freeze_backbone() was called and no PEFT modifications have
            re-enabled any parameters.
        """
        return sum(
            param.numel()
            for param in self.backbone.parameters()
            if param.requires_grad
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_architecture(self) -> None:
        """Validates that the loaded model matches the expected ViT-B/16 architecture.

        Checks that the backbone has the expected number of Transformer blocks
        and the correct embedding dimension. Logs a warning (rather than
        raising) if the architecture does not match, to allow flexibility
        with alternative ViT variants (e.g., ViT-L, ViT-H for Appendix C).

        The validation checks:
        1. backbone.blocks exists and has num_layers blocks
        2. backbone.embed_dim == VIT_EMBED_DIM (768)
        3. backbone.patch_embed.proj.kernel_size matches patch_size

        Raises:
            AttributeError: If the backbone does not have the expected
                timm VisionTransformer structure (blocks, embed_dim).
        """
        # ------------------------------------------------------------------
        # Check 1: Verify blocks attribute exists and has correct length.
        # ------------------------------------------------------------------
        if not hasattr(self.backbone, "blocks"):
            _logger.warning(
                "Loaded model '%s' does not have a 'blocks' attribute. "
                "Expected timm VisionTransformer structure. "
                "PEFT methods that use get_layer() may fail.",
                self.model_name,
            )
            return

        actual_num_layers: int = len(self.backbone.blocks)
        if actual_num_layers != self.num_layers:
            _logger.warning(
                "Model '%s' has %d Transformer blocks, expected %d. "
                "Architecture constants (num_layers=%d) may be incorrect for this model.",
                self.model_name,
                actual_num_layers,
                self.num_layers,
                self.num_layers,
            )
            # Update num_layers to match actual model for correct PEFT injection.
            self.num_layers = actual_num_layers
            _logger.info(
                "Updated num_layers to %d to match loaded model.",
                self.num_layers,
            )

        # ------------------------------------------------------------------
        # Check 2: Verify embedding dimension.
        # ------------------------------------------------------------------
        if hasattr(self.backbone, "embed_dim"):
            actual_embed_dim: int = self.backbone.embed_dim
            if actual_embed_dim != self.embed_dim:
                _logger.warning(
                    "Model '%s' has embed_dim=%d, expected %d. "
                    "Updating embed_dim to match loaded model.",
                    self.model_name,
                    actual_embed_dim,
                    self.embed_dim,
                )
                self.embed_dim = actual_embed_dim

        # ------------------------------------------------------------------
        # Check 3: Verify patch size via patch_embed.proj.kernel_size.
        # ------------------------------------------------------------------
        if hasattr(self.backbone, "patch_embed") and hasattr(
            self.backbone.patch_embed, "proj"
        ):
            proj = self.backbone.patch_embed.proj
            if hasattr(proj, "kernel_size"):
                kernel_size = proj.kernel_size
                # kernel_size is a tuple (P, P) for Conv2d.
                actual_patch_size: int = (
                    kernel_size[0]
                    if isinstance(kernel_size, (tuple, list))
                    else int(kernel_size)
                )
                if actual_patch_size != self.patch_size:
                    _logger.warning(
                        "Model '%s' has patch_size=%d, expected %d. "
                        "Updating patch_size to match loaded model.",
                        self.model_name,
                        actual_patch_size,
                        self.patch_size,
                    )
                    self.patch_size = actual_patch_size

        # ------------------------------------------------------------------
        # Check 4: Verify num_heads via first block's attention module.
        # ------------------------------------------------------------------
        if actual_num_layers > 0:
            first_block = self.backbone.blocks[0]
            if hasattr(first_block, "attn") and hasattr(first_block.attn, "num_heads"):
                actual_num_heads: int = first_block.attn.num_heads
                if actual_num_heads != self.num_heads:
                    _logger.warning(
                        "Model '%s' has num_heads=%d, expected %d. "
                        "Updating num_heads to match loaded model.",
                        self.model_name,
                        actual_num_heads,
                        self.num_heads,
                    )
                    self.num_heads = actual_num_heads

        _logger.debug(
            "Architecture validation complete: num_layers=%d, embed_dim=%d, "
            "patch_size=%d, num_heads=%d",
            self.num_layers,
            self.embed_dim,
            self.patch_size,
            self.num_heads,
        )
