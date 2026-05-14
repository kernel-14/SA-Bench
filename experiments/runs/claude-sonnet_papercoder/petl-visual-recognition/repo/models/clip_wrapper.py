## Code: models/clip_wrapper.py

```python
## models/clip_wrapper.py
"""CLIP ViT-B/16 wrapper for the PEFT Visual Recognition robustness experiment.

This module provides the CLIPWrapper class, which loads and manages the CLIP
ViT-B/16 model for the robustness evaluation described in Section 7 of the
paper: "How Robust are PEFT Methods to Distribution Shifts?"

The wrapper handles:
- Loading CLIP ViT-B/16 via open_clip
- Building the zero-shot classification head from 80 CLIP prompt templates
  (paper: "we ensemble the 80 prompts provided by CLIP")
- Exposing the visual encoder for PEFT module injection
- Freezing the text encoder after head construction
- Providing trainable parameter access for optimizer construction

Paper reference: "We focus on the CLIP ViT-B/16 model, which comprises a
visual encoder and a text encoder, pre-trained via contrastive learning on
image-text pairs. Following [96], we add an FC layer as the head initialized
using the class label text embedded by the text encoder. Subsequently, we
discard the text encoder and apply PEFT methods to the visual encoder."
(Section 7, Appendix A.1)

Config reference: config.yaml -> backbones.clip_vit

Typical usage:
    clip_wrapper = CLIPWrapper(model_name='ViT-B/16')
    clip_wrapper.freeze_text_encoder()
    zeroshot_weights = clip_wrapper.build_zeroshot_head(
        classnames=imagenet_classnames,
        templates=CLIP_IMAGENET_TEMPLATES,
        device='cuda',
    )
    visual_encoder = clip_wrapper.get_visual_encoder()
    trainable_params = clip_wrapper.get_trainable_params()
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLIP model name mapping: OpenAI convention -> open_clip convention
# config.yaml: backbones.clip_vit.name: 'ViT-B/16'
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME_MAP: Dict[str, str] = {
    "ViT-B/16": "ViT-B-16",
    "ViT-L/14": "ViT-L-14",
    "ViT-H/14": "ViT-H-14",
    "ViT-B/32": "ViT-B-32",
}

# ---------------------------------------------------------------------------
# Architecture constants for CLIP ViT-B/16
# config.yaml: backbones.clip_vit
# ---------------------------------------------------------------------------

# Joint image-text embedding dimension (output of visual projection head).
# config.yaml: backbones.clip_vit.embed_dim: 768
# Note: The config lists 768 as embed_dim, but CLIP ViT-B/16's joint space
# is 512-d. We store both: visual_embed_dim=768 (internal ViT) and
# embed_dim=512 (projected joint space for zero-shot head).
CLIP_JOINT_EMBED_DIM: int = 512

# Internal ViT embedding dimension (before projection).
CLIP_VISUAL_EMBED_DIM: int = 768

# Number of Transformer layers in the visual encoder.
# config.yaml: backbones.clip_vit.num_layers: 12
CLIP_NUM_LAYERS: int = 12

# Spatial patch size.
# config.yaml: backbones.clip_vit.patch_size: 16
CLIP_PATCH_SIZE: int = 16

# Default pretrained weights identifier for open_clip.
_DEFAULT_PRETRAINED: str = "openai"

# Default model name (OpenAI convention).
_DEFAULT_MODEL_NAME: str = "ViT-B/16"

# Batch size for text encoding during zero-shot head construction.
# Avoids OOM when encoding 1000 classes × 80 templates = 80,000 strings.
_TEXT_ENCODE_BATCH_SIZE: int = 256

# ---------------------------------------------------------------------------
# 80 CLIP ImageNet prompt templates
# Source: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
# config.yaml: robustness.num_clip_prompts: 80
# These are hardcoded to avoid runtime dependency on external URLs.
# ---------------------------------------------------------------------------
CLIP_IMAGENET_TEMPLATES: List[str] = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

# Verify we have exactly 80 templates (config.yaml: robustness.num_clip_prompts: 80).
assert len(CLIP_IMAGENET_TEMPLATES) == 80, (
    f"Expected 80 CLIP ImageNet templates, got {len(CLIP_IMAGENET_TEMPLATES)}."
)


# ---------------------------------------------------------------------------
# Public API: CLIPWrapper
# ---------------------------------------------------------------------------

class CLIPWrapper:
    """Wrapper for the CLIP ViT-B/16 model for the robustness experiment.

    Manages the CLIP model lifecycle for Section 7 of the paper:
    1. Loads CLIP ViT-B/16 via open_clip
    2. Builds the zero-shot classification head from 80 prompt templates
    3. Freezes the text encoder after head construction
    4. Exposes the visual encoder for PEFT module injection

    This class is NOT an nn.Module — it is a utility wrapper that manages
    the CLIP model object and provides access to its components. The actual
    nn.Module used in training is PEFTModel, which receives the visual
    encoder from get_visual_encoder().

    Architecture note on embed_dim:
        - self.embed_dim = 512: Joint image-text embedding space dimension.
          Used for zero-shot head construction (W_zeroshot shape: 512×num_classes).
        - self.visual_embed_dim = 768: Internal ViT-B/16 embedding dimension.
          Used by PEFT modules that hook into the Transformer blocks.

    Attributes:
        model_name: OpenAI-convention model name (e.g., 'ViT-B/16').
        open_clip_name: open_clip-convention model name (e.g., 'ViT-B-16').
        model: The full open_clip CLIP model (visual + text components).
        visual_encoder: The visual encoder module (model.visual).
        embed_dim: Joint embedding space dimension = 512.
        visual_embed_dim: Internal ViT embedding dimension = 768.
        num_layers: Number of Transformer blocks = 12.
        patch_size: Spatial patch size = 16.
        tokenizer: open_clip tokenizer for text encoding.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
    ) -> None:
        """Loads the CLIP ViT-B/16 model via open_clip.

        Loads the model with OpenAI pretrained weights. The visual encoder
        is extracted and stored separately for PEFT injection. The text
        encoder components remain accessible via self.model for zero-shot
        head construction.

        Args:
            model_name: CLIP model name in OpenAI convention.
                Default: 'ViT-B/16' (config.yaml: backbones.clip_vit.name).
                Supported values: 'ViT-B/16', 'ViT-L/14', 'ViT-H/14', 'ViT-B/32'.

        Raises:
            ImportError: If open_clip_torch is not installed.
            RuntimeError: If open_clip fails to load the model (e.g., network
                unavailable for pretrained weights download).
            ValueError: If model_name is not in CLIP_MODEL_NAME_MAP.
        """
        # ------------------------------------------------------------------
        # Step 1: Validate and map model name.
        # ------------------------------------------------------------------
        if model_name not in CLIP_MODEL_NAME_MAP:
            raise ValueError(
                f"Unknown CLIP model name: '{model_name}'. "
                f"Supported names: {list(CLIP_MODEL_NAME_MAP.keys())}"
            )

        self.model_name: str = model_name
        self.open_clip_name: str = CLIP_MODEL_NAME_MAP[model_name]

        # Architecture constants from config.yaml: backbones.clip_vit
        self.embed_dim: int = CLIP_JOINT_EMBED_DIM          # 512: joint space
        self.visual_embed_dim: int = CLIP_VISUAL_EMBED_DIM  # 768: internal ViT
        self.num_layers: int = CLIP_NUM_LAYERS               # 12
        self.patch_size: int = CLIP_PATCH_SIZE               # 16

        _logger.info(
            "Loading CLIP model: model_name='%s' (open_clip_name='%s'), "
            "pretrained='%s'",
            self.model_name,
            self.open_clip_name,
            _DEFAULT_PRETRAINED,
        )

        # ------------------------------------------------------------------
        # Step 2: Import open_clip and load the model.
        # ------------------------------------------------------------------
        try:
            import open_clip  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for the CLIP robustness experiment. "
                "Install with: pip install open_clip_torch==2.23.0"
            ) from exc

        try:
            self.model: nn.Module
            self.model, _, _ = open_clip.create_model_and_transforms(
                self.open_clip_name,
                pretrained=_DEFAULT_PRETRAINED,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Failed to load CLIP model '{self.open_clip_name}' "
                f"(pretrained='{_DEFAULT_PRETRAINED}'): {exc}\n"
                "Ensure open_clip_torch is installed and an internet connection "
                "is available for the first download."
            ) from exc

        # ------------------------------------------------------------------
        # Step 3: Extract the visual encoder.
        # In open_clip, the visual encoder is model.visual.
        # ------------------------------------------------------------------
        if not hasattr(self.model, "visual"):
            raise RuntimeError(
                f"Loaded CLIP model '{self.open_clip_name}' does not have a "
                "'visual' attribute. The open_clip API may have changed."
            )

        self.visual_encoder: nn.Module = self.model.visual

        # ------------------------------------------------------------------
        # Step 4: Load the tokenizer.
        # ------------------------------------------------------------------
        try:
            self.tokenizer = open_clip.get_tokenizer(self.open_clip_name)
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "Failed to load open_clip tokenizer for '%s': %s. "
                "Falling back to open_clip.tokenize().",
                self.open_clip_name,
                exc,
            )
            self.tokenizer = None

        # ------------------------------------------------------------------
        # Step 5: Validate and update architecture constants from loaded model.
        # ------------------------------------------------------------------
        self._validate_and_update_architecture()

        _logger.info(
            "CLIP model loaded successfully: embed_dim=%d, visual_embed_dim=%d, "
            "num_layers=%d, patch_size=%d",
            self.embed_dim,
            self.visual_embed_dim,
            self.num_layers,
            self.patch_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_zeroshot_head(
        self,
        classnames: List[str],
        templates: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> nn.Parameter:
        """Builds the zero-shot classification head from CLIP text embeddings.

        For each class, formats all 80 prompt templates with the class name,
        encodes them through the text encoder, normalizes, averages, and
        re-normalizes to produce one embedding per class. The resulting
        weight matrix W_zeroshot of shape (embed_dim, num_classes) = (512, 1000)
        serves as the initialization for the nn.Linear classification head
        in PEFTModel.

        Paper: "we add an FC layer as the head initialized using the class
        label text embedded by the text encoder. Subsequently, we discard
        the text encoder and apply PEFT methods to the visual encoder."
        (Section 7)

        Paper Appendix A.1: "we ensemble the 80 prompts provided by CLIP at
        https://github.com/openai/CLIP"
        Config: config.yaml -> robustness.num_clip_prompts: 80

        The returned parameter has requires_grad=False. PEFTFactory should
        create an nn.Linear head and initialize its weight with this matrix
        (transposed), then set requires_grad=True on the head parameters.

        Args:
            classnames: List of class name strings. For ImageNet-1K, this is
                a list of 1000 class names (e.g., ['tench', 'goldfish', ...]).
                The number of classes is inferred from len(classnames).
            templates: List of prompt template strings with a single '{}' 
                placeholder for the class name. Default: CLIP_IMAGENET_TEMPLATES
                (80 templates from the CLIP paper).
                Config: config.yaml -> robustness.num_clip_prompts: 80
            device: Device for text encoding computation. Default: 'cpu'.
                Should match the device used for training. The returned
                parameter is always on CPU.

        Returns:
            nn.Parameter of shape (embed_dim, num_classes) = (512, num_classes)
            with requires_grad=False. Each column is the normalized, averaged
            text embedding for one class across all prompt templates.

        Raises:
            RuntimeError: If text encoding fails.
            ValueError: If classnames is empty or templates is empty.
        """
        if not classnames:
            raise ValueError("classnames must be a non-empty list of class name strings.")

        if templates is None:
            templates = CLIP_IMAGENET_TEMPLATES

        if not templates:
            raise ValueError("templates must be a non-empty list of prompt template strings.")

        num_classes: int = len(classnames)
        num_templates: int = len(templates)

        _logger.info(
            "Building zero-shot head: %d classes × %d templates = %d text strings. "
            "Device: %s",
            num_classes,
            num_templates,
            num_classes * num_templates,
            device,
        )

        # ------------------------------------------------------------------
        # Step 1: Move model to the target device for encoding.
        # ------------------------------------------------------------------
        self.model.to(device)
        self.model.eval()

        # ------------------------------------------------------------------
        # Step 2: Import open_clip for tokenization.
        # ------------------------------------------------------------------
        try:
            import open_clip  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for zero-shot head construction."
            ) from exc

        # ------------------------------------------------------------------
        # Step 3: Encode text embeddings for all classes.
        # Process in batches to avoid OOM with 80,000 strings.
        # ------------------------------------------------------------------
        all_class_embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            for class_idx, classname in enumerate(classnames):
                # Format all templates for this class.
                # Handle class names with special characters gracefully.
                formatted_texts: List[str] = [
                    template.format(classname.replace("_", " ").lower())
                    for template in templates
                ]

                # Tokenize all formatted texts for this class.
                try:
                    if self.tokenizer is not None:
                        tokens: torch.Tensor = self.tokenizer(formatted_texts)
                    else:
                        tokens = open_clip.tokenize(formatted_texts)
                except Exception as exc:  # pylint: disable=broad-except
                    _logger.warning(
                        "Tokenization failed for class '%s': %s. "
                        "Attempting fallback tokenization.",
                        classname,
                        exc,
                    )
                    tokens = open_clip.tokenize(formatted_texts)

                tokens = tokens.to(device)

                # Encode in batches to avoid OOM.
                class_text_features: List[torch.Tensor] = []

                for batch_start in range(0, len(formatted_texts), _TEXT_ENCODE_BATCH_SIZE):
                    batch_tokens: torch.Tensor = tokens[
                        batch_start: batch_start + _TEXT_ENCODE_BATCH_SIZE
                    ]

                    # Encode text tokens through the CLIP text encoder.
                    # open_clip model.encode_text() returns normalized embeddings
                    # of shape (batch_size, embed_dim).
                    try:
                        batch_features: torch.Tensor = self.model.encode_text(batch_tokens)
                    except Exception as exc:  # pylint: disable=broad-except
                        raise RuntimeError(
                            f"Text encoding failed for class '{classname}' "
                            f"(batch_start={batch_start}): {exc}"
                        ) from exc

                    # Normalize each embedding to unit norm.
                    batch_features = batch_features / (
                        batch_features.norm(dim=-1, keepdim=True) + 1e-8
                    )
                    class_text_features.append(batch_features.cpu())

                # Concatenate all batches for this class: (num_templates, embed_dim)
                class_features: torch.Tensor = torch.cat(class_text_features, dim=0)

                # Average across all templates: (embed_dim,)
                mean_features: torch.Tensor = class_features.mean(dim=0)

                # Re-normalize the averaged embedding.
                mean_features = mean_features / (
                    mean_features.norm(dim=-1, keepdim=True) + 1e-8
                )

                all_class_embeddings.append(mean_features)

                if (class_idx + 1) % 100 == 0:
                    _logger.info(
                        "Zero-shot head: encoded %d / %d classes.",
                        class_idx + 1,
                        num_classes,
                    )

        # ------------------------------------------------------------------
        # Step 4: Stack all class embeddings into W_zeroshot.
        # Shape: (num_classes, embed_dim) -> transpose to (embed_dim, num_classes)
        # ------------------------------------------------------------------
        # Stack: (num_classes, embed_dim)
        zeroshot_weights: torch.Tensor = torch.stack(all_class_embeddings, dim=0)

        # Transpose to (embed_dim, num_classes) = (512, 1000) for ImageNet.
        # This matches the convention where nn.Linear(embed_dim, num_classes)
        # has weight shape (num_classes, embed_dim), so we store the transpose
        # here and the caller transposes back when initializing nn.Linear.weight.
        zeroshot_weights = zeroshot_weights.t().contiguous()  # (embed_dim, num_classes)

        _logger.info(
            "Zero-shot head built: W_zeroshot shape = %s (embed_dim=%d, num_classes=%d).",
            tuple(zeroshot_weights.shape),
            self.embed_dim,
            num_classes,
        )

        # ------------------------------------------------------------------
        # Step 5: Move model back to CPU to free GPU memory.
        # The caller will move the visual encoder to the target device.
        # ------------------------------------------------------------------
        self.model.cpu()

        # ------------------------------------------------------------------
        # Step 6: Return as nn.Parameter with requires_grad=False.
        # PEFTFactory will use this to initialize the head's weight matrix.
        # ------------------------------------------------------------------
        return nn.Parameter(zeroshot_weights.cpu(), requires_grad=False)

    def get_visual_encoder(self) -> nn.Module:
        """Returns the CLIP visual encoder module.

        The visual encoder is model.visual in open_clip. It is a ViT-B/16
        transformer that processes images and returns visual features.

        In open_clip, the visual encoder's forward() method returns the
        projected embedding of shape (B, embed_dim) = (B, 512) after the
        final projection layer (visual.proj). The internal CLS token feature
        before projection has shape (B, visual_embed_dim) = (B, 768).

        This module is passed to PEFTFactory.build() as the backbone for
        PEFT injection. PEFTFactory will deepcopy this module for each
        hyperparameter trial.

        Returns:
            The open_clip visual encoder nn.Module (model.visual).
            Its transformer blocks are accessible via:
                visual_encoder.transformer.resblocks[idx]
            for PEFT module injection.
        """
        return self.visual_encoder

    def get_layer(self, idx: int) -> nn.Module:
        """Returns the Transformer block at the given index in the visual encoder.

        In open_clip, the visual encoder's Transformer blocks are stored as:
            model.visual.transformer.resblocks (nn.Sequential of ResidualAttentionBlock)

        Each ResidualAttentionBlock has the structure:
            ResidualAttentionBlock
            ├── ln_1 (LayerNorm: 768)          — pre-attention LayerNorm
            ├── attn (MultiheadAttention)       — multi-head self-attention
            ├── ls_1 (LayerScale or Identity)   — layer scale
            ├── ln_2 (LayerNorm: 768)           — pre-MLP LayerNorm
            ├── mlp (nn.Sequential)             — MLP block
            │   ├── c_fc (Linear: 768→3072)
            │   ├── gelu (QuickGELU or GELU)
            │   └── c_proj (Linear: 3072→768)
            └── ls_2 (LayerScale or Identity)   — layer scale

        Note: This structure differs from timm's Block structure used in
        ViTWrapper. PEFTFactory must handle both structures when injecting
        PEFT modules. For CLIP experiments, PEFT modules are adapted to
        work with open_clip's ResidualAttentionBlock.

        Args:
            idx: Block index in [0, num_layers - 1]. Block 0 is the first
                Transformer layer (closest to the patch embedding).

        Returns:
            The ResidualAttentionBlock nn.Module at position idx.

        Raises:
            IndexError: If idx is outside [0, num_layers - 1].
            AttributeError: If the visual encoder does not have the expected
                transformer.resblocks structure.
        """
        if not (0 <= idx < self.num_layers):
            raise IndexError(
                f"Layer index {idx} is out of range. "
                f"CLIP ViT-B/16 has {self.num_layers} layers "
                f"(indices 0 to {self.num_layers - 1})."
            )

        # Navigate the open_clip visual encoder structure.
        if not hasattr(self.visual_encoder, "transformer"):
            raise AttributeError(
                "CLIP visual encoder does not have a 'transformer' attribute. "
                "The open_clip API may have changed."
            )

        transformer = self.visual_encoder.transformer

        if not hasattr(transformer, "resblocks"):
            raise AttributeError(
                "CLIP visual encoder's transformer does not have 'resblocks'. "
                "The open_clip API may have changed."
            )

        return transformer.resblocks[idx]

    def freeze_text_encoder(self) -> None:
        """Freezes all text encoder parameters by setting requires_grad=False.

        In open_clip, the text encoder components are:
        - model.transformer: The text Transformer
        - model.token_embedding: Token embedding table
        - model.positional_embedding: Positional embeddings
        - model.ln_final: Final LayerNorm for text features
        - model.text_projection: Text projection matrix (maps to joint space)

        After calling this method, only visual encoder parameters can be
        trained. The text encoder is effectively discarded from the training
        loop, consistent with the paper: "Subsequently, we discard the text
        encoder and apply PEFT methods to the visual encoder."

        Also freezes the visual projection layer (visual.proj) since the
        paper fine-tunes only "PEFT modules and the head", not the projection.

        Note: The logit_scale parameter (temperature) is also frozen to
        prevent it from being updated during PEFT fine-tuning.
        """
        frozen_count: int = 0

        # ------------------------------------------------------------------
        # Freeze text encoder components.
        # ------------------------------------------------------------------
        text_components: List[str] = [
            "transformer",       # Text Transformer blocks
            "token_embedding",   # Token embedding table
            "positional_embedding",  # Positional embeddings (may be a Parameter)
            "ln_final",          # Final LayerNorm for text
            "text_projection",   # Text projection to joint space
            "logit_scale",       # Temperature parameter
        ]

        for component_name in text_components:
            if hasattr(self.model, component_name):
                component = getattr(self.model, component_name)

                if isinstance(component, nn.Module):
                    for param in component.parameters():
                        param.requires_grad = False
                        frozen_count += 1
                elif isinstance(component, nn.Parameter):
                    component.requires_grad = False
                    frozen_count += 1
                elif isinstance(component, torch.Tensor):
                    # Some components may be registered as buffers, not parameters.
                    _logger.debug(
                        "Component '%s' is a Tensor (buffer), not a Parameter. "
                        "Buffers do not have requires_grad.",
                        component_name,
                    )
            else:
                _logger.debug(
                    "Text encoder component '%s' not found in CLIP model. "
                    "This may be expected for some open_clip versions.",
                    component_name,
                )

        # ------------------------------------------------------------------
        # Also freeze the visual projection layer (visual.proj).
        # Paper: fine-tunes "PEFT modules and the head" only.
        # ------------------------------------------------------------------
        if hasattr(self.visual_encoder, "proj") and isinstance(
            self.visual_encoder.proj, nn.Parameter
        ):
            self.visual_encoder.proj.requires_grad = False
            frozen_count += 1
            _logger.debug("Frozen visual projection (visual.proj).")

        _logger.info(
            "Text encoder frozen: %d parameter tensors set to requires_grad=False.",
            frozen_count,
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Returns all visual encoder parameters with requires_grad=True.

        This list is passed to the AdamW optimizer in Trainer._build_optimizer()
        for the CLIP robustness experiment. Must be called AFTER:
        1. freeze_text_encoder() — to freeze text components
        2. PEFT modifications are applied — to selectively re-enable requires_grad
           on PEFT-specific parameters in the visual encoder

        The visual projection layer (visual.proj) is excluded if it was frozen
        by freeze_text_encoder().

        Returns:
            List of nn.Parameter objects from the visual encoder with
            requires_grad=True. Empty if freeze_text_encoder() was called
            and no PEFT modifications have re-enabled any parameters.
        """
        return [
            param
            for param in self.visual_encoder.parameters()
            if param.requires_grad
        ]

    def count_trainable_params(self) -> int:
        """Returns the total number of trainable visual encoder parameters.

        Sums numel() for all visual encoder parameters with requires_grad=True.
        Used for logging and parameter cap enforcement.

        Returns:
            Integer count of trainable visual encoder parameters.
        """
        return sum(
            param.numel()
            for param in self.visual_encoder.parameters()
            if param.requires_grad
        )

    def get_model(self) -> nn.Module:
        """Returns the full CLIP model (visual + text components).

        Used by main.py to capture the pretrained state dict before PEFT
        modifications are applied, enabling WiSE weight interpolation:
            pretrained_state = copy.deepcopy(clip_wrapper.get_model().state_dict())

        Returns:
            The full open_clip CLIP model nn.Module.
        """
        return self.model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_and_update_architecture(self) -> None:
        """Validates and updates architecture constants from the loaded model.

        Checks that the visual encoder has the expected structure and updates
        self.num_layers, self.visual_embed_dim, and self.embed_dim if the
        loaded model differs from the defaults. Logs warnings for mismatches.
        """
        # ------------------------------------------------------------------
        # Check 1: Verify visual encoder has transformer.resblocks.
        # ------------------------------------------------------------------
        if not hasattr(self.visual_encoder, "transformer"):
            _logger.warning(
                "CLIP visual encoder does not have 'transformer' attribute. "
                "get_