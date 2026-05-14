## models/__init__.py
"""Public API for the SAM 2 models package.

This module re-exports every class that downstream code (training, evaluation,
main.py) needs to import from the models package. All imports are explicit
to keep the public API auditable and to surface sub-module errors early.

Import order follows the dependency graph bottom-up:
    positional_encoding → image_encoder, memory_attention, prompt_encoder,
    mask_decoder, memory_encoder, memory_bank → sam2

The SAM2Config dataclass is defined in models/sam2.py and is the single
source of truth for all architectural hyperparameters (Shared Knowledge §1).

PromptInput and SAM2FrameOutput are dataclasses defined in models/sam2.py
and consumed by datasets/prompt_sampler.py and all evaluators. Exporting
them here keeps the dependency direction clean: datasets and evaluation
depend on models, never the reverse (Shared Knowledge §3, §4).
"""

# ---------------------------------------------------------------------------
# Positional encoding primitives (no internal model dependencies)
# ---------------------------------------------------------------------------
from models.positional_encoding import (
    PositionEmbeddingRandom,
    RoPE2D,
    TemporalPositionalEncoding,
)

# ---------------------------------------------------------------------------
# Image encoder (depends on positional_encoding)
# ---------------------------------------------------------------------------
from models.image_encoder import (
    FeaturePyramidNetwork,
    HieraImageEncoder,
)

# ---------------------------------------------------------------------------
# Memory attention (depends on positional_encoding)
# ---------------------------------------------------------------------------
from models.memory_attention import (
    MemoryAttention,
    MemoryAttentionLayer,
)

# ---------------------------------------------------------------------------
# Prompt encoder (depends on positional_encoding)
# ---------------------------------------------------------------------------
from models.prompt_encoder import PromptEncoder

# ---------------------------------------------------------------------------
# Mask decoder (depends on standard torch only)
# ---------------------------------------------------------------------------
from models.mask_decoder import (
    MaskDecoder,
    TwoWayTransformer,
)

# ---------------------------------------------------------------------------
# Memory encoder (depends on standard torch only)
# ---------------------------------------------------------------------------
from models.memory_encoder import MemoryEncoder

# ---------------------------------------------------------------------------
# Memory bank (depends on positional_encoding)
# ---------------------------------------------------------------------------
from models.memory_bank import (
    MemoryBank,
    MemoryBankOutput,
)

# ---------------------------------------------------------------------------
# Top-level model, config, and shared dataclasses
# (depends on all of the above)
# ---------------------------------------------------------------------------
from models.sam2 import (
    PromptInput,
    SAM2Config,
    SAM2FrameOutput,
    SAM2Model,
)

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
__all__ = [
    # Config — single source of truth for all architectural hyperparameters
    "SAM2Config",
    # Top-level model
    "SAM2Model",
    # Shared dataclasses used by trainer, evaluators, and datasets
    "PromptInput",
    "SAM2FrameOutput",
    # Image encoder components
    "HieraImageEncoder",
    "FeaturePyramidNetwork",
    # Memory attention components
    "MemoryAttention",
    "MemoryAttentionLayer",
    # Positional encoding primitives
    "RoPE2D",
    "PositionEmbeddingRandom",
    "TemporalPositionalEncoding",
    # Prompt encoder
    "PromptEncoder",
    # Mask decoder components
    "MaskDecoder",
    "TwoWayTransformer",
    # Memory encoder
    "MemoryEncoder",
    # Memory bank components
    "MemoryBank",
    "MemoryBankOutput",
]
