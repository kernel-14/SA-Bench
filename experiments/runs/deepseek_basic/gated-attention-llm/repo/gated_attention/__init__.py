"""Gated Attention for Large Language Models.

This package implements the gated attention mechanism described in:
"Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"

Core contributions:
- Gated attention variants at multiple positions (G1-G5)
- Elementwise and headwise gating granularity
- Head-specific and head-shared gating
- Multiplicative and additive gating forms
- Analysis tools for sparsity and attention sink patterns
"""

from .modules.gating import (
    GatedAttention,
    GatingPosition,
    GatingGranularity,
    GatingMode,
    GatingScope,
    ActivationType,
)

__version__ = "0.1.0"
