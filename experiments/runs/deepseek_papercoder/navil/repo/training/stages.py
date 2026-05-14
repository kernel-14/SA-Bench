# training/stages.py

"""
Stage‑specific parameter freezing logic for the NaViL training pipeline.

This module provides a single public function
:func:`apply_freeze_pattern` that is called by :class:`Trainer` before each
of the three training stages (S1.1, S1.2, S2).  It resets the
``requires_grad`` flags of all model parameters and then selectively
freezes parameters whose names contain a given list of substring patterns.

The freeze patterns are taken directly from the ``freeze_pattern`` field of
``StageConfig`` (in turn sourced from ``config.yaml``).  The naming
convention of the model’s modules ensures that linguistic parameters carry
the substring ``"linguistic"``, while visual parameters (encoder, connector,
MoE visual experts) do *not*.  This convention is established by
:class:`MoELLM` during construction.

Patterns used in the paper’s stages
------------------------------------

- **S1.1** : ``["linguistic"]`` – freeze all linguistic parts.
- **S1.2** : ``["linguistic.ffn"]`` – freeze only linguistic FFN modules,
  leaving linguistic attention trainable.
- **S2** : ``[]`` – all parameters are trainable.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_freeze_pattern(model: nn.Module, freeze_pattern: Optional[List[str]] = None) -> None:
    """
    Reset ``requires_grad`` to ``True`` for every parameter of *model* and then
    freeze those whose name contains any of the strings in *freeze_pattern*.

    If *freeze_pattern* is ``None`` or an empty list, all parameters remain
    trainable.

    Args:
        model: The NaViL model (or any ``nn.Module``) whose parameters
            should be reconfigured.
        freeze_pattern: List of substrings; a parameter is frozen iff its
            ``name`` includes at least one of these substrings.  Defaults to
            ``[]`` (no freezing).
    """
    if freeze_pattern is None:
        freeze_pattern = []

    # Step 1 – unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

    # Step 2 – apply requested freezes
    if freeze_pattern:
        frozen_count = 0
        for name, p in model.named_parameters():
            if any(pattern in name for pattern in freeze_pattern):
                p.requires_grad = False
                frozen_count += 1
        trainable_count = sum(p.requires_grad for p in model.parameters())
        logger.info(
            "Freeze pattern applied: %s. Trainable params: %d, frozen: %d.",
            freeze_pattern,
            trainable_count,
            frozen_count,
        )
    else:
        logger.info("No freeze pattern – all %d parameters are trainable.",
                    sum(1 for _ in model.parameters()))


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Convenience helper that returns a flat list of all trainable parameters
    of *model*.  This is the list that should be passed to the optimizer
    after calling :func:`apply_freeze_pattern`.

    Args:
        model: The model whose parameters should be collected.

    Returns:
        List of ``nn.Parameter`` objects with ``requires_grad == True``.
    """
    return [p for p in model.parameters() if p.requires_grad]
