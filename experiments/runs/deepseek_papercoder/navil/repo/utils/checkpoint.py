# utils/checkpoint.py

"""
Checkpoint management utilities for the NaViL training pipeline.

This module provides a set of functions to save and restore full training
checkpoints in a distributed (DeepSpeed‑based) setting, as well as a
lightweight export of the model weights only.  All functions are designed
to be used by the ``Trainer`` class during the multi‑stage training described
in the paper.

Functions:
    - get_checkpoint_dir : construct a directory path for a given stage
      and step.
    - save_checkpoint : save the DeepSpeed engine state (model, optimizer,
      scheduler) together with a user‑defined client state.
    - load_checkpoint : load a DeepSpeed checkpoint and return the
      previously saved client state (step, stage, etc.).
    - save_model_weights_only : extract the bare PyTorch model from the
      engine and save its state dict for inference / evaluation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
import deepspeed

# ---------------------------------------------------------------------------
# Logger – will inherit configuration set up by ``utils.logging.setup_logging``
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_checkpoint_dir(
    config: Any,   # TrainingConfig or dict; must have ``output_dir`` if desired
    stage: str,
    step: int,
) -> str:
    """
    Build the checkpoint directory path for a given training stage and
    global step.

    The root output directory is taken from ``config.output_dir`` if present;
    otherwise falls back to ``"./checkpoints"``.

    Args:
        config: Training configuration object (e.g., ``TrainingConfig``
            instance) that may contain an ``output_dir`` attribute.
        stage: Stage identifier string (``"s1_1"``, ``"s1_2"``, ``"s2"``).
        step: Global training step (or epoch) number.

    Returns:
        Absolute or relative path to the checkpoint directory.
    """
    base_dir = getattr(config, "output_dir", "./checkpoints")
    # Sanitise the stage name for a filesystem‑safe name
    stage_safe = stage.replace("/", "_")
    return os.path.join(base_dir, f"stage-{stage_safe}", f"step_{step}")


def save_checkpoint(
    engine: deepspeed.DeepSpeedEngine,
    client_state: Dict[str, Any],
    stage_dir: str,
    global_step: int,
) -> None:
    """
    Save a full training checkpoint using the DeepSpeed engine.

    The checkpoint includes model parameters, optimizer states, and the
    learning rate scheduler state.  The *client_state* dictionary is
    serialised alongside the checkpoint and can be retrieved later via
    :func:`load_checkpoint`.

    **Distributed behaviour** – This function must be called on every
    process; DeepSpeed handles the persistence of sharded data correctly,
    with only rank 0 writing the shared JSON files.

    Args:
        engine: The DeepSpeed engine wrapping the NaViL model.
        client_state: Arbitrary dictionary containing user‑specific metadata
            to be saved (e.g., ``{"global_step": global_step, "stage": "s1_1"}``).
        stage_dir: Path to the directory where the checkpoint will be written
            (usually obtained from :func:`get_checkpoint_dir`).
        global_step: The current global training step (used both as the
            checkpoint tag and the printed log).
    """
    # Ensure the directory exists (rank‑0 does this to avoid races, but
    # makedirs is safe when called by all ranks as well).
    os.makedirs(stage_dir, exist_ok=True)

    # DeepSpeed checkpoint tag: we embed the step for human readability.
    tag = f"global_step{global_step}"
    logger.info("Saving checkpoint at step %d to %s (tag: %s)", global_step, stage_dir, tag)

    # DeepSpeed engine.save_checkpoint handles everything:
    #   - model zero shards
    #   - optimizer state
    #   - LR scheduler state
    #   - client_state written to "latest" file
    engine.save_checkpoint(
        save_dir=stage_dir,
        tag=tag,
        client_state=client_state,
    )

    # Additionally write a marker file indicating the latest step for easy
    # resumption logic.  Only rank 0 does this to avoid contention.
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    latest_path = os.path.join(stage_dir, "latest")
    with open(latest_path, "w") as f:
        f.write(f"{tag}\n")
    logger.debug("Updated latest checkpoint pointer: %s", latest_path)


def load_checkpoint(
    engine: deepspeed.DeepSpeedEngine,
    stage_dir: str,
    tag: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load a full training checkpoint (model, optimizer, scheduler) from disk.

    If *tag* is provided, the specific checkpoint with that tag is loaded;
    otherwise the checkpoint referenced by the ``"latest"`` file in
    *stage_dir* is used.

    Args:
        engine: The DeepSpeed engine (must be initialised, even if empty).
        stage_dir: Directory containing the checkpoint data.
        tag: Optional identifier of a specific checkpoint (e.g.,
            ``"global_step70000"``).  If ``None``, the function reads
            ``stage_dir/latest`` to determine the most recent checkpoint.

    Returns:
        The ``client_state`` dictionary that was saved with the checkpoint,
        or ``None`` if loading failed (e.g., directory does not exist).
    """
    if not os.path.isdir(stage_dir):
        logger.warning("Checkpoint directory %s not found; skipping load.", stage_dir)
        return None

    # If no tag given, try to infer from the "latest" file
    if tag is None:
        latest_file = os.path.join(stage_dir, "latest")
        if os.path.isfile(latest_file):
            with open(latest_file, "r") as f:
                tag = f.read().strip()
            logger.info("Resolved latest checkpoint tag: %s", tag)
        else:
            logger.warning("No 'latest' file in %s and no tag specified.", stage_dir)
            return None

    logger.info("Loading checkpoint from %s with tag %s", stage_dir, tag)
    # deepspeed.engine.load_checkpoint returns (path, client_state)
    try:
        _, client_state = engine.load_checkpoint(
            load_dir=stage_dir,
            tag=tag,
            load_module_strict=True,
        )
    except Exception as e:
        logger.error("Failed to load checkpoint: %s", e, exc_info=True)
        return None

    logger.info(
        "Checkpoint loaded successfully. global_step=%d",
        client_state.get("global_step", -1) if client_state else -1,
    )
    return client_state


def save_model_weights_only(
    model: deepspeed.DeepSpeedEngine,
    output_path: str,
) -> None:
    """
    Export only the model weights (state dict) to disk, suitable for
    inference or for transferring the trained model to a non‑DeepSpeed
    environment.

    The underlying PyTorch module is accessed via ``model.module``.

    Args:
        model: A DeepSpeed engine wrapping the NaViL model.
        output_path: Full path to the output file (e.g., ``"naVIL_2b.pt"``).
    """
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Retrieve the actual nn.Module
    raw_model = model.module
    logger.info("Saving bare model weights to %s", output_path)
    torch.save(raw_model.state_dict(), output_path)
    logger.debug("Model weights saved (size: %.2f MB)", os.path.getsize(output_path) / (1024**2))

