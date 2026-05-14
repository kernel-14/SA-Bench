```python
## utils/checkpointing.py
"""Checkpoint save/load utilities for Pyramidal Flow Matching.

Handles three distinct concerns:
1. Training checkpoint save/load — preserving and restoring full training
   state (model weights, optimizer, scheduler, step count) for resuming
   across multi-day, 128-GPU training runs.
2. Pretrained weight loading — initializing MM-DiT from SD3 Medium weights
   with key remapping (Paper: Section 4.1, Appendix B).
3. Distributed safety — only rank 0 writes to disk; all ranks can read.

Checkpoint directory structure per save:
    checkpoints/
      step_0050000/
        model.safetensors       # Model weights (safetensors format)
        optimizer.pt            # Optimizer state dict (torch.save)
        scheduler.pt            # Scheduler state dict (torch.save)
        training_state.json     # Step, stage, timestamp, config hash
        config_snapshot.yaml    # Copy of config at checkpoint time

Usage:
    from utils.checkpointing import (
        save_checkpoint,
        load_checkpoint,
        load_pretrained_sd3,
        get_latest_checkpoint,
    )
"""

import hashlib
import json
import os
import re
import shutil
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils.distributed import barrier, get_local_rank, is_main_process
from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)

## ---------------------------------------------------------------------------
## File name constants within each checkpoint directory
## ---------------------------------------------------------------------------
_MODEL_FILENAME: str = "model.safetensors"
_OPTIMIZER_FILENAME: str = "optimizer.pt"
_SCHEDULER_FILENAME: str = "scheduler.pt"
_TRAINING_STATE_FILENAME: str = "training_state.json"
_CONFIG_SNAPSHOT_FILENAME: str = "config_snapshot.yaml"

## ---------------------------------------------------------------------------
## SD3 Medium → MMDiT key remapping patterns
## ---------------------------------------------------------------------------
# Each entry is (sd3_pattern, our_replacement) using str.replace semantics.
# Applied in order; first match wins for each key.
# These patterns cover the SD3 Medium / diffusers naming conventions.
_SD3_KEY_REMAP_PATTERNS: Tuple[Tuple[str, str], ...] = (
    # Patch embedding
    ("pos_embed.proj.weight", "patch_embed.proj.weight"),
    ("pos_embed.proj.bias", "patch_embed.proj.bias"),
    # Timestep / text embedding
    ("time_text_embed.timestep_embedder.linear_1.weight",
     "timestep_embed.linear_1.weight"),
    ("time_text_embed.timestep_embedder.linear_1.bias",
     "timestep_embed.linear_1.bias"),
    ("time_text_embed.timestep_embedder.linear_2.weight",
     "timestep_embed.linear_2.weight"),
    ("time_text_embed.timestep_embedder.linear_2.bias",
     "timestep_embed.linear_2.bias"),
    ("time_text_embed.text_embedder.linear_1.weight",
     "text_embed.linear_1.weight"),
    ("time_text_embed.text_embedder.linear_1.bias",
     "text_embed.linear_1.bias"),
    ("time_text_embed.text_embedder.linear_2.weight",
     "text_embed.linear_2.weight"),
    ("time_text_embed.text_embedder.linear_2.bias",
     "text_embed.linear_2.bias"),
    # Context (text) projection
    ("context_embedder.weight", "context_proj.weight"),
    ("context_embedder.bias", "context_proj.bias"),
    # Output norm and projection
    ("norm_out.linear.weight", "output_norm.linear.weight"),
    ("norm_out.linear.bias", "output_norm.linear.bias"),
    ("proj_out.weight", "output_proj.weight"),
    ("proj_out.bias", "output_proj.bias"),
    # Transformer blocks: attention
    ("transformer_blocks.", "blocks."),
    (".attn.to_q.weight", ".self_attn.q_proj.weight"),
    (".attn.to_q.bias", ".self_attn.q_proj.bias"),
    (".attn.to_k.weight", ".self_attn.k_proj.weight"),
    (".attn.to_k.bias", ".self_attn.k_proj.bias"),
    (".attn.to_v.weight", ".self_attn.v_proj.weight"),
    (".attn.to_v.bias", ".self_attn.v_proj.bias"),
    (".attn.to_out.0.weight", ".self_attn.out_proj.weight"),
    (".attn.to_out.0.bias", ".self_attn.out_proj.bias"),
    # Context attention (dual-stream)
    (".attn.add_q_proj.weight", ".cross_attn.q_proj.weight"),
    (".attn.add_q_proj.bias", ".cross_attn.q_proj.bias"),
    (".attn.add_k_proj.weight", ".cross_attn.k_proj.weight"),
    (".attn.add_k_proj.bias", ".cross_attn.k_proj.bias"),
    (".attn.add_v_proj.weight", ".cross_attn.v_proj.weight"),
    (".attn.add_v_proj.bias", ".cross_attn.v_proj.bias"),
    (".attn.to_add_out.weight", ".cross_attn.out_proj.weight"),
    (".attn.to_add_out.bias", ".cross_attn.out_proj.bias"),
    # Feed-forward / MLP
    (".ff.net.0.proj.weight", ".mlp.fc1.weight"),
    (".ff.net.0.proj.bias", ".mlp.fc1.bias"),
    (".ff.net.2.weight", ".mlp.fc2.weight"),
    (".ff.net.2.bias", ".mlp.fc2.bias"),
    (".ff_context.net.0.proj.weight", ".mlp_context.fc1.weight"),
    (".ff_context.net.0.proj.bias", ".mlp_context.fc1.bias"),
    (".ff_context.net.2.weight", ".mlp_context.fc2.weight"),
    (".ff_context.net.2.bias", ".mlp_context.fc2.bias"),
    # Layer norms
    (".norm1.linear.weight", ".norm1.linear.weight"),
    (".norm1.linear.bias", ".norm1.linear.bias"),
    (".norm1_context.linear.weight", ".norm1_context.linear.weight"),
    (".norm1_context.linear.bias", ".norm1_context.linear.bias"),
    (".norm2.weight", ".norm2.weight"),
    (".norm2.bias", ".norm2.bias"),
    (".norm2_context.weight", ".norm2_context.weight"),
    (".norm2_context.bias", ".norm2_context.bias"),
)


## ---------------------------------------------------------------------------
## Private helpers
## ---------------------------------------------------------------------------


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwraps a DDP-wrapped model to access the underlying module.

    Args:
        model: A ``nn.Module``, possibly wrapped in
            ``torch.nn.parallel.DistributedDataParallel``.

    Returns:
        The underlying ``nn.Module`` without DDP wrapping.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _hash_config(config: Dict[str, Any]) -> str:
    """Computes a short SHA-256 hash of a config dictionary.

    Used to detect config changes between checkpoint saves for
    reproducibility auditing.

    Args:
        config: A (possibly nested) dictionary of configuration values.

    Returns:
        First 12 characters of the SHA-256 hex digest of the JSON-serialized
        config. Returns ``"unknown"`` if serialization fails.
    """
    try:
        config_str: str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "unknown"


def _remap_sd3_keys(
    sd3_state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remaps SD3 Medium checkpoint keys to our MMDiT naming convention.

    Applies the pattern-based remapping defined in ``_SD3_KEY_REMAP_PATTERNS``
    to translate SD3 key names to our model's key names. Only includes keys
    that exist in ``model_state_dict`` after remapping, and only when shapes
    match exactly.

    Args:
        sd3_state_dict: State dict loaded from the SD3 Medium checkpoint.
        model_state_dict: State dict of our MMDiT model (used for key
            validation and shape checking).

    Returns:
        A new ``OrderedDict`` containing only the remapped keys that are
        present in our model with matching shapes. Keys that cannot be
        remapped or have shape mismatches are logged and skipped.
    """
    remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    model_keys: set = set(model_state_dict.keys())

    skipped_not_found: int = 0
    skipped_shape_mismatch: int = 0
    loaded_count: int = 0

    for sd3_key, tensor in sd3_state_dict.items():
        # Apply remapping patterns sequentially
        our_key: str = sd3_key
        for sd3_pattern, our_replacement in _SD3_KEY_REMAP_PATTERNS:
            if sd3_pattern in our_key:
                our_key = our_key.replace(sd3_pattern, our_replacement)
                # Apply only the first matching pattern
                break

        # Check if the remapped key exists in our model
        if our_key not in model_keys:
            logger.debug(
                "SD3 key '%s' -> '%s' not found in model, skipping.",
                sd3_key,
                our_key,
            )
            skipped_not_found += 1
            continue

        # Validate shape compatibility
        model_tensor: torch.Tensor = model_state_dict[our_key]
        if tensor.shape != model_tensor.shape:
            logger.error(
                "Shape mismatch for key '%s' -> '%s': "
                "SD3 shape %s vs model shape %s. Skipping.",
                sd3_key,
                our_key,
                tuple(tensor.shape),
                tuple(model_tensor.shape),
            )
            skipped_shape_mismatch += 1
            continue

        remapped[our_key] = tensor
        loaded_count += 1

    logger.info(
        "SD3 key remapping summary: loaded=%d, skipped_not_found=%d, "
        "skipped_shape_mismatch=%d, total_sd3_keys=%d",
        loaded_count,
        skipped_not_found,
        skipped_shape_mismatch,
        len(sd3_state_dict),
    )

    return remapped


def _validate_checkpoint_dir(checkpoint_path: str) -> None:
    """Validates that a checkpoint directory contains all required files.

    Args:
        checkpoint_path: Path to the checkpoint directory to validate.

    Raises:
        FileNotFoundError: If the directory or any required file is missing.
    """
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint directory not found: '{checkpoint_path}'. "
            "Ensure the path points to a valid checkpoint directory "
            "(e.g., 'checkpoints/step_0050000/')."
        )

    required_files: Tuple[str, ...] = (
        _MODEL_FILENAME,
        _TRAINING_STATE_FILENAME,
    )
    for filename in required_files:
        full_path: str = os.path.join(checkpoint_path, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(
                f"Required checkpoint file missing: '{full_path}'. "
                f"The checkpoint at '{checkpoint_path}' may be incomplete "
                "or corrupt. Consider using an earlier checkpoint."
            )


## ---------------------------------------------------------------------------
## Public API
## ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    step: int,
    path: str,
    config: Optional[Dict[str, Any]] = None,
    stage: int = 1,
) -> None:
    """Saves a complete training checkpoint to disk.

    Only rank 0 performs the actual file I/O. All other ranks skip the save
    and wait at a distributed barrier until rank 0 finishes. This prevents
    race conditions in 128-GPU training.

    The checkpoint is saved as a directory containing separate files for
    model weights (safetensors), optimizer state, scheduler state, and a
    JSON training state file.

    Args:
        model: The model to checkpoint. May be DDP-wrapped; the underlying
            module is automatically extracted.
        optimizer: The AdamW optimizer whose state to save.
        scheduler: The LR scheduler whose state to save.
        step: Current global training step number. Used to name the
            checkpoint subdirectory (e.g., ``step_0050000``).
        path: Base checkpoint directory (``config.paths.checkpoint_dir``).
            A subdirectory ``step_{step:07d}`` is created inside it.
        config: Optional config dictionary to snapshot alongside the
            checkpoint for reproducibility. If None, no snapshot is saved.
        stage: Current training stage (1, 2, or 3). Stored in the training
            state JSON for auditing.

    Example:
        >>> save_checkpoint(
        ...     model=pyramid_flow_model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     step=50000,
        ...     path=config.paths.checkpoint_dir,
        ...     config=dict(config),
        ...     stage=1,
        ... )
    """
    # Only rank 0 writes to disk
    if not is_main_process():
        # Non-rank-0 processes wait at the barrier below
        barrier()
        return

    # Construct checkpoint subdirectory path
    checkpoint_dir: str = os.path.join(path, f"step_{step:07d}")

    # Use a temporary directory during writing to prevent partial checkpoints
    # from being mistaken for valid ones if the process is interrupted
    tmp_dir: str = checkpoint_dir + ".tmp"

    try:
        os.makedirs(tmp_dir, exist_ok=True)

        # ----------------------------------------------------------------
        # 1. Save model weights using safetensors
        # ----------------------------------------------------------------
        # Unwrap DDP to access the underlying module's state dict
        unwrapped_model: nn.Module = _unwrap_model(model)
        model_state_dict: Dict[str, torch.Tensor] = (
            unwrapped_model.state_dict()
        )

        model_path: str = os.path.join(tmp_dir, _MODEL_FILENAME)
        try:
            from safetensors.torch import save_file as safetensors_save_file
            safetensors_save_file(model_state_dict, model_path)
        except ImportError:
            # Fallback to torch.save if safetensors is not installed
            logger.warning(
                "safetensors not installed. Falling back to torch.save "
                "for model weights. Install with: pip install safetensors"
            )
            torch.save(model_state_dict, model_path.replace(".safetensors", ".pt"))

        # ----------------------------------------------------------------
        # 2. Save optimizer state
        # ----------------------------------------------------------------
        optimizer_path: str = os.path.join(tmp_dir, _OPTIMIZER_FILENAME)
        torch.save(optimizer.state_dict(), optimizer_path)

        # ----------------------------------------------------------------
        # 3. Save scheduler state
        # ----------------------------------------------------------------
        scheduler_path: str = os.path.join(tmp_dir, _SCHEDULER_FILENAME)
        if scheduler is not None:
            torch.save(scheduler.state_dict(), scheduler_path)
        else:
            # Write an empty dict so load_checkpoint doesn't fail
            torch.save({}, scheduler_path)

        # ----------------------------------------------------------------
        # 4. Save training state JSON
        # ----------------------------------------------------------------
        training_state: Dict[str, Any] = {
            "step": step,
            "stage": stage,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "config_hash": _hash_config(config) if config is not None else "none",
        }
        training_state_path: str = os.path.join(
            tmp_dir, _TRAINING_STATE_FILENAME
        )
        with open(training_state_path, "w", encoding="utf-8") as f:
            json.dump(training_state, f, indent=2)

        # ----------------------------------------------------------------
        # 5. Save config snapshot (optional, for reproducibility)
        # ----------------------------------------------------------------
        if config is not None:
            config_snapshot_path: str = os.path.join(
                tmp_dir, _CONFIG_SNAPSHOT_FILENAME
            )
            try:
                import yaml  # type: ignore[import]
                with open(config_snapshot_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False)
            except ImportError:
                # Fall back to JSON if PyYAML is not available
                config_json_path: str = os.path.join(
                    tmp_dir, "config_snapshot.json"
                )
                with open(config_json_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2, default=str)

        # ----------------------------------------------------------------
        # 6. Atomically rename tmp_dir -> checkpoint_dir
        # ----------------------------------------------------------------
        # Remove existing checkpoint dir if it exists (e.g., from a
        # previous interrupted save at the same step)
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.rename(tmp_dir, checkpoint_dir)

        logger.info(
            "Checkpoint saved: step=%d, stage=%d, path='%s'",
            step,
            stage,
            checkpoint_dir,
        )

    except Exception as exc:
        # Clean up the incomplete temporary directory to avoid corruption
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except Exception as cleanup_exc:
                logger.error(
                    "Failed to clean up incomplete checkpoint at '%s': %s",
                    tmp_dir,
                    cleanup_exc,
                )
        raise RuntimeError(
            f"Failed to save checkpoint at step {step} to '{path}': {exc}"
        ) from exc

    finally:
        # All ranks synchronize here: non-rank-0 processes (which returned
        # early above) also call barrier(), so this barrier pairs with theirs.
        barrier()


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[Any],
    path: str,
    strict: bool = True,
) -> int:
    """Loads a training checkpoint and restores full training state.

    All ranks load the checkpoint independently (read-only, no race
    condition). Model weights are loaded to CPU first to avoid GPU OOM,
    then transferred to the model's current device via ``load_state_dict``.

    Args:
        model: The model to restore weights into. May be DDP-wrapped.
        optimizer: The optimizer to restore state into. If ``None``,
            optimizer state loading is skipped (useful for inference).
        scheduler: The LR scheduler to restore state into. If ``None``,
            scheduler state loading is skipped.
        path: Path to the checkpoint directory (e.g.,
            ``'checkpoints/step_0050000'``).
        strict: Whether to require an exact match between the checkpoint's
            state dict keys and the model's keys. Defaults to ``True``.
            Set to ``False`` when loading partial checkpoints (e.g., for
            fine-tuning with architectural changes).

    Returns:
        The global training step number at which the checkpoint was saved.
        The trainer uses this to resume from the correct step.

    Raises:
        FileNotFoundError: If the checkpoint directory or required files
            are missing.
        RuntimeError: If loading fails due to corrupt files or key
            mismatches (when ``strict=True``).

    Example:
        >>> resumed_step = load_checkpoint(
        ...     model=pyramid_flow_model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     path="checkpoints/step_0050000",
        ... )
        >>> print(f"Resuming from step {resumed_step}")
    """
    # Validate checkpoint directory before loading any tensors
    _validate_checkpoint_dir(path)

    # ----------------------------------------------------------------
    # 1. Load training state JSON first (cheap, validates checkpoint)
    # ----------------------------------------------------------------
    training_state_path: str = os.path.join(path, _TRAINING_STATE_FILENAME)
    with open(training_state_path, "r", encoding="utf-8") as f:
        training_state: Dict[str, Any] = json.load(f)

    step: int = int(training_state.get("step", 0))
    stage: int = int(training_state.get("stage", 1))
    timestamp: str = training_state.get("timestamp", "unknown")

    logger.info(
        "Loading checkpoint: step=%d, stage=%d, saved_at='%s', path='%s'",
        step,
        stage,
        timestamp,
        path,
    )

    # ----------------------------------------------------------------
    # 2. Load model weights
    # ----------------------------------------------------------------
    model_path: str = os.path.join(path, _MODEL_FILENAME)
    fallback_model_path: str = model_path.replace(".safetensors", ".pt")

    if os.path.isfile(model_path):
        try:
            from safetensors.torch import load_file as safetensors_load_file
            # Load to CPU first to avoid GPU OOM spikes
            state_dict: Dict[str, torch.Tensor] = safetensors_load_file(
                model_path, device="cpu"
            )
        except ImportError:
            logger.warning(
                "safetensors not installed. Attempting torch.load fallback."
            )
            state_dict = torch.load(
                fallback_model_path,
                map_location="cpu",
                weights_only=True,
            )
    elif os.path.isfile(fallback_model_path):
        logger.warning(
            "model.safetensors not found; loading from fallback '%s'.",
            fallback_model_path,
        )
        state_dict = torch.load(
            fallback_model_path,
            map_location="cpu",
            weights_only=True,
        )
    else:
        raise FileNotFoundError(
            f"No model weights file found in checkpoint '{path}'. "
            f"Expected '{model_path}' or '{fallback_model_path}'."
        )

    # Unwrap DDP before loading state dict
    unwrapped_model: nn.Module = _unwrap_model(model)

    try:
        missing_keys, unexpected_keys = unwrapped_model.load_state_dict(
            state_dict, strict=strict
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load model state dict from '{model_path}' "
            f"(strict={strict}). This may indicate an architecture mismatch. "
            f"Original error: {exc}"
        ) from exc

    if missing_keys:
        logger.warning(
            "Missing keys when loading checkpoint (not in checkpoint): %s",
            missing_keys[:20],  # Truncate for readability
        )
    if unexpected_keys:
        logger.warning(
            "Unexpected keys when loading checkpoint (not in model): %s",
            unexpected_keys[:20],
        )

    logger.info(
        "Model weights loaded: %d tensors, missing=%d, unexpected=%d",
        len(state_dict),
        len(missing_keys),
        len(unexpected_keys),
    )

    # ----------------------------------------------------------------
    # 3. Load optimizer state
    # ----------------------------------------------------------------
    if optimizer is not None:
        optimizer_path: str = os.path.join(path, _OPTIMIZER_FILENAME)
        if os.path.isfile(optimizer_path):
            try:
                optimizer_state: Dict[str, Any] = torch.load(
                    optimizer_path,
                    map_location="cpu",
                    weights_only=False,
                )
                optimizer.load_state_dict(optimizer_state)
                logger.info("Optimizer state loaded from '%s'.", optimizer_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load optimizer state from '%s': %s. "
                    "Optimizer will start from scratch.",
                    optimizer_path,
                    exc,
                )
        else:
            logger.warning(
                "Optimizer state file not found at '%s'. "
                "Optimizer will start from scratch.",
                optimizer_path,
            )

    # ----------------------------------------------------------------
    # 4. Load scheduler state
    # ----------------------------------------------------------------
    if scheduler is not None:
        scheduler_path: str = os.path.join(path, _SCHEDULER_FILENAME)
        if os.path.isfile(scheduler_path):
            try:
                scheduler_state: Dict[str, Any] = torch.load(
                    scheduler_path,
                    map_location="cpu",
                    weights_only=False,
                )
                # Only load if non-empty (empty dict = scheduler was None at save time)
                if scheduler_state:
                    scheduler.load_state_dict(scheduler_state)
                    logger.info(
                        "Scheduler state loaded from '%s'.", scheduler_path
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to load scheduler state from '%s': %s. "
                    "Scheduler will start from scratch.",
                    scheduler_path,
                    exc,
                )
        else:
            logger.warning(
                "Scheduler state file not found at '%s'. "
                "Scheduler will start from scratch.",
                scheduler_path,
            )

    logger.info(
        "Checkpoint loaded successfully. Resuming from step %d (stage %d).",
        step,
        stage,
    )

    return step


def load_pretrained_sd3(
    model: nn.Module,
    sd3_path: Optional[str],
) -> None:
    """Initializes MM-DiT weights from a pretrained SD3 Medium checkpoint.

    Loads SD3 Medium weights and remaps key names to match our MMDiT
    naming convention. New parameters (1D RoPE, stage embedding) that
    have no SD3 equivalent are left at their random initialization.

    This function implements the initialization described in the paper:
    "The weights of the MM-DiT are initialized from the SD3 medium."
    (Paper: Appendix B)

    Args:
        model: The MMDiT model to initialize. May be DDP-wrapped.
        sd3_path: Path to the SD3 Medium checkpoint. Can be:
            - A ``.safetensors`` file (e.g., from Hugging Face Hub)
            - A directory in diffusers format containing
              ``model.safetensors``
            - ``None``: Skip SD3 initialization (model trains from
              random init). A warning is logged.

    Raises:
        FileNotFoundError: If ``sd3_path`` is not None but the file/
            directory does not exist.
        RuntimeError: If the checkpoint file cannot be loaded.

    Example:
        >>> load_pretrained_sd3(
        ...     model=mmdit,
        ...     sd3_path="checkpoints/sd3_medium.safetensors",
        ... )
    """
    # Handle null path: skip initialization with a warning
    if sd3_path is None:
        logger.warning(
            "config.training.pretrained_sd3_path is null. "
            "MM-DiT will train from random initialization. "
            "Set pretrained_sd3_path to the SD3 Medium checkpoint path "
            "to reproduce the paper's initialization (Paper: Appendix B)."
        )
        return

    # Resolve the actual .safetensors file path
    if os.path.isdir(sd3_path):
        # Diffusers format: look for model.safetensors inside the directory
        candidate_paths: Tuple[str, ...] = (
            os.path.join(sd3_path, "model.safetensors"),
            os.path.join(sd3_path, "transformer", "model.safetensors"),
            os.path.join(sd3_path, "sd3_medium.safetensors"),
        )
        resolved_path: Optional[str] = None
        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                resolved_path = candidate
                break
        if resolved_path is None:
            raise FileNotFoundError(
                f"SD3 checkpoint directory '{sd3_path}' does not contain "
                f"a recognized model file. Searched: {candidate_paths}"
            )
        sd3_path = resolved_path
    elif not os.path.isfile(sd3_path):
        raise FileNotFoundError(
            f"SD3 checkpoint file not found: '{sd3_path}'. "
            "Ensure the path points to a valid .safetensors file or "
            "a diffusers-format directory."
        )

    logger.info("Loading SD3 Medium pretrained weights from '%s'.", sd3_path)

    # ----------------------------------------------------------------
    # 1. Load SD3 checkpoint to CPU
    # ----------------------------------------------------------------
    try:
        from safetensors.torch import load_file as safetensors_load_file
        sd3_state_dict: Dict[str, torch.Tensor] = safetensors_load_file(
            sd3_path, device="cpu"
        )
    except ImportError:
        logger.warning(
            "safetensors not installed. Attempting torch.load for SD3 weights."
        )
        try:
            sd3_state_dict = torch.load(
                sd3_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SD3 checkpoint from '{sd3_path}': {exc}"
            ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load SD3 checkpoint from '{sd3_path}': {exc}"
        ) from exc

    logger.info(
        "SD3 checkpoint loaded: %d keys, total size ~%.1f GB",
        len(sd3_state_dict),
        sum(t.numel() * t.element_size() for t in sd3_state_dict.values())
        / (1024 ** 3),
    )

    # ----------------------------------------------------------------
    # 2. Remap SD3 keys to our MMDiT naming convention
    # ----------------------------------------------------------------
    unwrapped_model: nn.Module = _unwrap_model(model)
    model_state_dict: Dict[str, torch.Tensor] = unwrapped_model.state_dict()

    remapped_state_dict: Dict[str, torch.Tensor] = _remap_sd3_keys(
        sd3_state_dict, model_state_dict
    )

    # ----------------------------------------------------------------
    # 3. Load remapped weights (strict=False: new params keep random init)
    # ----------------------------------------------------------------
    missing_keys, unexpected_keys = unwrapped_model.