## Code: utils/checkpoint.py

```python
## utils/checkpoint.py
"""Checkpoint management for OLMoE pretraining and adaptation.

Handles saving and loading of model weights, optimizer state, and scheduler
state across all training phases. Supports both FSDP-wrapped models (for
distributed pretraining on 256 H100s) and plain nn.Module models (for
single-GPU debugging and analysis).

Key design decisions:
  - FULL_STATE_DICT for all saves: portable checkpoints loadable without FSDP
  - safetensors for model weights: safe, memory-mappable, HuggingFace-compatible
  - torch.save for optimizer/scheduler: only needed for training resumption
  - Zero-padded step numbers: correct lexicographic sorting
  - Barriers before and after save: prevents race conditions in distributed training
  - HuggingFace format fallback in load_model_only(): enables loading released checkpoints

Configuration values used (from config.yaml):
  pretraining.save_every_steps: 5000
  pretraining.output_dir: "outputs"
  artifacts.olmoe_pretrained: "allenai/OLMoE-1B-7B-0924"
  analysis.checkpoint_fractions: [0.01, 0.10, 0.20, 0.40, 1.0]

Checkpoint directory structure:
  {output_dir}/checkpoint-{step:08d}/
  ├── model.safetensors    # Model weights (rank 0 only, CPU tensors)
  ├── optimizer.pt         # Optimizer state dict (rank 0 only)
  ├── scheduler.pt         # LR scheduler state dict (rank 0 only)
  └── metadata.json        # step, metrics, timestamp, config hash
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

# ---------------------------------------------------------------------------
# Optional safetensors import.
# safetensors is the preferred format for model weights: safe (no arbitrary
# code execution), memory-mappable, and compatible with HuggingFace ecosystem.
# Falls back to torch.save if unavailable.
# ---------------------------------------------------------------------------
try:
    from safetensors.torch import load_file as safetensors_load_file
    from safetensors.torch import save_file as safetensors_save_file
    SAFETENSORS_AVAILABLE: bool = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    safetensors_load_file = None  # type: ignore[assignment]
    safetensors_save_file = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional FSDP imports.
# FSDP is used for distributed training on 256 H100s (config.yaml: fsdp: true).
# Falls back gracefully for single-GPU debugging.
# ---------------------------------------------------------------------------
try:
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp import (
        FullOptimStateDictConfig,
        FullStateDictConfig,
        StateDictType,
    )
    FSDP_AVAILABLE: bool = True
except ImportError:
    FSDP_AVAILABLE = False
    FullyShardedDataParallel = None  # type: ignore[assignment,misc]
    FullStateDictConfig = None  # type: ignore[assignment]
    FullOptimStateDictConfig = None  # type: ignore[assignment]
    StateDictType = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional HuggingFace transformers import.
# Used as a fallback in load_model_only() to load released HuggingFace
# checkpoints (e.g., allenai/OLMoE-1B-7B-0924) directly.
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE: bool = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]

from utils.distributed import DistributedUtils
from utils.logging_utils import get_logger

logger: logging.Logger = get_logger("olmoe.checkpoint")

# ---------------------------------------------------------------------------
# Checkpoint directory naming pattern.
# Zero-padded to 8 digits for correct lexicographic sorting.
# Example: "checkpoint-00005000" for step 5000.
# ---------------------------------------------------------------------------
CHECKPOINT_DIR_PATTERN: str = "checkpoint-{step:08d}"
CHECKPOINT_DIR_PREFIX: str = "checkpoint-"

# ---------------------------------------------------------------------------
# Files within each checkpoint directory.
# ---------------------------------------------------------------------------
MODEL_WEIGHTS_FILENAME: str = "model.safetensors"
MODEL_WEIGHTS_FALLBACK_FILENAME: str = "model.pt"
OPTIMIZER_STATE_FILENAME: str = "optimizer.pt"
SCHEDULER_STATE_FILENAME: str = "scheduler.pt"
METADATA_FILENAME: str = "metadata.json"

# ---------------------------------------------------------------------------
# FSDP key prefix that may appear in state dicts saved from FSDP-wrapped models.
# When loading into a non-FSDP model, these prefixes must be stripped.
# ---------------------------------------------------------------------------
FSDP_KEY_PREFIX: str = "_fsdp_wrapped_module."


class CheckpointManager:
    """Manages saving and loading of OLMoE training checkpoints.

    Handles the full checkpoint lifecycle for pretraining (save every 5,000
    steps), adaptation (load pretrained checkpoint, save SFT/DPO checkpoints),
    and analysis (load intermediate checkpoints for routing analysis).

    Supports both FSDP-wrapped models (distributed training) and plain
    nn.Module models (single-GPU debugging and analysis). FSDP handling uses
    FULL_STATE_DICT to produce portable checkpoints loadable without FSDP.

    Attributes:
        output_dir: Base directory for saving checkpoints.
        max_checkpoints: Maximum number of checkpoints to retain. None means
                         keep all (default, matching paper's full release).
                         Set to a finite number for disk-constrained environments.

    Example (pretraining):
        >>> manager = CheckpointManager(output_dir="outputs", max_checkpoints=None)
        >>> # Every 5000 steps:
        >>> manager.save(model, optimizer, scheduler, step=5000, metrics={"loss": 2.1})
        >>> # Resume training:
        >>> step = manager.load("outputs/checkpoint-00005000", model, optimizer, scheduler)

    Example (analysis):
        >>> manager = CheckpointManager(output_dir="outputs")
        >>> manager.load_model_only("outputs/checkpoint-00005000", model)
        >>> # Or load from HuggingFace:
        >>> manager.load_model_only("allenai/OLMoE-1B-7B-0924", model)
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        max_checkpoints: Optional[int] = None,
    ) -> None:
        """Initialize CheckpointManager.

        Args:
            output_dir: Base directory for saving checkpoints. Will be created
                        if it does not exist. Subdirectories are created per
                        checkpoint step: {output_dir}/checkpoint-{step:08d}/.
                        Default: "outputs" (config.yaml: pretraining.output_dir).
            max_checkpoints: Maximum number of checkpoints to retain on disk.
                             When exceeded, the oldest checkpoints are deleted.
                             None means keep all checkpoints (default), matching
                             the paper's release of all intermediate checkpoints
                             every 5,000 steps.
                             Set to e.g. 5 for disk-constrained environments.

        Raises:
            ValueError: If max_checkpoints is provided and is not a positive integer.
        """
        if max_checkpoints is not None and max_checkpoints <= 0:
            raise ValueError(
                f"max_checkpoints must be a positive integer or None, "
                f"got {max_checkpoints}."
            )

        self.output_dir: str = output_dir
        self.max_checkpoints: Optional[int] = max_checkpoints

        # Create output directory on rank 0 only.
        # Other ranks will see it after the barrier in save().
        if DistributedUtils.is_main_process():
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"CheckpointManager initialized: "
            f"output_dir='{output_dir}', "
            f"max_checkpoints={max_checkpoints}, "
            f"safetensors_available={SAFETENSORS_AVAILABLE}, "
            f"fsdp_available={FSDP_AVAILABLE}"
        )

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Save a complete training checkpoint.

        Saves model weights, optimizer state, scheduler state, and metadata
        to a new checkpoint directory. Handles both FSDP-wrapped and plain
        nn.Module models.

        For FSDP models, uses FULL_STATE_DICT to gather all shards to rank 0
        before saving. This produces portable checkpoints loadable without FSDP.
        All ranks must call this method simultaneously (collective FSDP operation).

        The save sequence:
          1. Barrier: all ranks synchronize before save begins
          2. Gather model state dict (all ranks participate, rank 0 gets result)
          3. Rank 0 writes model.safetensors
          4. Gather optimizer state dict (all ranks participate, rank 0 gets result)
          5. Rank 0 writes optimizer.pt, scheduler.pt, metadata.json
          6. Rank 0 cleans up old checkpoints if max_checkpoints is set
          7. Barrier: all ranks wait for rank 0 to finish writing

        Args:
            model: The OLMoEModel (or FSDP-wrapped OLMoEModel) to save.
                   All ranks must pass the same model instance.
            optimizer: The AdamW (or RMSprop for KTO) optimizer to save.
                       All ranks must pass the same optimizer instance.
            scheduler: The LRScheduler or ConstantLRScheduler to save.
                       Must have a state_dict() method. All ranks must pass
                       the same scheduler instance.
            step: Current global training step (0-indexed). Used to name
                  the checkpoint directory: checkpoint-{step:08d}.
            metrics: Optional dictionary of metric values to store in metadata.
                     Example: {"ce_loss": 2.1, "lb_loss": 0.12, "grad_norm": 0.8}.
                     Stored in metadata.json for reference. Default: None.

        Raises:
            RuntimeError: If the checkpoint directory cannot be created.
            OSError: If writing checkpoint files fails on rank 0.
        """
        # -----------------------------------------------------------------------
        # Barrier 1: All ranks must reach this point before rank 0 starts writing.
        # This ensures all ranks have completed their current training step and
        # the model/optimizer state is consistent before saving.
        # -----------------------------------------------------------------------
        DistributedUtils.barrier()

        # Build checkpoint directory path.
        checkpoint_dir: Path = (
            Path(self.output_dir) / CHECKPOINT_DIR_PATTERN.format(step=step)
        )

        logger.info(
            f"Saving checkpoint at step {step:,} to '{checkpoint_dir}' "
            f"(rank={DistributedUtils.get_rank()})"
        )

        # -----------------------------------------------------------------------
        # Step 1: Gather model state dict.
        # All ranks participate in the FSDP gather even though only rank 0
        # receives the full state dict. Non-FSDP models use standard state_dict().
        # -----------------------------------------------------------------------
        model_state_dict: Dict[str, torch.Tensor] = self._gather_model_state_dict(model)

        # -----------------------------------------------------------------------
        # Step 2: Gather optimizer state dict.
        # FSDP requires a special API to gather sharded optimizer states.
        # -----------------------------------------------------------------------
        optimizer_state_dict: Optional[Dict[str, Any]] = self._gather_optimizer_state_dict(
            model, optimizer
        )

        # -----------------------------------------------------------------------
        # Steps 3-6: Rank 0 writes all checkpoint files.
        # -----------------------------------------------------------------------
        if DistributedUtils.is_main_process():
            # Create checkpoint directory.
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Write model weights.
            self._save_model_weights(model_state_dict, checkpoint_dir)

            # Write optimizer state.
            if optimizer_state_dict is not None:
                optimizer_path: Path = checkpoint_dir / OPTIMIZER_STATE_FILENAME
                torch.save(optimizer_state_dict, optimizer_path)
                logger.debug(f"Saved optimizer state to '{optimizer_path}'")

            # Write scheduler state.
            scheduler_path: Path = checkpoint_dir / SCHEDULER_STATE_FILENAME
            scheduler_state: Dict[str, Any] = self._get_scheduler_state(scheduler)
            torch.save(scheduler_state, scheduler_path)
            logger.debug(f"Saved scheduler state to '{scheduler_path}'")

            # Write metadata.
            metadata: Dict[str, Any] = {
                "step": step,
                "metrics": metrics if metrics is not None else {},
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "world_size": DistributedUtils.get_world_size(),
                "safetensors_format": SAFETENSORS_AVAILABLE,
            }
            metadata_path: Path = checkpoint_dir / METADATA_FILENAME
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata to '{metadata_path}'")

            # Clean up old checkpoints if max_checkpoints is set.
            self._cleanup_old_checkpoints()

            logger.info(
                f"Checkpoint saved successfully: step={step:,}, "
                f"dir='{checkpoint_dir}'"
            )

        # -----------------------------------------------------------------------
        # Barrier 2: All ranks wait for rank 0 to finish writing.
        # This prevents any rank from proceeding (and potentially triggering
        # another save) before the checkpoint is fully written to disk.
        # -----------------------------------------------------------------------
        DistributedUtils.barrier()

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
    ) -> int:
        """Load a complete training checkpoint for resuming training.

        Restores model weights, optimizer state, and scheduler state from a
        checkpoint directory. Returns the step number so the trainer knows
        where to resume.

        All ranks must call this method simultaneously for FSDP models.
        The model state dict is broadcast from rank 0 to all ranks by FSDP's
        load_state_dict() internally.

        Args:
            path: Path to the checkpoint directory (e.g., "outputs/checkpoint-00005000").
                  Must contain model.safetensors (or model.pt), optimizer.pt,
                  scheduler.pt, and metadata.json.
            model: The OLMoEModel (or FSDP-wrapped OLMoEModel) to load weights into.
                   All ranks must pass the same model instance.
            optimizer: The optimizer to restore state into.
                       All ranks must pass the same optimizer instance.
            scheduler: The LR scheduler to restore state into.
                       Must have a load_state_dict() method.

        Returns:
            The training step number from the checkpoint metadata. The trainer
            should resume training from step + 1.

        Raises:
            FileNotFoundError: If the checkpoint directory or required files
                               do not exist.
            RuntimeError: If checkpoint files are incomplete (interrupted save).
        """
        checkpoint_dir: Path = Path(path)

        # Validate checkpoint directory exists.
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: '{checkpoint_dir}'. "
                f"Available checkpoints: {self.list_checkpoints()}"
            )

        # Validate all required files are present (guards against interrupted saves).
        self._validate_checkpoint_files(checkpoint_dir)

        # Load metadata to get step number.
        metadata: Dict[str, Any] = self._load_metadata(checkpoint_dir)
        step: int = metadata.get("step", 0)

        logger.info(
            f"Loading checkpoint from '{checkpoint_dir}' "
            f"(step={step:,}, rank={DistributedUtils.get_rank()})"
        )

        # -----------------------------------------------------------------------
        # Load model weights.
        # All ranks participate; FSDP distributes shards internally.
        # -----------------------------------------------------------------------
        self._load_model_weights_into(checkpoint_dir, model)

        # -----------------------------------------------------------------------
        # Load optimizer state.
        # FSDP requires converting the full optimizer state dict back to
        # the sharded format expected by the optimizer.
        # -----------------------------------------------------------------------
        optimizer_path: Path = checkpoint_dir / OPTIMIZER_STATE_FILENAME
        if optimizer_path.exists():
            self._load_optimizer_state(optimizer_path, model, optimizer)
        else:
            logger.warning(
                f"Optimizer state file not found at '{optimizer_path}'. "
                f"Optimizer will start from scratch."
            )

        # -----------------------------------------------------------------------
        # Load scheduler state.
        # Not FSDP-specific — same on all ranks.
        # -----------------------------------------------------------------------
        scheduler_path: Path = checkpoint_dir / SCHEDULER_STATE_FILENAME
        if scheduler_path.exists():
            scheduler_state: Dict[str, Any] = torch.load(
                scheduler_path,
                map_location="cpu",
                weights_only=False,
            )
            self._load_scheduler_state(scheduler, scheduler_state)
            logger.debug(f"Loaded scheduler state from '{scheduler_path}'")
        else:
            logger.warning(
                f"Scheduler state file not found at '{scheduler_path}'. "
                f"Scheduler will start from scratch."
            )

        logger.info(
            f"Checkpoint loaded successfully: step={step:,}, "
            f"dir='{checkpoint_dir}'"
        )

        return step

    def load_model_only(
        self,
        path: str,
        model: nn.Module,
        strict: bool = True,
    ) -> None:
        """Load only model weights from a checkpoint.

        Used by analysis modules (router saturation, co-activation, domain/
        vocabulary specialization) and adaptation trainers (SFT, DPO) that
        need model weights but not optimizer/scheduler state.

        Supports three checkpoint formats:
          1. Local checkpoint directory (from save()): contains model.safetensors
          2. HuggingFace Hub model ID (e.g., "allenai/OLMoE-1B-7B-0924"):
             downloads and loads via transformers.AutoModelForCausalLM
          3. Local HuggingFace-format directory (contains config.json):
             loads via transformers.AutoModelForCausalLM.from_pretrained()

        All ranks must call this method simultaneously for FSDP models.

        Args:
            path: One of:
                  - Local checkpoint directory path (e.g., "outputs/checkpoint-00005000")
                  - HuggingFace Hub model ID (e.g., "allenai/OLMoE-1B-7B-0924")
                  - Local HuggingFace-format directory (contains config.json)
            model: The OLMoEModel (or FSDP-wrapped OLMoEModel) to load weights into.
                   All ranks must pass the same model instance.
            strict: Whether to require exact key matching between the checkpoint
                    and the model. Default: True. Set to False for loading
                    checkpoints with slightly different architectures (ablations).

        Raises:
            FileNotFoundError: If the path does not exist and is not a valid
                               HuggingFace Hub model ID.
            RuntimeError: If the checkpoint format is unrecognized.
        """
        checkpoint_path: Path = Path(path)

        # -----------------------------------------------------------------------
        # Determine checkpoint format and load accordingly.
        # -----------------------------------------------------------------------

        # Case 1: Local checkpoint directory from save() (contains model.safetensors or model.pt)
        model_weights_path: Path = checkpoint_path / MODEL_WEIGHTS_FILENAME
        model_weights_fallback_path: Path = checkpoint_path / MODEL_WEIGHTS_FALLBACK_FILENAME

        if checkpoint_path.is_dir() and (
            model_weights_path.exists() or model_weights_fallback_path.exists()
        ):
            logger.info(
                f"Loading model weights from local checkpoint: '{checkpoint_path}' "
                f"(rank={DistributedUtils.get_rank()})"
            )
            self._load_model_weights_into(checkpoint_path, model, strict=strict)
            return

        # Case 2: HuggingFace-format directory (contains config.json) or Hub model ID.
        # Detect HuggingFace format by checking for config.json in local dir,
        # or by treating the path as a Hub model ID if it doesn't exist locally.
        hf_config_path: Path = checkpoint_path / "config.json"
        is_hf_local: bool = checkpoint_path.is_dir() and hf_config_path.exists()
        is_hf_hub: bool = not checkpoint_path.exists() and "/" in path

        if is_hf_local or is_hf_hub:
            logger.info(
                f"Loading model weights from HuggingFace format: '{path}' "
                f"(rank={DistributedUtils.get_rank()})"
            )
            self._load_from_huggingface(path, model, strict=strict)
            return

        # Case 3: Path doesn't exist and doesn't look like a Hub ID.
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint path not found: '{path}'. "
                f"Expected either:\n"
                f"  - A local checkpoint directory containing '{MODEL_WEIGHTS_FILENAME}'\n"
                f"  - A HuggingFace Hub model ID (e.g., 'allenai/OLMoE-1B-7B-0924')\n"
                f"  - A local HuggingFace-format directory containing 'config.json'\n"
                f"Available local checkpoints: {self.list_checkpoints()}"
            )

        # Case 4: Path exists but format is unrecognized.
        raise RuntimeError(
            f"Unrecognized checkpoint format at '{path}'. "
            f"Expected a directory containing '{MODEL_WEIGHTS_FILENAME}', "
            f"'{MODEL_WEIGHTS_FALLBACK_FILENAME}', or 'config.json'."
        )

    def list_checkpoints(self) -> List[str]:
        """List all checkpoint directories sorted by step number.

        Scans the output_dir for directories matching the checkpoint naming
        pattern (checkpoint-{8 digits}) and returns them sorted by step number.

        Returns:
            List of full checkpoint directory paths as strings, sorted from
            oldest (lowest step) to newest (highest step).
            Returns an empty list if output_dir doesn't exist or contains
            no checkpoints.

        Example:
            >>> manager = CheckpointManager("outputs")
            >>> manager.list_checkpoints()
            ['outputs/checkpoint-00005000',
             'outputs/checkpoint-00010000',
             'outputs/checkpoint-00015000']
        """
        output_path: Path = Path(self.output_dir)

        if not output_path.exists():
            return []

        checkpoints: List[Path] = []
        for entry in output_path.iterdir():
            if (
                entry.is_dir()
                and entry.name.startswith(CHECKPOINT_DIR_PREFIX)
            ):
                # Validate the suffix is an 8-digit number.
                suffix: str = entry.name[len(CHECKPOINT_DIR_PREFIX):]
                if suffix.isdigit() and len(suffix) == 8:
                    checkpoints.append(entry)

        # Sort by step number (extracted from directory name).
        # Zero-padding makes lexicographic and numeric sort equivalent,
        # but we extract the integer for clarity and robustness.
        checkpoints.sort(key=lambda p: int(p.name[len(CHECKPOINT_DIR_PREFIX):]))

        return [str(p) for p in checkpoints]

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _is_fsdp_model(self, model: nn.Module) -> bool:
        """Check if the model is wrapped with FSDP.

        Args:
            model: The model to check.

        Returns:
            True if the model is an FSDP-wrapped module, False otherwise.
        """
        if not FSDP_AVAILABLE:
            return False
        return isinstance(model, FullyShardedDataParallel)

    def _gather_model_state_dict(
        self, model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Gather model state dict from all FSDP shards to rank 0.

        For FSDP models: uses FULL_STATE_DICT context to gather all shards.
        All ranks must call this method; only rank 0 receives the full dict.

        For non-FSDP models: standard model.state_dict() on all ranks.

        Args:
            model: The model (FSDP-wrapped or plain nn.Module).

        Returns:
            Full model state dict on rank 0 (CPU tensors).
            Empty dict on non-rank-0 processes for FSDP models.
        """
        if self._is_fsdp_model(model):
            # FSDP full state dict: gathers all shards to rank 0.
            # offload_to_cpu=True: prevents OOM when gathering 6.9B params.
            # rank0_only=True: only rank 0 receives the full dict.
            fsdp_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )
            with FullyShardedDataParallel.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                fsdp_config,
            ):
                # All ranks call state_dict() — FSDP coordinates the gather.
                # Non-rank-0 processes receive an empty dict.
                state_dict: Dict[str, torch.Tensor] = model.state_dict()
            return state_dict
        else:
            # Non-FSDP: standard state dict, move to CPU for consistency.
            state_dict = {
                k: v.cpu() for k, v in model.state_dict().items()
            }
            return state_dict

    def _gather_optimizer_state_dict(
        self,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> Optional[Dict[str, Any]]:
        """Gather optimizer state dict from all FSDP shards to rank 0.

        For FSDP models: uses FSDP.optim_state_dict() to gather sharded
        optimizer states. All ranks must call this method.

        For non-FSDP models: standard optimizer.state_dict() on rank 0 only.

        Args:
            model: The model (FSDP-wrapped or plain nn.Module).
            optimizer: The optimizer whose state to gather.

        Returns:
            Full optimizer state dict on rank 0.
            None on non-rank-0 processes for FSDP models.
        """
        if self._is_fsdp_model(model):
            # FSDP optimizer state dict: gathers sharded optimizer states.
            # All ranks must call this; only rank 0 gets the full result.
            optim_config = FullOptimStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )
            with FullyShardedDataParallel.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                None,  # model state dict config (not needed here)
                optim_config,
            ):
                optim_state: Dict[str, Any] = FullyShardedDataParallel.optim_state_dict(
                    model, optimizer
                )
            return optim_state
        else:
            # Non-FSDP: standard optimizer state dict on rank 0 only.
            if DistributedUtils.is_main_process():
                return optimizer.state_dict()
            return None

    def _save_model_weights(
        self,
        state_dict: Dict[str, torch.Tensor],
        checkpoint_dir: Path,
    ) -> None:
        """Save model weights to disk using safetensors or torch.save fallback.

        Only called on rank 0. The state_dict contains CPU tensors gathered
        by _gather_model_state_dict().

        Args:
            state_dict: Model state dict with CPU tensor values.
            checkpoint_dir: Directory to save the weights file into.
        """
        if not state_dict:
            logger.warning(
                "Model state dict is empty. Skipping model weights save. "
                "This is expected on non-rank-0 processes for FSDP models."
            )
            return

        if SAFETENSORS_AVAILABLE:
            weights_path: Path = checkpoint_dir / MODEL_WEIGHTS_FILENAME
            # safetensors requires all tensors to be contiguous.
            contiguous_state_dict: Dict[str, torch.Tensor] = {
                k: v.contiguous() for k, v in state_dict.items()
            }
            safetensors_save_file(contiguous_state_dict, str(weights_path))
            logger.debug(
                f"Saved model weights (safetensors) to '{weights_path}' "
                f"({len(state_dict)} tensors)"
            )
        else:
            # Fallback to torch.save if safetensors is not available.
            weights_path = checkpoint_dir / MODEL_WEIGHTS_FALLBACK_FILENAME
            torch.save(state_dict, weights_path)
            logger.debug(
                f"Saved model weights (torch.save fallback) to '{weights_path}' "
                f"({len(state_dict)} tensors)"
            )

    def _load_model_weights_into(
        self,
        checkpoint_dir: Path,
        model: nn.Module,
        strict: bool = True,
    ) -> None:
        """Load model weights from a checkpoint directory into the model.

        Handles both FSDP and non-FSDP models. For FSDP models, uses the
        FULL_STATE_DICT context so FSDP distributes the loaded weights to
        the appropriate shards on each rank.

        All ranks must call this method for FSDP models.

        Args:
            checkpoint_dir: Directory containing model.safetensors or model.pt.
            model: The model to load weights into.
            strict: Whether to require exact key matching. Default: True.
        """
        # Load state dict from disk (rank 0 loads, then FSDP broadcasts).
        state_dict: Dict[str, torch.Tensor] = self._read_model_weights(checkpoint_dir)

        # Strip FSDP key prefixes if present (from checkpoints saved with FSDP).
        state_dict = self._strip_fsdp_prefixes(state_dict)

        if self._is_fsdp_model(model):
            # FSDP load: use FULL_STATE_DICT context.
            # FSDP's load_state_dict() broadcasts from rank 0 to all ranks.
            fsdp_config = FullStateDictConfig(
                offload_to_cpu=True,
                rank0_only=False,  # All ranks need the state dict for loading
            )
            with FullyShardedDataParallel.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                fsdp_config,
            ):