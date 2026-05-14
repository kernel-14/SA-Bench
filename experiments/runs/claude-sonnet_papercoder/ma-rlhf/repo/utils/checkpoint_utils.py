## utils/checkpoint_utils.py
"""Checkpoint saving and loading utilities for the MA-RLHF pipeline.

This module provides CheckpointUtils, a stateless collection of static
methods for saving and restoring training state across all three stages
(SFT, RM, MA-PPO). It handles both plain PyTorch/HuggingFace models and
DeepSpeed-wrapped engines transparently.

Key design decisions:
  - All methods are static: no instance state, purely functional utilities.
  - DeepSpeed detection via hasattr(model, 'save_checkpoint').
  - trainer_state.json is always written as the canonical step record,
    enabling get_latest_checkpoint() to work regardless of backend.
  - Atomic writes for non-DeepSpeed saves: write to .tmp then os.replace.
  - Scheduler=None is handled gracefully throughout.
  - No Config dependency at runtime — operates on objects and paths only.

File layout on disk:
    {output_dir}/
      sft/
        step_500/
          model.pt          (non-DeepSpeed)
          optimizer.pt      (non-DeepSpeed)
          scheduler.pt      (non-DeepSpeed, if scheduler is not None)
          trainer_state.json
        step_1000/
          ...
      rm/
        step_500/
          ...
      ppo/
        policy/
          step_500/
            ...
        critic/
          step_500/
            ...

Dependencies:
    Standard library: os, re, json, logging, pathlib, typing
    External: torch
    Internal: none (leaf module in the dependency graph)
"""

import json
import logging
import os
import pathlib
import re
import tempfile
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class CheckpointUtils:
    """Stateless utilities for saving and loading training checkpoints.

    All methods are static. This class is never instantiated — it serves
    as a namespace for checkpoint-related operations used by SFTTrainer,
    RMTrainer, MAPPOTrainer, and main.py.

    Supports two backends transparently:
      1. Plain PyTorch / HuggingFace models: saves state dicts as .pt files.
      2. DeepSpeed engines: delegates to model.save_checkpoint() /
         model.load_checkpoint() which handle ZeRO sharding internally.

    The canonical step record is always trainer_state.json, written in
    both backends, so get_latest_checkpoint() works uniformly.
    """

    # Filename constants for non-DeepSpeed checkpoints.
    _MODEL_FILENAME: str = "model.pt"
    _OPTIMIZER_FILENAME: str = "optimizer.pt"
    _SCHEDULER_FILENAME: str = "scheduler.pt"
    _TRAINER_STATE_FILENAME: str = "trainer_state.json"

    # Pattern for checkpoint subdirectory names.
    _STEP_DIR_PATTERN: re.Pattern = re.compile(r"^step_(\d+)$")

    # ------------------------------------------------------------------
    # Public static methods
    # ------------------------------------------------------------------

    @staticmethod
    def save_checkpoint(
        model: Any,
        optimizer: Any,
        scheduler: Optional[Any],
        step: int,
        path: str,
    ) -> None:
        """Save training state to a checkpoint directory.

        Creates a subdirectory named ``step_{step}`` under ``path`` and
        writes the model, optimizer, and scheduler state. The exact files
        written depend on whether ``model`` is a DeepSpeed engine or a
        plain PyTorch model.

        For DeepSpeed engines, model.save_checkpoint() is called with
        save_zero_checkpoint=True to ensure ZeRO-3 shards are included.
        Optimizer and scheduler state are managed by DeepSpeed internally.

        For plain models, atomic writes are used: each file is written to
        a temporary path first, then renamed via os.replace() to prevent
        partial writes from corrupting a checkpoint if the process is
        killed mid-save.

        This method never raises — failures are logged as warnings so that
        an expensive training run is not aborted by a checkpoint error.

        Args:
            model: The model to checkpoint. Either a plain
                torch.nn.Module / HuggingFace PreTrainedModel, or a
                DeepSpeed engine (detected via hasattr check).
            optimizer: The optimizer whose state should be saved. May be
                a plain torch.optim.Optimizer or a DeepSpeed-wrapped one.
                For DeepSpeed, this argument is ignored (DS saves it).
            scheduler: The LR scheduler whose state should be saved.
                Pass None if no scheduler is used (constant LR runs).
                For DeepSpeed, this argument is ignored.
            step: The current global training step number. Used as the
                subdirectory name (``step_{step}``) and stored in
                trainer_state.json.
            path: Parent directory under which the ``step_{step}``
                subdirectory will be created. For example:
                ``./outputs/ppo/policy/`` → saves to
                ``./outputs/ppo/policy/step_2000/``.
        """
        checkpoint_dir: str = os.path.join(path, f"step_{step}")

        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to create checkpoint directory '%s': %s. "
                "Skipping checkpoint save at step %d.",
                checkpoint_dir,
                exc,
                step,
            )
            return

        logger.info(
            "Saving checkpoint at step %d to '%s'.", step, checkpoint_dir
        )

        # Detect DeepSpeed engine by checking for its save_checkpoint method.
        # This is more robust than checking the module name since it works
        # with any DeepSpeed version and custom wrappers.
        is_deepspeed: bool = hasattr(model, "save_checkpoint") and hasattr(
            model, "load_checkpoint"
        )

        if is_deepspeed:
            CheckpointUtils._save_deepspeed_checkpoint(
                model=model,
                step=step,
                path=path,
                checkpoint_dir=checkpoint_dir,
            )
        else:
            CheckpointUtils._save_pytorch_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                checkpoint_dir=checkpoint_dir,
            )

    @staticmethod
    def load_checkpoint(
        model: Any,
        optimizer: Any,
        scheduler: Optional[Any],
        path: str,
        strict: bool = True,
    ) -> int:
        """Load training state from a specific checkpoint directory.

        ``path`` must point to a specific checkpoint subdirectory (e.g.,
        ``./outputs/ppo/policy/step_2000/``), not the parent directory.
        Use get_latest_checkpoint() to resolve the latest checkpoint path
        before calling this method.

        For DeepSpeed engines, model.load_checkpoint() is called with the
        parent directory and the tag extracted from the path basename.
        DeepSpeed restores optimizer and scheduler state automatically.

        For plain models, state dicts are loaded from .pt files. The model
        state dict is loaded with map_location='cpu' first to avoid GPU
        memory issues, then moved to the model's current device.

        Args:
            model: The model to restore. Either a plain torch.nn.Module /
                HuggingFace PreTrainedModel, or a DeepSpeed engine.
            optimizer: The optimizer to restore. Pass None to skip
                optimizer state restoration (e.g., evaluation-only loading).
                For DeepSpeed, this argument is ignored.
            scheduler: The LR scheduler to restore. Pass None if no
                scheduler is used or to skip restoration.
                For DeepSpeed, this argument is ignored.
            path: Path to the specific checkpoint subdirectory to load,
                e.g., ``./outputs/ppo/policy/step_2000/``.
            strict: Whether to enforce strict state dict matching for
                non-DeepSpeed models. Set to False when loading a base
                model checkpoint into a model with additional heads
                (e.g., loading SFT weights into RewardModel). Defaults
                to True.

        Returns:
            The integer step number read from trainer_state.json.
            Returns 0 if trainer_state.json is missing or malformed,
            with a warning logged.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            RuntimeError: If model.pt is missing for non-DeepSpeed models.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: '{path}'. "
                f"Use get_latest_checkpoint() to find a valid checkpoint."
            )

        logger.info("Loading checkpoint from '%s'.", path)

        # Detect DeepSpeed engine.
        is_deepspeed: bool = hasattr(model, "save_checkpoint") and hasattr(
            model, "load_checkpoint"
        )

        if is_deepspeed:
            step: int = CheckpointUtils._load_deepspeed_checkpoint(
                model=model,
                path=path,
            )
        else:
            step = CheckpointUtils._load_pytorch_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                path=path,
                strict=strict,
            )

        logger.info(
            "Checkpoint loaded successfully from '%s' (step=%d).",
            path,
            step,
        )

        return step

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Find the path to the most recent valid checkpoint.

        Scans ``checkpoint_dir`` for subdirectories matching the pattern
        ``step_{N}`` and returns the path to the one with the highest N.
        A checkpoint is considered valid only if it contains a
        trainer_state.json file (guards against partially written
        checkpoints from interrupted saves).

        Args:
            checkpoint_dir: Parent directory to scan, e.g.,
                ``./outputs/ppo/policy/``. Returns None if this directory
                does not exist or contains no valid checkpoints.

        Returns:
            The full path to the latest valid checkpoint subdirectory,
            e.g., ``./outputs/ppo/policy/step_2000/``.
            Returns None if no valid checkpoints are found.
        """
        if not os.path.exists(checkpoint_dir):
            logger.debug(
                "Checkpoint directory '%s' does not exist; "
                "no checkpoint to resume from.",
                checkpoint_dir,
            )
            return None

        if not os.path.isdir(checkpoint_dir):
            logger.warning(
                "'%s' is not a directory; cannot scan for checkpoints.",
                checkpoint_dir,
            )
            return None

        # Scan for subdirectories matching step_{N}.
        valid_checkpoints: list[Tuple[int, str]] = []

        try:
            entries: list[str] = os.listdir(checkpoint_dir)
        except OSError as exc:
            logger.warning(
                "Failed to list checkpoint directory '%s': %s.",
                checkpoint_dir,
                exc,
            )
            return None

        for entry in entries:
            match: Optional[re.Match] = CheckpointUtils._STEP_DIR_PATTERN.match(
                entry
            )
            if match is None:
                # Not a step directory (e.g., wandb/, logs/, etc.).
                continue

            step_num: int = int(match.group(1))
            candidate_path: str = os.path.join(checkpoint_dir, entry)

            if not os.path.isdir(candidate_path):
                continue

            # Validate: trainer_state.json must exist to confirm the
            # checkpoint was fully written.
            state_file: str = os.path.join(
                candidate_path, CheckpointUtils._TRAINER_STATE_FILENAME
            )
            if not os.path.exists(state_file):
                logger.debug(
                    "Skipping incomplete checkpoint '%s' "
                    "(missing trainer_state.json).",
                    candidate_path,
                )
                continue

            valid_checkpoints.append((step_num, candidate_path))

        if not valid_checkpoints:
            logger.info(
                "No valid checkpoints found in '%s'.", checkpoint_dir
            )
            return None

        # Return the checkpoint with the highest step number.
        valid_checkpoints.sort(key=lambda x: x[0], reverse=True)
        latest_step, latest_path = valid_checkpoints[0]

        logger.info(
            "Latest checkpoint found: '%s' (step=%d).",
            latest_path,
            latest_step,
        )

        return latest_path

    # ------------------------------------------------------------------
    # Private helpers: DeepSpeed backend
    # ------------------------------------------------------------------

    @staticmethod
    def _save_deepspeed_checkpoint(
        model: Any,
        step: int,
        path: str,
        checkpoint_dir: str,
    ) -> None:
        """Save a DeepSpeed engine checkpoint.

        Calls model.save_checkpoint() with the parent path and a step tag.
        DeepSpeed creates its own subdirectory structure under path/step_{step}/.
        save_zero_checkpoint=True ensures ZeRO-3 optimizer shards are saved.

        Also writes trainer_state.json for get_latest_checkpoint() discovery.

        Args:
            model: DeepSpeed engine instance.
            step: Current training step.
            path: Parent directory (DeepSpeed uses this as the base dir).
            checkpoint_dir: Full path to step_{step}/ subdirectory.
                Used only for writing trainer_state.json.
        """
        tag: str = f"step_{step}"

        try:
            # save_zero_checkpoint=True is required for ZeRO-3 to save
            # the full model weights alongside the optimizer shards.
            # For ZeRO-2, it is a no-op but harmless.
            model.save_checkpoint(
                path,
                tag=tag,
                save_zero_checkpoint=True,
            )
            logger.info(
                "DeepSpeed checkpoint saved: path='%s', tag='%s'.",
                path,
                tag,
            )
        except Exception as exc:
            logger.warning(
                "DeepSpeed model.save_checkpoint() failed at step %d: %s. "
                "Checkpoint may be incomplete.",
                step,
                exc,
            )
            return

        # Write trainer_state.json for checkpoint discovery.
        CheckpointUtils._write_trainer_state(checkpoint_dir, step)

    @staticmethod
    def _load_deepspeed_checkpoint(
        model: Any,
        path: str,
    ) -> int:
        """Load a DeepSpeed engine checkpoint.

        Extracts the tag from the path basename and calls
        model.load_checkpoint() with the parent directory and tag.
        DeepSpeed restores optimizer and scheduler state automatically.

        Args:
            model: DeepSpeed engine instance.
            path: Full path to the step_{N}/ checkpoint directory.

        Returns:
            The step number read from trainer_state.json, or 0 on failure.
        """
        tag: str = os.path.basename(path.rstrip(os.sep))
        parent_dir: str = os.path.dirname(path.rstrip(os.sep))

        try:
            model.load_checkpoint(parent_dir, tag=tag)
            logger.info(
                "DeepSpeed checkpoint loaded: parent='%s', tag='%s'.",
                parent_dir,
                tag,
            )
        except Exception as exc:
            logger.warning(
                "DeepSpeed model.load_checkpoint() failed for tag='%s': %s.",
                tag,
                exc,
            )

        return CheckpointUtils._read_trainer_state(path)

    # ------------------------------------------------------------------
    # Private helpers: plain PyTorch backend
    # ------------------------------------------------------------------

    @staticmethod
    def _save_pytorch_checkpoint(
        model: Any,
        optimizer: Any,
        scheduler: Optional[Any],
        step: int,
        checkpoint_dir: str,
    ) -> None:
        """Save a plain PyTorch model checkpoint using atomic writes.

        Each file is written to a temporary path in the same directory,
        then renamed via os.replace() for atomicity. This prevents partial
        writes from corrupting a checkpoint if the process is killed
        mid-save.

        Args:
            model: torch.nn.Module or HuggingFace PreTrainedModel.
            optimizer: torch.optim.Optimizer instance.
            scheduler: LR scheduler instance, or None.
            step: Current training step.
            checkpoint_dir: Full path to the step_{N}/ directory.
        """
        # --- Model state dict ---
        model_path: str = os.path.join(
            checkpoint_dir, CheckpointUtils._MODEL_FILENAME
        )
        try:
            CheckpointUtils._atomic_torch_save(
                model.state_dict(), model_path
            )
            logger.debug("Model state dict saved to '%s'.", model_path)
        except Exception as exc:
            logger.warning(
                "Failed to save model state dict at step %d: %s.",
                step,
                exc,
            )
            return

        # --- Optimizer state dict ---
        if optimizer is not None:
            optimizer_path: str = os.path.join(
                checkpoint_dir, CheckpointUtils._OPTIMIZER_FILENAME
            )
            try:
                CheckpointUtils._atomic_torch_save(
                    optimizer.state_dict(), optimizer_path
                )
                logger.debug(
                    "Optimizer state dict saved to '%s'.", optimizer_path
                )
            except Exception as exc:
                logger.warning(
                    "Failed to save optimizer state dict at step %d: %s.",
                    step,
                    exc,
                )

        # --- Scheduler state dict (optional) ---
        if scheduler is not None:
            scheduler_path: str = os.path.join(
                checkpoint_dir, CheckpointUtils._SCHEDULER_FILENAME
            )
            try:
                CheckpointUtils._atomic_torch_save(
                    scheduler.state_dict(), scheduler_path
                )
                logger.debug(
                    "Scheduler state dict saved to '%s'.", scheduler_path
                )
            except Exception as exc:
                logger.warning(
                    "Failed to save scheduler state dict at step %d: %s.",
                    step,
                    exc,
                )

        # --- trainer_state.json (always last, marks checkpoint as complete) ---
        CheckpointUtils._write_trainer_state(checkpoint_dir, step)

        logger.info(
            "PyTorch checkpoint saved at step %d to '%s'.",
            step,
            checkpoint_dir,
        )

    @staticmethod
    def _load_pytorch_checkpoint(
        model: Any,
        optimizer: Optional[Any],
        scheduler: Optional[Any],
        path: str,
        strict: bool = True,
    ) -> int:
        """Load a plain PyTorch model checkpoint.

        Loads model weights with map_location='cpu' first to avoid GPU
        memory fragmentation, then moves to the model's current device.

        Args:
            model: torch.nn.Module or HuggingFace PreTrainedModel.
            optimizer: torch.optim.Optimizer, or None to skip.
            scheduler: LR scheduler, or None to skip.
            path: Full path to the step_{N}/ checkpoint directory.
            strict: Whether to enforce strict state dict matching.

        Returns:
            The step number from trainer_state.json, or 0 on failure.

        Raises:
            RuntimeError: If model.pt is missing.
        """
        model_path: str = os.path.join(path, CheckpointUtils._MODEL_FILENAME)

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model checkpoint file not found: '{model_path}'. "
                f"The checkpoint at '{path}' may be incomplete or corrupted."
            )

        # --- Model state dict ---
        # Load to CPU first to avoid GPU memory issues during loading.
        # The model's parameters will be on the correct device already;
        # load_state_dict() moves the loaded tensors to match.
        try:
            state_dict: dict = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict, strict=strict)
            logger.debug(
                "Model state dict loaded from '%s' (strict=%s).",
                model_path,
                strict,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load model state dict from '%s': %s.",
                model_path,
                exc,
            )
            raise

        # --- Optimizer state dict ---
        if optimizer is not None:
            optimizer_path: str = os.path.join(
                path, CheckpointUtils._OPTIMIZER_FILENAME
            )
            if os.path.exists(optimizer_path):
                try:
                    opt_state: dict = torch.load(
                        optimizer_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    optimizer.load_state_dict(opt_state)
                    logger.debug(
                        "Optimizer state dict loaded from '%s'.",
                        optimizer_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load optimizer state dict from '%s': %s. "
                        "Optimizer state will be reset.",
                        optimizer_path,
                        exc,
                    )
            else:
                logger.warning(
                    "Optimizer checkpoint not found at '%s'. "
                    "Optimizer state will be reset.",
                    optimizer_path,
                )

        # --- Scheduler state dict ---
        if scheduler is not None:
            scheduler_path: str = os.path.join(
                path, CheckpointUtils._SCHEDULER_FILENAME
            )
            if os.path.exists(scheduler_path):
                try:
                    sched_state: dict = torch.load(
                        scheduler_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    scheduler.load_state_dict(sched_state)
                    logger.debug(
                        "Scheduler state dict loaded from '%s'.",
                        scheduler_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load scheduler state dict from '%s': %s. "
                        "Scheduler state will be reset.",
                        scheduler_path,
                        exc,
                    )
            else:
                logger.debug(
                    "Scheduler checkpoint not found at '%s'; "
                    "scheduler state will be reset.",
                    scheduler_path,
                )

        return CheckpointUtils._read_trainer_state(path)

    # ------------------------------------------------------------------
    # Private helpers: trainer_state.json I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _write_trainer_state(checkpoint_dir: str, step: int) -> None:
        """Write trainer_state.json to mark a checkpoint as complete.

        This file is always written last, after all model/optimizer/scheduler
        files. Its presence signals to get_latest_checkpoint() that the
        checkpoint is valid and fully written.

        Uses an atomic write (temp file + os.replace) to prevent a partial
        JSON file from being mistaken for a valid checkpoint.

        Args:
            checkpoint_dir: Directory where trainer_state.json will be written.
            step: Current training step to record.
        """
        state_path: str = os.path.join(
            checkpoint_dir, CheckpointUtils._TRAINER_STATE_FILENAME
        )
        state_data: dict = {"step": step}

        try:
            # Write to a temp file in the same directory for atomic rename.
            dir_path: str = os.path.dirname(state_path)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=dir_path,
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as tmp_file:
                json.dump(state_data, tmp_file, indent=2)
                tmp_path: str = tmp_file.name

            # Atomic rename: replaces state_path if it already exists.
            os.replace(tmp_path, state_path)

            logger.debug(
                "trainer_state.json written: step=%d, path='%s'.",
                step,
                state_path,
            )
        except Exception as exc:
            logger.warning(
                "Failed to write trainer_state.json at '%s': %s.",
                state_path,
                exc,
            )
            # Clean up temp file if rename failed.
            try:
                if "tmp_path" in dir() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _read_trainer_state(checkpoint_dir: str) -> int:
        """Read the step number from trainer_state.json.

        Args:
            checkpoint_dir: Directory containing trainer_state.json.

        Returns:
            The integer step number, or 0 if the file is missing or
            malformed (with a warning logged).
        """
        state_path: str = os.path.join(
            checkpoint_dir, CheckpointUtils._TRAINER_STATE_FILENAME
        )

        if not os.path.exists(state_path):
            logger.warning(
                "trainer_state.json not found at '%s'. "
                "Returning step=0 as default.",
                state_path,
            )
            return 0

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state_data: dict = json.load(f)

            step: int = int(state_data.get("step", 0))
            logger.debug(
                "trainer_state.json read: step=%d from '%s'.",
                step,
                state_path,
            )
            return step

        except (json.JSONDecodeError, ValueError, TypeError, OSError) as exc:
            logger.warning(
                "Failed to read trainer_state.json from '%s': %s. "
                "Returning step=0 as default.",
                state_path,
                exc,
            )
            return 0

    # ------------------------------------------------------------------
    # Private helpers: atomic file I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_torch_save(obj: Any, path: str) -> None:
        """Save a PyTorch object atomically using a temp file + rename.

        Writes to a temporary file in the same directory as ``path``,
        then renames it to ``path`` via os.replace(). This ensures that
        ``path`` is never in a partially written state, even if the
        process is killed during the save.

        Args:
            obj: Any object serializable by torch.save (typically a
                state dict: Dict[str, torch.Tensor]).
            path: Target file path. The directory must already exist.

        Raises:
            OSError: If the directory does not exist or is not writable.
            RuntimeError: If torch.save fails.
        """
        dir_path: str = os.path.dirname(os.path.abspath(path))

        # Write to a temp file in the same directory to ensure the rename
        # is atomic (same filesystem, no cross-device move).
        with tempfile.NamedTemporaryFile(
            dir=dir_path,
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path: str = tmp_file.name

        try:
            torch.save(obj, tmp_path)
            # Atomic rename: replaces path if it already exists.
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure.
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise
