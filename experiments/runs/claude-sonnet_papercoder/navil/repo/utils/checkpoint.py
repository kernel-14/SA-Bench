## utils/checkpoint.py
"""Checkpoint management for NaViL training.

This module provides the ``CheckpointManager`` class, which handles
persisting and restoring the full training state (model weights,
optimizer state, scheduler state, and metadata) to/from disk.

Directory layout per checkpoint::

    checkpoints/navil_2b/
    ├── step_000005000/
    │   ├── model/
    │   │   ├── config.json
    │   │   └── model.safetensors
    │   ├── optimizer.pt
    │   ├── scheduler.pt
    │   └── metadata.json
    └── step_000010000/
        └── ...

Step numbers are zero-padded to 9 digits so that lexicographic sort
equals numeric sort, enabling correct ordering in ``list_checkpoints``.

Config alignment:
    - ``output.checkpoint_dir: "checkpoints/navil_2b"``
    - ``output.max_checkpoints: 3``
    - ``output.save_every_steps: 5000``

Design constraints:
    - No internal project dependencies (leaf utility module).
    - ``torch.save`` for optimizer/scheduler (contain non-tensor Python objects).
    - safetensors for model weights (delegated to ``NaViLModel.save_pretrained``).
    - Cleanup is called automatically inside ``save`` to maintain the
      ``max_checkpoints`` invariant without requiring caller management.
"""

import datetime
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Module-level logger — no handlers attached here; callers configure logging
# via utils/logging_utils.py. Using __name__ gives "utils.checkpoint".
logger: logging.Logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading of NaViL training checkpoints.

    Each checkpoint captures the complete training state needed to resume
    training without loss spikes or step-count discrepancies:

    - Model weights (via ``NaViLModel.save_pretrained`` / safetensors)
    - Optimizer state dict (momentum buffers, step counters)
    - Scheduler state (``current_step`` of the custom ``LRScheduler``)
    - Metadata (step, stage, global_step, timestamp)

    The manager enforces a maximum checkpoint count by deleting the oldest
    checkpoints after each save.

    Args:
        output_dir:      Root directory under which per-step checkpoint
                         subdirectories are created. Created automatically
                         if it does not exist.
                         Example: ``"checkpoints/navil_2b"``
        max_checkpoints: Maximum number of checkpoint directories to retain.
                         When a new checkpoint is saved and the total count
                         exceeds this value, the oldest checkpoints are
                         deleted. Set to ``0`` to keep all checkpoints
                         (no cleanup). Defaults to ``3``.

    Example::

        manager = CheckpointManager("checkpoints/navil_2b", max_checkpoints=3)
        # During training:
        ckpt_path = manager.save(model, trainer, step=5000)
        # On resume:
        resumed_step = manager.load("checkpoints/navil_2b/step_000005000",
                                    model, trainer)
    """

    def __init__(
        self,
        output_dir: str,
        max_checkpoints: int = 3,
    ) -> None:
        """Initialise the CheckpointManager and ensure the output directory exists.

        Args:
            output_dir:      Root directory for checkpoint subdirectories.
            max_checkpoints: Maximum number of checkpoints to retain.
                             ``0`` means keep all (no cleanup).
        """
        self.output_dir: Path = Path(output_dir)
        self.max_checkpoints: int = max_checkpoints

        # Create the root checkpoint directory immediately so that callers
        # do not need to handle directory creation themselves.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "CheckpointManager initialised. output_dir=%s, max_checkpoints=%d",
            self.output_dir,
            self.max_checkpoints,
        )

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def save(
        self,
        model: Any,
        trainer: Any,
        step: int,
    ) -> str:
        """Persist the full training state to a new checkpoint directory.

        Saves model weights, optimizer state, scheduler state, and a
        metadata JSON file. After saving, calls ``cleanup_old_checkpoints``
        to enforce the ``max_checkpoints`` limit.

        Args:
            model:   A ``NaViLModel`` instance. Must expose a
                     ``save_pretrained(path: str)`` method that writes
                     model weights (preferably in safetensors format) and
                     a ``config.json`` to the given directory.
            trainer: A ``NaViLTrainer`` instance. Must expose:
                     - ``trainer.optimizer``: a ``torch.optim.Optimizer``
                     - ``trainer.scheduler``: a custom ``LRScheduler``
                       with a ``current_step: int`` attribute
                     - ``trainer.current_stage: str``
                     - ``trainer.global_step: int``
            step:    The current global training step (0-indexed integer).
                     Used to name the checkpoint subdirectory.

        Returns:
            The absolute path to the created checkpoint directory as a
            string, e.g. ``"checkpoints/navil_2b/step_000005000"``.

        Raises:
            RuntimeError: If model weight saving fails and no fallback
                          ``state_dict`` is available.

        Note:
            Re-saving to the same step (same ``step`` value) is safe —
            ``mkdir(exist_ok=True)`` and file overwrites handle it
            gracefully.
        """
        # ------------------------------------------------------------------ #
        # 1. Construct and create the checkpoint subdirectory.                #
        # ------------------------------------------------------------------ #
        # Zero-pad to 9 digits: step=5000 → "step_000005000"
        # This ensures lexicographic sort == numeric sort in list_checkpoints.
        ckpt_dir_name: str = f"step_{step:09d}"
        ckpt_path: Path = self.output_dir / ckpt_dir_name
        ckpt_path.mkdir(parents=True, exist_ok=True)

        logger.info("Saving checkpoint at step %d to %s", step, ckpt_path)

        # ------------------------------------------------------------------ #
        # 2. Save model weights.                                               #
        # ------------------------------------------------------------------ #
        # Delegate to NaViLModel.save_pretrained, which writes safetensors
        # weight files and a config.json into the "model" subdirectory.
        # Fall back to torch.save(state_dict) if save_pretrained is absent.
        model_dir: Path = ckpt_path / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained") and callable(
            getattr(model, "save_pretrained")
        ):
            try:
                model.save_pretrained(str(model_dir))
                logger.debug("Model weights saved via save_pretrained to %s", model_dir)
            except Exception as exc:
                # Log the error and attempt the fallback before re-raising.
                logger.warning(
                    "model.save_pretrained failed (%s); falling back to "
                    "torch.save(state_dict).",
                    exc,
                )
                fallback_path: Path = ckpt_path / "model.pt"
                torch.save(model.state_dict(), str(fallback_path))
                logger.debug("Model weights saved via state_dict to %s", fallback_path)
        else:
            # NaViLModel.save_pretrained not yet implemented — use fallback.
            fallback_path = ckpt_path / "model.pt"
            torch.save(model.state_dict(), str(fallback_path))
            logger.debug(
                "model.save_pretrained not found; saved state_dict to %s",
                fallback_path,
            )

        # ------------------------------------------------------------------ #
        # 3. Save optimizer state.                                             #
        # ------------------------------------------------------------------ #
        # torch.save (pickle) is required here because optimizer state dicts
        # contain Python objects (step counters, hyperparameter dicts) that
        # safetensors cannot serialize.
        optimizer_path: Path = ckpt_path / "optimizer.pt"
        if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            torch.save(trainer.optimizer.state_dict(), str(optimizer_path))
            logger.debug("Optimizer state saved to %s", optimizer_path)
        else:
            logger.warning(
                "trainer.optimizer is None or missing; optimizer state not saved."
            )

        # ------------------------------------------------------------------ #
        # 4. Save scheduler state.                                             #
        # ------------------------------------------------------------------ #
        # The custom LRScheduler stores its position as current_step (int).
        # We save a lightweight dict rather than a full PyTorch scheduler
        # state_dict because LRScheduler is a custom class, not a
        # torch.optim.lr_scheduler subclass.
        scheduler_path: Path = ckpt_path / "scheduler.pt"
        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            scheduler_state: Dict[str, Any] = {
                "current_step": getattr(trainer.scheduler, "current_step", 0),
            }
            torch.save(scheduler_state, str(scheduler_path))
            logger.debug("Scheduler state saved to %s", scheduler_path)
        else:
            logger.warning(
                "trainer.scheduler is None or missing; scheduler state not saved."
            )

        # ------------------------------------------------------------------ #
        # 5. Write metadata JSON.                                              #
        # ------------------------------------------------------------------ #
        metadata: Dict[str, Any] = {
            "step": step,
            "stage": getattr(trainer, "current_stage", None),
            "global_step": getattr(trainer, "global_step", step),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        metadata_path: Path = ckpt_path / "metadata.json"
        with open(str(metadata_path), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.debug("Metadata written to %s: %s", metadata_path, metadata)

        # ------------------------------------------------------------------ #
        # 6. Enforce max_checkpoints by removing oldest checkpoints.          #
        # ------------------------------------------------------------------ #
        self.cleanup_old_checkpoints()

        logger.info("Checkpoint saved successfully: %s", ckpt_path)
        return str(ckpt_path)

    def load(
        self,
        path: str,
        model: Any,
        trainer: Any,
    ) -> int:
        """Restore training state from a checkpoint directory.

        Loads model weights, optimizer state, and scheduler state.
        Returns the global step stored in the checkpoint's metadata so
        the trainer can resume from the correct iteration.

        Args:
            path:    Path to the checkpoint directory to load from.
                     Example: ``"checkpoints/navil_2b/step_000005000"``.
            model:   A ``NaViLModel`` instance. Must expose either a
                     ``load_pretrained(path: str)`` method or a standard
                     ``load_state_dict`` method.
            trainer: A ``NaViLTrainer`` instance. Must expose:
                     - ``trainer.optimizer``: restored in-place if present.
                     - ``trainer.scheduler``: ``current_step`` restored if
                       present.
                     - ``trainer.current_stage``: restored from metadata.
                     - ``trainer.global_step``: restored from metadata.

        Returns:
            The integer global step recorded in the checkpoint's
            ``metadata.json``. The caller (``main.py``) uses this to
            skip already-completed training steps.

        Raises:
            FileNotFoundError: If ``path`` does not exist or is not a
                               directory.
            ValueError:        If ``metadata.json`` is missing or cannot
                               be parsed as valid JSON.

        Note:
            Model weights are loaded to CPU (``map_location="cpu"``) to
            avoid device placement issues. The ``accelerator.prepare``
            call in ``NaViLTrainer.setup_stage`` moves the model to the
            correct device after loading.
        """
        ckpt_path: Path = Path(path)

        # ------------------------------------------------------------------ #
        # 1. Validate the checkpoint directory.                                #
        # ------------------------------------------------------------------ #
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {ckpt_path}. "
                "Verify the path passed to --checkpoint."
            )
        if not ckpt_path.is_dir():
            raise FileNotFoundError(
                f"Checkpoint path is not a directory: {ckpt_path}. "
                "Expected a directory created by CheckpointManager.save()."
            )

        logger.info("Loading checkpoint from %s", ckpt_path)

        # ------------------------------------------------------------------ #
        # 2. Read metadata first (needed for step and stage restoration).      #
        # ------------------------------------------------------------------ #
        metadata_path: Path = ckpt_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(
                f"metadata.json not found in checkpoint directory: {ckpt_path}. "
                "The checkpoint may be incomplete or corrupted."
            )

        try:
            with open(str(metadata_path), "r", encoding="utf-8") as f:
                metadata: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse metadata.json in {ckpt_path}: {exc}. "
                "The checkpoint metadata may be corrupted."
            ) from exc

        restored_step: int = int(metadata.get("step", 0))
        restored_stage: Optional[str] = metadata.get("stage", None)
        restored_global_step: int = int(metadata.get("global_step", restored_step))

        logger.debug(
            "Checkpoint metadata: step=%d, stage=%s, global_step=%d",
            restored_step,
            restored_stage,
            restored_global_step,
        )

        # ------------------------------------------------------------------ #
        # 3. Load model weights.                                               #
        # ------------------------------------------------------------------ #
        model_dir: Path = ckpt_path / "model"
        model_pt_path: Path = ckpt_path / "model.pt"

        if model_dir.exists() and model_dir.is_dir():
            # Preferred path: NaViLModel.load_pretrained (safetensors)
            if hasattr(model, "load_pretrained") and callable(
                getattr(model, "load_pretrained")
            ):
                try:
                    model.load_pretrained(str(model_dir))
                    logger.debug(
                        "Model weights loaded via load_pretrained from %s", model_dir
                    )
                except Exception as exc:
                    logger.warning(
                        "model.load_pretrained failed (%s); attempting "
                        "load_state_dict fallback.",
                        exc,
                    )
                    self._load_state_dict_fallback(model, model_dir, model_pt_path)
            else:
                # load_pretrained not available — try load_state_dict
                self._load_state_dict_fallback(model, model_dir, model_pt_path)
        elif model_pt_path.exists():
            # Fallback: plain torch.save checkpoint
            state_dict: Dict[str, torch.Tensor] = torch.load(
                str(model_pt_path), map_location="cpu"
            )
            model.load_state_dict(state_dict)
            logger.debug(
                "Model weights loaded via state_dict from %s", model_pt_path
            )
        else:
            logger.warning(
                "No model weights found in checkpoint directory %s. "
                "Neither 'model/' subdirectory nor 'model.pt' exists. "
                "Model weights will not be restored.",
                ckpt_path,
            )

        # ------------------------------------------------------------------ #
        # 4. Load optimizer state.                                             #
        # ------------------------------------------------------------------ #
        optimizer_path: Path = ckpt_path / "optimizer.pt"
        if optimizer_path.exists():
            if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                optimizer_state: Dict[str, Any] = torch.load(
                    str(optimizer_path), map_location="cpu"
                )
                trainer.optimizer.load_state_dict(optimizer_state)
                logger.debug("Optimizer state restored from %s", optimizer_path)
            else:
                logger.warning(
                    "optimizer.pt found at %s but trainer.optimizer is None "
                    "or missing. Call trainer.setup_optimizer() before loading "
                    "a checkpoint to restore optimizer state.",
                    optimizer_path,
                )
        else:
            logger.warning(
                "optimizer.pt not found in %s; optimizer state not restored.",
                ckpt_path,
            )

        # ------------------------------------------------------------------ #
        # 5. Load scheduler state.                                             #
        # ------------------------------------------------------------------ #
        scheduler_path: Path = ckpt_path / "scheduler.pt"
        if scheduler_path.exists():
            if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
                scheduler_state_dict: Dict[str, Any] = torch.load(
                    str(scheduler_path), map_location="cpu"
                )
                restored_scheduler_step: int = int(
                    scheduler_state_dict.get("current_step", 0)
                )
                trainer.scheduler.current_step = restored_scheduler_step
                logger.debug(
                    "Scheduler current_step restored to %d from %s",
                    restored_scheduler_step,
                    scheduler_path,
                )
            else:
                logger.warning(
                    "scheduler.pt found at %s but trainer.scheduler is None "
                    "or missing. Scheduler state not restored.",
                    scheduler_path,
                )
        else:
            logger.warning(
                "scheduler.pt not found in %s; scheduler state not restored.",
                ckpt_path,
            )

        # ------------------------------------------------------------------ #
        # 6. Restore trainer metadata fields.                                  #
        # ------------------------------------------------------------------ #
        if restored_stage is not None and hasattr(trainer, "current_stage"):
            trainer.current_stage = restored_stage
            logger.debug("trainer.current_stage restored to '%s'", restored_stage)

        if hasattr(trainer, "global_step"):
            trainer.global_step = restored_global_step
            logger.debug(
                "trainer.global_step restored to %d", restored_global_step
            )

        logger.info(
            "Checkpoint loaded successfully from %s (step=%d, stage=%s)",
            ckpt_path,
            restored_step,
            restored_stage,
        )

        return restored_step

    def list_checkpoints(self) -> List[str]:
        """Return a sorted list of valid checkpoint directory paths.

        Scans ``output_dir`` for subdirectories matching the ``step_*``
        naming pattern. Only directories that contain a ``metadata.json``
        file are included (incomplete or corrupted checkpoints are
        excluded). Results are sorted in ascending step order (oldest
        first) using lexicographic sort on zero-padded step names.

        Returns:
            A list of absolute path strings for each valid checkpoint
            directory, sorted from oldest (lowest step) to newest
            (highest step). Returns an empty list if no valid checkpoints
            exist.

        Example::

            checkpoints = manager.list_checkpoints()
            # ["checkpoints/navil_2b/step_000005000",
            #  "checkpoints/navil_2b/step_000010000",
            #  "checkpoints/navil_2b/step_000015000"]
        """
        # glob("step_*") returns Path objects in filesystem order (not sorted).
        # sorted() on Path objects uses lexicographic comparison on the full
        # path string, which equals numeric order due to zero-padding.
        candidate_dirs: List[Path] = sorted(
            self.output_dir.glob("step_*")
        )

        valid_checkpoints: List[str] = []
        for candidate in candidate_dirs:
            # Only include directories (not stray files named step_*)
            if not candidate.is_dir():
                continue
            # Only include directories with a valid metadata.json
            # (guards against partial/interrupted saves)
            if not (candidate / "metadata.json").exists():
                logger.debug(
                    "Skipping candidate checkpoint %s: metadata.json missing.",
                    candidate,
                )
                continue
            valid_checkpoints.append(str(candidate))

        return valid_checkpoints

    def cleanup_old_checkpoints(self) -> None:
        """Remove the oldest checkpoints when the total count exceeds the limit.

        Called automatically at the end of every ``save`` call. Also safe
        to call manually.

        If ``max_checkpoints`` is ``0``, no cleanup is performed (keep all).
        If the current checkpoint count is within the limit, returns
        immediately without any filesystem operations.

        The oldest checkpoints (lowest step numbers, at the front of the
        sorted list from ``list_checkpoints``) are removed first.

        Returns:
            None.
        """
        # max_checkpoints=0 means "keep all" — skip cleanup entirely.
        if self.max_checkpoints <= 0:
            return

        existing_checkpoints: List[str] = self.list_checkpoints()
        num_existing: int = len(existing_checkpoints)

        if num_existing <= self.max_checkpoints:
            # Within the limit — nothing to remove.
            return

        # Determine how many to remove: oldest ones at the front of the list.
        num_to_remove: int = num_existing - self.max_checkpoints
        checkpoints_to_remove: List[str] = existing_checkpoints[:num_to_remove]

        for ckpt_path_str in checkpoints_to_remove:
            ckpt_path: Path = Path(ckpt_path_str)
            try:
                shutil.rmtree(str(ckpt_path))
                logger.info(
                    "Removed old checkpoint: %s (keeping %d most recent)",
                    ckpt_path,
                    self.max_checkpoints,
                )
            except OSError as exc:
                # Log but do not raise — a failed cleanup should not abort
                # training. The next save will attempt cleanup again.
                logger.warning(
                    "Failed to remove old checkpoint %s: %s",
                    ckpt_path,
                    exc,
                )

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _load_state_dict_fallback(
        self,
        model: Any,
        model_dir: Path,
        model_pt_path: Path,
    ) -> None:
        """Attempt to load model weights via ``load_state_dict`` as a fallback.

        Tries ``model_dir / "model.pt"`` first (if it exists inside the
        model subdirectory), then the top-level ``model_pt_path``.

        Args:
            model:         The model instance to load weights into.
            model_dir:     The ``model/`` subdirectory of the checkpoint.
            model_pt_path: The top-level ``model.pt`` fallback path.

        Returns:
            None.

        Raises:
            RuntimeError: If no loadable weight file is found.
        """
        # Check for model.pt inside the model/ subdirectory first.
        inner_model_pt: Path = model_dir / "model.pt"

        if inner_model_pt.exists():
            state_dict: Dict[str, torch.Tensor] = torch.load(
                str(inner_model_pt), map_location="cpu"
            )
            model.load_state_dict(state_dict)
            logger.debug(
                "Model weights loaded via state_dict from %s", inner_model_pt
            )
        elif model_pt_path.exists():
            state_dict = torch.load(str(model_pt_path), map_location="cpu")
            model.load_state_dict(state_dict)
            logger.debug(
                "Model weights loaded via state_dict from %s", model_pt_path
            )
        else:
            raise RuntimeError(
                f"Cannot load model weights: neither load_pretrained nor "
                f"state_dict fallback succeeded. Checked paths: "
                f"{inner_model_pt}, {model_pt_path}."
            )
