## utils/checkpoint.py
"""Checkpoint management for gated attention experiment training runs.

This module implements CheckpointManager, which handles saving and restoring
training state (model weights, optimizer state, step counter, and config) during
the long pretraining runs described in the paper "Gated Attention for Large
Language Models: Non-linearity, Sparsity, and Attention-Sink-Free".

Key responsibilities:
    - Saving checkpoints every save_interval steps (5000 per config.yaml)
    - Restoring training state for resumption or evaluation/analysis
    - Maintaining a 'latest' pointer to the most recent checkpoint
    - Handling distributed training (FSDP/DDP) by unwrapping model containers
    - Rank-0-only writes to avoid race conditions in multi-GPU training

Config values used (from config.yaml):
    logging.save_dir: 'outputs/checkpoints' — directory for checkpoint files
    training.save_interval: 5000 — steps between checkpoint saves
    hardware.distributed: 'fsdp' — distributed strategy (affects model unwrapping)
    hardware.num_gpus: 8 — number of GPUs (affects rank-0 guard logic)

Integration points:
    Trainer.train()              → save(model, optimizer, step, config)
    Trainer.load_checkpoint()    → load(model, optimizer, path)
    Main.run_evaluation()        → get_latest() → load(model, optimizer, path)
    Main.run_analysis()          → get_latest() → load(model, optimizer, path)
    Main.run_context_extension() → get_latest() → load(model, optimizer, path)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    Provides a clean interface for persisting and restoring complete training
    state, including model weights, optimizer state, step counter, and the
    experiment configuration. Handles distributed training contexts (FSDP/DDP)
    by unwrapping model containers before accessing state dicts.

    Checkpoint files are named `checkpoint_{step}.pt` and stored in save_dir.
    A `latest.pt` symlink (or `latest.txt` fallback on Windows) always points
    to the most recently saved checkpoint.

    Only rank 0 writes checkpoints in distributed training to avoid race
    conditions and redundant I/O. All ranks can read checkpoints.

    Attributes:
        save_dir: Path object pointing to the checkpoint directory.
            Created at construction time if it does not exist.
    """

    def __init__(self, save_dir: str = "outputs/checkpoints") -> None:
        """Initialize CheckpointManager and create the checkpoint directory.

        Args:
            save_dir: Directory path where checkpoint files will be stored.
                From config.yaml logging.save_dir = 'outputs/checkpoints'.
                Created with parents=True, exist_ok=True if it does not exist.
                Default 'outputs/checkpoints' matches config.yaml default.
        """
        self.save_dir: Path = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CheckpointManager initialized. save_dir='%s'", self.save_dir)

    # ---------------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Unwrap a distributed model container to access the underlying module.

        FSDP (FullyShardedDataParallel) and DDP (DistributedDataParallel) both
        wrap the user model in a container accessible via model.module. This
        method unwraps the container to get the actual model for state_dict access.

        The paper uses FSDP for the 15B MoE model (config hardware.distributed: 'fsdp')
        and DDP for dense models. Both expose the underlying model via .module.

        Args:
            model: Potentially wrapped model (FSDP, DDP, or plain nn.Module).

        Returns:
            The underlying nn.Module, unwrapped from any distributed container.
            If the model is not wrapped, returns it unchanged.
        """
        # Check for DDP or FSDP wrapper by looking for the .module attribute.
        # Both torch.nn.parallel.DistributedDataParallel and
        # torch.distributed.fsdp.FullyShardedDataParallel expose .module.
        if hasattr(model, "module"):
            return model.module  # type: ignore[return-value]
        return model

    @staticmethod
    def _is_rank_zero() -> bool:
        """Check if the current process is rank 0 in distributed training.

        Returns True if:
            - torch.distributed is not initialized (single-GPU or CPU training), OR
            - the current process rank is 0.

        Returns False only when distributed is initialized AND rank > 0.
        This ensures single-GPU runs always write checkpoints.

        Returns:
            True if this process should write checkpoints, False otherwise.
        """
        if not torch.distributed.is_available():
            return True
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _update_latest_pointer(self, checkpoint_filename: str) -> None:
        """Update the 'latest' pointer to the most recently saved checkpoint.

        Attempts to create a symlink `latest.pt -> checkpoint_{step}.pt`.
        Falls back to writing a `latest.txt` text file on systems where
        symlinks are unavailable (e.g., Windows without elevated permissions).

        Args:
            checkpoint_filename: The filename (not full path) of the checkpoint
                to point to, e.g., 'checkpoint_5000.pt'.
        """
        latest_symlink: Path = self.save_dir / "latest.pt"
        latest_txt: Path = self.save_dir / "latest.txt"

        # Attempt symlink approach (preferred on Linux/Mac HPC clusters)
        try:
            # Remove existing symlink or file if present
            if latest_symlink.exists() or latest_symlink.is_symlink():
                latest_symlink.unlink()

            # Create new symlink: latest.pt -> checkpoint_{step}.pt
            # Use relative symlink so the directory can be moved without breaking it
            latest_symlink.symlink_to(checkpoint_filename)
            logger.debug(
                "Updated latest.pt symlink → '%s'", checkpoint_filename
            )
            return

        except (OSError, NotImplementedError) as exc:
            # Symlink creation failed (e.g., Windows without elevated permissions,
            # or filesystem that doesn't support symlinks)
            logger.debug(
                "Symlink creation failed (%s). Falling back to latest.txt.", str(exc)
            )

        # Fallback: write checkpoint filename to latest.txt
        try:
            latest_txt.write_text(checkpoint_filename, encoding="utf-8")
            logger.debug(
                "Updated latest.txt with checkpoint filename '%s'", checkpoint_filename
            )
        except OSError as exc:
            # Even the text file fallback failed — log a warning but don't crash
            logger.warning(
                "Failed to update latest pointer (both symlink and txt): %s", str(exc)
            )

    def _move_optimizer_state_to_device(
        self,
        optimizer: Optimizer,
        device: Union[torch.device, str],
    ) -> None:
        """Move optimizer state tensors to the specified device.

        When loading a checkpoint with map_location='cpu', optimizer state
        tensors (momentum buffers, second moment estimates, etc.) are loaded
        to CPU. This method moves them to the target device so they are
        co-located with the model parameters for correct gradient computation.

        Args:
            optimizer: The optimizer whose state tensors should be moved.
            device: Target device (e.g., torch.device('cuda:0') or 'cuda').
        """
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def _infer_model_device(self, model: nn.Module) -> torch.device:
        """Infer the device of a model by checking its first parameter.

        Used after loading a checkpoint to determine where to move optimizer
        state tensors. Falls back to CPU if the model has no parameters.

        Args:
            model: The model whose device should be inferred.

        Returns:
            torch.device of the model's first parameter, or torch.device('cpu')
            if the model has no parameters.
        """
        try:
            first_param = next(iter(model.parameters()))
            return first_param.device
        except StopIteration:
            return torch.device("cpu")

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        step: int,
        config: Any,
    ) -> None:
        """Save a complete training checkpoint to disk.

        Persists model weights, optimizer state, step counter, and experiment
        configuration. Only rank 0 writes in distributed training to avoid
        race conditions and redundant I/O.

        The checkpoint dictionary contains:
            - 'model_state_dict': Unwrapped model state dict (handles FSDP/DDP)
            - 'optimizer_state_dict': Full optimizer state (momentum, etc.)
            - 'step': Current optimization step (int)
            - 'config': Experiment configuration as a plain dict

        After saving, updates the 'latest.pt' symlink (or 'latest.txt' fallback)
        to point to the newly saved checkpoint.

        Args:
            model: The model to checkpoint. May be wrapped in FSDP or DDP;
                the underlying module is automatically unwrapped for state_dict
                access. From Trainer: self.model (potentially FSDP/DDP wrapped).
            optimizer: The AdamW optimizer instance. From Trainer: self.optimizer.
                Saves the full optimizer state including momentum buffers and
                second moment estimates (required for exact training resumption).
            step: Current optimization step (0-based). From Trainer: self.global_step.
                Stored in the checkpoint so load() can return it for LR schedule
                resumption. Paper trains for up to 100k steps (MoE) or equivalent
                for dense models.
            config: Experiment configuration object with a to_dict() method.
                From Main: self.config (OmegaConf DictConfig or compatible).
                Stored as a plain dict for portability (no OmegaConf dependency
                when loading). Enables self-describing checkpoints — each saved
                model records exactly what hyperparameters produced it.

        Note:
            In distributed training (FSDP/DDP), this method is a no-op for
            all ranks except rank 0. All ranks should call this method; the
            rank guard is internal. This avoids the need for rank checks in
            the Trainer.

        Example:
            >>> ckpt_mgr = CheckpointManager(save_dir='outputs/checkpoints')
            >>> ckpt_mgr.save(model, optimizer, step=5000, config=config)
            # Creates: outputs/checkpoints/checkpoint_5000.pt
            # Updates: outputs/checkpoints/latest.pt → checkpoint_5000.pt
        """
        # Rank-0-only guard: only the primary process writes checkpoints
        # to avoid race conditions and redundant I/O in multi-GPU training.
        # Paper config hardware.distributed: 'fsdp', hardware.num_gpus: 8.
        if not self._is_rank_zero():
            logger.debug(
                "Rank %d: skipping checkpoint save at step %d (rank-0 only).",
                torch.distributed.get_rank(),
                step,
            )
            return

        # Unwrap distributed model container (FSDP/DDP → underlying nn.Module)
        unwrapped_model: nn.Module = self._unwrap_model(model)

        # Serialize config to a plain Python dict for portability.
        # config.to_dict() is defined in the Config class interface (design spec).
        # Falls back to vars(config) or str(config) if to_dict() is unavailable.
        config_dict: Dict[str, Any]
        if hasattr(config, "to_dict"):
            config_dict = config.to_dict()
        elif hasattr(config, "__dict__"):
            config_dict = dict(vars(config))
        else:
            # Last resort: store string representation
            config_dict = {"config_str": str(config)}

        # Build checkpoint dictionary
        checkpoint: Dict[str, Any] = {
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "config": config_dict,
        }

        # Determine checkpoint file path: save_dir/checkpoint_{step}.pt
        checkpoint_filename: str = f"checkpoint_{step}.pt"
        checkpoint_path: Path = self.save_dir / checkpoint_filename

        # Save checkpoint to disk using atomic write pattern:
        # Write to a temporary file first, then rename to avoid partial writes
        # that could corrupt the checkpoint if the process is interrupted.
        tmp_path: Path = self.save_dir / f"checkpoint_{step}.pt.tmp"
        try:
            torch.save(checkpoint, tmp_path)
            # Atomic rename: on POSIX systems, os.rename is atomic within the
            # same filesystem. This prevents reading a partially written checkpoint.
            os.replace(str(tmp_path), str(checkpoint_path))
        except Exception as exc:
            # Clean up temporary file if save failed
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(
                f"Failed to save checkpoint at step {step} to '{checkpoint_path}': {exc}"
            ) from exc

        logger.info(
            "Checkpoint saved: '%s' (step=%d, model_params=%d)",
            checkpoint_path,
            step,
            sum(p.numel() for p in unwrapped_model.parameters()),
        )

        # Update latest pointer (symlink or txt fallback)
        self._update_latest_pointer(checkpoint_filename)

    def load(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        path: str,
    ) -> int:
        """Load a checkpoint and restore model and optimizer state.

        Restores model weights and optimizer state from a checkpoint file.
        Returns the step number so the Trainer can resume the LR schedule
        (WarmupCosineScheduler.current_step) and token counter from the
        correct position.

        Loading strategy:
            1. Load checkpoint to CPU (map_location='cpu') for device-agnostic
               loading — safe regardless of how many GPUs were used during saving.
            2. Unwrap distributed model container (FSDP/DDP → underlying module).
            3. Restore model state dict with strict=True (catches architecture mismatches).
            4. Restore optimizer state dict.
            5. Move optimizer state tensors to the model's device (they were loaded
               to CPU in step 1).
            6. Return the step number from the checkpoint.

        Args:
            model: The model to restore weights into. May be wrapped in FSDP or DDP;
                the underlying module is automatically unwrapped. The model should
                already be on the correct device before calling load() — this method
                does not move the model itself, only the optimizer state tensors.
            optimizer: The optimizer to restore state into. The optimizer should
                already be configured with the same parameter groups as when the
                checkpoint was saved (same model, same weight decay groups).
            path: Full path to the checkpoint file to load.
                Typically obtained from get_latest() or list_checkpoints().
                Example: 'outputs/checkpoints/checkpoint_5000.pt'

        Returns:
            The optimization step number stored in the checkpoint (int).
            The Trainer uses this to set self.global_step and resume the
            WarmupCosineScheduler from the correct LR position.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist at path.
            KeyError: If the checkpoint dict is missing required keys
                ('model_state_dict', 'optimizer_state_dict', 'step').
            RuntimeError: If model state dict loading fails (e.g., architecture
                mismatch between the checkpoint and the current model).

        Example:
            >>> ckpt_mgr = CheckpointManager(save_dir='outputs/checkpoints')
            >>> step = ckpt_mgr.load(model, optimizer, 'outputs/checkpoints/checkpoint_5000.pt')
            >>> print(f"Resumed from step {step}")
            Resumed from step 5000
        """
        checkpoint_path: Path = Path(path)

        # Validate that the checkpoint file exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: '{checkpoint_path}'. "
                f"Available checkpoints: {self.list_checkpoints()}"
            )

        logger.info("Loading checkpoint from '%s'...", checkpoint_path)

        # Load checkpoint to CPU first for device-agnostic loading.
        # This is safe regardless of how many GPUs were used during saving
        # (paper trains on 8 GPUs, but evaluation/analysis may use fewer).
        # weights_only=False is needed to load the config dict (non-tensor data).
        try:
            checkpoint: Dict[str, Any] = torch.load(
                str(checkpoint_path),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint from '{checkpoint_path}': {exc}"
            ) from exc

        # Validate required keys are present in the checkpoint dict
        required_keys = {"model_state_dict", "optimizer_state_dict", "step"}
        missing_keys = required_keys - set(checkpoint.keys())
        if missing_keys:
            raise KeyError(
                f"Checkpoint at '{checkpoint_path}' is missing required keys: "
                f"{missing_keys}. Available keys: {set(checkpoint.keys())}"
            )

        # Unwrap distributed model container (FSDP/DDP → underlying nn.Module)
        unwrapped_model: nn.Module = self._unwrap_model(model)

        # Restore model state dict with strict=True to catch architecture mismatches.
        # strict=True (default) raises RuntimeError if keys don't match exactly,
        # which is the desired behavior — a mismatch indicates a config error.
        try:
            unwrapped_model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load model state dict from '{checkpoint_path}'. "
                f"This likely indicates an architecture mismatch between the "
                f"checkpoint and the current model configuration. "
                f"Original error: {exc}"
            ) from exc

        logger.info(
            "Model state dict restored (strict=True). "
            "Model params: %d",
            sum(p.numel() for p in unwrapped_model.parameters()),
        )

        # Restore optimizer state dict.
        # The optimizer state is loaded to CPU (from map_location='cpu' above).
        # We move it to the model's device after loading.
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load optimizer state dict from '{checkpoint_path}': {exc}"
            ) from exc

        # Move optimizer state tensors to the model's device.
        # After torch.load with map_location='cpu', all optimizer state tensors
        # (momentum buffers, second moment estimates for AdamW) are on CPU.
        # They must be co-located with the model parameters for correct training.
        model_device: torch.device = self._infer_model_device(unwrapped_model)
        if model_device != torch.device("cpu"):
            self._move_optimizer_state_to_device(optimizer, model_device)
            logger.debug(
                "Moved optimizer state tensors to device '%s'.", model_device
            )

        # Extract and return the step number
        step: int = int(checkpoint["step"])

        logger.info(
            "Checkpoint loaded successfully from '%s'. "
            "Resuming from step %d.",
            checkpoint_path,
            step,
        )

        return step

    def list_checkpoints(self) -> List[str]:
        """Return a sorted list of all checkpoint file paths in save_dir.

        Finds all files matching the pattern `checkpoint_*.pt` in save_dir,
        excluding `latest.pt` (which is a symlink or alias, not a numbered
        checkpoint). Sorts by step number in ascending order.

        Returns:
            List of checkpoint file paths as strings, sorted by step number
            in ascending order. Returns an empty list if no checkpoints exist.

        Example:
            >>> ckpt_mgr = CheckpointManager(save_dir='outputs/checkpoints')
            >>> ckpt_mgr.list_checkpoints()
            ['outputs/checkpoints/checkpoint_5000.pt',
             'outputs/checkpoints/checkpoint_10000.pt',
             'outputs/checkpoints/checkpoint_15000.pt']
        """
        checkpoint_paths: List[Path] = []

        # Glob for all checkpoint_*.pt files in save_dir
        for path in self.save_dir.glob("checkpoint_*.pt"):
            # Exclude latest.pt (symlink or alias) and any .tmp files
            if path.name == "latest.pt" or path.suffix == ".tmp":
                continue
            # Validate that the filename matches the expected pattern
            # (checkpoint_{integer}.pt) to exclude malformed files
            stem: str = path.stem  # e.g., 'checkpoint_5000'
            parts: List[str] = stem.split("_")
            if len(parts) == 2 and parts[0] == "checkpoint":
                try:
                    int(parts[1])  # Validate that the step is an integer
                    checkpoint_paths.append(path)
                except ValueError:
                    # Filename doesn't match expected pattern — skip
                    logger.debug(
                        "Skipping non-standard checkpoint file: '%s'", path.name
                    )

        # Sort by step number (ascending)
        checkpoint_paths.sort(
            key=lambda p: int(p.stem.split("_")[1])
        )

        return [str(p) for p in checkpoint_paths]

    def get_latest(self) -> str:
        """Return the path to the most recently saved checkpoint.

        Lookup strategy (in order):
            1. Check if `save_dir/latest.pt` exists as a valid file or symlink.
               If it resolves to an existing file, return its resolved path.
            2. Check if `save_dir/latest.txt` exists (Windows fallback).
               If it contains a valid checkpoint filename, return the full path.
            3. Fall back to list_checkpoints() and return the last element
               (highest step number).

        Returns:
            Full path to the most recent checkpoint file as a string.

        Raises:
            FileNotFoundError: If no checkpoints exist in save_dir via any
                of the three lookup strategies.

        Example:
            >>> ckpt_mgr = CheckpointManager(save_dir='outputs/checkpoints')
            >>> ckpt_mgr.get_latest()
            'outputs/checkpoints/checkpoint_15000.pt'
        """
        # Strategy 1: Check latest.pt symlink or file
        latest_symlink: Path = self.save_dir / "latest.pt"
        if latest_symlink.exists() or latest_symlink.is_symlink():
            try:
                # Resolve the symlink to get the actual checkpoint path
                resolved: Path = latest_symlink.resolve()
                if resolved.exists():
                    logger.debug(
                        "get_latest(): resolved latest.pt → '%s'", resolved
                    )
                    return str(resolved)
                else:
                    logger.warning(
                        "latest.pt symlink exists but target '%s' does not. "
                        "Falling back to list_checkpoints().",
                        resolved,
                    )
            except OSError as exc:
                logger.warning(
                    "Failed to resolve latest.pt symlink: %s. "
                    "Falling back to latest.txt.",
                    str(exc),
                )

        # Strategy 2: Check latest.txt (Windows fallback)
        latest_txt: Path = self.save_dir / "latest.txt"
        if latest_txt.exists():
            try:
                checkpoint_filename: str = latest_txt.read_text(encoding="utf-8").strip()
                if checkpoint_filename:
                    # Construct full path from the stored filename
                    candidate: Path = self.save_dir / checkpoint_filename
                    if candidate.exists():
                        logger.debug(
                            "get_latest(): found via latest.txt → '%s'", candidate
                        )
                        return str(candidate)
                    else:
                        logger.warning(
                            "latest.txt points to '%s' which does not exist. "
                            "Falling back to list_checkpoints().",
                            candidate,
                        )
            except OSError as exc:
                logger.warning(
                    "Failed to read latest.txt: %s. "
                    "Falling back to list_checkpoints().",
                    str(exc),
                )

        # Strategy 3: Fall back to list_checkpoints() — return highest step
        all_checkpoints: List[str] = self.list_checkpoints()
        if not all_checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found in '{self.save_dir}'. "
                f"Ensure training has been run and checkpoints have been saved "
                f"(config.yaml training.save_interval: 5000)."
            )

        latest: str = all_checkpoints[-1]  # Last element = highest step number
        logger.debug(
            "get_latest(): found via list_checkpoints() → '%s'", latest
        )
        return latest

    def __repr__(self) -> str:
        """Return a human-readable string representation of the CheckpointManager.

        Returns:
            String summarizing the save directory and available checkpoints.
        """
        checkpoints: List[str] = self.list_checkpoints()
        return (
            f"CheckpointManager("
            f"save_dir='{self.save_dir}', "
            f"num_checkpoints={len(checkpoints)}"
            f")"
        )
