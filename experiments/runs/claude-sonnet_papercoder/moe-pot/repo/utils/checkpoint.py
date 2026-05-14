# utils/checkpoint.py
"""Checkpoint saving and loading utility for the MoE-POT training pipeline.

Provides a Checkpointer class that handles persisting and restoring model
training state for both training continuity (full state) and fine-tuning
initialization (weights only).
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


class Checkpointer:
    """Manages saving and loading of model checkpoints.

    Supports two use cases:
      1. Full training state (model + optimizer + epoch + metrics) for
         resuming interrupted pre-training runs.
      2. Model weights only for initializing fine-tuning from a
         pre-trained checkpoint.

    Automatically tracks the best validation L2RE seen during a training
    run and writes a separate ``best.pt`` whenever a new best is achieved.

    Attributes:
        checkpoint_dir: Path object pointing to the checkpoint directory.
        best_val_l2re: Best validation L2RE seen so far in the current
            training run. Initialized to ``float('inf')``.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        """Initializes the Checkpointer and creates the checkpoint directory.

        Args:
            checkpoint_dir: Path to the directory where checkpoint files
                will be written. Created (including parents) if it does
                not already exist.
        """
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_l2re: float = float("inf")

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> None:
        """Saves the full training state to a checkpoint file.

        Persists model weights, optimizer state, current epoch, and
        metrics. Handles ``DistributedDataParallel``-wrapped models by
        extracting the underlying module's state dict.

        If ``metrics`` contains ``'val_l2re'`` and its value is lower
        than the best seen so far, an additional copy is written to
        ``best.pt`` in the same directory.

        Args:
            model: The model to checkpoint. May be a raw ``nn.Module``
                or a ``DistributedDataParallel``-wrapped module.
            optimizer: The optimizer whose state should be saved.
            epoch: Current epoch number (0-indexed or 1-indexed,
                consistent with the caller's convention).
            metrics: Dictionary of metric values to store alongside the
                checkpoint. Should contain at least ``'val_l2re'`` for
                best-model tracking.
            filename: Name of the checkpoint file. Defaults to
                ``'epoch_{epoch}.pt'`` if ``None``.
        """
        # Resolve the target filename.
        if filename is None:
            filename = f"epoch_{epoch}.pt"

        save_path: Path = self.checkpoint_dir / filename

        # Extract state dict, unwrapping DDP if necessary.
        state_dict: dict = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        torch.save(checkpoint, save_path)

        # Check whether this is a new best model.
        val_l2re: Optional[float] = metrics.get("val_l2re", None)
        if val_l2re is not None and val_l2re < self.best_val_l2re:
            self.best_val_l2re = val_l2re
            best_path: Path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> Tuple[int, Dict[str, Any]]:
        """Loads a full training state from a checkpoint file.

        Restores model weights, optimizer state, and returns the saved
        epoch and metrics so the caller can resume training from the
        correct position.

        Uses ``map_location='cpu'`` universally to ensure portability
        across machines with different GPU configurations. PyTorch moves
        optimizer tensors to the correct device automatically on the
        first optimizer step.

        Args:
            path: Path to the checkpoint file to load.
            model: Model whose weights will be restored. May be DDP-
                wrapped.
            optimizer: Optimizer whose state will be restored.

        Returns:
            A tuple ``(epoch, metrics)`` where ``epoch`` is the integer
            epoch at which the checkpoint was saved and ``metrics`` is
            the dictionary of metric values stored in the checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist at
                ``path``.
            KeyError: If the checkpoint file is missing expected keys
                (``'epoch'``, ``'model_state_dict'``,
                ``'optimizer_state_dict'``, ``'metrics'``).
        """
        checkpoint_path: Path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )

        checkpoint: Dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu"
        )

        # Validate required keys before attempting to restore state.
        required_keys = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "metrics",
        }
        missing_keys = required_keys - set(checkpoint.keys())
        if missing_keys:
            raise KeyError(
                f"Checkpoint at '{path}' is missing required keys: "
                f"{missing_keys}. Found keys: {set(checkpoint.keys())}"
            )

        # Restore model weights, unwrapping DDP if necessary.
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state.
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch: int = checkpoint["epoch"]
        metrics: Dict[str, Any] = checkpoint["metrics"]

        return epoch, metrics

    def load_model_only(self, path: str, model: nn.Module) -> None:
        """Loads only model weights from a checkpoint file.

        Used for fine-tuning initialization: restores pre-trained weights
        without touching optimizer state. The caller creates a fresh
        optimizer and scheduler starting from epoch 0.

        Uses ``strict=True`` (PyTorch default) so any architecture
        mismatch between the checkpoint and the current model raises an
        error immediately, ensuring reproducibility.

        Args:
            path: Path to the checkpoint file to load.
            model: Model whose weights will be restored in-place. May be
                DDP-wrapped.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist at
                ``path``.
            KeyError: If the checkpoint file does not contain
                ``'model_state_dict'``.
            RuntimeError: If the model architecture does not match the
                checkpoint (strict loading).
        """
        checkpoint_path: Path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )

        checkpoint: Dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu"
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint at '{path}' does not contain 'model_state_dict'. "
                f"Found keys: {set(checkpoint.keys())}"
            )

        # Restore model weights only, unwrapping DDP if necessary.
        if hasattr(model, "module"):
            model.module.load_state_dict(
                checkpoint["model_state_dict"], strict=True
            )
        else:
            model.load_state_dict(
                checkpoint["model_state_dict"], strict=True
            )
