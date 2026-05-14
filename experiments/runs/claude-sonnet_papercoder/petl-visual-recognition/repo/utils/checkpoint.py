## utils/checkpoint.py
"""Checkpoint utility for the PEFT Visual Recognition reproduction study.

This module provides the Checkpoint class, which handles two distinct types
of persistence:

1. Model checkpoints — saved during training to preserve the best-performing
   model state for each (method, dataset, hyperparameter config) combination.
   Only trainable parameters are saved to keep files small and to correctly
   separate pretrained knowledge from learned PEFT adaptation.

2. Prediction artifacts — saved after evaluation to enable the diversity
   analysis (Section 4 of the paper) and ensemble evaluation, which require
   predictions from all 14 PEFT methods on the same test set.

Typical usage:
    checkpoint = Checkpoint(save_dir="./outputs/vtab/dtd/lora")

    # During training (called by Trainer):
    checkpoint.save(model, optimizer, epoch=50, val_acc=72.1, config=cfg)

    # For evaluation / WiSE interpolation:
    model, meta = checkpoint.load("./outputs/vtab/dtd/lora/best_checkpoint.pth", model)

    # After all methods evaluated (called by main.py):
    checkpoint.save_predictions(
        {"lora_preds": pred_tensor, "lora_logits": logit_tensor},
        path="./outputs/vtab/dtd/predictions.pkl",
    )
    predictions = checkpoint.load_predictions("./outputs/vtab/dtd/predictions.pkl")
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# Module-level logger — messages flow through the root logger configured by
# utils/logger.py (which sets up console + file handlers).
_logger: logging.Logger = logging.getLogger(__name__)


class Checkpoint:
    """Persistence layer for model states and prediction artifacts.

    Handles saving and loading of:
    - Best model checkpoints (trainable parameters only) during training.
    - Optimizer states for potential training resumption.
    - Prediction tensors and logits for post-hoc diversity and ensemble analysis.

    Attributes:
        save_dir: Root directory where all checkpoint files are written.
    """

    def __init__(self, save_dir: str) -> None:
        """Initialises the checkpoint manager and creates the save directory.

        Args:
            save_dir: Root directory for checkpoint files. Created if it does
                not already exist. The caller is responsible for structuring
                this path to avoid collisions across experiments, e.g.:
                ``outputs/vtab/caltech101/lora/``.
        """
        self.save_dir: str = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        _logger.info("Checkpoint manager initialised. Save directory: %s", self.save_dir)

    # ------------------------------------------------------------------
    # Model checkpoint methods
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_acc: float,
        config: Any,
    ) -> None:
        """Saves the current best model state to disk.

        Only trainable parameters are persisted. This keeps checkpoint files
        small (e.g., ~0.1 MB for BitFit vs ~330 MB for full ViT-B/16) and
        correctly separates pretrained backbone weights from learned PEFT
        adaptation weights. For full fine-tuning, all parameters have
        ``requires_grad=True``, so the full state is saved automatically.

        The checkpoint is always written to ``best_checkpoint.pth`` within
        ``self.save_dir``, overwriting any previous best. The caller
        (``Trainer``) is responsible for only calling this method when
        validation accuracy improves.

        Args:
            model: The PEFTModel (or any nn.Module) being trained.
            optimizer: The AdamW optimizer used for training.
            epoch: Current epoch number (0-indexed or 1-indexed, consistent
                with the caller's convention).
            val_acc: Validation accuracy at this checkpoint (0–100 scale).
            config: Experiment configuration object. Must support either
                ``config.to_dict()``, ``vars(config)``, or be a plain dict.
        """
        # ------------------------------------------------------------------
        # Extract only trainable parameters.
        # strict=False in load() relies on this being a subset of the full
        # model state_dict.
        # ------------------------------------------------------------------
        trainable_state: Dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # ------------------------------------------------------------------
        # Serialise config to a plain dict to avoid pickle dependency on the
        # Config class definition when loading in a different environment.
        # ------------------------------------------------------------------
        config_dict: Dict[str, Any] = self._serialise_config(config)

        # ------------------------------------------------------------------
        # Build the checkpoint payload.
        # ------------------------------------------------------------------
        checkpoint: Dict[str, Any] = {
            "trainable_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": float(val_acc),
            "config": config_dict,
            "method": config_dict.get("peft_method", "unknown"),
            "dataset": config_dict.get("dataset", "unknown"),
        }

        filepath: str = os.path.join(self.save_dir, "best_checkpoint.pth")

        try:
            torch.save(checkpoint, filepath)
            _logger.info(
                "Checkpoint saved — epoch: %d, val_acc: %.4f, path: %s",
                epoch,
                val_acc,
                filepath,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.error(
                "Failed to save checkpoint to %s: %s", filepath, exc
            )

    def load(
        self,
        path: str,
        model: nn.Module,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Restores a saved model state from a checkpoint file.

        Uses ``strict=False`` when loading the state dict because the
        checkpoint contains only trainable parameters (a subset of the full
        model state). Frozen backbone parameters retain their pretrained
        values loaded by ``ViTWrapper`` or ``CLIPWrapper``; only PEFT-specific
        parameters are updated from the checkpoint.

        This method is also used by ``training/wise.py`` to obtain both the
        pretrained state (before fine-tuning) and the finetuned state for
        WiSE weight interpolation.

        Args:
            path: Absolute or relative path to the ``.pth`` checkpoint file.
            model: The model instance whose trainable parameters will be
                updated in-place. The model should already have the correct
                PEFT architecture applied (i.e., the same PEFT method and
                hyperparameters as when the checkpoint was saved).

        Returns:
            A tuple ``(model, meta_dict)`` where:
            - ``model`` is the input model with trainable parameters restored
              (modified in-place and also returned for convenience).
            - ``meta_dict`` is a dict containing ``epoch``, ``val_acc``,
              ``config``, ``method``, and ``dataset`` from the checkpoint.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If the checkpoint file does not contain the expected
                ``trainable_state_dict`` key.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {path}"
            )

        # Always load to CPU first to avoid device mismatch; the caller moves
        # the model to the target device after loading.
        checkpoint: Dict[str, Any] = torch.load(path, map_location="cpu")

        if "trainable_state_dict" not in checkpoint:
            raise KeyError(
                f"Checkpoint at '{path}' does not contain 'trainable_state_dict'. "
                f"Available keys: {list(checkpoint.keys())}"
            )

        # ------------------------------------------------------------------
        # Restore trainable parameters.
        # strict=False: only keys present in the checkpoint are updated;
        # frozen params retain their pretrained values.
        # ------------------------------------------------------------------
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["trainable_state_dict"], strict=False
        )

        if unexpected_keys:
            _logger.warning(
                "Unexpected keys in checkpoint (ignored): %s", unexpected_keys
            )
        if missing_keys:
            _logger.debug(
                "Keys not in checkpoint (retained from pretrained): %d params",
                len(missing_keys),
            )

        # ------------------------------------------------------------------
        # Build metadata dict (everything except the large tensor dicts).
        # ------------------------------------------------------------------
        meta_dict: Dict[str, Any] = {
            key: value
            for key, value in checkpoint.items()
            if key not in ("trainable_state_dict", "optimizer_state_dict")
        }

        _logger.info(
            "Checkpoint loaded — epoch: %s, val_acc: %s, path: %s",
            meta_dict.get("epoch", "N/A"),
            meta_dict.get("val_acc", "N/A"),
            path,
        )

        return model, meta_dict

    def load_optimizer_state(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        """Restores optimizer state from a checkpoint for training resumption.

        This is a convenience method separate from ``load()`` because optimizer
        state restoration is only needed when resuming training, not during
        evaluation or WiSE interpolation.

        Args:
            path: Path to the ``.pth`` checkpoint file.
            optimizer: The optimizer instance to restore state into.

        Returns:
            The optimizer with restored state (modified in-place and returned).

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {path}"
            )

        checkpoint: Dict[str, Any] = torch.load(path, map_location="cpu")

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _logger.info("Optimizer state restored from %s", path)
        else:
            _logger.warning(
                "No optimizer state found in checkpoint at %s; "
                "optimizer state not restored.",
                path,
            )

        return optimizer

    # ------------------------------------------------------------------
    # Prediction artifact methods
    # ------------------------------------------------------------------

    def save_predictions(
        self,
        predictions: Dict[str, Any],
        path: str,
    ) -> None:
        """Persists prediction tensors and logits for post-hoc analysis.

        Saves a dict mapping method names (or artifact names) to tensors.
        Used by ``main.py`` after evaluating all PEFT methods on a test set
        to enable the diversity analysis (Section 4) and ensemble evaluation.

        The dict may contain any combination of:
        - ``{method}_preds``: (N,) int tensor of predicted class indices.
        - ``{method}_logits``: (N, C) float tensor of raw logits.
        - ``{method}_confs``: (N,) float tensor of max softmax probabilities.
        - ``labels``: (N,) int tensor of ground-truth labels.

        All tensors are moved to CPU before saving.

        Args:
            predictions: Dict mapping string keys to tensors or other
                serialisable values.
            path: Destination file path. Relative paths are used as-is;
                parent directories are created if they do not exist.
        """
        # Move all tensors to CPU before serialisation.
        cpu_predictions: Dict[str, Any] = {
            key: value.cpu() if isinstance(value, torch.Tensor) else value
            for key, value in predictions.items()
        }

        # Ensure parent directory exists.
        parent_dir: str = os.path.dirname(os.path.abspath(path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        try:
            torch.save(cpu_predictions, path)
            num_keys: int = len(cpu_predictions)
            _logger.info(
                "Predictions saved (%d artifact(s)) to %s", num_keys, path
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.error(
                "Failed to save predictions to %s: %s", path, exc
            )

    def load_predictions(self, path: str) -> Dict[str, Any]:
        """Loads saved prediction artifacts for diversity and ensemble analysis.

        Args:
            path: Path to the predictions file saved by ``save_predictions()``.

        Returns:
            Dict mapping string keys to tensors or other values, as originally
            passed to ``save_predictions()``. All tensors are on CPU.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Predictions file not found: {path}"
            )

        predictions: Any = torch.load(path, map_location="cpu")

        if not isinstance(predictions, dict):
            _logger.warning(
                "Loaded predictions from %s is not a dict (type: %s). "
                "Downstream analysis may fail.",
                path,
                type(predictions).__name__,
            )

        num_keys: int = len(predictions) if isinstance(predictions, dict) else 0
        _logger.info(
            "Predictions loaded (%d artifact(s)) from %s", num_keys, path
        )

        return predictions  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Returns the path to the best checkpoint file if it exists.

        Returns:
            Absolute path to ``best_checkpoint.pth`` within ``self.save_dir``,
            or ``None`` if no checkpoint has been saved yet.
        """
        path: str = os.path.join(self.save_dir, "best_checkpoint.pth")
        return path if os.path.exists(path) else None

    def checkpoint_exists(self) -> bool:
        """Checks whether a best checkpoint file exists in ``self.save_dir``.

        Returns:
            True if ``best_checkpoint.pth`` exists, False otherwise.
        """
        return self.get_best_checkpoint_path() is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_config(config: Any) -> Dict[str, Any]:
        """Converts a config object to a plain Python dict for serialisation.

        Supports Config dataclasses (with ``to_dict``), omegaconf DictConfig,
        and plain dicts. Falls back to ``vars()`` and then ``str()`` for
        unknown types.

        Args:
            config: Experiment configuration object or dict.

        Returns:
            A plain Python dict representation of the config.
        """
        if isinstance(config, dict):
            return config

        if hasattr(config, "to_dict") and callable(config.to_dict):
            return config.to_dict()

        # Try omegaconf DictConfig.
        try:
            from omegaconf import OmegaConf  # type: ignore

            return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
        except (ImportError, Exception):
            pass

        # Fall back to vars() for dataclasses / simple objects.
        try:
            return vars(config)
        except TypeError:
            return {"config_repr": str(config)}
