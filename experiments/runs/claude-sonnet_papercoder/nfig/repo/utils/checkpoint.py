## utils/checkpoint.py
"""Checkpoint management utilities for the NFIG framework.

Provides the CheckpointManager class for saving and loading model checkpoints
during FR-VAE and NFIG Transformer training. Maintains a JSON registry of all
saved checkpoints with associated metrics, enabling metric-based best checkpoint
retrieval without scanning file contents.

Used by:
    - FRVAETrainer: saves/loads FRVAE and DINODiscriminator checkpoints
    - NFIGTrainer: saves NFIG Transformer checkpoints; loads frozen FR-VAE
    - main.py evaluation path: loads best checkpoints for both models

Config values used (config.yaml training section):
    checkpoint_dir:          'checkpoints'
    keep_last_n_checkpoints: 5
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Module-level logger.
logger: logging.Logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoint saving, loading, and metric-based retrieval.

    Maintains a JSON registry file (`checkpoint_registry.json`) inside the
    checkpoint directory that tracks all saved checkpoints and their associated
    metrics. This enables `get_best_checkpoint()` to find the optimal checkpoint
    for a given metric without loading any model weights.

    Pruning is applied per `name` prefix: when more than `keep_last_n`
    checkpoints with the same name exist, the oldest (by epoch) is deleted.
    This prevents FR-VAE checkpoints from crowding out NFIG Transformer
    checkpoints and vice versa.

    Registry structure (checkpoint_registry.json):
        {
          "frvae_epoch_0010": {
            "path": "checkpoints/frvae_epoch_0010.pt",
            "epoch": 10,
            "metrics": {"rfid": 1.20, "loss": 0.45},
            "name": "frvae",
            "timestamp": "2024-01-01T00:00:00.000000"
          },
          ...
        }

    Attributes:
        checkpoint_dir: Directory where checkpoint files and the registry are stored.
        keep_last_n: Maximum number of checkpoints to retain per name prefix.
            Older checkpoints are deleted when this limit is exceeded.
            From config.training.keep_last_n_checkpoints = 5.
        _registry_path: Full path to the JSON registry file.
        _registry: In-memory copy of the checkpoint registry dict.
    """

    # Registry filename within checkpoint_dir.
    _REGISTRY_FILENAME: str = "checkpoint_registry.json"

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5,
    ) -> None:
        """Initialize the CheckpointManager.

        Creates the checkpoint directory if it does not exist, and loads
        the existing registry from disk if one is present. If the registry
        file is corrupted (invalid JSON), logs a warning and starts fresh.

        Args:
            checkpoint_dir: Path to the directory where checkpoints and the
                registry file will be stored.
                From config.training.checkpoint_dir = 'checkpoints'.
            keep_last_n: Maximum number of checkpoints to retain per name
                prefix before pruning the oldest.
                From config.training.keep_last_n_checkpoints = 5.
                Must be a positive integer.

        Raises:
            ValueError: If keep_last_n is not a positive integer.
        """
        if keep_last_n < 1:
            raise ValueError(
                f"keep_last_n must be a positive integer, got {keep_last_n}."
            )

        self.checkpoint_dir: str = checkpoint_dir
        self.keep_last_n: int = keep_last_n

        # Create checkpoint directory if it does not exist.
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Path to the JSON registry file.
        self._registry_path: str = os.path.join(
            checkpoint_dir, self._REGISTRY_FILENAME
        )

        # Load existing registry from disk, or start with an empty dict.
        self._registry: Dict[str, Dict] = self._load_registry()

        logger.info(
            "CheckpointManager initialized. "
            "checkpoint_dir='%s', keep_last_n=%d, "
            "existing_checkpoints=%d",
            checkpoint_dir,
            keep_last_n,
            len(self._registry),
        )

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
        epoch: int,
        metrics: Dict,
        name: str,
    ) -> str:
        """Save a model checkpoint with associated training state and metrics.

        Constructs a checkpoint file containing the model state dict, optional
        optimizer state dict, epoch number, and metrics dict. Updates the
        in-memory registry and persists it to disk. Prunes old checkpoints
        for the same `name` prefix if the count exceeds `keep_last_n`.

        Args:
            model: The nn.Module whose state_dict will be saved.
                Can be a raw model or a DDP-wrapped model (the trainer is
                responsible for passing the unwrapped model via model.module).
            optimizer: Optional optimizer whose state_dict will be saved.
                Pass None when saving a model for inference-only use (e.g.,
                saving a frozen FR-VAE that will be loaded without resuming
                optimizer state). The checkpoint will store None for the
                optimizer state in this case.
            epoch: Current training epoch index (0-based).
                Used in the checkpoint filename and registry for sorting.
            metrics: Dictionary of metric values to store alongside the
                checkpoint. Examples:
                  - FR-VAE: {"rfid": 0.85, "val_rec_loss": 0.12}
                  - NFIG Transformer: {"loss": 2.1, "val_loss": 2.3, "lr": 8e-5}
                Stored in the registry for `get_best_checkpoint()` lookups.
            name: Checkpoint name prefix used in the filename and for
                per-prefix pruning. Examples: "frvae", "disc", "nfig_transformer".
                Must be a non-empty string without path separators.

        Returns:
            Full path to the saved checkpoint file as a string.
            Example: "checkpoints/frvae_epoch_0010.pt"

        Raises:
            ValueError: If name is empty or contains path separators.
            OSError: If the checkpoint file cannot be written to disk.
        """
        if not name:
            raise ValueError("name must be a non-empty string.")
        if os.sep in name or "/" in name:
            raise ValueError(
                f"name must not contain path separators. Got: '{name}'. "
                "Use a simple identifier like 'frvae' or 'nfig_transformer'."
            )

        # Construct checkpoint filename with zero-padded epoch for lexicographic sorting.
        filename: str = f"{name}_epoch_{epoch:04d}.pt"
        checkpoint_path: str = os.path.join(self.checkpoint_dir, filename)

        # Build the state dict to save.
        state: Dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "metrics": metrics,
            "name": name,
        }

        # Save checkpoint to disk using atomic write (temp file + rename).
        # This prevents corruption if the process is killed mid-write.
        tmp_path: str = checkpoint_path + ".tmp"
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, checkpoint_path)
        except Exception:
            # Clean up temp file if save failed.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # Build registry key: unique identifier for this checkpoint.
        # Format: "{name}_epoch_{epoch:04d}" — matches the filename stem.
        registry_key: str = f"{name}_epoch_{epoch:04d}"

        # Update in-memory registry with checkpoint metadata.
        self._registry[registry_key] = {
            "path": checkpoint_path,
            "epoch": epoch,
            "metrics": dict(metrics),  # Defensive copy
            "name": name,
            "timestamp": datetime.now().isoformat(),
        }

        # Persist registry to disk atomically.
        self._save_registry()

        logger.info(
            "Checkpoint saved: '%s' (epoch=%d, metrics=%s)",
            checkpoint_path,
            epoch,
            {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()},
        )

        # Prune old checkpoints for this name prefix.
        self._prune_old_checkpoints(name=name)

        return checkpoint_path

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer],
    ) -> Tuple[int, Dict]:
        """Load a model checkpoint and restore model (and optionally optimizer) state.

        Always loads the checkpoint to CPU first to avoid GPU memory issues
        when loading on a different device than the checkpoint was saved on.
        The caller is responsible for moving the model to the correct device
        after loading (e.g., model.to(device)).

        Does NOT call model.eval() or model.train() — that is the caller's
        responsibility. This keeps the method focused on state restoration only.

        Args:
            path: Full path to the checkpoint file to load.
                Must be an existing .pt file saved by CheckpointManager.save().
            model: The nn.Module to load weights into.
                Must have the same architecture as when the checkpoint was saved.
                The model's state_dict is updated in-place.
            optimizer: Optional optimizer to restore state into.
                Pass None when loading a model for inference-only use (e.g.,
                loading a frozen FR-VAE in NFIGTrainer.__init__).
                If not None and the checkpoint contains optimizer state,
                the optimizer's state_dict is updated in-place.

        Returns:
            Tuple of:
                - epoch: The epoch number at which the checkpoint was saved (int).
                  Callers typically use this as `start_epoch + 1` to resume
                  training from the next epoch.
                - metrics: Dictionary of metrics stored with the checkpoint.
                  May be empty ({}) if no metrics were saved.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist at `path`.
            RuntimeError: If the model architecture does not match the saved
                state_dict (propagated from model.load_state_dict with strict=True).
            RuntimeError: If the checkpoint file is corrupted or not a valid
                PyTorch checkpoint (propagated from torch.load).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint file not found: '{path}'. "
                "Ensure the path is correct and the file has not been deleted. "
                f"Available checkpoints: {self.list_checkpoints()}"
            )

        # Load checkpoint to CPU first to avoid device placement issues.
        # The caller moves the model to the correct device after loading.
        state: Dict = torch.load(path, map_location="cpu")

        # Restore model weights with strict=True to catch architecture mismatches.
        model.load_state_dict(state["model_state_dict"], strict=True)

        # Restore optimizer state if both optimizer and saved state are available.
        if optimizer is not None and state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        elif optimizer is not None and state.get("optimizer_state_dict") is None:
            logger.warning(
                "Optimizer provided but checkpoint '%s' contains no optimizer state. "
                "Optimizer state will not be restored.",
                path,
            )

        # Extract epoch and metrics from the checkpoint.
        epoch: int = state.get("epoch", 0)
        metrics: Dict = state.get("metrics", {})

        logger.info(
            "Checkpoint loaded: '%s' (epoch=%d, metrics=%s)",
            path,
            epoch,
            {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()},
        )

        return epoch, metrics

    def get_best_checkpoint(
        self,
        metric: str,
        mode: str = "min",
    ) -> str:
        """Find and return the path of the best checkpoint for a given metric.

        Scans the in-memory registry for all checkpoints that contain the
        specified metric, then selects the one with the best value according
        to `mode`. Only checkpoints whose files actually exist on disk are
        considered.

        Common usage patterns:
            - get_best_checkpoint("rfid", "min") → best FR-VAE by reconstruction FID
            - get_best_checkpoint("gfid", "min") → best NFIG Transformer by generation FID
            - get_best_checkpoint("loss", "min") → best checkpoint by training loss
            - get_best_checkpoint("is", "max")   → best checkpoint by Inception Score

        Args:
            metric: The metric key to look up in each checkpoint's metrics dict.
                Must be a key that was included in the `metrics` argument to
                `save()`. Examples: "rfid", "gfid", "loss", "val_loss", "is".
            mode: Selection criterion. Must be either:
                - "min": Select the checkpoint with the lowest metric value.
                  Use for FID, loss, and other metrics where lower is better.
                - "max": Select the checkpoint with the highest metric value.
                  Use for IS, Precision, Recall, and other metrics where
                  higher is better.

        Returns:
            Full path to the best checkpoint file as a string.
            The file is guaranteed to exist on disk at the time of this call.

        Raises:
            ValueError: If mode is not "min" or "max".
            ValueError: If no checkpoints in the registry contain the specified
                metric, or if all matching checkpoints have been deleted from disk.
        """
        if mode not in ("min", "max"):
            raise ValueError(
                f"mode must be 'min' or 'max', got '{mode}'. "
                "Use 'min' for metrics where lower is better (FID, loss) "
                "and 'max' for metrics where higher is better (IS, Precision)."
            )

        # Reload registry from disk to ensure freshness across process restarts.
        self._registry = self._load_registry()

        # Filter registry entries that:
        # 1. Contain the requested metric in their metrics dict.
        # 2. Have a checkpoint file that actually exists on disk.
        valid_entries: Dict[str, Dict] = {}
        for key, entry in self._registry.items():
            entry_metrics: Dict = entry.get("metrics", {})
            entry_path: str = entry.get("path", "")

            if metric not in entry_metrics:
                continue  # This checkpoint does not have the requested metric.

            if not os.path.exists(entry_path):
                logger.warning(
                    "Checkpoint file '%s' (key='%s') is in the registry but "
                    "does not exist on disk. Skipping for best checkpoint selection.",
                    entry_path,
                    key,
                )
                continue

            valid_entries[key] = entry

        if not valid_entries:
            raise ValueError(
                f"No valid checkpoints found with metric '{metric}'. "
                f"Available checkpoints in registry: {list(self._registry.keys())}. "
                f"Metrics available across all checkpoints: "
                f"{sorted(set(k for e in self._registry.values() for k in e.get('metrics', {}).keys()))}. "
                "Ensure the metric name matches exactly what was passed to save()."
            )

        # Select the best checkpoint based on mode.
        if mode == "min":
            best_key: str = min(
                valid_entries,
                key=lambda k: valid_entries[k]["metrics"][metric],
            )
        else:  # mode == "max"
            best_key = max(
                valid_entries,
                key=lambda k: valid_entries[k]["metrics"][metric],
            )

        best_path: str = valid_entries[best_key]["path"]
        best_value: float = valid_entries[best_key]["metrics"][metric]

        logger.info(
            "Best checkpoint for metric='%s' (mode='%s'): "
            "path='%s', value=%.6f (key='%s')",
            metric,
            mode,
            best_path,
            best_value,
            best_key,
        )

        return best_path

    def list_checkpoints(self) -> List[str]:
        """Return a sorted list of all checkpoint file paths in the registry.

        Scans the in-memory registry and returns paths for checkpoints that
        actually exist on disk, sorted by epoch number in ascending order.
        Stale registry entries (files that have been manually deleted) are
        excluded from the result but remain in the registry.

        Returns:
            List of checkpoint file path strings, sorted by epoch (ascending).
            Returns an empty list if no checkpoints exist.

        Example:
            >>> manager = CheckpointManager("checkpoints")
            >>> manager.list_checkpoints()
            [
                "checkpoints/frvae_epoch_0010.pt",
                "checkpoints/frvae_epoch_0020.pt",
                "checkpoints/nfig_transformer_epoch_0050.pt",
            ]
        """
        # Reload registry from disk to ensure freshness.
        self._registry = self._load_registry()

        # Collect valid (existing) checkpoint entries with their epoch numbers.
        valid_entries: List[Tuple[int, str]] = []  # (epoch, path)

        for key, entry in self._registry.items():
            entry_path: str = entry.get("path", "")
            entry_epoch: int = entry.get("epoch", 0)

            if not os.path.exists(entry_path):
                # Log stale entries but do not remove them from the registry here.
                # Stale entries are cleaned up during save() via pruning.
                logger.debug(
                    "Stale registry entry: key='%s', path='%s' does not exist on disk.",
                    key,
                    entry_path,
                )
                continue

            valid_entries.append((entry_epoch, entry_path))

        # Sort by epoch number (ascending) for chronological ordering.
        valid_entries.sort(key=lambda x: x[0])

        # Return only the paths (epoch used only for sorting).
        return [path for _epoch, path in valid_entries]

    # ---------------------------------------------------------------------- #
    # Private helper methods
    # ---------------------------------------------------------------------- #

    def _load_registry(self) -> Dict[str, Dict]:
        """Load the checkpoint registry from disk.

        Reads the JSON registry file from `self._registry_path`. If the file
        does not exist, returns an empty dict. If the file is corrupted
        (invalid JSON), logs a warning and returns an empty dict rather than
        crashing — this allows training to continue with a fresh registry.

        Returns:
            Dictionary mapping checkpoint keys to their metadata dicts.
            Returns an empty dict if the registry file does not exist or
            is corrupted.
        """
        if not os.path.exists(self._registry_path):
            return {}

        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                registry: Dict[str, Dict] = json.load(f)

            # Validate that the loaded data is a dict (basic sanity check).
            if not isinstance(registry, dict):
                logger.warning(
                    "Registry file '%s' contains invalid data (expected dict, "
                    "got %s). Starting with empty registry.",
                    self._registry_path,
                    type(registry).__name__,
                )
                return {}

            return registry

        except json.JSONDecodeError as exc:
            logger.warning(
                "Registry file '%s' is corrupted (JSON decode error: %s). "
                "Starting with empty registry. "
                "Existing checkpoints on disk are not affected.",
                self._registry_path,
                exc,
            )
            return {}

        except OSError as exc:
            logger.warning(
                "Could not read registry file '%s' (OS error: %s). "
                "Starting with empty registry.",
                self._registry_path,
                exc,
            )
            return {}

    def _save_registry(self) -> None:
        """Persist the in-memory registry to disk atomically.

        Uses a temp-file-then-rename strategy to prevent registry corruption
        if the process is killed mid-write. The temp file is written first,
        then atomically renamed to the final registry path.

        Side effects:
            Writes `self._registry` to `self._registry_path` as JSON.
            Logs a warning if the write fails (does not raise — a failed
            registry write should not crash training).
        """
        tmp_path: str = self._registry_path + ".tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2, ensure_ascii=False)

            # Atomic rename: replaces the registry file in a single OS operation.
            os.replace(tmp_path, self._registry_path)

        except OSError as exc:
            logger.warning(
                "Failed to save checkpoint registry to '%s' (OS error: %s). "
                "The registry may be out of sync with saved checkpoints. "
                "Training will continue but get_best_checkpoint() may be unreliable.",
                self._registry_path,
                exc,
            )
            # Clean up temp file if it was created.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass  # Best-effort cleanup; ignore secondary errors.

    def _prune_old_checkpoints(self, name: str) -> None:
        """Delete the oldest checkpoints for a given name prefix.

        Finds all registry entries with the given `name` prefix, sorts them
        by epoch (ascending), and deletes the oldest ones if the count exceeds
        `self.keep_last_n`. Both the checkpoint file and the registry entry
        are removed.

        Checkpoints marked as "best" (metrics dict contains "is_best": True)
        are exempt from pruning to preserve the best-performing checkpoint
        regardless of age.

        Args:
            name: The checkpoint name prefix to prune (e.g., "frvae",
                "nfig_transformer"). Only checkpoints with this exact name
                are considered for pruning.
        """
        # Find all registry entries for this name prefix.
        name_entries: List[Tuple[int, str]] = []  # (epoch, registry_key)

        for key, entry in self._registry.items():
            if entry.get("name") == name:
                epoch: int = entry.get("epoch", 0)
                name_entries.append((epoch, key))

        # Sort by epoch ascending (oldest first).
        name_entries.sort(key=lambda x: x[0])

        # Determine how many to prune.
        num_to_prune: int = len(name_entries) - self.keep_last_n

        if num_to_prune <= 0:
            return  # No pruning needed.

        # Prune the oldest checkpoints, skipping "best" checkpoints.
        pruned_count: int = 0
        for epoch_val, key in name_entries:
            if pruned_count >= num_to_prune:
                break

            entry: Dict = self._registry.get(key, {})

            # Skip checkpoints explicitly marked as best.
            if entry.get("metrics", {}).get("is_best", False):
                logger.debug(
                    "Skipping pruning of best checkpoint: key='%s', epoch=%d",
                    key,
                    epoch_val,
                )
                continue

            # Delete the checkpoint file from disk.
            checkpoint_path: str = entry.get("path", "")
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                    logger.info(
                        "Pruned old checkpoint: '%s' (epoch=%d, name='%s')",
                        checkpoint_path,
                        epoch_val,
                        name,
                    )
                except OSError as exc:
                    logger.warning(
                        "Failed to delete checkpoint file '%s' during pruning "
                        "(OS error: %s). Registry entry will still be removed.",
                        checkpoint_path,
                        exc,
                    )
            elif checkpoint_path:
                logger.debug(
                    "Checkpoint file '%s' already missing during pruning. "
                    "Removing stale registry entry.",
                    checkpoint_path,
                )

            # Remove the registry entry.
            del self._registry[key]
            pruned_count += 1

        # Persist the updated registry after pruning.
        if pruned_count > 0:
            self._save_registry()
            logger.debug(
                "Pruned %d old checkpoint(s) for name='%s'. "
                "Remaining: %d",
                pruned_count,
                name,
                len([k for k, e in self._registry.items() if e.get("name") == name]),
            )

    def __repr__(self) -> str:
        """Return a human-readable string representation of the CheckpointManager.

        Returns:
            String describing the checkpoint directory, keep_last_n setting,
            and current number of registered checkpoints.
        """
        return (
            f"CheckpointManager("
            f"checkpoint_dir='{self.checkpoint_dir}', "
            f"keep_last_n={self.keep_last_n}, "
            f"num_checkpoints={len(self._registry)})"
        )
