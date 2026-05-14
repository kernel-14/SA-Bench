## utils/checkpointing.py
"""Checkpointing utilities for saving and loading model training state.

This module provides the Checkpointer class for managing training checkpoints
and the load_pretrained_mdm function for loading pretrained MDM models from
local paths or HuggingFace Hub.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Checkpointer:
    """Manages saving and loading of model training checkpoints.

    Handles atomic writes to prevent corruption, maintains an in-memory
    metadata registry for fast best-checkpoint lookup, and persists metadata
    to a sidecar JSON file for cross-process resumability.

    Attributes:
        checkpoint_dir: Full path to the directory where checkpoints are stored.
        experiment_name: Name of the experiment, used in log messages.
    """

    _METADATA_FILENAME: str = "metadata.json"
    _CHECKPOINT_PREFIX: str = "checkpoint_"
    _CHECKPOINT_SUFFIX: str = ".pt"

    def __init__(self, save_dir: str, experiment_name: str) -> None:
        """Initializes the Checkpointer and scans for existing checkpoints.

        Args:
            save_dir: Base directory under which experiment checkpoints live.
            experiment_name: Name of the experiment; checkpoints are stored
                under ``save_dir/experiment_name/``.
        """
        self.checkpoint_dir: str = os.path.join(save_dir, experiment_name)
        self.experiment_name: str = experiment_name
        self._metadata: Dict[str, Dict[str, Any]] = {}

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._load_or_scan_metadata()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        optimizer: Any,
        epoch_or_iter: int,
        metrics: Dict[str, Any],
        scheduler: Optional[Any] = None,
    ) -> None:
        """Saves a training checkpoint atomically.

        Writes to a temporary file first, then renames to the final path to
        prevent corruption if the process is killed mid-write.

        Args:
            model: The model whose state dict will be saved.  If the model is
                wrapped in ``nn.DataParallel`` or ``DistributedDataParallel``,
                the inner ``module`` state dict is saved automatically.
            optimizer: The optimizer whose state dict will be saved.
            epoch_or_iter: Current epoch or iteration number, used in the
                checkpoint filename.
            metrics: Arbitrary metrics dict (e.g. ``{'val_loss': 2.34}``).
                Stored in the checkpoint and in the sidecar metadata file.
            scheduler: Optional LR scheduler.  If provided, its state dict is
                included in the checkpoint.
        """
        filename: str = self._make_filename(epoch_or_iter)
        final_path: str = os.path.join(self.checkpoint_dir, filename)
        tmp_path: str = final_path + ".tmp"

        # Unwrap DataParallel / DistributedDataParallel if necessary.
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_state: Dict[str, Any] = model.module.state_dict()
        else:
            model_state = model.state_dict()

        state: Dict[str, Any] = {
            "epoch_or_iter": epoch_or_iter,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, tmp_path)
        os.replace(tmp_path, final_path)  # atomic on POSIX; best-effort on Windows

        # Update in-memory registry and persist sidecar.
        self._metadata[filename] = {
            "epoch_or_iter": epoch_or_iter,
            "metrics": metrics,
        }
        self._persist_metadata()

        logger.info(
            "Saved checkpoint '%s' (epoch/iter=%d, metrics=%s)",
            final_path,
            epoch_or_iter,
            metrics,
        )

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Loads a checkpoint into the given model (and optionally optimizer).

        Always loads weights to CPU first to avoid device mismatches; the
        caller is responsible for moving the model to the target device
        afterwards.

        Args:
            path: Full path to the ``.pt`` checkpoint file.
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state into.
            scheduler: Optional LR scheduler to restore state into.

        Returns:
            A tuple ``(epoch_or_iter, metrics)`` extracted from the checkpoint.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint: Dict[str, Any] = torch.load(path, map_location="cpu")

        # Load model weights — fall back to non-strict on key mismatch.
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as exc:
            logger.warning(
                "Strict load failed (%s); retrying with strict=False.  "
                "Some weights may not be restored.",
                exc,
            )
            missing, unexpected = model.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            if missing:
                logger.warning("Missing keys: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys: %s", unexpected)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch_or_iter: int = checkpoint.get("epoch_or_iter", 0)
        metrics: Dict[str, Any] = checkpoint.get("metrics", {})

        logger.info(
            "Loaded checkpoint '%s' (epoch/iter=%d, metrics=%s)",
            path,
            epoch_or_iter,
            metrics,
        )
        return epoch_or_iter, metrics

    def get_best_checkpoint(
        self, metric: str = "val_loss", mode: str = "min"
    ) -> str:
        """Returns the path of the checkpoint with the best metric value.

        Args:
            metric: Key to look up in each checkpoint's ``metrics`` dict.
            mode: ``'min'`` if lower is better (e.g. loss), ``'max'`` if
                higher is better (e.g. accuracy).

        Returns:
            Full path to the best checkpoint file.

        Raises:
            ValueError: If no checkpoints are found, or none contain the
                requested metric key.
        """
        if not self._metadata:
            raise ValueError(
                f"No checkpoints found in '{self.checkpoint_dir}'."
            )

        best_path: Optional[str] = None
        best_value: Optional[float] = None

        for filename, meta in self._metadata.items():
            metrics: Dict[str, Any] = meta.get("metrics", {})
            if metric not in metrics:
                logger.debug(
                    "Checkpoint '%s' does not contain metric '%s'; skipping.",
                    filename,
                    metric,
                )
                continue

            value: float = float(metrics[metric])
            if best_value is None:
                best_value = value
                best_path = filename
            elif mode == "min" and value < best_value:
                best_value = value
                best_path = filename
            elif mode == "max" and value > best_value:
                best_value = value
                best_path = filename

        if best_path is None:
            raise ValueError(
                f"No checkpoint contains metric '{metric}' in "
                f"'{self.checkpoint_dir}'."
            )

        full_path: str = os.path.join(self.checkpoint_dir, best_path)
        logger.info(
            "Best checkpoint by '%s' (%s): '%s' (value=%.6f)",
            metric,
            mode,
            full_path,
            best_value,  # type: ignore[arg-type]
        )
        return full_path

    def list_checkpoints(self) -> List[str]:
        """Returns a sorted list of full paths to all checkpoint files.

        Sorting is lexicographic on the filename, which equals numeric order
        because filenames are zero-padded (``checkpoint_00001000.pt``).

        Returns:
            Sorted list of full checkpoint paths.  Empty list if none exist.
        """
        if not os.path.isdir(self.checkpoint_dir):
            return []

        filenames: List[str] = sorted(
            f
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith(self._CHECKPOINT_PREFIX)
            and f.endswith(self._CHECKPOINT_SUFFIX)
        )
        return [os.path.join(self.checkpoint_dir, f) for f in filenames]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_filename(self, epoch_or_iter: int) -> str:
        """Constructs a zero-padded checkpoint filename."""
        return f"{self._CHECKPOINT_PREFIX}{epoch_or_iter:08d}{self._CHECKPOINT_SUFFIX}"

    def _metadata_path(self) -> str:
        """Returns the full path to the sidecar metadata JSON file."""
        return os.path.join(self.checkpoint_dir, self._METADATA_FILENAME)

    def _persist_metadata(self) -> None:
        """Writes the in-memory metadata registry to the sidecar JSON file."""
        tmp_path: str = self._metadata_path() + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, indent=2, default=str)
        os.replace(tmp_path, self._metadata_path())

    def _load_or_scan_metadata(self) -> None:
        """Populates ``self._metadata`` from the sidecar file or by scanning.

        Prefers the sidecar JSON (fast).  Falls back to scanning ``.pt`` files
        and reading only their ``metrics`` and ``epoch_or_iter`` fields (slow
        but robust after a crash that left the sidecar stale).
        """
        sidecar: str = self._metadata_path()
        if os.path.isfile(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as fh:
                    self._metadata = json.load(fh)
                logger.debug(
                    "Loaded metadata for %d checkpoints from '%s'.",
                    len(self._metadata),
                    sidecar,
                )
                return
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Could not read sidecar metadata (%s); scanning .pt files.",
                    exc,
                )

        # Fallback: scan existing checkpoint files.
        self._metadata = {}
        for full_path in self.list_checkpoints():
            filename: str = os.path.basename(full_path)
            try:
                ckpt: Dict[str, Any] = torch.load(
                    full_path, map_location="cpu"
                )
                self._metadata[filename] = {
                    "epoch_or_iter": ckpt.get("epoch_or_iter", 0),
                    "metrics": ckpt.get("metrics", {}),
                }
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Could not read checkpoint '%s' (%s); skipping.",
                    full_path,
                    exc,
                )

        if self._metadata:
            self._persist_metadata()
            logger.debug(
                "Scanned and cached metadata for %d checkpoints.",
                len(self._metadata),
            )


# ---------------------------------------------------------------------------
# Standalone helper: load a pretrained MDM from a local path or HuggingFace
# ---------------------------------------------------------------------------


def load_pretrained_mdm(
    model_name_or_path: str,
    config: Any,  # ModelConfig from models/mdm_transformer.py
    device: str = "cpu",
) -> "nn.Module":
    """Loads a pretrained MDM from a local path or HuggingFace Hub.

    Handles two cases:

    1. **Local ``.pt`` file** — loads ``model_state_dict`` from the checkpoint
       into a freshly instantiated ``MDMTransformer(config)``.
    2. **Local directory** — looks for ``pytorch_model.bin``,
       ``model.safetensors``, or the lexicographically last
       ``checkpoint_*.pt`` file.
    3. **HuggingFace model ID** — uses
       ``transformers.AutoModel.from_pretrained`` for standard HuggingFace
       models (e.g. LLaDA-8B).  Returns the raw HuggingFace model; the caller
       is responsible for wrapping it if needed.

    Note:
        For LLaDA-8B (``GSAI-ML/LLaDA-8B-Instruct``), this function returns
        the raw HuggingFace model object, *not* an ``MDMTransformer`` instance.
        ``LLaDAEvaluator.load_model()`` calls this function and handles the
        returned object appropriately.

    Args:
        model_name_or_path: Local file/directory path or HuggingFace model ID.
        config: A ``ModelConfig``-like object with fields ``n_layers``,
            ``n_heads``, ``d_model``, ``d_ff``, ``vocab_size``,
            ``max_seq_len``, ``dropout``, ``pos_emb_type``,
            ``time_conditioned``.  Used only for local checkpoints.
        device: Target device string (e.g. ``'cpu'``, ``'cuda'``).

    Returns:
        A loaded model (``MDMTransformer`` for local checkpoints, or a
        HuggingFace ``PreTrainedModel`` for Hub IDs).

    Raises:
        FileNotFoundError: If a local path is given but no model file is found.
        ImportError: If ``transformers`` is not installed and a HuggingFace ID
            is provided.
    """
    # ------------------------------------------------------------------ #
    # Case 1 & 2: local path                                              #
    # ------------------------------------------------------------------ #
    if os.path.exists(model_name_or_path):
        return _load_local_mdm(model_name_or_path, config, device)

    # ------------------------------------------------------------------ #
    # Case 3: HuggingFace Hub model ID                                    #
    # ------------------------------------------------------------------ #
    return _load_hf_mdm(model_name_or_path, device)


# ------------------------------------------------------------------
# Private helpers for load_pretrained_mdm
# ------------------------------------------------------------------


def _load_local_mdm(
    path: str,
    config: Any,
    device: str,
) -> nn.Module:
    """Loads an MDMTransformer from a local file or directory.

    Args:
        path: Local ``.pt`` file or directory containing model weights.
        config: ModelConfig used to instantiate ``MDMTransformer``.
        device: Target device.

    Returns:
        An ``MDMTransformer`` with loaded weights, moved to ``device``.

    Raises:
        FileNotFoundError: If no recognisable model file is found.
    """
    # Lazy import to avoid circular dependency at module load time.
    from models.mdm_transformer import MDMTransformer  # noqa: PLC0415

    model: MDMTransformer = MDMTransformer(config)

    if os.path.isfile(path):
        # Direct .pt file — may be a raw state dict or a full checkpoint.
        raw: Any = torch.load(path, map_location="cpu")
        state_dict: Dict[str, Any] = (
            raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw
            else raw
        )
        _load_state_dict_flexible(model, state_dict)
        logger.info("Loaded pretrained MDM from file '%s'.", path)

    elif os.path.isdir(path):
        # Directory — try common filenames in priority order.
        candidates: List[str] = [
            os.path.join(path, "pytorch_model.bin"),
            os.path.join(path, "model.safetensors"),
        ]
        # Also consider the latest checkpoint_*.pt in the directory.
        pt_files: List[str] = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        )
        if pt_files:
            candidates.append(pt_files[-1])

        loaded: bool = False
        for candidate in candidates:
            if os.path.isfile(candidate):
                if candidate.endswith(".safetensors"):
                    state_dict = _load_safetensors(candidate)
                else:
                    raw = torch.load(candidate, map_location="cpu")
                    state_dict = (
                        raw["model_state_dict"]
                        if isinstance(raw, dict) and "model_state_dict" in raw
                        else raw
                    )
                _load_state_dict_flexible(model, state_dict)
                logger.info(
                    "Loaded pretrained MDM from directory file '%s'.", candidate
                )
                loaded = True
                break

        if not loaded:
            raise FileNotFoundError(
                f"No recognisable model file found in directory '{path}'. "
                "Expected pytorch_model.bin, model.safetensors, or checkpoint_*.pt."
            )
    else:
        raise FileNotFoundError(f"Local path does not exist: '{path}'.")

    model = model.to(device)
    model.eval()
    return model


def _load_hf_mdm(model_name_or_path: str, device: str) -> nn.Module:
    """Loads a model from the HuggingFace Hub.

    Args:
        model_name_or_path: HuggingFace model ID (e.g.
            ``'GSAI-ML/LLaDA-8B-Instruct'``).
        device: Target device string.

    Returns:
        A HuggingFace ``PreTrainedModel`` moved to ``device``.

    Raises:
        ImportError: If ``transformers`` is not installed.
    """
    try:
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required to load models from "
            "HuggingFace Hub.  Install it with: pip install transformers"
        ) from exc

    logger.info(
        "Loading pretrained model '%s' from HuggingFace Hub …",
        model_name_or_path,
    )

    # Attempt to load with trust_remote_code for custom model classes
    # (e.g. LLaDA uses a custom architecture).
    try:
        hf_model: nn.Module = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "AutoModel.from_pretrained failed (%s); retrying without "
            "trust_remote_code.",
            exc,
        )
        hf_model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
        )

    hf_model = hf_model.to(device)
    hf_model.eval()
    logger.info(
        "Loaded HuggingFace model '%s' on device '%s'.",
        model_name_or_path,
        device,
    )
    return hf_model


def _load_state_dict_flexible(model: nn.Module, state_dict: Dict[str, Any]) -> None:
    """Loads a state dict into a model, falling back to non-strict on mismatch.

    Args:
        model: Target model.
        state_dict: State dict to load.
    """
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        logger.warning(
            "Strict state dict load failed (%s); retrying with strict=False.",
            exc,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys in state dict: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in state dict: %s", unexpected)


def _load_safetensors(path: str) -> Dict[str, Any]:
    """Loads a safetensors file into a plain state dict.

    Args:
        path: Path to the ``.safetensors`` file.

    Returns:
        A dict mapping parameter names to tensors.

    Raises:
        ImportError: If the ``safetensors`` package is not installed.
    """
    try:
        from safetensors.torch import load_file  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'safetensors' package is required to load .safetensors files. "
            "Install it with: pip install safetensors"
        ) from exc

    return load_file(path, device="cpu")
