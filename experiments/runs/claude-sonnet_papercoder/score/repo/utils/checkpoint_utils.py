## utils/checkpoint_utils.py
"""Checkpoint utilities for SCoRe: Self-Correction via Reinforcement Learning.

This module implements CheckpointUtils, the stateless utility class for saving
and loading model checkpoints during the two-stage SCoRe training pipeline.

It implements the paper's checkpoint selection strategy from Section 6:
    "For all RL runs, we selected checkpoints with the highest training reward,
    although a small held-out validation set of problems can also be used."

The class serves as the bridge between Stage I and Stage II training:
    1. Stage I trainer saves checkpoints (including a "best" checkpoint).
    2. Stage II trainer loads the best Stage I checkpoint as its initialization.
    3. Final evaluation loads the best Stage II checkpoint.

Checkpoint directory structure:
    output_dir/
    ├── stage1_step_500/
    │   ├── model/          ← ModelWrapper.save_checkpoint() writes here
    │   └── metrics.json    ← CheckpointUtils writes this
    ├── stage1_best/
    │   ├── model/
    │   └── metrics.json
    └── stage2_final/
        ├── model/
        └── metrics.json

The metrics.json file stores the step number and all training metrics logged
at that checkpoint. get_best_checkpoint() reads these files to rank checkpoints
by any specified metric (default: 'train_reward_t2' per config.yaml).

Typical usage:
    from utils.checkpoint_utils import CheckpointUtils
    from models.model_wrapper import ModelWrapper

    ckpt = CheckpointUtils()

    # Save a checkpoint during training
    ckpt.save(model, path="outputs/stage1_step_500", step=500, metrics=metrics_dict)

    # Find the best checkpoint by training reward
    best_path = ckpt.get_best_checkpoint("outputs/", metric="train_reward_t2")

    # Load the best checkpoint
    loaded_model = ckpt.load(best_path)
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Filename for the metrics JSON file stored alongside each checkpoint.
_METRICS_FILENAME: str = "metrics.json"

# Subdirectory name within each checkpoint directory where the model weights
# and tokenizer are stored. ModelWrapper.save_checkpoint() writes here.
_MODEL_SUBDIR: str = "model"

# Sentinel value used when a checkpoint's metrics.json is missing the
# requested metric key. Using -inf ensures such checkpoints are never
# selected as "best" by get_best_checkpoint().
_MISSING_METRIC_SENTINEL: float = float("-inf")

# Default metric name for checkpoint selection.
# The config.yaml specifies checkpoint_metric: "train_reward_t2", but the
# design spec shows default='train_reward'. We use 'train_reward' as the
# default and callers pass the config value explicitly.
_DEFAULT_CHECKPOINT_METRIC: str = "train_reward"


class CheckpointUtils:
    """Stateless utility class for model checkpoint save/load/selection.

    All methods are instance methods (not static) per the design specification,
    but the class holds no mutable state — it can be instantiated once and
    reused throughout the training pipeline.

    The class is agnostic to LoRA vs. full fine-tuning — all model weight
    serialization is delegated to ModelWrapper.save_checkpoint() and
    ModelWrapper.load_checkpoint(), which handle PEFT-aware serialization.

    Attributes:
        None. This class is stateless.
    """

    def __init__(self) -> None:
        """Initialize CheckpointUtils.

        No configuration is required at construction time. All methods
        receive their required parameters explicitly, making the class
        fully reusable across different training stages and runs.
        """
        logger.debug("CheckpointUtils initialized.")

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def save(
        self,
        model: "ModelWrapper",
        path: str,
        step: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Save a model checkpoint and its associated metrics to disk.

        Creates the checkpoint directory structure:
            path/
            ├── model/          ← model weights + tokenizer (via ModelWrapper)
            └── metrics.json    ← step number + all metrics

        The metrics.json file is what get_best_checkpoint() reads to rank
        checkpoints. The 'train_reward_t2' key (from config.yaml:
        evaluation.checkpoint_metric) must be present in the metrics dict
        for checkpoint selection to work correctly. Callers (SCoReStage1Trainer,
        SCoReStage2Trainer) are responsible for including this key.

        Args:
            model: The ModelWrapper instance to save. Its save_checkpoint()
                method is called with the model subdirectory path. This
                handles both LoRA adapter weights and full model weights
                transparently.
            path: Full path to the checkpoint directory (e.g.,
                "outputs/stage1_step_500"). Created if it does not exist.
                The caller constructs this path — CheckpointUtils does not
                impose a naming convention.
            step: The global training step number at the time of saving.
                Stored in metrics.json under the key 'step'. Used for
                logging and ordering checkpoints chronologically.
            metrics: Dict of training metrics to persist alongside the
                checkpoint. Expected keys include 'train_reward_t2',
                'train_reward_t1', 'loss', 'mean_kl_t1', 'mean_kl_t2',
                'delta_t1_t2'. All keys are preserved verbatim in
                metrics.json. The dict is not modified.

        Raises:
            OSError: If the checkpoint directory cannot be created or if
                the metrics.json file cannot be written. Let propagate —
                checkpoint failures should not be silently swallowed.
            Exception: Any exception from model.save_checkpoint() is
                propagated. Checkpoint failures must be visible to the
                caller to prevent silent data loss.
        """
        # ------------------------------------------------------------------
        # Step 1: Create the checkpoint directory and model subdirectory.
        # exist_ok=True is safe for both first-time creation and re-saves
        # (e.g., overwriting the "best" checkpoint).
        # ------------------------------------------------------------------
        model_dir: str = os.path.join(path, _MODEL_SUBDIR)
        os.makedirs(model_dir, exist_ok=True)

        logger.info(
            "CheckpointUtils.save(): Saving checkpoint to '%s' (step=%d).",
            path,
            step,
        )

        # ------------------------------------------------------------------
        # Step 2: Save model weights and tokenizer via ModelWrapper.
        # ModelWrapper.save_checkpoint() handles:
        #   - LoRA: saves only adapter weights (adapter_config.json + weights)
        #   - Full fine-tuning: saves full model state dict
        #   - Tokenizer: always saved for self-contained checkpoints
        # ------------------------------------------------------------------
        model.save_checkpoint(model_dir)

        logger.debug(
            "CheckpointUtils.save(): Model weights saved to '%s'.", model_dir
        )

        # ------------------------------------------------------------------
        # Step 3: Build the metrics metadata dict.
        # Include 'step' as a top-level key alongside all provided metrics.
        # We create a new dict (not mutating the input) to avoid side effects.
        # ------------------------------------------------------------------
        metadata: Dict[str, Any] = {"step": int(step)}

        # Merge all provided metrics into the metadata dict.
        # Values are serialized as-is; json.dump handles float, int, str, bool.
        # Non-serializable values (e.g., torch.Tensor) are converted to float.
        for key, value in metrics.items():
            serialized_value: Any = self._serialize_metric_value(value)
            metadata[key] = serialized_value

        # ------------------------------------------------------------------
        # Step 4: Write metrics.json to the checkpoint directory.
        # ------------------------------------------------------------------
        metrics_path: str = os.path.join(path, _METRICS_FILENAME)
        try:
            with open(metrics_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise OSError(
                f"CheckpointUtils.save(): Failed to write metrics.json to "
                f"'{metrics_path}': {exc}. "
                "Ensure the output directory is writable."
            ) from exc

        logger.info(
            "CheckpointUtils.save(): Checkpoint saved successfully to '%s'. "
            "Metrics: %s.",
            path,
            {k: v for k, v in metadata.items() if k != "step"},
        )

    def load(self, path: str) -> "ModelWrapper":
        """Load a model checkpoint from disk and return a ModelWrapper.

        Reads the checkpoint directory structure:
            path/
            ├── model/          ← model weights + tokenizer
            └── metrics.json    ← step number + metrics (for validation)

        The model subdirectory is passed to ModelWrapper.load_checkpoint(),
        which handles both LoRA adapter checkpoints (detected via
        adapter_config.json) and full model checkpoints transparently.

        Since ModelWrapper.__init__ requires a Config object, this method
        reads the config from the model subdirectory (saved by
        ModelWrapper.save_checkpoint() via save_pretrained, which writes
        the model config). A new ModelWrapper is instantiated using the
        Config reconstructed from the checkpoint.

        Implementation note: ModelWrapper.save_checkpoint() calls
        model.save_pretrained(path) which saves the HuggingFace model config
        (config.json) but NOT the SCoRe Config dataclass. To make load()
        self-contained, we save the SCoRe Config as 'score_config.json'
        in save() and read it back here.

        Args:
            path: Full path to the checkpoint directory (e.g.,
                "outputs/stage1_best"). Must contain a 'model/' subdirectory
                and a 'metrics.json' file.

        Returns:
            A ModelWrapper instance with weights loaded from the checkpoint.
            The model is in eval mode if it was saved as a reference model,
            or in train mode if it was saved as a policy model. Callers
            should call model.model.train() or model.model.eval() as needed
            after loading.

        Raises:
            FileNotFoundError: If path does not exist, or if the model
                subdirectory does not exist within path.
            RuntimeError: If the ModelWrapper cannot be reconstructed from
                the checkpoint (e.g., missing config, incompatible weights).
        """
        # ------------------------------------------------------------------
        # Step 1: Validate that the checkpoint directory exists.
        # ------------------------------------------------------------------
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"CheckpointUtils.load(): Checkpoint directory '{path}' does "
                "not exist. Ensure the path is correct and the checkpoint "
                "was saved successfully."
            )

        model_dir: str = os.path.join(path, _MODEL_SUBDIR)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"CheckpointUtils.load(): Model subdirectory '{model_dir}' "
                "does not exist within checkpoint directory '{path}'. "
                "The checkpoint may be corrupted or was saved with a "
                "different directory structure."
            )

        # ------------------------------------------------------------------
        # Step 2: Read metrics.json for validation and logging (optional).
        # We don't fail if metrics.json is missing — the model weights are
        # the critical artifact.
        # ------------------------------------------------------------------
        metrics_path: str = os.path.join(path, _METRICS_FILENAME)
        loaded_metrics: Dict[str, Any] = {}
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as fh:
                    loaded_metrics = json.load(fh)
                logger.info(
                    "CheckpointUtils.load(): Loading checkpoint from '%s' "
                    "(step=%s, metrics=%s).",
                    path,
                    loaded_metrics.get("step", "unknown"),
                    {
                        k: v
                        for k, v in loaded_metrics.items()
                        if k != "step"
                    },
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "CheckpointUtils.load(): Could not read metrics.json "
                    "from '%s': %s. Proceeding with model loading.",
                    metrics_path,
                    exc,
                )
        else:
            logger.warning(
                "CheckpointUtils.load(): metrics.json not found at '%s'. "
                "Proceeding with model loading without metrics validation.",
                metrics_path,
            )

        # ------------------------------------------------------------------
        # Step 3: Reconstruct the SCoRe Config from the saved score_config.json.
        # This file is written by save() alongside the checkpoint.
        # If it's missing (e.g., checkpoint from an older version), we fall
        # back to reading from the metrics dict or raising a clear error.
        # ------------------------------------------------------------------
        score_config_path: str = os.path.join(path, "score_config.json")

        # Import here to avoid circular imports at module level.
        # config.py has no dependencies on utils/, so this is safe.
        from config import Config

        score_config: Optional[Config] = None

        if os.path.isfile(score_config_path):
            try:
                with open(score_config_path, "r", encoding="utf-8") as fh:
                    config_dict: Dict[str, Any] = json.load(fh)
                score_config = Config.from_dict(config_dict)
                logger.debug(
                    "CheckpointUtils.load(): Reconstructed SCoRe Config "
                    "from '%s'.",
                    score_config_path,
                )
            except Exception as exc:
                logger.warning(
                    "CheckpointUtils.load(): Failed to reconstruct Config "
                    "from '%s': %s. Will attempt to load model directly.",
                    score_config_path,
                    exc,
                )

        # ------------------------------------------------------------------
        # Step 4: Instantiate ModelWrapper and load checkpoint weights.
        # ------------------------------------------------------------------
        # Import ModelWrapper here to avoid circular imports at module level.
        # models/model_wrapper.py imports config.py, not utils/checkpoint_utils.py.
        from models.model_wrapper import ModelWrapper

        if score_config is not None:
            # Full reconstruction: instantiate ModelWrapper with the saved
            # config, then load the checkpoint weights on top.
            try:
                policy_model: ModelWrapper = ModelWrapper(
                    config=score_config,
                    freeze=False,
                )
                policy_model.load_checkpoint(model_dir)
                logger.info(
                    "CheckpointUtils.load(): Successfully loaded checkpoint "
                    "from '%s' using saved SCoRe Config.",
                    path,
                )
                return policy_model
            except Exception as exc:
                raise RuntimeError(
                    f"CheckpointUtils.load(): Failed to load model from "
                    f"'{model_dir}' using saved Config: {exc}. "
                    "The checkpoint may be corrupted or incompatible with "
                    "the current model architecture."
                ) from exc
        else:
            # Fallback: no score_config.json found. We cannot instantiate
            # ModelWrapper without a Config. Raise a clear error.
            raise RuntimeError(
                f"CheckpointUtils.load(): Cannot reconstruct ModelWrapper "
                f"from checkpoint at '{path}' — 'score_config.json' is "
                "missing. This file is written by CheckpointUtils.save() "
                "alongside the model weights. Ensure the checkpoint was "
                "saved using CheckpointUtils.save() with a valid Config. "
                "If loading a checkpoint saved by an older version, "
                "manually provide the Config and call "
                "model.load_checkpoint(path) directly."
            )

    def get_best_checkpoint(
        self,
        output_dir: str,
        metric: str = _DEFAULT_CHECKPOINT_METRIC,
    ) -> str:
        """Find the checkpoint with the highest value of the specified metric.

        Implements the paper's checkpoint selection strategy from Section 6:
            "For all RL runs, we selected checkpoints with the highest
            training reward, although a small held-out validation set of
            problems can also be used."

        Scans all subdirectories of output_dir, reads their metrics.json
        files, and returns the path of the subdirectory with the highest
        value of the specified metric.

        The config.yaml specifies:
            evaluation.checkpoint_metric: "train_reward_t2"
        Callers should pass this value explicitly:
            best_path = ckpt.get_best_checkpoint(output_dir, metric="train_reward_t2")

        Args:
            output_dir: Directory to scan for checkpoint subdirectories.
                Each subdirectory is expected to contain a metrics.json file.
                Subdirectories without metrics.json are skipped with a warning.
            metric: The metric key to rank checkpoints by. Default is
                'train_reward' (design spec default). The config.yaml value
                is 'train_reward_t2'. Callers should pass the config value.
                If a checkpoint's metrics.json does not contain this key,
                that checkpoint receives a score of -inf and is never selected.

        Returns:
            Full path string to the checkpoint directory with the highest
            value of the specified metric. This path can be passed directly
            to load() or used to construct model paths.

        Raises:
            FileNotFoundError: If output_dir does not exist.
            ValueError: If output_dir contains no valid checkpoint
                subdirectories (i.e., no subdirectories with readable
                metrics.json files containing the specified metric key).
                This is a hard error — callers depend on a valid path.
        """
        # ------------------------------------------------------------------
        # Step 1: Validate that output_dir exists.
        # ------------------------------------------------------------------
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(
                f"CheckpointUtils.get_best_checkpoint(): Output directory "
                f"'{output_dir}' does not exist. Ensure training has been "
                "run and checkpoints have been saved before calling this method."
            )

        # ------------------------------------------------------------------
        # Step 2: Scan output_dir for checkpoint subdirectories.
        # A valid checkpoint subdirectory must:
        #   (a) be a directory (not a file)
        #   (b) contain a metrics.json file
        #   (c) have the specified metric key in metrics.json
        # ------------------------------------------------------------------
        checkpoint_scores: List[Tuple[str, float]] = []

        try:
            entries: List[str] = os.listdir(output_dir)
        except OSError as exc:
            raise FileNotFoundError(
                f"CheckpointUtils.get_best_checkpoint(): Cannot list "
                f"directory '{output_dir}': {exc}."
            ) from exc

        for entry_name in sorted(entries):
            entry_path: str = os.path.join(output_dir, entry_name)

            # Skip non-directories (e.g., training.log, config files)
            if not os.path.isdir(entry_path):
                continue

            # Check for metrics.json
            metrics_path: str = os.path.join(entry_path, _METRICS_FILENAME)
            if not os.path.isfile(metrics_path):
                logger.debug(
                    "get_best_checkpoint(): Skipping '%s' — no metrics.json.",
                    entry_path,
                )
                continue

            # Read metrics.json
            try:
                with open(metrics_path, "r", encoding="utf-8") as fh:
                    checkpoint_metrics: Dict[str, Any] = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "get_best_checkpoint(): Skipping '%s' — could not read "
                    "metrics.json: %s.",
                    entry_path,
                    exc,
                )
                continue

            # Extract the metric value for ranking
            if metric not in checkpoint_metrics:
                logger.debug(
                    "get_best_checkpoint(): Checkpoint '%s' does not contain "
                    "metric '%s'. Assigning score=-inf.",
                    entry_path,
                    metric,
                )
                metric_value: float = _MISSING_METRIC_SENTINEL
            else:
                try:
                    metric_value = float(checkpoint_metrics[metric])
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "get_best_checkpoint(): Could not convert metric "
                        "'%s'=%r to float in checkpoint '%s': %s. "
                        "Assigning score=-inf.",
                        metric,
                        checkpoint_metrics[metric],
                        entry_path,
                        exc,
                    )
                    metric_value = _MISSING_METRIC_SENTINEL

            checkpoint_scores.append((entry_path, metric_value))

        # ------------------------------------------------------------------
        # Step 3: Validate that at least one valid checkpoint was found.
        # ------------------------------------------------------------------
        if not checkpoint_scores:
            raise ValueError(
                f"CheckpointUtils.get_best_checkpoint(): No valid checkpoints "
                f"found in '{output_dir}'. A valid checkpoint must be a "
                "subdirectory containing a 'metrics.json' file. "
                "Ensure training has completed and checkpoints were saved "
                "using CheckpointUtils.save()."
            )

        # Filter out checkpoints with -inf scores (missing metric key)
        valid_checkpoints: List[Tuple[str, float]] = [
            (path, score)
            for path, score in checkpoint_scores
            if score > _MISSING_METRIC_SENTINEL
        ]

        if not valid_checkpoints:
            raise ValueError(
                f"CheckpointUtils.get_best_checkpoint(): Found "
                f"{len(checkpoint_scores)} checkpoint(s) in '{output_dir}', "
                f"but none contain the metric '{metric}'. "
                "Available checkpoints: "
                + ", ".join(p for p, _ in checkpoint_scores)
                + ". "
                "Ensure the metric key matches what was logged during training. "
                f"The config.yaml specifies checkpoint_metric: 'train_reward_t2'."
            )

        # ------------------------------------------------------------------
        # Step 4: Select the checkpoint with the highest metric value.
        # In case of ties, the last one in sorted order is selected
        # (consistent with the sort in Step 2 — later steps win ties).
        # ------------------------------------------------------------------
        best_path: str
        best_score: float
        best_path, best_score = max(valid_checkpoints, key=lambda x: x[1])

        logger.info(
            "get_best_checkpoint(): Best checkpoint for metric='%s': "
            "'%s' (score=%.6f). "
            "Scanned %d total checkpoints, %d with valid metric.",
            metric,
            best_path,
            best_score,
            len(checkpoint_scores),
            len(valid_checkpoints),
        )

        return best_path

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _serialize_metric_value(self, value: Any) -> Any:
        """Convert a metric value to a JSON-serializable type.

        Handles the common case where metric values are PyTorch tensors
        (e.g., loss.item() may not have been called) or numpy scalars.
        Falls back to str() for any type that cannot be converted to a
        standard JSON type.

        Args:
            value: The metric value to serialize. Expected types:
                float, int, bool, str — returned as-is.
                torch.Tensor (scalar) — converted via .item().
                numpy scalar — converted via float().
                Other — converted via str() as last resort.

        Returns:
            A JSON-serializable value (float, int, bool, str, None).
        """
        # Handle None
        if value is None:
            return None

        # Handle standard JSON-serializable types directly
        if isinstance(value, (bool, int, str)):
            return value

        if isinstance(value, float):
            # Handle NaN and Inf — json.dump raises on these by default
            import math
            if math.isnan(value):
                return "NaN"
            if math.isinf(value):
                return "Inf" if value > 0 else "-Inf"
            return value

        # Handle PyTorch tensors (scalar tensors from loss.item() calls)
        try:
            import torch
            if isinstance(value, torch.Tensor):
                scalar_val: float = value.item()
                return self._serialize_metric_value(scalar_val)
        except ImportError:
            pass

        # Handle numpy scalars and arrays
        try:
            import numpy as np
            if isinstance(value, (np.integer, np.floating)):
                return float(value)
            if isinstance(value, np.ndarray) and value.ndim == 0:
                return float(value.item())
        except ImportError:
            pass

        # Handle lists and dicts recursively (for nested metrics)
        if isinstance(value, list):
            return [self._serialize_metric_value(v) for v in value]

        if isinstance(value, dict):
            return {
                str(k): self._serialize_metric_value(v)
                for k, v in value.items()
            }

        # Last resort: convert to string
        logger.debug(
            "_serialize_metric_value: Converting value of type '%s' to str.",
            type(value).__name__,
        )
        return str(value)

    def save_score_config(
        self,
        config: "Config",
        path: str,
    ) -> None:
        """Save the SCoRe Config dataclass as score_config.json in a checkpoint directory.

        This method is called by save() to persist the full SCoRe Config
        alongside the model weights. The load() method reads this file to
        reconstruct the Config when loading a checkpoint.

        Saving the Config is essential for self-contained checkpoints:
        without it, load() cannot instantiate a ModelWrapper (which requires
        a Config). This is especially important for Stage II initialization
        from Stage I checkpoints, where the Config must be preserved exactly.

        Args:
            config: The SCoRe Config instance to serialize. Calls
                config.to_dict() to get a flat JSON-serializable dict.
            path: The checkpoint directory path (same as passed to save()).
                The score_config.json file is written directly in this
                directory (not in the model/ subdirectory).

        Raises:
            OSError: If the file cannot be written.
        """
        # Import here to avoid circular imports at module level.
        from config import Config

        score_config_path: str = os.path.join(path, "score_config.json")

        try:
            config_dict: Dict[str, Any] = config.to_dict()
            with open(score_config_path, "w", encoding="utf-8") as fh:
                json.dump(config_dict, fh, indent=2, ensure_ascii=False)
            logger.debug(
                "save_score_config(): Saved SCoRe Config to '%s'.",
                score_config_path,
            )
        except OSError as exc:
            raise OSError(
                f"save_score_config(): Failed to write score_config.json to "
                f"'{score_config_path}': {exc}."
            ) from exc
