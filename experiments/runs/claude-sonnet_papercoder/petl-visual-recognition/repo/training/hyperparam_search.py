## Code: training/hyperparam_search.py
```python
## training/hyperparam_search.py
"""Hyperparameter grid search for the PEFT Visual Recognition reproduction study.

This module implements the grid search described in Appendix A.1 of the paper:

    "Lessons and Insights from a Unifying Study of Parameter-Efficient
    Fine-Tuning (PEFT) in Visual Recognition"

For each PEFT method on each dataset, HyperparamSearch exhaustively searches
over learning rate, weight decay, drop path rate, and method-specific parameters
(e.g., bottleneck dimension, scale factor). The best configuration found on the
80/20 validation split is returned for use in the final training run.

Paper: "We systematically tune 1) learning rate, 2) weight decay, and 3)
method-specifics like the PEFT parameter sizes." (Section 3, Appendix A.1)

Config references (config.yaml):
    vtab.hyperparam_search.learning_rate: [0.001, 0.01]
    vtab.hyperparam_search.weight_decay: [0.0001, 0.001]
    vtab.hyperparam_search.drop_path_rate: [0.0, 0.1]
    manyshot.hyperparam_search.learning_rate: [0.0005, 0.001]
    manyshot.hyperparam_search.weight_decay: [0.0001, 0.001]
    peft_param_cap.ratio: 0.015
    backbones.imagenet21k_vit.total_params: 86_000_000
    peft_methods.*.search_grid: method-specific hyperparameter grids

Typical usage (called by main.py):
    search = HyperparamSearch(
        base_config=config,
        method='lora',
        dataset_name='dtd',
        backbone_wrapper=vit_wrapper,
        num_classes=47,
        device='cuda',
    )
    best_config = search.run_search(train_loader, val_loader)
    search.save_results('./outputs/vtab/dtd/lora/search_results.csv')
"""

import copy
import itertools
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.peft_factory import PEFTFactory, PEFTModel, PEFT_PARAM_CAP_RATIO, TOTAL_VIT_B16_PARAMS
from training.trainer import Trainer
from utils.checkpoint import Checkpoint
from utils.logger import Logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter cap constants (config.yaml: peft_param_cap)
# ---------------------------------------------------------------------------
_DEFAULT_CAP_RATIO: float = PEFT_PARAM_CAP_RATIO          # 0.015
_DEFAULT_TOTAL_PARAMS: int = TOTAL_VIT_B16_PARAMS          # 86_000_000

# ---------------------------------------------------------------------------
# Methods with no method-specific hyperparameters (empty search_grid).
# For these, the method-specific grid is a single empty dict {}.
# config.yaml: peft_methods.{method}.search_grid: {}
# ---------------------------------------------------------------------------
_NO_HPARAM_METHODS: set = {"bitfit", "layernorm", "difffit", "ssf", "linear", "full"}

# ---------------------------------------------------------------------------
# Default training config values used when config attributes are missing.
# These match config.yaml defaults.
# ---------------------------------------------------------------------------
_DEFAULT_EPOCHS_VTAB: int = 100
_DEFAULT_EPOCHS_MANYSHOT: int = 40
_DEFAULT_BATCH_SIZE: int = 64
_DEFAULT_MIXED_PRECISION: bool = True
_DEFAULT_LOG_INTERVAL: int = 10
_DEFAULT_GRADIENT_CLIP: float = 1.0


class _TrialConfig:
    """Lightweight configuration object for a single hyperparameter trial.

    Holds all training hyperparameters needed by Trainer and PEFTFactory
    for one grid search trial. Created fresh for each trial by HyperparamSearch
    to avoid state leakage between trials.

    Attributes:
        experiment: Experiment type string ('vtab', 'manyshot', 'robustness').
        peft_method: PEFT method name string.
        dataset: Dataset name string.
        lr: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        drop_path_rate: Stochastic depth drop rate for ViT backbone.
        epochs: Number of training epochs.
        batch_size: Batch size (informational; DataLoaders are pre-built).
        mixed_precision: Whether to use AMP (torch.cuda.amp).
        log_interval: Log every N batches.
        gradient_clip_max_norm: Max gradient norm for clipping.
        peft_params: Dict of method-specific hyperparameters.
        output_dir: Output directory for this trial's artifacts.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        experiment: str = "vtab",
        peft_method: str = "lora",
        dataset: str = "dtd",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        drop_path_rate: float = 0.0,
        epochs: int = _DEFAULT_EPOCHS_VTAB,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        mixed_precision: bool = _DEFAULT_MIXED_PRECISION,
        log_interval: int = _DEFAULT_LOG_INTERVAL,
        gradient_clip_max_norm: float = _DEFAULT_GRADIENT_CLIP,
        peft_params: Optional[Dict[str, Any]] = None,
        output_dir: str = "./outputs",
        seed: int = 42,
    ) -> None:
        """Initialises the trial configuration.

        Args:
            experiment: Experiment type. One of 'vtab', 'manyshot', 'robustness'.
            peft_method: PEFT method name. One of SUPPORTED_METHODS.
            dataset: Dataset name (e.g., 'dtd', 'cifar100').
            lr: Learning rate for AdamW.
            weight_decay: Weight decay for AdamW.
            drop_path_rate: Stochastic depth drop rate.
            epochs: Number of training epochs.
            batch_size: Batch size (informational).
            mixed_precision: Enable AMP.
            log_interval: Log every N batches.
            gradient_clip_max_norm: Max gradient norm for clipping.
            peft_params: Method-specific hyperparameter dict.
            output_dir: Output directory for artifacts.
            seed: Random seed.
        """
        self.experiment: str = experiment
        self.peft_method: str = peft_method
        self.dataset: str = dataset
        self.lr: float = lr
        self.weight_decay: float = weight_decay
        self.drop_path_rate: float = drop_path_rate
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.mixed_precision: bool = mixed_precision
        self.log_interval: int = log_interval
        self.gradient_clip_max_norm: float = gradient_clip_max_norm
        self.peft_params: Dict[str, Any] = peft_params if peft_params is not None else {}
        self.output_dir: str = output_dir
        self.seed: int = seed

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the config to a plain Python dict.

        Used by Checkpoint._serialise_config() for checkpoint metadata.

        Returns:
            Dict representation of all config attributes.
        """
        return {
            "experiment": self.experiment,
            "peft_method": self.peft_method,
            "dataset": self.dataset,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "drop_path_rate": self.drop_path_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "mixed_precision": self.mixed_precision,
            "log_interval": self.log_interval,
            "gradient_clip_max_norm": self.gradient_clip_max_norm,
            "peft_params": self.peft_params,
            "output_dir": self.output_dir,
            "seed": self.seed,
        }


class HyperparamSearch:
    """Grid search engine for PEFT hyperparameter tuning.

    Exhaustively searches over learning rate, weight decay, drop path rate,
    and method-specific parameters. For each combination, trains a PEFT model
    on the training split and evaluates on the validation split. The best
    configuration (highest validation accuracy) is returned for use in the
    final training run.

    Paper: "We systematically tune 1) learning rate, 2) weight decay, and
    3) method-specifics like the PEFT parameter sizes, which are often left
    intact in previous studies." (Section 3, Appendix A.1)

    Attributes:
        base_config: The loaded YAML configuration object. Never mutated.
        method: PEFT method name string (one of SUPPORTED_METHODS).
        dataset_name: Dataset name string (e.g., 'dtd', 'cifar100').
        backbone_wrapper: ViTWrapper or CLIPWrapper instance. Never mutated.
        num_classes: Number of output classes for the current dataset.
        device: Target device string ('cuda' or 'cpu').
        results: List of trial results, each a dict with 'params' and 'val_acc'.
        best_config: Best hyperparameter dict found by run_search(). None until
            run_search() completes.
        factory: PEFTFactory instance reused across all trials.
    """

    def __init__(
        self,
        base_config: Any,
        method: str,
        dataset_name: str,
        backbone_wrapper: Any,
        num_classes: int,
        device: str = "cuda",
    ) -> None:
        """Initialises the hyperparameter search engine.

        Args:
            base_config: The loaded YAML configuration object (omegaconf
                DictConfig or plain dict). Must expose experiment-specific
                hyperparam_search grids and peft_methods search grids.
                Never mutated — all modifications are made on deep copies.
            method: PEFT method name. Must be one of SUPPORTED_METHODS.
                Example: 'lora', 'houlsby_adapter', 'bitfit'.
            dataset_name: Dataset name for this search. Used for logging
                and output directory naming.
                Example: 'dtd', 'cifar100', 'clevr_distance'.
            backbone_wrapper: A ViTWrapper or CLIPWrapper instance providing
                get_backbone() and architecture constants. Never modified —
                each trial deep-copies the backbone from this wrapper.
            num_classes: Number of output classes for the classification head.
                Varies per dataset (e.g., 47 for DTD, 100 for CIFAR-100).
            device: Target device for training. Default: 'cuda'.
                Falls back to 'cpu' if CUDA is unavailable.
        """
        self.base_config: Any = base_config
        self.method: str = method
        self.dataset_name: str = dataset_name
        self.backbone_wrapper: Any = backbone_wrapper
        self.num_classes: int = num_classes
        self.device: str = device

        # Resolve device: fall back to CPU if CUDA requested but unavailable.
        if device == "cuda" and not torch.cuda.is_available():
            _logger.warning(
                "CUDA requested but not available. Falling back to CPU for "
                "hyperparameter search."
            )
            self.device = "cpu"

        # Trial results: list of {'params': dict, 'val_acc': float}.
        self.results: List[Dict[str, Any]] = []

        # Best config found by run_search(). None until search completes.
        self.best_config: Optional[Dict[str, Any]] = None

        # PEFTFactory instance reused across all trials (stateless).
        self.factory: PEFTFactory = PEFTFactory()

        _logger.info(
            "HyperparamSearch initialised: method='%s', dataset='%s', "
            "num_classes=%d, device='%s'",
            self.method,
            self.dataset_name,
            self.num_classes,
            self.device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_search_space(self) -> List[Dict[str, Any]]:
        """Builds the complete hyperparameter search space as a list of dicts.

        Constructs the Cartesian product of:
        1. Common grids: learning rate × weight decay × drop path rate
        2. Method-specific grids: from config.yaml peft_methods.{method}.search_grid

        Paper Table 3 and Appendix A.1 define the exact search grids for each
        method. This method reads them directly from the config.

        Config references:
            vtab.hyperparam_search.learning_rate: [0.001, 0.01]
            vtab.hyperparam_search.weight_decay: [0.0001, 0.001]
            vtab.hyperparam_search.drop_path_rate: [0.0, 0.1]
            manyshot.hyperparam_search.learning_rate: [0.0005, 0.001]
            peft_methods.lora.search_grid.lora_rank: [1, 8, 16, 32]
            peft_methods.houlsby_adapter.search_grid.adapter_scale: [0.01, 0.1, 1.0, 10.0]
            ... (see config.yaml for all method-specific grids)

        Returns:
            List of dicts, each representing one complete hyperparameter
            combination. Each dict has keys:
            - 'lr': float — learning rate
            - 'weight_decay': float — weight decay
            - 'drop_path_rate': float — stochastic depth rate
            - method-specific keys (e.g., 'lora_rank', 'adapter_bottleneck')

            Example for LoRA (2×2×2×4 = 32 combinations):
                [
                    {'lr': 0.001, 'weight_decay': 0.0001, 'drop_path_rate': 0.0, 'lora_rank': 1},
                    {'lr': 0.001, 'weight_decay': 0.0001, 'drop_path_rate': 0.0, 'lora_rank': 8},
                    ...
                ]

            Example for BitFit (2×2×2×1 = 8 combinations):
                [
                    {'lr': 0.001, 'weight_decay': 0.0001, 'drop_path_rate': 0.0},
                    {'lr': 0.001, 'weight_decay': 0.0001, 'drop_path_rate': 0.1},
                    ...
                ]
        """
        # ------------------------------------------------------------------
        # Step 1: Read common grids from config.
        # Detect experiment type to read from the correct config section.
        # ------------------------------------------------------------------
        experiment: str = self._get_config_value(
            self.base_config, "experiment", default="vtab"
        )

        # Determine which hyperparam_search section to use.
        if experiment == "manyshot":
            search_section_key: str = "manyshot"
        else:
            # Default to vtab for both vtab and robustness experiments.
            search_section_key = "vtab"

        # Read common grids with fallback defaults matching config.yaml.
        lr_values: List[float] = self._get_nested_config_value(
            self.base_config,
            [search_section_key, "hyperparam_search", "learning_rate"],
            default=[0.001, 0.01],
        )
        wd_values: List[float] = self._get_nested_config_value(
            self.base_config,
            [search_section_key, "hyperparam_search", "weight_decay"],
            default=[0.0001, 0.001],
        )
        drop_path_values: List[float] = self._get_nested_config_value(
            self.base_config,
            [search_section_key, "hyperparam_search", "drop_path_rate"],
            default=[0.0, 0.1],
        )

        # Ensure all values are Python floats (not numpy or omegaconf types).
        lr_values = [float(v) for v in lr_values]
        wd_values = [float(v) for v in wd_values]
        drop_path_values = [float(v) for v in drop_path_values]

        _logger.info(
            "Common search grids: lr=%s, wd=%s, drop_path=%s",
            lr_values,
            wd_values,
            drop_path_values,
        )

        # ------------------------------------------------------------------
        # Step 2: Read method-specific grids from config.
        # ------------------------------------------------------------------
        method_specific_combos: List[Dict[str, Any]] = self._build_method_specific_combos()

        _logger.info(
            "Method-specific combos for '%s': %d combinations",
            self.method,
            len(method_specific_combos),
        )

        # ------------------------------------------------------------------
        # Step 3: Build the full Cartesian product.
        # ------------------------------------------------------------------
        search_space: List[Dict[str, Any]] = []

        for lr, wd, dp, method_combo in itertools.product(
            lr_values,
            wd_values,
            drop_path_values,
            method_specific_combos,
        ):
            combo: Dict[str, Any] = {
                "lr": lr,
                "weight_decay": wd,
                "drop_path_rate": dp,
            }
            # Merge method-specific params into the combo dict.
            combo.update(method_combo)
            search_space.append(combo)

        _logger.info(
            "Search space built: %d total combinations for method='%s', "
            "dataset='%s'.",
            len(search_space),
            self.method,
            self.dataset_name,
        )

        return search_space

    def run_search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Runs the full hyperparameter grid search.

        Iterates over all combinations in the search space, trains a PEFT
        model for each, evaluates on the validation set, and records results.
        Returns the best configuration found.

        Paper: "We systematically tune 1) learning rate, 2) weight decay, and
        3) method-specifics like the PEFT parameter sizes." (Appendix A.1)

        Args:
            train_loader: DataLoader for the training split (80% of 1000 VTAB
                samples, or 90% of many-shot training data). Pre-built by the
                caller (VTABLoader or ManyShotLoader).
            val_loader: DataLoader for the validation split (20% of 1000 VTAB
                samples, or 10% of many-shot training data). Pre-built by the
                caller.

        Returns:
            Dict containing the best hyperparameter combination found, with
            keys: 'lr', 'weight_decay', 'drop_path_rate', and any
            method-specific keys. Same format as entries in build_search_space().

        Raises:
            RuntimeError: If no valid trials complete (all exceed param cap
                or all fail with errors).
        """
        # ------------------------------------------------------------------
        # Step 1: Build the search space.
        # ------------------------------------------------------------------
        search_space: List[Dict[str, Any]] = self.build_search_space()
        total_trials: int = len(search_space)

        _logger.info(
            "Starting hyperparameter search: method='%s', dataset='%s', "
            "%d total trials.",
            self.method,
            self.dataset_name,
            total_trials,
        )

        # ------------------------------------------------------------------
        # Step 2: Run each trial.
        # ------------------------------------------------------------------
        num_skipped_cap: int = 0
        num_failed: int = 0
        num_completed: int = 0

        for trial_idx, params in enumerate(search_space):
            _logger.info(
                "Trial %d/%d: %s",
                trial_idx + 1,
                total_trials,
                params,
            )

            try:
                val_acc: Optional[float] = self._run_single(
                    params=params,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    trial_idx=trial_idx,
                )
            except Exception as exc:  # pylint: disable=broad-except
                _logger.error(
                    "Trial %d/%d failed with error: %s. Skipping.",
                    trial_idx + 1,
                    total_trials,
                    exc,
                    exc_info=True,
                )
                num_failed += 1
                # Free GPU memory after failed trial.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if val_acc is None:
                # Param cap exceeded — skip this combo.
                _logger.warning(
                    "Trial %d/%d skipped: param cap exceeded for params=%s",
                    trial_idx + 1,
                    total_trials,
                    params,
                )
                num_skipped_cap += 1
                continue

            # Record successful trial result.
            self.results.append({"params": params, "val_acc": val_acc})
            num_completed += 1

            _logger.info(
                "Trial %d/%d completed: val_acc=%.4f (%.2f%%), params=%s",
                trial_idx + 1,
                total_trials,
                val_acc,
                val_acc * 100.0,
                params,
            )

        # ------------------------------------------------------------------
        # Step 3: Validate that at least one trial completed.
        # ------------------------------------------------------------------
        _logger.info(
            "Search complete: %d completed, %d skipped (cap), %d failed.",
            num_completed,
            num_skipped_cap,
            num_failed,
        )

        if num_completed == 0:
            raise RuntimeError(
                f"Hyperparameter search for method='{self.method}', "
                f"dataset='{self.dataset_name}' produced no valid trials. "
                f"All {total_trials} trials were either skipped (param cap: "
                f"{num_skipped_cap}) or failed (errors: {num_failed}). "
                "Check the param cap ratio and method-specific search grids."
            )

        # ------------------------------------------------------------------
        # Step 4: Find and return the best config.
        # ------------------------------------------------------------------
        best_config: Dict[str, Any] = self.get_best_config()

        _logger.info(
            "Best config found: val_acc=%.4f (%.2f%%), params=%s",
            max(r["val_acc"] for r in self.results),
            max(r["val_acc"] for r in self.results) * 100.0,
            best_config,
        )

        return best_config

    def get_best_config(self) -> Dict[str, Any]:
        """Returns the hyperparameter combination with the highest validation accuracy.

        Scans self.results to find the trial with the maximum val_acc and
        returns its params dict. Also caches the result in self.best_config.

        Returns:
            Dict containing the best hyperparameter combination. Keys include
            'lr', 'weight_decay', 'drop_path_rate', and any method-specific
            keys (e.g., 'lora_rank', 'adapter_bottleneck').

        Raises:
            RuntimeError: If self.results is empty (run_search() has not been
                called or all trials failed).
        """
        if not self.results:
            raise RuntimeError(
                "No search results available. Call run_search() before "
                "get_best_config(). If run_search() was called, all trials "
                "may have failed or been skipped."
            )

        # Find the trial with the highest validation accuracy.
        best_trial: Dict[str, Any] = max(
            self.results, key=lambda x: x["val_acc"]
        )
        self.best_config = best_trial["params"]

        return dict(self.best_config)  # Return a copy to prevent mutation.

    def save_results(self, path: str) -> None:
        """Saves all trial results to a CSV file.

        Converts self.results to a flat DataFrame where each row is one trial,
        with columns for all hyperparameter keys and the val_acc. Adds metadata
        columns for method and dataset_name.

        Args:
            path: Destination CSV file path. Parent directories are created
                if they do not exist. If the file already exists, it is
                overwritten.
        """
        if not self.results:
            _logger.warning(
                "save_results called with empty results. Nothing to save."
            )
            return

        # ------------------------------------------------------------------
        # Flatten each result entry into a single dict.
        # Merge params dict with val_acc and metadata.
        # ------------------------------------------------------------------
        flat_rows: List[Dict[str, Any]] = []

        for result in self.results:
            row: Dict[str, Any] = {
                "method": self.method,
                "dataset": self.dataset_name,
                "val_acc": result["val_acc"],
                "val_acc_pct": result["val_acc"] * 100.0,
            }
            # Merge all hyperparameter keys.
            row.update(result["params"])
            flat_rows.append(row)

        # ------------------------------------------------------------------
        # Create DataFrame and save to CSV.
        # ------------------------------------------------------------------
        df: pd.DataFrame = pd.DataFrame(flat_rows)

        # Sort by val_acc descending so the best trial is at the top.
        df = df.sort_values("val_acc", ascending=False).reset_index(drop=True)

        # Ensure parent directory exists.
        parent_dir: str = os.path.dirname(os.path.abspath(path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        df.to_csv(path, index=False)

        _logger.info(
            "Search results saved: %d trials → %s",
            len(flat_rows),
            path,
        )

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _run_single(
        self,
        params: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        trial_idx: int = 0,
    ) -> Optional[float]:
        """Runs a single hyperparameter trial.

        Performs the full trial pipeline:
        1. Deep-copy the backbone and apply drop_path_rate
        2. Build the PEFT model via PEFTFactory
        3. Check param cap (return None if exceeded)
        4. Create trial-scoped Logger and Checkpoint
        5. Train with Trainer and return best val_acc
        6. Clean up GPU memory

        Args:
            params: Complete hyperparameter dict for this trial. Contains
                'lr', 'weight_decay', 'drop_path_rate', and method-specific keys.
            train_loader: DataLoader for the training split.
            val_loader: DataLoader for the validation split.
            trial_idx: Trial index for logging and output directory naming.

        Returns:
            Best validation accuracy as a float in [0.0, 1.0], or None if
            the param cap is exceeded.

        Raises:
            Any exception from Trainer.train() or PEFTFactory.build() that
            is not caught internally. The caller (run_search) handles these.
        """
        # ------------------------------------------------------------------
        # Step 1: Extract hyperparameters from params dict.
        # ------------------------------------------------------------------
        lr: float = float(params.get("lr", 1e-3))
        weight_decay: float = float(params.get("weight_decay", 1e-4))
        drop_path_rate: float = float(params.get("drop_path_rate", 0.0))

        # Method-specific params: everything except the common keys.
        _common_keys: set = {"lr", "weight_decay", "drop_path_rate"}
        peft_params: Dict[str, Any] = {
            k: v for k, v in params.items() if k not in _common_keys
        }

        # ------------------------------------------------------------------
        # Step 2: Deep-copy the backbone and apply drop_path_rate.
        # The backbone_wrapper is never mutated — we always work on a copy.
        # ------------------------------------------------------------------
        backbone_copy: Any = copy.deepcopy(self.backbone_wrapper.get_backbone())
        self._apply_drop_path_rate(backbone_copy, drop_path_rate)

        # ------------------------------------------------------------------
        # Step 3: Create a temporary backbone wrapper proxy that returns
        # the modified backbone copy. PEFTFactory.build() calls
        # backbone_wrapper.get_backbone() and deepcopies it internally.
        # We need to pass the already-modified backbone copy, so we create
        # a lightweight proxy that returns it directly.
        # ------------------------------------------------------------------
        backbone_proxy: _BackboneProxy = _BackboneProxy(
            backbone=backbone_copy,
            embed_dim=getattr(self.backbone_wrapper, "embed_dim", 768),
            num_layers=getattr(self.backbone_wrapper, "num_layers", 12),
        )

        # ------------------------------------------------------------------
        # Step 4: Build the PEFT model via PEFTFactory.
        # PEFTFactory.build() will deepcopy the backbone from the proxy.
        # ------------------------------------------------------------------
        model: PEFTModel = self.factory.build(
            method=self.method,
            backbone_wrapper=backbone_proxy,
            num_classes=self.num_classes,
            peft_params=peft_params,
            device=self.device,
        )

        # ------------------------------------------------------------------
        # Step 5: Check param cap.
        # Paper: "We set a cap for PEFT size ≤ 1.5% of ViT-B/16."
        # config.yaml: peft_param_cap.ratio: 0.015
        # Methods 'linear' and 'full' are exempt from the cap.
        # ------------------------------------------------------------------
        if self.method not in {"linear", "full"}:
            cap_ratio: float = self._get_nested_config_value(
                self.base_config,
                ["peft_param_cap", "ratio"],
                default=_DEFAULT_CAP_RATIO,
            )
            total_params: int = self._get_nested_config_value(
                self.base_config,
                ["peft_param_cap", "absolute_max_params"],
                default=_DEFAULT_TOTAL_PARAMS,
            )

            if not self.factory.check_param_cap(
                model,
                cap_ratio=float(cap_ratio),
                total_params=int(total_params),
            ):
                # Param cap exceeded — clean up and return None.
                trainable_count: int = model.count_trainable_params()
                max_allowed: int = int(float(cap_ratio) * int(total_params))
                _logger.debug(
                    "Param cap exceeded: trainable=%d > max=%d (%.4fM > %.4fM). "
                    "Skipping trial.",
                    trainable_count,
                    max_allowed,
                    trainable_count / 1_000_000,
                    max_allowed / 1_000_000,
                )
                # Free memory.
                del model
                del backbone_copy
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None

        # ------------------------------------------------------------------
        # Step 6: Create trial-scoped Logger and Checkpoint.
        # Each trial gets its own subdirectory to avoid log collisions.
        # ------------------------------------------------------------------
        output_dir: str = self._get_config_value(
            self.