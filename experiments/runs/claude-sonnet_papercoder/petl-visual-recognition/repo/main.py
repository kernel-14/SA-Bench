## Code: main.py

```python
## main.py
"""Main entry point for the PEFT Visual Recognition reproduction study.

This module orchestrates all five experiment types described in the paper:
    "Lessons and Insights from a Unifying Study of Parameter-Efficient
    Fine-Tuning (PEFT) in Visual Recognition"

Experiment types:
    vtab       - Low-shot evaluation on VTAB-1K (19 tasks, 1000 samples each)
    manyshot   - Many-shot evaluation on CIFAR-100, RESISC45, Clevr-Distance
    robustness - Distribution shift robustness with CLIP ViT-B/16
    diversity  - Prediction diversity analysis across PEFT methods
    ensemble   - Ensemble evaluation (majority vote / average logits)
    wise       - Weight-Space Ensemble (WiSE) robustness sweep

Usage:
    python main.py --config config.yaml --experiment vtab --method all --dataset all
    python main.py --config config.yaml --experiment manyshot --method lora --dataset cifar100
    python main.py --config config.yaml --experiment robustness --method lora
    python main.py --config config.yaml --experiment diversity
    python main.py --config config.yaml --experiment ensemble
"""

import argparse
import copy
import json
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# ---------------------------------------------------------------------------
# Project module imports
# ---------------------------------------------------------------------------
from datasets.vtab_loader import (
    VTABLoader,
    VTAB_TASK_NAMES,
    VTAB_GROUPS,
    VTAB_NUM_CLASSES,
)
from datasets.manyshot_loader import ManyShotLoader, SUPPORTED_DATASETS as MANYSHOT_DATASETS
from datasets.imagenet_loader import ImageNetLoader
from models.vit_wrapper import ViTWrapper
from models.clip_wrapper import CLIPWrapper, CLIP_IMAGENET_TEMPLATES
from models.peft_factory import PEFTFactory, SUPPORTED_METHODS, PEFTModel
from training.trainer import Trainer
from training.hyperparam_search import HyperparamSearch
from training.wise import WiSE
from evaluation.metrics import Metrics
from evaluation.diversity import DiversityAnalysis
from evaluation.ensemble import EnsembleEvaluator
from utils.logger import Logger
from utils.checkpoint import Checkpoint

# ---------------------------------------------------------------------------
# Module-level logger (root logger; experiment-specific loggers are created
# inside each run_* function via Logger class)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid experiment names
# ---------------------------------------------------------------------------
VALID_EXPERIMENTS: List[str] = [
    "vtab",
    "manyshot",
    "robustness",
    "diversity",
    "ensemble",
    "wise",
]

# ---------------------------------------------------------------------------
# Default values from config.yaml
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH: str = "config.yaml"
_DEFAULT_EXPERIMENT: str = "vtab"
_DEFAULT_METHOD: str = "all"
_DEFAULT_DATASET: str = "all"
_DEFAULT_DEVICE: str = "cuda"
_DEFAULT_SEED: int = 42
_DEFAULT_OUTPUT_DIR: str = "./outputs"

# ---------------------------------------------------------------------------
# ImageNet class names file (downloaded or bundled)
# ---------------------------------------------------------------------------
_IMAGENET_CLASSNAMES_URL: str = (
    "https://raw.githubusercontent.com/openai/CLIP/main/notebooks/imagenet_classes.py"
)


# ===========================================================================
# CLI Argument Parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the PEFT reproduction study.

    Returns:
        argparse.Namespace with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="PEFT Visual Recognition Reproduction Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help="Path to the unified config.yaml file.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=_DEFAULT_EXPERIMENT,
        choices=VALID_EXPERIMENTS,
        help=(
            "Experiment type to run. One of: "
            + ", ".join(VALID_EXPERIMENTS)
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default=_DEFAULT_METHOD,
        help=(
            "PEFT method name or 'all'. "
            f"Valid methods: {', '.join(SUPPORTED_METHODS)}"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=_DEFAULT_DATASET,
        help=(
            "Dataset name or 'all'. "
            "For vtab: one of 19 VTAB task names. "
            "For manyshot: cifar100, resisc45, clevr_distance."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_DEFAULT_DEVICE,
        choices=["cuda", "cpu"],
        help="Compute device.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override config.output.base_dir if specified.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config.seed if specified.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run experiments even if results already exist on disk.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory for datasets.",
    )

    return parser.parse_args()


# ===========================================================================
# Config Loading
# ===========================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the YAML configuration file into a nested dict.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Nested dict representation of the config.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: '{config_path}'. "
            "Please provide a valid path via --config."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    _logger.info("Config loaded from: %s", config_path)
    return config


def _get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely retrieves a nested value from a config dict.

    Args:
        config: Nested dict.
        *keys: Sequence of keys to traverse.
        default: Default value if any key is missing.

    Returns:
        The value at the nested path, or default if not found.
    """
    current: Any = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


# ===========================================================================
# Device Setup
# ===========================================================================

def _setup_device(device_str: str) -> str:
    """Validates and resolves the compute device.

    Falls back to CPU if CUDA is requested but unavailable.

    Args:
        device_str: Requested device string ('cuda' or 'cpu').

    Returns:
        Resolved device string ('cuda' or 'cpu').
    """
    if device_str == "cuda" and not torch.cuda.is_available():
        _logger.warning(
            "CUDA requested but not available. Falling back to CPU."
        )
        return "cpu"
    return device_str


# ===========================================================================
# Reproducibility Setup
# ===========================================================================

def _setup_reproducibility(seed: int, device: str) -> None:
    """Sets random seeds for reproducibility.

    Args:
        seed: Random seed value. config.yaml: seed: 42
        device: Resolved device string.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _logger.info("Reproducibility configured: seed=%d, device=%s", seed, device)


# ===========================================================================
# Method and Dataset Expansion
# ===========================================================================

def _expand_methods(
    method_arg: str,
    experiment: str,
    config: Dict[str, Any],
) -> List[str]:
    """Expands 'all' to the full list of methods for the given experiment.

    Args:
        method_arg: Method name or 'all'.
        experiment: Experiment type string.
        config: Loaded config dict.

    Returns:
        List of method name strings to run.

    Raises:
        ValueError: If method_arg is not 'all' and not in SUPPORTED_METHODS.
    """
    if method_arg == "all":
        if experiment == "robustness":
            # config.yaml: robustness.methods_evaluated
            methods: List[str] = _get_nested(
                config, "robustness", "methods_evaluated",
                default=["full", "bitfit", "layernorm", "houlsby_adapter",
                         "adaptformer", "repadapter", "convpass", "lora", "fact_tk"],
            )
            return [str(m) for m in methods]
        else:
            return list(SUPPORTED_METHODS)
    else:
        if method_arg not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method: '{method_arg}'. "
                f"Valid methods: {SUPPORTED_METHODS}"
            )
        return [method_arg]


def _expand_datasets(
    dataset_arg: str,
    experiment: str,
    config: Dict[str, Any],
) -> List[str]:
    """Expands 'all' to the full list of datasets for the given experiment.

    Args:
        dataset_arg: Dataset name or 'all'.
        experiment: Experiment type string.
        config: Loaded config dict.

    Returns:
        List of dataset name strings to run.
    """
    if dataset_arg == "all":
        if experiment == "vtab":
            return list(VTAB_TASK_NAMES)
        elif experiment == "manyshot":
            return list(MANYSHOT_DATASETS)
        elif experiment in ("robustness", "wise"):
            return ["imagenet"]
        else:
            return ["all"]
    else:
        return [dataset_arg]


# ===========================================================================
# Helper: Build training config namespace
# ===========================================================================

def _build_trial_config(
    config: Dict[str, Any],
    experiment: str,
    method: str,
    dataset: str,
    lr: float,
    weight_decay: float,
    drop_path_rate: float,
    peft_params: Dict[str, Any],
    output_dir: str,
    seed: int,
) -> SimpleNamespace:
    """Builds a SimpleNamespace config for a single training trial.

    Args:
        config: Full config dict.
        experiment: Experiment type ('vtab', 'manyshot', 'robustness').
        method: PEFT method name.
        dataset: Dataset name.
        lr: Learning rate.
        weight_decay: Weight decay.
        drop_path_rate: Stochastic depth rate.
        peft_params: Method-specific hyperparameters.
        output_dir: Output directory for this trial.
        seed: Random seed.

    Returns:
        SimpleNamespace with all training config attributes.
    """
    if experiment == "manyshot":
        epochs: int = int(_get_nested(config, "manyshot", "training", "epochs", default=40))
    elif experiment == "robustness":
        epochs = int(_get_nested(config, "robustness", "training", "epochs", default=10))
    else:
        epochs = int(_get_nested(config, "vtab", "training", "epochs", default=100))

    batch_size: int = int(_get_nested(config, "vtab", "training", "batch_size", default=64))
    mixed_precision: bool = bool(_get_nested(config, "compute", "mixed_precision", default=True))
    log_interval: int = int(_get_nested(config, "output", "log_interval", default=10))
    gradient_clip: float = float(
        _get_nested(config, "vtab", "training", "gradient_clip_max_norm", default=1.0)
    )

    trial_cfg = SimpleNamespace(
        experiment=experiment,
        peft_method=method,
        dataset=dataset,
        lr=lr,
        weight_decay=weight_decay,
        drop_path_rate=drop_path_rate,
        epochs=epochs,
        batch_size=batch_size,
        mixed_precision=mixed_precision,
        log_interval=log_interval,
        gradient_clip_max_norm=gradient_clip,
        peft_params=peft_params,
        output_dir=output_dir,
        seed=seed,
    )

    # Add to_dict method for Checkpoint serialization
    def _to_dict() -> Dict[str, Any]:
        return {
            "experiment": trial_cfg.experiment,
            "peft_method": trial_cfg.peft_method,
            "dataset": trial_cfg.dataset,
            "lr": trial_cfg.lr,
            "weight_decay": trial_cfg.weight_decay,
            "drop_path_rate": trial_cfg.drop_path_rate,
            "epochs": trial_cfg.epochs,
            "batch_size": trial_cfg.batch_size,
            "mixed_precision": trial_cfg.mixed_precision,
            "log_interval": trial_cfg.log_interval,
            "gradient_clip_max_norm": trial_cfg.gradient_clip_max_norm,
            "peft_params": trial_cfg.peft_params,
            "output_dir": trial_cfg.output_dir,
            "seed": trial_cfg.seed,
        }

    trial_cfg.to_dict = _to_dict
    return trial_cfg


# ===========================================================================
# Helper: Find bottleneck dimension for target parameter count
# ===========================================================================

def _find_bottleneck_for_target(
    method: str,
    target_params: int,
    embed_dim: int = 768,
    num_layers: int = 12,
) -> int:
    """Finds the bottleneck dimension closest to a target parameter count.

    Uses closed-form parameter count formulas per method to find the
    bottleneck dimension that achieves approximately target_params trainable
    parameters.

    Args:
        method: PEFT method name.
        target_params: Target number of trainable parameters.
        embed_dim: Token embedding dimension D (768 for ViT-B/16).
        num_layers: Number of Transformer layers (12 for ViT-B/16).

    Returns:
        Integer bottleneck dimension. Returns 8 as a safe default if the
        method does not use a bottleneck.
    """
    # Candidate bottleneck dimensions to search over.
    candidates: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    def _count_params(bottleneck: int) -> int:
        """Estimates parameter count for a given bottleneck dimension."""
        if method in ("pfeiffer_adapter", "adaptformer"):
            # 1 adapter per layer: 2 * D * r per adapter
            return num_layers * 2 * embed_dim * bottleneck
        elif method == "houlsby_adapter":
            # 2 adapters per layer: 4 * D * r per layer
            return num_layers * 4 * embed_dim * bottleneck
        elif method == "convpass":
            # 2 convpass per layer: 2 * (D*r + r*r*9 + r*D) per layer
            return num_layers * 2 * (embed_dim * bottleneck + bottleneck * bottleneck * 9 + bottleneck * embed_dim)
        elif method == "repadapter":
            # 2 repadapters per layer: 2 * (D*r + D*r) per layer
            return num_layers * 2 * 2 * embed_dim * bottleneck
        elif method == "lora":
            # Q and V projections: 4 * D * r per layer (W_down_Q, W_up_Q, W_down_V, W_up_V)
            return num_layers * 4 * embed_dim * bottleneck
        elif method == "fact_tt":
            # U, V: 2 * D * r; Sigma: 12*L * r * r
            return 2 * embed_dim * bottleneck + 12 * num_layers * bottleneck * bottleneck
        elif method == "fact_tk":
            # U, V: 2 * D * r; B: 12*L * r; A: r^3
            return 2 * embed_dim * bottleneck + 12 * num_layers * bottleneck + bottleneck ** 3
        elif method in ("vpt_shallow",):
            # num_prompts * D
            return bottleneck * embed_dim
        elif method in ("vpt_deep",):
            # num_prompts * D * num_layers
            return bottleneck * embed_dim * num_layers
        else:
            # Default: linear in bottleneck
            return num_layers * 2 * embed_dim * bottleneck

    # Find the candidate closest to target_params.
    best_bottleneck: int = candidates[0]
    best_diff: int = abs(_count_params(candidates[0]) - target_params)

    for candidate in candidates[1:]:
        diff: int = abs(_count_params(candidate) - target_params)
        if diff < best_diff:
            best_diff = diff
            best_bottleneck = candidate

    return best_bottleneck


# ===========================================================================
# Helper: Load ImageNet class names
# ===========================================================================

def _load_imagenet_classnames() -> List[str]:
    """Loads the 1000 ImageNet class names.

    Tries to load from a local file first, then falls back to a hardcoded
    minimal list for testing. In production, the full 1000-class list should
    be available at data/imagenet_classes.txt.

    Returns:
        List of 1000 ImageNet class name strings.
    """
    # Try local file first.
    local_paths: List[str] = [
        "data/imagenet_classes.txt",
        "data/imagenet/imagenet_classes.txt",
        os.path.join(os.path.dirname(__file__), "data", "imagenet_classes.txt"),
    ]

    for path in local_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                classnames: List[str] = [line.strip() for line in f if line.strip()]
            if len(classnames) == 1000:
                _logger.info("Loaded %d ImageNet class names from %s", len(classnames), path)
                return classnames
            _logger.warning(
                "File %s has %d class names (expected 1000). Continuing search.",
                path,
                len(classnames),
            )

    # Try to import from torchvision.
    try:
        from torchvision.datasets import ImageNet  # type: ignore
        # torchvision doesn't expose class names directly without a dataset instance.
        # Fall through to the hardcoded approach.
    except ImportError:
        pass

    # Fallback: generate placeholder class names.
    # In a real experiment, the user must provide the actual class names.
    _logger.warning(
        "ImageNet class names file not found. Using placeholder names. "
        "For accurate zero-shot head initialization, provide "
        "data/imagenet_classes.txt with 1000 class names."
    )
    return [f"class_{i}" for i in range(1000)]


# ===========================================================================
# Helper: Save VTAB summary table
# ===========================================================================

def _save_vtab_summary(
    results: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
    output_dir: str,
    logger: Logger,
) -> None:
    """Saves the VTAB-1K results as a summary CSV matching Table 1 structure.

    Args:
        results: Nested dict {dataset: {method: accuracy}}.
        config: Full config dict.
        output_dir: Output directory for the summary CSV.
        logger: Logger instance.
    """
    if not results:
        logger.info("No VTAB results to save.")
        return

    # Build DataFrame: rows = methods, columns = datasets.
    all_methods: List[str] = sorted(
        set(m for task_results in results.values() for m in task_results.keys())
    )
    all_datasets: List[str] = list(VTAB_TASK_NAMES)

    rows: List[Dict[str, Any]] = []
    for method in all_methods:
        row: Dict[str, Any] = {"method": method}
        task_accs: Dict[str, float] = {}

        for dataset in all_datasets:
            acc: float = results.get(dataset, {}).get(method, float("nan"))
            row[dataset] = round(acc * 100.0, 1) if not np.isnan(acc) else float("nan")
            if not np.isnan(acc):
                task_accs[dataset] = acc

        # Compute group averages.
        metrics_obj: Metrics = Metrics()
        group_avgs: Dict[str, float] = metrics_obj.compute_group_avg(task_accs)
        for group_name, avg in group_avgs.items():
            row[f"avg_{group_name}"] = round(avg * 100.0, 1)

        rows.append(row)

    df: pd.DataFrame = pd.DataFrame(rows)

    # Compute relative standard deviations per task (across PEFT methods only).
    peft_only_results: Dict[str, Dict[str, float]] = {
        dataset: {
            m: results[dataset][m]
            for m in results.get(dataset, {})
            if m not in ("linear", "full")
        }
        for dataset in results
    }

    rsd_row: Dict[str, Any] = {"method": "Relative Std Dev"}
    metrics_obj2: Metrics = Metrics()
    for dataset in all_datasets:
        method_accs_for_task: List[float] = [
            peft_only_results.get(dataset, {}).get(m, float("nan"))
            for m in all_methods
            if m not in ("linear", "full")
            and not np.isnan(peft_only_results.get(dataset, {}).get(m, float("nan")))
        ]
        if method_accs_for_task:
            rsd: float = metrics_obj2.relative_std_dev(method_accs_for_task)
            rsd_row[dataset] = round(rsd, 2)
        else:
            rsd_row[dataset] = float("nan")

    df = pd.concat([df, pd.DataFrame([rsd_row])], ignore_index=True)

    # Save to CSV.
    os.makedirs(output_dir, exist_ok=True)
    summary_path: str = os.path.join(output_dir, "vtab_summary_table1.csv")
    df.to_csv(summary_path, index=False)
    logger.info("VTAB summary table saved to: %s", summary_path)

    # Also save as JSON for programmatic access.
    json_path: str = os.path.join(output_dir, "vtab_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                dataset: {
                    method: float(acc) if not np.isnan(acc) else None
                    for method, acc in task_results.items()
                }
                for dataset, task_results in results.items()
            },
            f,
            indent=2,
        )
    logger.info("VTAB results JSON saved to: %s", json_path)


# ===========================================================================
# Helper: Load dataset predictions from disk
# ===========================================================================

def _load_dataset_predictions(
    predictions_dir: str,
    dataset: str,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Loads all method predictions for a given dataset from disk.

    Args:
        predictions_dir: Base directory containing prediction files.
        dataset: Dataset name (e.g., 'dtd').
        methods: List of method names to load. If None, loads all available.

    Returns:
        Dict with keys:
        - 'predictions': Dict[str, Tensor] — method -> (N,) int tensor
        - 'confidences': Dict[str, Tensor] — method -> (N,) float tensor (max conf)
        - 'logits': Dict[str, Tensor] — method -> (N, C) float tensor
        - 'labels': (N,) int tensor
    """
    if methods is None:
        methods = list(SUPPORTED_METHODS)

    all_predictions: Dict[str, torch.Tensor] = {}
    all_confidences: Dict[str, torch.Tensor] = {}
    all_logits: Dict[str, torch.Tensor] = {}
    labels: Optional[torch.Tensor] = None

    dataset_dir: str = os.path.join(predictions_dir, dataset)

    for method in methods:
        pred_path: str = os.path.join(dataset_dir, f"{method}_predictions.pkl")
        if not os.path.exists(pred_path):
            _logger.debug(
                "Predictions not found for method='%s', dataset='%s': %s",
                method,
                dataset,
                pred_path,
            )
            continue

        try:
            ckp: Checkpoint = Checkpoint(save_dir=dataset_dir)
            data: Dict[str, Any] = ckp.load_predictions(pred_path)

            if "predictions" in data:
                all_predictions[method] = data["predictions"]
            if "confidences" in data:
                # confidences may be (N, C) or (N,) — extract max if needed
                conf_tensor: torch.Tensor = data["confidences"]
                if conf_tensor.dim() == 2:
                    all_confidences[method] = conf_tensor.max(dim=1).values
                else:
                    all_confidences[method] = conf_tensor
            if "logits" in data:
                all_logits[method] = data["logits"]
            if labels is None and "labels" in data:
                labels = data["labels"]

        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "Failed to load predictions for method='%s', dataset='%s': %s",
                method,
                dataset,
                exc,
            )

    return {
        "predictions": all_predictions,
        "confidences": all_confidences,
        "logits": all_logits,
        "labels": labels,
    }


# ===========================================================================
# Helper: Load all VTAB results from disk
# ===========================================================================

def _load_all_vtab_results(
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    """Loads all saved VTAB accuracy results from disk.

    Looks for vtab_results.json in the output directory.

    Args:
        output_dir: Base output directory.

    Returns:
        Nested dict {dataset: {method: accuracy}} with accuracies in [0, 1].
        Returns empty dict if no results file is found.
    """
    json_path: str = os.path.join(output_dir, "vtab_results.json")

    if not os.path.exists(json_path):
        _logger.warning(
            "VTAB results file not found: %s. "
            "Run the vtab experiment first.",
            json_path,
        )
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    # Convert None back to NaN and ensure float values.
    results: Dict[str, Dict[str, float]] = {}
    for dataset, method_accs in raw.items():
        results[dataset] = {}
        for method, acc in method_accs.items():
            results[dataset][method] = float(acc) if acc is not None else float("nan")

    return results


# ===========================================================================
# Experiment Runner: VTAB-1K
# ===========================================================================

def run_vtab_experiment(
    config: Dict[str, Any],
    method: str,
    dataset: str,
    device: str,
    logger: Logger,
    output_dir: str,
    force: bool = False,
    data_dir: Optional[str] = None,
) -> float:
    """Runs the full VTAB-1K evaluation pipeline for one (method, dataset) pair.

    Pipeline:
    1. Load VTAB data (1000 samples, 80/20 split for HP search)
    2. Build ViT-B/16 backbone
    3. Run hyperparameter grid search
    4. Retrain on full 1000 samples with best config
    5. Evaluate on test set
    6. Save predictions and logits for diversity/ensemble analysis

    Paper: "We employ the ViT-B/16 pre-trained on ImageNet-21K as the backbone.
    We systematically tune 1) learning rate, 2) weight decay, and 3) method-
    specifics like the PEFT parameter sizes." (Section 3, Appendix A.1)

    Args:
        config: Full config dict loaded from config.yaml.
        method: PEFT method name (one of SUPPORTED_METHODS).
        dataset: VTAB task name (one of VTAB_TASK_NAMES).
        device: Compute device string ('cuda' or 'cpu').
        logger: Logger instance for this experiment.
        output_dir: Base output directory.
        force: If True, re-run even if results exist on disk.
        data_dir: Override data directory. If None, uses config default.

    Returns:
        Test Top-1 accuracy as a float in [0.0, 1.0].
        Returns 0.0 if the experiment fails.
    """
    # ------------------------------------------------------------------
    # Check if results already exist (skip if not forced).
    # ------------------------------------------------------------------
    result_path: str = os.path.join(output_dir, "vtab", dataset, f"{method}_predictions.pkl")
    if os.path.exists(result_path) and not force:
        logger.info(
            "Results already exist for method='%s', dataset='%s'. "
            "Loading from disk (use --force to re-run).",
            method,
            dataset,
        )
        try:
            ckp: Checkpoint = Checkpoint(save_dir=os.path.join(output_dir, "vtab", dataset))
            data: Dict[str, Any] = ckp.load_predictions(result_path)
            if "test_acc" in data:
                return float(data["test_acc"])
        except Exception:  # pylint: disable=broad-except
            pass

    # ------------------------------------------------------------------
    # Step 1: Load VTAB data.
    # ------------------------------------------------------------------
    vtab_cfg