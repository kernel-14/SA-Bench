## Code: main.py

```python
## main.py
"""Entry point for SC-FNO experiments.

Orchestrates the full experimental pipeline for reproducing the SC-FNO paper:
  - Dataset generation (differentiable solver or finite differences)
  - Training all four model variants (FNO, SC-FNO, FNO-PINN, SC-FNO-PINN)
  - Evaluation: forward quality, perturbation robustness, parameter inversion,
    data scaling, high-dimensional parameters, AD vs FD comparison

Usage examples:
    # Generate dataset for PDE1
    python main.py --equation pde1 --mode generate

    # Train SC-FNO on PDE1
    python main.py --equation pde1 --mode train --variant sc_fno

    # Evaluate forward quality
    python main.py --equation pde1 --mode evaluate --variant sc_fno --experiment forward

    # Run all experiments for PDE1
    python main.py --equation pde1 --mode all --variant all --experiment all

    # Reproduce the full paper
    python main.py --mode all --variant all --experiment all

References:
    - SC-FNO paper: "Sensitivity-Constrained Fourier Neural Operators for
      Forward and Inverse Problems in Parametric Differential Equations"
    - config.yaml: master configuration file
"""

import argparse
import copy
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import SCFNODataset
from data.generators.ode1_generator import ODE1Generator
from data.generators.ode2_generator import ODE2Generator
from data.generators.pde1_generator import PDE1Generator
from data.generators.pde2_generator import PDE2Generator
from data.generators.pde3_generator import PDE3Generator
from data.generators.pde4_generator import PDE4Generator
from evaluation.evaluator import Evaluator
from evaluation.metrics import Metrics
from models.sc_fno import VALID_VARIANTS, build_model
from training.inversion import Inverter
from training.trainer import Trainer
from utils.config_loader import ConfigLoader
from utils.logger import Logger
from utils.visualization import Visualizer

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: All valid equation identifiers.
VALID_EQUATIONS: List[str] = [
    "ode1", "ode2", "pde1", "pde2", "pde2_zoned", "pde3", "pde4"
]

#: All valid pipeline modes.
VALID_MODES: List[str] = ["generate", "train", "evaluate", "invert", "all"]

#: All valid experiment types.
VALID_EXPERIMENTS: List[str] = [
    "forward", "perturbation", "inversion", "scaling",
    "high_dim", "fd_comparison", "all"
]

#: Default path to the master configuration file.
DEFAULT_CONFIG_PATH: str = "config.yaml"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the SC-FNO pipeline.

    Returns:
        Parsed argument namespace with attributes:
          - config: str, path to master config YAML
          - equation: str, equation identifier or 'all'
          - mode: str, pipeline stage
          - variant: str, model variant or 'all'
          - experiment: str, evaluation experiment or 'all'
          - force_regenerate: bool, re-generate even if dataset exists
          - force_retrain: bool, re-train even if checkpoint exists
    """
    parser = argparse.ArgumentParser(
        description="SC-FNO: Sensitivity-Constrained Fourier Neural Operators",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the master YAML configuration file.",
    )

    parser.add_argument(
        "--equation",
        type=str,
        default="pde1",
        choices=VALID_EQUATIONS + ["all"],
        help="Equation to run. Use 'all' to run all equations sequentially.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=VALID_MODES,
        help=(
            "Pipeline stage: 'generate' (dataset), 'train', 'evaluate', "
            "'invert' (standalone inversion), or 'all' (full pipeline)."
        ),
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="sc_fno",
        choices=VALID_VARIANTS + ["all"],
        help="Model variant. Use 'all' to run all variants sequentially.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="forward",
        choices=VALID_EXPERIMENTS,
        help=(
            "Evaluation experiment to run. Only used when --mode is "
            "'evaluate' or 'all'."
        ),
    )

    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        default=False,
        help="Re-generate datasets even if .pt files already exist.",
    )

    parser.add_argument(
        "--force_retrain",
        action="store_true",
        default=False,
        help="Re-train models even if checkpoints already exist.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def setup_environment(cfg: dict) -> torch.device:
    """Sets random seeds, validates device, and creates output directories.

    Args:
        cfg: The master configuration dictionary loaded from config.yaml.
             Reads 'seed', 'device', 'output_dir', 'log_dir',
             'checkpoint_dir', 'results_dir', 'figures_dir'.

    Returns:
        The resolved torch.device (CPU or CUDA).

    Side effects:
        - Sets torch, numpy, and Python random seeds.
        - Creates all output directories.
        - Updates cfg['device'] to the resolved device string.
    """
    seed: int = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Resolve device — fall back to CPU if CUDA is requested but unavailable.
    device_str: str = str(cfg.get("device", "cpu"))
    if device_str == "cuda" and not torch.cuda.is_available():
        print(
            "[main] WARNING: CUDA requested but not available. "
            "Falling back to CPU."
        )
        device_str = "cpu"
    device: torch.device = torch.device(device_str)

    # Normalize device string in cfg so all downstream modules see a consistent value.
    cfg["device"] = str(device)

    # Create all output directories.
    dir_keys: List[str] = [
        "output_dir", "log_dir", "checkpoint_dir", "results_dir", "figures_dir"
    ]
    for key in dir_keys:
        dir_path: str = str(cfg.get(key, f"outputs/{key.replace('_dir', '')}"))
        os.makedirs(dir_path, exist_ok=True)

    print(
        f"[main] Environment: device={device} | seed={seed} | "
        f"output_dir='{cfg.get('output_dir', 'outputs')}'"
    )

    return device


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_eq_cfg(master_cfg: dict, equation_name: str) -> dict:
    """Resolves the effective configuration for a specific equation.

    Extracts the equation-specific sub-dict from the master config and merges
    global defaults for keys not present in the equation sub-config. The
    equation-level keys always win over global defaults.

    Args:
        master_cfg: The full master configuration dictionary loaded from
                    config.yaml by ConfigLoader.
        equation_name: Equation identifier string. One of VALID_EQUATIONS.
                       For 'pde2_zoned', reads from master_cfg['pde2_zoned'].

    Returns:
        A deep-copied, merged configuration dict for the specified equation.
        Contains all keys needed by generators, models, trainers, and evaluators.

    Raises:
        KeyError: If equation_name is not found in master_cfg.
    """
    # Map equation name to config key (pde2_zoned is a special case).
    config_key: str = equation_name

    if config_key not in master_cfg:
        raise KeyError(
            f"resolve_eq_cfg: equation '{equation_name}' not found in master "
            f"config. Available keys: {sorted(master_cfg.keys())}. "
            f"Check config.yaml."
        )

    # Deep copy to avoid mutating the master config.
    eq_cfg: dict = copy.deepcopy(master_cfg[config_key])

    # ------------------------------------------------------------------
    # Merge global defaults into equation sub-config.
    # Equation-level keys override global keys (eq_cfg wins).
    # ------------------------------------------------------------------

    # Training settings: inherit global defaults for missing keys.
    global_training: dict = copy.deepcopy(master_cfg.get("training", {}))
    eq_training: dict = eq_cfg.get("training", {})
    merged_training: dict = {**global_training, **eq_training}
    eq_cfg["training"] = merged_training

    # Model settings: inherit global defaults for missing keys.
    global_model: dict = copy.deepcopy(master_cfg.get("model", {}))
    eq_model: dict = eq_cfg.get("model", {})
    merged_model: dict = {**global_model, **eq_model}
    eq_cfg["model"] = merged_model

    # Data settings: inherit global defaults for missing keys.
    global_data: dict = copy.deepcopy(master_cfg.get("data", {}))
    eq_data: dict = eq_cfg.get("data", {})
    merged_data: dict = {**global_data, **eq_data}
    eq_cfg["data"] = merged_data

    # Inversion settings: inherit global defaults.
    global_inversion: dict = copy.deepcopy(master_cfg.get("inversion", {}))
    eq_inversion: dict = eq_cfg.get("inversion", {})
    merged_inversion: dict = {**global_inversion, **eq_inversion}
    eq_cfg["inversion"] = merged_inversion

    # ------------------------------------------------------------------
    # Always inject global settings that all modules need.
    # ------------------------------------------------------------------
    eq_cfg["seed"] = int(master_cfg.get("seed", 42))
    eq_cfg["device"] = str(master_cfg.get("device", "cpu"))
    eq_cfg["output_dir"] = str(master_cfg.get("output_dir", "outputs"))
    eq_cfg["log_dir"] = str(master_cfg.get("log_dir", "outputs/logs"))
    eq_cfg["checkpoint_dir"] = str(
        master_cfg.get("checkpoint_dir", "outputs/checkpoints")
    )
    eq_cfg["results_dir"] = str(
        master_cfg.get("results_dir", "outputs/results")
    )
    eq_cfg["figures_dir"] = str(
        master_cfg.get("figures_dir", "outputs/figures")
    )

    # PINN settings (used by FNO-PINN and SC-FNO-PINN).
    eq_cfg["pinn"] = copy.deepcopy(master_cfg.get("pinn", {
        "n_colloc": 256,
        "alpha_weight": 1.0,
    }))

    # Perturbation experiment settings.
    eq_cfg["perturbation"] = copy.deepcopy(master_cfg.get("perturbation", {
        "lambda_values": [0.1, 0.2, 0.3, 0.4]
    }))

    # Data scaling experiment settings.
    eq_cfg["data_scaling"] = copy.deepcopy(master_cfg.get("data_scaling", {
        "sample_sizes": [100, 200, 500, 1000, 2000],
        "equation": "pde1",
    }))

    # Variants list (used by run_all_experiments).
    eq_cfg["variants"] = copy.deepcopy(master_cfg.get("variants", [
        {"name": "fno", "use_sensitivity": False, "use_pinn": False},
        {"name": "sc_fno", "use_sensitivity": True, "use_pinn": False},
        {"name": "fno_pinn", "use_sensitivity": False, "use_pinn": True},
        {"name": "sc_fno_pinn", "use_sensitivity": True, "use_pinn": True},
    ]))

    return eq_cfg


# ---------------------------------------------------------------------------
# Checkpoint path helper
# ---------------------------------------------------------------------------

def _checkpoint_path(eq_cfg: dict, variant: str) -> str:
    """Returns the deterministic checkpoint path for an equation + variant.

    Args:
        eq_cfg: Resolved equation configuration dict.
        variant: Model variant string (e.g., 'sc_fno').

    Returns:
        Full path string for the checkpoint .pt file.
    """
    equation: str = str(eq_cfg.get("equation", "unknown"))
    checkpoint_dir: str = str(eq_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    return os.path.join(checkpoint_dir, f"{equation}_{variant}.pt")


# ---------------------------------------------------------------------------
# Dataset path helper
# ---------------------------------------------------------------------------

def _dataset_path(eq_cfg: dict) -> str:
    """Returns the primary dataset save path for an equation.

    For equations with multiple sample sizes (pde4, pde2_zoned), returns
    the path for the largest sample size.

    Args:
        eq_cfg: Resolved equation configuration dict.

    Returns:
        Full path string for the primary .pt dataset file.
    """
    data_cfg: dict = eq_cfg.get("data", {})

    # Check for n_samples_list (pde4, pde2_zoned).
    n_samples_list: Optional[List[int]] = data_cfg.get("n_samples_list", None)
    if n_samples_list:
        # Use the largest sample size as the primary dataset.
        max_n: int = max(n_samples_list)
        # Try to find the corresponding save path key.
        path_key: str = f"save_path_{max_n}"
        if path_key in data_cfg:
            return str(data_cfg[path_key])
        # Fallback: construct path from equation name.
        equation: str = str(eq_cfg.get("equation", "unknown"))
        return os.path.join("data", "datasets", f"{equation}_{max_n}.pt")

    # Standard case: single save_path.
    return str(data_cfg.get("save_path", "data/datasets/dataset.pt"))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(
    eq_cfg: dict,
    force_regenerate: bool = False,
) -> None:
    """Generates the dataset for the specified equation.

    Dispatches to the appropriate generator class based on eq_cfg['equation'].
    Skips generation if the dataset file already exists and force_regenerate
    is False.

    For equations with multiple sample sizes (pde4, pde2_zoned), generates
    one dataset per size in n_samples_list.

    Args:
        eq_cfg: Resolved equation configuration dict. Must contain 'equation'
                and 'data' sub-dict with 'save_path' (or 'n_samples_list' +
                'save_path_{N}' for multi-size equations).
        force_regenerate: If True, regenerate even if the file exists.
                          Default False.
    """
    equation: str = str(eq_cfg.get("equation", "pde1")).lower()
    data_cfg: dict = eq_cfg.get("data", {})

    print(f"\n[main] === Generating dataset for equation='{equation}' ===")

    # ------------------------------------------------------------------
    # Handle equations with multiple sample sizes.
    # ------------------------------------------------------------------
    n_samples_list: Optional[List[int]] = data_cfg.get("n_samples_list", None)

    if n_samples_list and equation in ("pde4", "pde2"):
        # Generate one dataset per sample size.
        for n_samples in n_samples_list:
            # Determine save path for this sample size.
            path_key: str = f"save_path_{n_samples}"
            if path_key in data_cfg:
                save_path: str = str(data_cfg[path_key])
            else:
                save_path = os.path.join(
                    "data", "datasets", f"{equation}_{n_samples}.pt"
                )

            if not force_regenerate and os.path.exists(save_path):
                print(
                    f"[main] Dataset already exists: '{save_path}'. "
                    f"Skipping (use --force_regenerate to override)."
                )
                continue

            # Create a modified eq_cfg for this sample size.
            eq_cfg_n: dict = copy.deepcopy(eq_cfg)
            eq_cfg_n["data"]["n_samples"] = n_samples
            eq_cfg_n["data"]["save_path"] = save_path

            _run_generator(equation, eq_cfg_n)

        return

    # ------------------------------------------------------------------
    # Standard case: single dataset.
    # ------------------------------------------------------------------
    save_path = str(data_cfg.get("save_path", f"data/datasets/{equation}.pt"))

    if not force_regenerate and os.path.exists(save_path):
        print(
            f"[main] Dataset already exists: '{save_path}'. "
            f"Skipping (use --force_regenerate to override)."
        )
        return

    _run_generator(equation, eq_cfg)


def _run_generator(equation: str, eq_cfg: dict) -> None:
    """Instantiates and runs the appropriate generator for the given equation.

    Args:
        equation: Equation identifier string.
        eq_cfg: Resolved equation configuration dict (may be modified for
                specific sample sizes).
    """
    # Build the master-style config that generators expect.
    # Generators access their config via cfg[equation_key], so we wrap
    # eq_cfg in a dict keyed by the equation name.
    generator_cfg: dict = {equation: eq_cfg, **eq_cfg}

    # Dispatch to the appropriate generator class.
    if equation == "ode1":
        generator = ODE1Generator(generator_cfg)
    elif equation == "ode2":
        generator = ODE2Generator(generator_cfg)
    elif equation == "pde1":
        generator = PDE1Generator(generator_cfg)
    elif equation in ("pde2", "pde2_zoned"):
        zoned: bool = bool(eq_cfg.get("zoned", equation == "pde2_zoned"))
        n_samples: int = int(eq_cfg.get("data", {}).get("n_samples", 2000))
        generator = PDE2Generator(generator_cfg, zoned=zoned, n_samples_override=n_samples)
    elif equation == "pde3":
        generator = PDE3Generator(generator_cfg)
    elif equation == "pde4":
        generator = PDE4Generator(generator_cfg)
    else:
        raise ValueError(
            f"_run_generator: unknown equation '{equation}'. "
            f"Must be one of {VALID_EQUATIONS}."
        )

    generator.generate_all()


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _build_model_for_equation(eq_cfg: dict, variant: str) -> torch.nn.Module:
    """Builds an FNO model for the given equation and variant.

    Injects the variant tag into eq_cfg before calling build_model so that
    the factory can validate it. The variant is also stamped onto the returned
    model instance.

    Args:
        eq_cfg: Resolved equation configuration dict. Must contain 'equation',
                'n_params', 'model', 'discretization', 'params'.
        variant: Model variant string. One of VALID_VARIANTS.

    Returns:
        An FNO model instance with model.variant set to variant.
    """
    # Inject variant into eq_cfg for the factory.
    eq_cfg_with_variant: dict = copy.deepcopy(eq_cfg)
    eq_cfg_with_variant["variant"] = variant

    model: torch.nn.Module = build_model(eq_cfg_with_variant)
    model.variant = variant  # type: ignore[attr-defined]

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    eq_cfg: dict,
    variant: str,
    force_retrain: bool = False,
    n_samples_override: Optional[int] = None,
    dataset_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Trains a model for the given equation and variant.

    Loads the pre-generated dataset, builds the FNO model, initializes the
    Trainer, and runs the training loop. Saves the best checkpoint to
    cfg['checkpoint_dir']/{equation}_{variant}.pt.

    Args:
        eq_cfg: Resolved equation configuration dict.
        variant: Model variant string. One of VALID_VARIANTS.
        force_retrain: If True, re-train even if a checkpoint exists.
                       Default False.
        n_samples_override: If provided, use only the first n_samples_override
                            samples from the training dataset. Used by the
                            data scaling experiment.
        dataset_path_override: If provided, use this path instead of the
                               default dataset path. Used for FD comparison.

    Returns:
        Training history dict with keys 'train_loss', 'val_loss', etc.
        Returns an empty dict if training was skipped (checkpoint exists).
    """
    equation: str = str(eq_cfg.get("equation", "pde1")).lower()
    run_name: str = f"{equation}_{variant}"

    print(f"\n[main] === Training: equation='{equation}', variant='{variant}' ===")

    # ------------------------------------------------------------------
    # Check if checkpoint already exists.
    # ------------------------------------------------------------------
    ckpt_path: str = _checkpoint_path(eq_cfg, variant)
    if not force_retrain and os.path.exists(ckpt_path):
        print(
            f"[main] Checkpoint already exists: '{ckpt_path}'. "
            f"Skipping training (use --force_retrain to override)."
        )
        return {}

    # ------------------------------------------------------------------
    # Set random seed for reproducibility.
    # ------------------------------------------------------------------
    seed: int = int(eq_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ------------------------------------------------------------------
    # Resolve dataset path.
    # ------------------------------------------------------------------
    if dataset_path_override is not None:
        data_path: str = dataset_path_override
    else:
        data_path = _dataset_path(eq_cfg)

    if not os.path.exists(data_path):
        print(
            f"[main] ERROR: Dataset not found at '{data_path}'. "
            f"Run with --mode generate first."
        )
        return {}

    # ------------------------------------------------------------------
    # Determine whether to load Jacobians.
    # Only SC variants need Jacobians during training.
    # ------------------------------------------------------------------
    use_jacobian: bool = variant in ("sc_fno", "sc_fno_pinn")

    # ------------------------------------------------------------------
    # Load datasets.
    # ------------------------------------------------------------------
    try:
        train_ds: SCFNODataset = SCFNODataset(
            data_path=data_path,
            split="train",
            use_jacobian=use_jacobian,
        )
        val_ds: SCFNODataset = SCFNODataset(
            data_path=data_path,
            split="val",
            use_jacobian=use_jacobian,
        )
    except Exception as exc:
        print(f"[main] ERROR loading dataset: {exc}")
        return {}

    # ------------------------------------------------------------------
    # Apply n_samples_override for data scaling experiment.
    # ------------------------------------------------------------------
    if n_samples_override is not None and n_samples_override < len(train_ds):
        from torch.utils.data import Subset  # pylint: disable=import-outside-toplevel
        # Use the first n_samples_override samples (already sorted by param norm).
        subset_indices: List[int] = list(range(n_samples_override))
        train_ds_effective = Subset(train_ds, subset_indices)
        print(
            f"[main] Data scaling: using {n_samples_override} of "
            f"{len(train_ds)} training samples."
        )
    else:
        train_ds_effective = train_ds

    # ------------------------------------------------------------------
    # Create DataLoaders.
    # ------------------------------------------------------------------
    training_cfg: dict = eq_cfg.get("training", {})
    batch_size: int = int(training_cfg.get("batch_size", 4))

    train_loader: DataLoader = DataLoader(
        train_ds_effective,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader: DataLoader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Build model.
    # ------------------------------------------------------------------
    model: torch.nn.Module = _build_model_for_equation(eq_cfg, variant)

    # ------------------------------------------------------------------
    # Initialize Trainer and run training.
    # ------------------------------------------------------------------
    trainer: Trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=eq_cfg,
        run_name=run_name,
    )

    n_epochs: int = int(training_cfg.get("n_epochs", 500))
    history: Dict[str, Any] = trainer.train(n_epochs=n_epochs)

    # ------------------------------------------------------------------
    # Save final checkpoint with deterministic name.
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    trainer.save_checkpoint(best=False)

    # Also copy the best checkpoint to the deterministic path.
    best_ckpt_src: str = os.path.join(
        eq_cfg.get("checkpoint_dir", "outputs/checkpoints"),
        "best_model.pt",
    )
    if os.path.exists(best_ckpt_src):
        import shutil  # pylint: disable=import-outside-toplevel
        shutil.copy2(best_ckpt_src, ckpt_path)
        print(f"[main] Best checkpoint saved to: '{ckpt_path}'")

    # ------------------------------------------------------------------
    # Save training history as JSON.
    # ------------------------------------------------------------------
    results_dir: str = str(eq_cfg.get("results_dir", "outputs/results"))
    os.makedirs(results_dir, exist_ok=True)
    history_path: str = os.path.join(results_dir, f"{run_name}_training_history.json")
    _save_json(history, history_path)

    print(
        f"[main] Training complete: equation='{equation}', variant='{variant}' | "
        f"best_val_loss={trainer.best_val_loss:.6f}"
    )

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    eq_cfg: dict,
    variant: str,
    experiment: str,
    master_cfg: Optional[dict] = None,
    dataset_path_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluates a trained model for the given equation, variant, and experiment.

    Loads the checkpoint, builds the model, creates the test DataLoader, and
    dispatches to the appropriate Evaluator method.

    Args:
        eq_cfg: Resolved equation configuration dict.
        variant: Model variant string. One of VALID_VARIANTS.
        experiment: Evaluation experiment. One of VALID_EXPERIMENTS.
        master_cfg: The full master configuration dict. Required for 'scaling'
                    and 'fd_comparison' experiments that need cross-equation
                    settings. Default None.
        dataset_path_override: If provided, use this path instead of the
                               default dataset path. Used for FD comparison.

    Returns:
        Nested dict of evaluation results. Structure depends on experiment type.
        Returns empty dict if the checkpoint is not found.
    """
    equation: str = str(eq_cfg.get("equation", "pde1")).lower()
    run_name: str = f"{equation}_{variant}_{experiment}"

    print(
        f"\n[main] === Evaluating: equation='{equation}', "
        f"variant='{variant}', experiment='{experiment}' ==="
    )

    # ------------------------------------------------------------------
    # Check if checkpoint exists.
    # ------------------------------------------------------------------
    ckpt_path: str = _checkpoint_path(eq_cfg, variant)
    if not os.path.exists(ckpt_path):
        print(
            f"[main] WARNING: Checkpoint not found at '{ckpt_path}'. "
            f"Run training first (--mode train --variant {variant})."
        )
        return {}

    # ------------------------------------------------------------------
    # Resolve dataset path.
    # ------------------------------------------------------------------
    if dataset_path_override is not None:
        data_path: str = dataset_path_override
    else:
        data_path = _dataset_path(eq_cfg)

    if not os.path.exists(data_path):
        print(
            f"[main] ERROR: Dataset not found at '{data_path}'. "
            f"Run with --mode generate first."
        )
        return {}

    # ------------------------------------------------------------------
    # Build model and load checkpoint.
    # ------------------------------------------------------------------
    model: torch.nn.Module = _build_model_for_equation(eq_cfg, variant)

    try:
        checkpoint: dict = torch.load(
            ckpt_path,
            map_location=eq_cfg.get("device", "cpu"),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[main] Loaded checkpoint from '{ckpt_path}'.")
    except Exception as exc:
        print(f"[main] ERROR loading checkpoint: {exc}")
        return {}

    model.eval()

    # ------------------------------------------------------------------
    # Load test dataset (always with Jacobians for evaluation).
    # ------------------------------------------------------------------
    try:
        test_ds: SCFNODataset = SCFNODataset(
            data_path=data_path,
            split="test",
            use_jacobian=True,
        )
    except Exception as exc:
        print(f"[main] ERROR loading test dataset: {exc}")
        return {}

    training_cfg: dict = eq_cfg.get("training", {})
    batch_size: int = int(training_cfg.get("batch_size", 4))

    test_loader: DataLoader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Initialize Evaluator.
    # ------------------------------------------------------------------
    evaluator: Evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        cfg=eq_cfg,
    )

    # ------------------------------------------------------------------
    # Dispatch to the appropriate experiment.
    # ------------------------------------------------------------------
    results: Dict[str, Any] = {}

    if experiment == "forward":
        results = evaluator.evaluate_forward()

    elif experiment == "perturbation":
        lambda_vals: List[float] = list(
            eq_cfg.get("perturbation", {}).get("lambda_values", [0.1, 0.2, 0.3, 0.4])
        )
        results = evaluator.evaluate_perturbation(lambda_vals=lambda_vals)

    elif experiment == "inversion":
        inverter: