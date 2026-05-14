# main.py
# ============================================================================
# Main entry point for reproducing the SC‑FNO experiments.
#
# This script loads a YAML configuration file, selects the appropriate
# differentiable solver, generates (or loads) the required datasets,
# instantiates the FNO model, and runs training, evaluation, parameter
# inversion, and optional data‑volume studies in accordance with the
# paper's methodology.  All experiment‑specific choices are driven by
# the configuration file, making the script fully self‑contained and
# reproducible.
# ============================================================================

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project‑local imports
# ---------------------------------------------------------------------------
from config import Config
from data_generator import DataGenerator
from dataset import PDEDataset
from evaluation import Evaluator
from inversion import Inversion
from models.fno import FNO
from solver import (
    AllenCahnSolver,
    BurgersSolver,
    BurgersZonedSolver,
    DampedWaveSolver,
    DuffingSolver,
    FiniteDifferenceSolver,
    HarmonicOscillatorSolver,
    NavierStokesSolver,
    Solver,
)
from training import SC_NO_Trainer
from utils import set_seed

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: instantiate the appropriate solver
# ---------------------------------------------------------------------------
def _build_solver(config: Config) -> Solver:
    """Create a solver instance based on the equation and data parameters.

    Args:
        config: Global configuration object.

    Returns:
        A concrete Solver instance, possibly wrapped in FiniteDifferenceSolver
        if the ``gradient_method`` is ``"FD"``.
    """
    eq_name: str = config.equation
    eq_cfg: Dict[str, Any] = config.sol_params
    data_cfg: Dict[str, Any] = config.data_params

    solver_method = data_cfg.get("solver", "dopri5")
    rtol = data_cfg.get("rtol", 1e-5)
    atol = data_cfg.get("atol", 1e-7)

    base_solver: Solver
    if eq_name == "ode1":
        base_solver = HarmonicOscillatorSolver(
            rtol=rtol, atol=atol, method=solver_method
        )
    elif eq_name == "ode2":
        base_solver = DuffingSolver(rtol=rtol, atol=atol, method=solver_method)
    elif eq_name == "pde1":
        base_solver = DampedWaveSolver(rtol=rtol, atol=atol, method=solver_method)
    elif eq_name == "pde2":
        base_solver = BurgersSolver(rtol=rtol, atol=atol, method=solver_method)
    elif eq_name == "pde2_zoned":
        num_zones: int = eq_cfg["num_zones"]
        base_solver = BurgersZonedSolver(
            num_zones=num_zones, rtol=rtol, atol=atol, method=solver_method
        )
    elif eq_name == "pde3":
        Re = eq_cfg.get("Re", 1000.0)
        base_solver = NavierStokesSolver(Re=Re, rtol=rtol, atol=atol, method=solver_method)
    elif eq_name == "pde4":
        base_solver = AllenCahnSolver(rtol=rtol, atol=atol, method=solver_method)
    else:
        raise ValueError(f"Unsupported equation '{eq_name}'.")

    # Optional finite‑difference wrapper
    if data_cfg.get("gradient_method", "AD") == "FD":
        eps = data_cfg.get("fd_epsilon", 1e-4)
        base_solver = FiniteDifferenceSolver(base_solver, eps=eps)

    return base_solver

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main(config_path: str = "config.yaml") -> None:
    """Orchestrate the reproduction pipeline.

    1. Load configuration.
    2. Set randomness and device.
    3. Build solver.
    4. Generate / load datasets.
    5. Create datasets (train, val, test, perturbed).
    6. Optionally run data‑volume experiments.
    7. Otherwise, train a single model.
    8. Evaluate on test (and perturbed) data.
    9. If enabled, run parameter inversion.
    10. Save all quantitative results in the output directory.
    """
    # -----------------------------------------------------------------------
    # 1. Load configuration
    # -----------------------------------------------------------------------
    logger.info(f"Loading configuration from '{config_path}'")
    cfg = Config.from_yaml(config_path)

    # -----------------------------------------------------------------------
    # 2. Basic environment setup
    # -----------------------------------------------------------------------
    out_dir = Path(cfg.global_params["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.global_params["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.global_params["seed"])
    device = torch.device(cfg.global_params["device"])
    logger.info(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # 3. Build solver
    # -----------------------------------------------------------------------
    solver = _build_solver(cfg)
    logger.info(f"Solver: {solver.__class__.__name__}")

    # -----------------------------------------------------------------------
    # 4. Data generation / loading
    # -----------------------------------------------------------------------
    eq_name = cfg.equation
    data_file = data_dir / f"{eq_name}_data.h5"
    perturbed_file = data_dir / f"{eq_name}_perturbed_test.h5"
    perturbed_enabled = cfg.data_params.get("perturbed_test", False)

    # If any required data file is missing, generate it.
    if not data_file.exists() or (perturbed_enabled and not perturbed_file.exists()):
        logger.info("Generating missing datasets...")
        dgen = DataGenerator(solver, cfg)
        dgen.save()   # writes to data_dir using the naming scheme of DataGenerator

    # -----------------------------------------------------------------------
    # 5. Create PDEDataset objects
    # -----------------------------------------------------------------------
    logger.info("Creating dataset splits.")
    train_dataset = PDEDataset(cfg, split="train")
    val_dataset = PDEDataset(cfg, split="val")
    test_dataset = PDEDataset(cfg, split="test")

    perturbed_test_dataset: Optional[PDEDataset] = None
    if perturbed_enabled:
        perturbed_test_dataset = PDEDataset(
            cfg,
            split="test",
            file_path=str(perturbed_file),
        )

    # -----------------------------------------------------------------------
    # 6. Data‑volume experiments (optional)
    # -----------------------------------------------------------------------
    data_vol = cfg.data_volume_params
    if data_vol and data_vol.get("enabled", False):
        logger.info("Starting data‑volume experiments.")
        training_sizes = data_vol["training_sizes"]
        results: List[Dict[str, Any]] = []

        for n_train in training_sizes:
            logger.info(f"Training on {n_train} samples...")
            # -------- fresh model and limited training set -----------------
            model = FNO(cfg)
            train_sub = PDEDataset(cfg, split="train", n_train_samples=n_train)

            # -------- trainer (val set is created inside) -------------------
            trainer = SC_NO_Trainer(model, train_sub, cfg)
            train_hist, val_hist = trainer.run_training()

            # -------- evaluation on full test sets --------------------------
            evaluator = Evaluator(model, test_dataset, cfg)
            test_metrics = evaluator.compute_metrics()
            metrics_dict = {
                "train_size": n_train,
                "test_u_rel_l2": test_metrics["u_relative_l2"],
                "test_u_r2": test_metrics["u_r2"],
                "test_avg_dp_rel_l2": test_metrics["avg_dp_relative_l2"],
                "test_avg_dp_r2": test_metrics["avg_dp_r2"],
            }

            if perturbed_test_dataset is not None:
                evaluator_pert = Evaluator(model, perturbed_test_dataset, cfg)
                pert_metrics = evaluator_pert.compute_metrics()
                metrics_dict["perturbed_u_rel_l2"] = pert_metrics["u_relative_l2"]
                metrics_dict["perturbed_u_r2"] = pert_metrics["u_r2"]
                metrics_dict["perturbed_avg_dp_rel_l2"] = pert_metrics["avg_dp_relative_l2"]
                metrics_dict["perturbed_avg_dp_r2"] = pert_metrics["avg_dp_r2"]

            results.append(metrics_dict)

        # -------- Save summary --------------------------------------------------
        results_path = out_dir / "data_volume_metrics.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Data‑volume results saved to {results_path}")

        # Nothing else to do after this experiment; return early.
        logger.info("Data‑volume experiments complete.  Exiting.")
        return

    # -----------------------------------------------------------------------
    # 7. Standard single‑run training
    # -----------------------------------------------------------------------
    logger.info("Starting standard training.")
    model = FNO(cfg)
    trainer = SC_NO_Trainer(model, train_dataset, cfg)
    train_hist, val_hist = trainer.run_training()
    logger.info("Training finished.")

    # -----------------------------------------------------------------------
    # 8. Evaluation on test set(s)
    # -----------------------------------------------------------------------
    logger.info("Evaluating on original test set.")
    evaluator = Evaluator(model, test_dataset, cfg)
    test_metrics = evaluator.compute_metrics()
    logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")

    perturbed_metrics: Optional[Dict[str, float]] = None
    if perturbed_test_dataset is not None:
        logger.info("Evaluating on perturbed test set.")
        evaluator_pert = Evaluator(model, perturbed_test_dataset, cfg)
        perturbed_metrics = evaluator_pert.compute_metrics()
        logger.info(f"Perturbed test metrics: {json.dumps(perturbed_metrics, indent=2)}")

    # Save combined metrics
    metrics_summary: Dict[str, Any] = {
        "test": test_metrics,
        "perturbed": perturbed_metrics,
    }
    metrics_path = out_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Evaluation metrics saved to {metrics_path}")

    # (Optional) sample plots
    if cfg.eval_params.get("save_plots", False):
        evaluator.plot_sample(sample_idx=0, save_path=str(out_dir / "sample_test"))
        if perturbed_test_dataset is not None:
            evaluator_pert.plot_sample(
                sample_idx=0,
                save_path=str(out_dir / "sample_perturbed_test")
            )

    # -----------------------------------------------------------------------
    # 9. Parameter inversion (optional)
    # -----------------------------------------------------------------------
    inv_cfg = cfg.inversion_params
    if inv_cfg.get("enabled", False):
        logger.info("Starting parameter inversion.")
        # The ground‑trutog solver (non‑FD) is the base solver extracted earlier.
        # This ensures exact Jacobians from the differentiable solver.
        ground_solver = solver
        if isinstance(ground_solver, FiniteDifferenceSolver):
            # Invert to the original base solver inside
            ground_solver = ground_solver.base_solver   # type: ignore[attr-defined]

        inv = Inversion(model, ground_solver, cfg)
        inv_results = inv.evaluate_inversion(
            test_dataset,
            num_experiments=inv_cfg.get("num_experiments", 20),
        )
        inv_path = out_dir / "inversion_metrics.json"
        with open(inv_path, "w") as f:
            json.dump(inv_results, f, indent=2)
        logger.info(f"Inversion results saved to {inv_path}")

    logger.info("==== Pipeline completed successfully. ====")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce SC‑FNO experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    args = parser.parse_args()
    main(args.config)
