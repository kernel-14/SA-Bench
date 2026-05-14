## main.py

"""
main.py – Orchestrator for reproducing the PEFT study.

Supports three experiment modes:
    vtab        – low‑shot VTAB‑1K (19 tasks)
    many_shot   – many‑shot datasets (CIFAR‑100, RESISC45, Clevr‑Distance)
    robustness  – distribution‑shift evaluation with CLIP and WiSE

Command‑line arguments:
    --config   path to config.yaml (required)
    --mode     one of {vtab, many_shot, robustness} (default vtab)
    --method   PEFT method name (e.g., lora); if not given, loops over all methods
    --task     specific VTAB task or many‑shot dataset name (optional)
    --tune     flag to enable hyperparameter tuning (otherwise loads saved best params)
"""

import argparse
import copy
import itertools
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project‑local modules (expected to be in PYTHONPATH)
from config import Config
from dataset_manager import DatasetManager
from model_builder import PEFTModel
from trainer import Trainer
from evaluation import Evaluator
from utils import set_seed, compute_trainable_params

# ------------------------------------------------------------------------ #
#  Logging setup
# ------------------------------------------------------------------------ #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------ #
#  VTAB task list (order as in paper, 19 tasks)
# ------------------------------------------------------------------------ #
VTAB_TASKS = [
    "caltech101", "cifar100", "dtd", "flowers102", "pets", "sun397",
    "svhn",  # Natural (7)
    "camelyon", "eurosat", "resisc45", "retinopathy",  # Specialized (4)
    "clevr_count", "clevr_distance", "dmlab", "kitti",
    "dsprite_ori", "smallnorb_azim", "smallnorb_ele",  # Structured (8)
]

MANY_SHOT_DATASETS = ["cifar100", "resisc45", "clevr_distance"]


# ------------------------------------------------------------------------ #
#  Helper: directory creation
# ------------------------------------------------------------------------ #
def ensure_dir(path: str) -> None:
    """Create directory if it doesn’t exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------------ #
#  HyperparameterTuner
# ------------------------------------------------------------------------ #
class HyperparamTuner:
    """
    Grid‑search hyperparameter tuner for a single PEFT method on a given dataset.

    It builds candidate hyperparameter combinations from:
        - learning rate values (list)
        - weight decay values (list)
        - method‑specific hyperparameters (Cartesian product provided by `method_hparam_combos`)
        - drop path rate values (list)
    Excludes any combination where the number of trainable parameters exceeds `param_cap`.

    The tuning is performed on a provided training split (train_loader) with a dedicated
    validation set (val_loader). The model with the highest validation accuracy is
    selected.
    """

    def __init__(
        self,
        config: Config,
        backbone: str,
        num_classes: int,
        method: str,
        method_hparam_combos: List[Dict[str, Any]],
        lr_values: List[float],
        wd_values: List[float],
        drop_path_values: List[float],
        param_cap: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tuning_epochs: int,
        device: torch.device,
    ) -> None:
        """
        Args:
            config:                 global experiment Config.
            backbone:               backbone name string.
            num_classes:            number of output classes.
            method:                 PEFT method key.
            method_hparam_combos:   list of dicts, each a method‑specific hyperparam set.
            lr_values:              list of learning rates to try.
            wd_values:              list of weight decays to try.
            drop_path_values:       list of drop path rates to try.
            param_cap:              maximum allowed trainable parameters (int).
            train_loader:           DataLoader for the hyperparameter‑tuning training set.
            val_loader:             DataLoader for the validation set.
            tuning_epochs:          number of epochs per trial.
            device:                 torch device.
        """
        self.config = config
        self.backbone = backbone
        self.num_classes = num_classes
        self.method = method
        self.method_hparam_combos = method_hparam_combos
        self.lr_values = lr_values
        self.wd_values = wd_values
        self.drop_path_values = drop_path_values
        self.param_cap = param_cap
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tuning_epochs = tuning_epochs
        self.device = device

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_val_acc: float = -1.0

    def _params_ok(self, model: PEFTModel) -> bool:
        """Check whether the model’s trainable parameters are within the cap."""
        trainable = compute_trainable_params(model)
        return trainable <= self.param_cap

    def _build_trainer(self, model: PEFTModel) -> Trainer:
        """Create a Trainer for a trial with the fixed val_loader."""
        # We create a small temporary config copy to pass the learning rate/wd to Trainer.
        trial_config = copy.deepcopy(self.config)
        # We'll set the training parameters directly in the copy's training.vtab (or generic).
        trial_config.training["learning_rate"] = model.lr  # will be set during trial
        trial_config.training["weight_decay"] = model.wd
        trial_config.training["epochs"] = self.tuning_epochs
        trial_config.training["drop_path_rate"] = model.drop_path_rate
        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=trial_config,
        )
        return trainer

    def grid_search(self) -> Dict[str, Any]:
        """
        Execute the grid search and return the best hyperparameter combination.
        """
        logger.info(f"Starting hyperparameter search for method '{self.method}' "
                     f"on {len(self.lr_values)} LR x {len(self.wd_values)} WD x "
                     f"{len(self.method_hparam_combos)} method combos x "
                     f"{len(self.drop_path_values)} drop path = "
                     f"{len(self.lr_values)*len(self.wd_values)*len(self.method_hparam_combos)*len(self.drop_path_values)} trials.")

        for lr in self.lr_values:
            for wd in self.wd_values:
                for mh in self.method_hparam_combos:
                    for dp in self.drop_path_values:
                        # Assemble full hyperparameter dict
                        full_params = copy.deepcopy(mh)
                        full_params["lr"] = lr
                        full_params["weight_decay"] = wd
                        full_params["drop_path_rate"] = dp

                        # Build model
                        try:
                            model = PEFTModel(
                                backbone_name=self.backbone,
                                num_classes=self.num_classes,
                                peft_method=self.method,
                                peft_hyperparams=full_params,
                                drop_path_rate=dp,  # PEFTModel may set it internally
                            )
                            model.lr = lr          # attach for Trainer convenience
                            model.wd = wd
                            model.drop_path_rate = dp
                        except Exception as e:
                            logger.warning(f"Failed to build model with {full_params}: {e}")
                            continue

                        # Check parameter cap
                        if not self._params_ok(model):
                            logger.debug(f"Skipping combination {full_params}: "
                                         f"trainable params {compute_trainable_params(model)} > cap {self.param_cap}")
                            del model
                            torch.cuda.empty_cache()
                            continue

                        # Train for one trial
                        trainer = self._build_trainer(model)
                        val_acc = trainer.train()   # returns best val accuracy
                        logger.info(f"Trial {full_params}: val accuracy = {val_acc:.2f}%")

                        if val_acc > self.best_val_acc:
                            self.best_val_acc = val_acc
                            self.best_params = full_params
                            logger.info(f"  New best! {self.best_val_acc:.2f}%")
                        # Cleanup to free GPU memory
                        del model
                        torch.cuda.empty_cache()

        if self.best_params is None:
            raise RuntimeError("No valid hyperparameter combination found (all exceeded param cap or failed).")

        logger.info(f"Best hyperparameters: {self.best_params} with accuracy {self.best_val_acc:.2f}%")
        return self.best_params


# ------------------------------------------------------------------------ #
#  Main experiment routines
# ------------------------------------------------------------------------ #
def run_vtab_experiment(args, cfg: Config) -> None:
    """Run VTAB‑1K evaluation for the specified method(s) and task(s)."""
    # Determine tasks to run
    if args.task:
        tasks = [args.task]
        if args.task not in VTAB_TASKS:
            raise ValueError(f"Unknown VTAB task '{args.task}'. Choose from {VTAB_TASKS}")
    else:
        tasks = VTAB_TASKS

    # Determine method(s)
    methods = [args.method] if args.method else list(cfg.peft_methods.keys())
    if not methods:
        logger.error("No PEFT methods specified. Use --method or define in config.")
        return

    # Prepare backbone info
    backbone_name = cfg.get_backbone_name()
    param_cap = int(cfg.backbone["param_total"] * cfg.training["vtab"]["param_cap_percent"])
    drop_path_values = cfg.training["vtab"]["drop_path_tuning"]
    lr_values = cfg.training["vtab"]["learning_rate"]
    wd_values = cfg.training["vtab"]["weight_decay"]
    tuning_epochs = cfg.training["vtab"]["epochs"]  # we use same epochs for tuning as full training? paper: 100 epochs.
    device = torch.device(cfg.misc.get("device", "cuda"))
    set_seed(cfg.misc["seed"])

    dataset_mgr = DatasetManager(cfg)

    # Per method results
    all_results: Dict[str, Dict[str, float]] = {}  # {method: {task: acc}}

    for method in methods:
        logger.info(f"=== Starting VTAB experiments for method: {method} ===")
        method_results = {}
        method_hparam_combos = cfg.get_method_hyperparam_combinations(method)

        # Best‑params persistence directory
        best_dir = os.path.join("best_params", "vtab", method)
        ensure_dir(best_dir)

        for task in tasks:
            logger.info(f"Task: {task}")

            # 1. Load data for tuning (train+val) and test
            tune_train_loader, val_loader = dataset_mgr.load_vtab(task, split="train")
            # test loader
            test_loader, _ = dataset_mgr.load_vtab(task, split="test")

            # 2. Determine best hyperparameters
            if args.tune:
                # Perform grid search
                # Need number of classes: from val_loader dataset
                num_classes = len(val_loader.dataset.dataset.classes)  # Subset -> .dataset.dataset.classes
                tuner = HyperparamTuner(
                    config=cfg,
                    backbone=backbone_name,
                    num_classes=num_classes,
                    method=method,
                    method_hparam_combos=method_hparam_combos,
                    lr_values=lr_values,
                    wd_values=wd_values,
                    drop_path_values=drop_path_values,
                    param_cap=param_cap,
                    train_loader=tune_train_loader,
                    val_loader=val_loader,
                    tuning_epochs=tuning_epochs,
                    device=device,
                )
                best_params = tuner.grid_search()
                # Save best params
                param_path = os.path.join(best_dir, f"{task}.json")
                with open(param_path, 'w') as f:
                    json.dump(best_params, f, indent=2)
            else:
                # Load saved best params
                param_path = os.path.join(best_dir, f"{task}.json")
                if not os.path.exists(param_path):
                    logger.error(f"Best params file not found: {param_path}. Run with --tune first.")
                    continue
                with open(param_path, 'r') as f:
                    best_params = json.load(f)

            # 3. Final training on full 1000 samples
            full_train_loader, _ = dataset_mgr.load_vtab(task, split="full")
            final_model = PEFTModel(
                backbone_name=backbone_name,
                num_classes=num_classes,
                peft_method=method,
                peft_hyperparams=best_params,
                drop_path_rate=best_params["drop_path_rate"],
            ).to(device)

            # Build a config copy with best LR/WD for Trainer
            trial_config = copy.deepcopy(cfg)
            trial_config.training["learning_rate"] = best_params["lr"]
            trial_config.training["weight_decay"] = best_params["weight_decay"]
            trial_config.training["epochs"] = cfg.training["vtab"]["epochs"]   # 100 epochs
            trial_config.training["drop_path_rate"] = best_params["drop_path_rate"]

            trainer = Trainer(final_model, full_train_loader, None, trial_config)
            trainer.train()   # no validation, only logs training loss

            # 4. Evaluate on test set
            evaluator = Evaluator(final_model, test_loader)
            test_acc = evaluator.evaluate_accuracy()
            logger.info(f"Task {task} final test accuracy: {test_acc:.2f}%")
            method_results[task] = test_acc

            # Cleanup
            del final_model, trainer, evaluator
            torch.cuda.empty_cache()

        all_results[method] = method_results

    # Save summary results
    results_dir = "results/vtab"
    ensure_dir(results_dir)
    out_file = os.path.join(results_dir, "summary.json")
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"VTAB results saved to {out_file}")


def run_many_shot_experiment(args, cfg: Config) -> None:
    """Run many‑shot evaluation for a specific dataset (or all if none given)."""
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = MANY_SHOT_DATASETS

    methods = [args.method] if args.method else list(cfg.peft_methods.keys())
    if not methods:
        logger.error("No PEFT methods specified. Use --method or define in config.")
        return

    backbone_name = cfg.get_backbone_name()
    param_cap = int(cfg.backbone["param_total"] * cfg.training["many_shot"].get("param_cap_percent", 0.05))  # paper says 2-5% param
    lr_values = cfg.training["many_shot"]["learning_rate"]
    wd_values = cfg.training["many_shot"]["weight_decay"]
    drop_path_values = cfg.training["many_shot"].get("drop_path_tuning", [0.0])  # Many-shot probably uses 0.0? Not explicitly mentioned.
    tuning_epochs = cfg.training["many_shot"]["epochs"]  # 40
    device = torch.device(cfg.misc.get("device", "cuda"))
    set_seed(cfg.misc["seed"])

    dataset_mgr = DatasetManager(cfg)

    for method in methods:
        logger.info(f"=== Starting many‑shot experiments for method: {method} ===")
        method_hparam_combos = cfg.get_method_hyperparam_combinations(method)
        best_dir = os.path.join("best_params", "many_shot", method)
        ensure_dir(best_dir)

        for dset in datasets:
            logger.info(f"Dataset: {dset}")

            # Load tuning splits
            tune_train_loader, val_loader = dataset_mgr.load_many_shot(dset, split="train")
            # Test loader
            test_loader, _ = dataset_mgr.load_many_shot(dset, split="test")

            # Number of classes
            if hasattr(tune_train_loader.dataset, 'classes'):
                num_classes = len(tune_train_loader.dataset.classes)
            elif hasattr(tune_train_loader.dataset, 'class_to_idx'):
                num_classes = len(tune_train_loader.dataset.class_to_idx)
            else:
                # fallback: deduce from the dataset
                # For CIFAR100 it's 100, for RESISC45 45, for Clevr-Distance 6.
                if dset == "cifar100": num_classes = 100
                elif dset == "resisc45": num_classes = 45
                elif dset == "clevr_distance": num_classes = 6
                else: raise ValueError("Cannot determine num_classes")

            # Hyperparameter tuning (optional)
            if args.tune:
                tuner = HyperparamTuner(
                    config=cfg,
                    backbone=backbone_name,
                    num_classes=num_classes,
                    method=method,
                    method_hparam_combos=method_hparam_combos,
                    lr_values=lr_values,
                    wd_values=wd_values,
                    drop_path_values=drop_path_values,
                    param_cap=param_cap,
                    train_loader=tune_train_loader,
                    val_loader=val_loader,
                    tuning_epochs=tuning_epochs,
                    device=device,
                )
                best_params = tuner.grid_search()
                # Save
                param_path = os.path.join(best_dir, f"{dset}.json")
                with open(param_path, 'w') as f:
                    json.dump(best_params, f, indent=2)
            else:
                param_path = os.path.join(best_dir, f"{dset}.json")
                if not os.path.exists(param_path):
                    logger.error(f"Best params file not found: {param_path}. Run with --tune first.")
                    continue
                with open(param_path, 'r') as f:
                    best_params = json.load(f)

            # Final training on full dataset
            full_train_loader, _ = dataset_mgr.load_many_shot(dset, split="full")
            final_model = PEFTModel(
                backbone_name=backbone_name,
                num_classes=num_classes,
                peft_method=method,
                peft_hyperparams=best_params,
                drop_path_rate=best_params["drop_path_rate"],
            ).to(device)

            trial_config = copy.deepcopy(cfg)
            trial_config.training["learning_rate"] = best_params["lr"]
            trial_config.training["weight_decay"] = best_params["weight_decay"]
            trial_config.training["epochs"] = cfg.training["many_shot"]["epochs"]
            trial_config.training["drop_path_rate"] = best_params["drop_path_rate"]

            trainer = Trainer(final_model, full_train_loader, None, trial_config)
            trainer.train()
            evaluator = Evaluator(final_model, test_loader)
            test_acc = evaluator.evaluate_accuracy()
            logger.info(f"Dataset {dset} final test accuracy: {test_acc:.2f}%")

            # Save per‑dataset result
            result_dir = os.path.join("results", "many_shot", method)
            ensure_dir(result_dir)
            with open(os.path.join(result_dir, f"{dset}.txt"), 'w') as f:
                f.write(str(test_acc))

            del final_model, trainer, evaluator
            torch.cuda.empty_cache()


def run_robustness_experiment(args, cfg: Config) -> None:
    """
    Robustness experiment with CLIP ViT‑B/16.
    Fine‑tunes on 100‑shot ImageNet and evaluates target and distribution shift accuracy,
    then performs WiSE parameter interpolation.
    """
    # In the paper, they didn't tune PEFT hyperparameters for robustness separately;
    # instead they reused best settings from VTAB. We'll load pre‑saved best params
    # for the given method (expects a VTAB calibration on a standard task like caltech101).
    method = args.method
    if method is None:
        logger.error("Robustness mode requires a single --method (e.g., lora)")
        return

    # Determine a reference task to retrieve best params. The paper doesn't specify,
    # so we use "caltech101" as a representative natural image task.
    if args.tune:
        logger.warning("Tuning not implemented for robustness; using saved best params for caltech101.")
    param_path = os.path.join("best_params", "vtab", method, "caltech101.json")
    if not os.path.exists(param_path):
        logger.error(f"Best params for method {method} not found at {param_path}. Please run VTAB tuning first (caltech101).")
        return
    with open(param_path, 'r') as f:
        best_params = json.load(f)

    # Override with robustness‑specific training hyperparameters (learning rate, weight decay)
    train_config = cfg.training["robustness"]
    lr = train_config["learning_rate"]
    wd = train_config["weight_decay"]
    epochs = train_config["epochs"]
    alphas = train_config["wise_alphas"]
    device = torch.device(cfg.misc.get("device", "cuda"))
    set_seed(cfg.misc["seed"])

    # Load data
    dataset_mgr = DatasetManager(cfg)
    (target_train_loader, target_val_loader, shift_loaders) = dataset_mgr.load_distribution_shift()
    # shift_loaders order: V2, R, Sketch, A

    # Build model (CLIP ViT‑B/16 backbone with linear head initialised from text embeddings)
    # PEFTModel for robustness expects backbone_name = "ViT-B/16" (as used in config)
    # We'll set num_classes = 1000
    backbone_name = "ViT-B/16"  # must match how model_builder interprets it (CLIP).
    model = PEFTModel(
        backbone_name=backbone_name,
        num_classes=1000,
        peft_method=method,
        peft_hyperparams=best_params,
        drop_path_rate=0.0,   # no mention of drop path for robustness
    ).to(device)

    # Override some training parameters in the config for Trainer
    trial_config = copy.deepcopy(cfg)
    trial_config.training["learning_rate"] = lr
    trial_config.training["weight_decay"] = wd
    trial_config.training["epochs"] = epochs
    trial_config.training["drop_path_rate"] = 0.0

    # Trainer: we pass target_val_loader to allow validation logging
    trainer = Trainer(model, target_train_loader, target_val_loader, trial_config)
    trainer.train()

    # --- Evaluation before WiSE ---
    evaluator = Evaluator(model, target_val_loader)
    target_acc = evaluator.evaluate_accuracy()
    shift_accs = evaluator.evaluate_distribution_shift(shift_loaders)
    avg_shift_acc = np.mean(shift_accs)
    logger.info(f"Before WiSE: Target acc {target_acc:.2f}, Avg shift acc {avg_shift_acc:.2f}")

    # --- WiSE sweep ---
    wise_results = {}
    for alpha in alphas:
        model.apply_wise(alpha)
        t_acc = evaluator.evaluate_accuracy()
        s_accs = evaluator.evaluate_distribution_shift(shift_loaders)
        avg_s = np.mean(s_accs)
        wise_results[alpha] = {"target": t_acc, "avg_shift": avg_s}
        logger.info(f"WiSE α={alpha:.1f}: target={t_acc:.2f}, avg_shift={avg_s:.2f}")

    # Reset model to fine‑tuned state (alpha=1.0)
    model.apply_wise(1.0)

    # Save results
    result_dir = os.path.join("results", "robustness", method)
    ensure_dir(result_dir)
    with open(os.path.join(result_dir, "wise_curve.json"), 'w') as f:
        json.dump(wise_results, f, indent=2)
    with open(os.path.join(result_dir, "summary.txt"), 'w') as f:
        f.write(f"Target accuracy (α=1): {target_acc:.2f}\n")
        f.write(f"Average shift accuracy (α=1): {avg_shift_acc:.2f}\n")

    logger.info("Robustness evaluation complete.")


# ======================================================================== #
#  Command‑line interface and entry point
# ======================================================================== #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PEFT Unifying Study Reproduction")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--mode", type=str, default="vtab",
                        choices=["vtab", "many_shot", "robustness"],
                        help="Experiment mode")
    parser.add_argument("--method", type=str, default=None,
                        help="PEFT method to evaluate (if not given, loops over all methods)")
    parser.add_argument("--task", type=str, default=None,
                        help="For VTAB: a specific task name; for many_shot: leave None (use --dataset)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="For many_shot: dataset name (cifar100, resisc45, clevr_distance)")
    parser.add_argument("--tune", action="store_true",
                        help="Perform hyperparameter tuning (otherwise expects saved best_params)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Config file {args.config} not found.")
        sys.exit(1)
    cfg = Config.from_file(args.config)

    # Ensure necessary directories exist
    os.makedirs("best_params", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Set seed at the very beginning
    set_seed(cfg.misc["seed"])

    if args.mode == "vtab":
        run_vtab_experiment(args, cfg)
    elif args.mode == "many_shot":
        run_many_shot_experiment(args, cfg)
    elif args.mode == "robustness":
        run_robustness_experiment(args, cfg)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
