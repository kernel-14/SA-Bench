# main.py
"""
main.py — Orchestrator for "Towards Universal Neural Operators through
Multiphysics Pretraining" reproduction.

Runs all transfer‑learning experiments (Tables 1 & 2) across all neural
operator architectures (FNO, Mamba‑FNO, Perceiver IO, Swin‑v2, CoDA‑NO),
performing pretraining (multi‑physics), fine‑tuning, and scratch baselines.
All hyperparameters are read from config.yaml via the Config class.
"""

import argparse
import copy
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Local project imports (assuming all modules are in the same directory)
from config import Config
from data_utils import DataUtils, Dataset, MultiPhysicsLoader
from models import ModelBase
from trainer import Trainer
from evaluator import Evaluator

# ----------------------------------------------------------------------
# Reproducibility seeds
# ----------------------------------------------------------------------
RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set deterministic seeds for all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Additional: deterministic backend (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# Experiment definition helper – returns list of experiment descriptors
# ----------------------------------------------------------------------
def build_experiment_list(config: Config) -> List[Dict[str, Any]]:
    """
    Construct a list containing all transfer‑learning experiments required
    to fill Tables 1 and 2.
    Each dictionary specifies the PDE, parameter values for pretraining
    and fine‑tuning, the number of channels, and spatial dimension.
    """
    data_cfg = config.data_params

    experiments = [
        # ------------------------------------------------------------------
        # Table 1 – out‑of‑sample parameter values
        # ------------------------------------------------------------------
        {
            "id": "burgers_out_of_sample",
            "table": 1,
            "pde": "burgers",
            "dim": 1,
            "input_channels": 1,
            "output_channels": 1,
            "pretrain_params": {"nu_list": data_cfg["burgers"]["nu_pretrain"]},
            "finetune_params": {"nu_list": [data_cfg["burgers"]["nu_finetune"]]},
            "problem_prefix": "burgers_nu",
        },
        {
            "id": "grayscott_out_of_sample",
            "table": 1,
            "pde": "grayscott",
            "dim": 2,
            "input_channels": 2,
            "output_channels": 2,
            "pretrain_params": {
                "params_list": [
                    (f, k)
                    for f, k in zip(
                        data_cfg["grayscott"]["F_pretrain"],
                        data_cfg["grayscott"]["k_pretrain"],
                    )
                ]
            },
            "finetune_params": {
                "params_list": [
                    (
                        data_cfg["grayscott"]["F_finetune"],
                        data_cfg["grayscott"]["k_finetune"],
                    )
                ]
            },
            "problem_prefix": "grayscott",
        },
        {
            "id": "navierstokes_out_of_sample",
            "table": 1,
            "pde": "navierstokes",
            "dim": 2,
            "input_channels": 1,
            "output_channels": 1,
            "pretrain_params": {"Re_list": data_cfg["navierstokes"]["Re_pretrain"]},
            "finetune_params": {"Re_list": [data_cfg["navierstokes"]["Re_finetune"]]},
            "problem_prefix": "ns_Re",
        },
        # ------------------------------------------------------------------
        # Table 2 – input function set extension & multi‑physics learning
        # ------------------------------------------------------------------
        {
            "id": "heat_extension",
            "table": 2,
            "pde": "heat",
            "dim": 2,
            "pretrain_params": {"include_advection": False},
            "finetune_params": {"include_advection": True},
            "input_channels_pretrain": 1,
            "input_channels_finetune": 3,
            "output_channels": 1,
            "problem_prefix": "heat",
        },
        {
            "id": "reactdiff_extension",
            "table": 2,
            "pde": "reactdiff",
            "dim": 2,
            "pretrain_params": {"include_advection": False},
            "finetune_params": {"include_advection": True},
            "input_channels_pretrain": 2,
            "input_channels_finetune": 4,
            "output_channels": 2,
            "problem_prefix": "reactdiff",
        },
        {
            "id": "pdebench_transfer",
            "table": 2,
            "pde": "pdebench",
            "dim": 2,
            "pretrain_tasks": data_cfg["pdebench_pretrain_tasks"],
            "finetune_task": data_cfg["pdebench_finetune_task"],
            # Channels will be inferred from loaded data.
            "problem_prefix": "pdebench",
        },
    ]
    return experiments


# ----------------------------------------------------------------------
# Dataset preparation for a single experiment
# ----------------------------------------------------------------------
def prepare_datasets(
    exp_cfg: Dict[str, Any],
    config: Config,
) -> Tuple[
    List[Tuple[str, Dataset, Dataset, Dataset]],  # pretrain: (name, train, val, test)
    Tuple[str, Dataset, Dataset, Dataset],         # finetune: (name, train, val, test)
]:
    """
    Generate (or load from PDEBench) all datasets required by an experiment.
    Returns pretrain entries (each a tuple of name, train, val, test) and
    the finetune entry (name, train, val, test). The test splits of pretrain
    datasets are kept for recording validation metrics but are otherwise unused.
    """
    data_utils = DataUtils()
    data_cfg = config.data_params
    pde = exp_cfg["pde"]
    dim = exp_cfg["dim"]
    grid_size = data_cfg["grid_size_1d"] if dim == 1 else data_cfg["grid_size_2d"]
    n_train = data_cfg["n_train_samples"]
    n_val = data_cfg["n_val_samples"]
    n_test = data_cfg["n_test_samples"]
    n_total = n_train + n_val + n_test

    # Helper to split a raw Dataset
    def split_ds(raw: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        return DataUtils.split_dataset(raw, ratios=(n_train / n_total, n_val / n_total, n_test / n_total))

    # -------------------------------- Pretrain datasets --------------------------------
    pretrain_entries = []
    if pde == "pdebench":
        # Load PDEBench tasks
        for task_name in exp_cfg["pretrain_tasks"]:
            raw = DataUtils.load_pdebench(task_name)
            train_ds, val_ds, test_ds = split_ds(raw)
            pretrain_entries.append((task_name, train_ds, val_ds, test_ds))
    else:
        # Custom PDE solver
        param_list = _param_list_from_cfg(exp_cfg, "pretrain")
        if pde == "burgers":
            for nu in param_list:
                raw = DataUtils.generate_burgers([nu], n_total, grid_size)
                train, val, test = split_ds(raw)
                name = f"{exp_cfg['problem_prefix']}{nu}"
                pretrain_entries.append((name, train, val, test))
        elif pde == "grayscott":
            for (F, k) in param_list:
                raw = DataUtils.generate_grayscott([(F, k)], n_total, grid_size)
                train, val, test = split_ds(raw)
                name = f"{exp_cfg['problem_prefix']}_F{F}_k{k}"
                pretrain_entries.append((name, train, val, test))
        elif pde == "navierstokes":
            for Re in param_list:
                raw = DataUtils.generate_navierstokes([Re], n_total, grid_size)
                train, val, test = split_ds(raw)
                name = f"{exp_cfg['problem_prefix']}{Re}"
                pretrain_entries.append((name, train, val, test))
        elif pde == "heat":
            raw = DataUtils.generate_heat(
                n_total, grid_size, include_advection=exp_cfg["pretrain_params"].get("include_advection", False)
            )
            train, val, test = split_ds(raw)
            name = f"{exp_cfg['problem_prefix']}_base"
            pretrain_entries.append((name, train, val, test))
        elif pde == "reactdiff":
            raw = DataUtils.generate_reactdiff(
                n_total, grid_size, include_advection=exp_cfg["pretrain_params"].get("include_advection", False)
            )
            train, val, test = split_ds(raw)
            name = f"{exp_cfg['problem_prefix']}_base"
            pretrain_entries.append((name, train, val, test))
        else:
            raise ValueError(f"Unknown PDE type for pretraining: {pde}")

    # -------------------------------- Finetune dataset --------------------------------
    if pde == "pdebench":
        raw_ft = DataUtils.load_pdebench(exp_cfg["finetune_task"])
        ft_train, ft_val, ft_test = split_ds(raw_ft)
        ft_name = exp_cfg["finetune_task"]
    else:
        param_ft = _param_list_from_cfg(exp_cfg, "finetune")
        if pde == "burgers":
            raw_ft = DataUtils.generate_burgers(param_ft, n_total, grid_size)
            ft_name = f"{exp_cfg['problem_prefix']}{param_ft[0]}"
        elif pde == "grayscott":
            (F, k) = param_ft[0]
            raw_ft = DataUtils.generate_grayscott([(F, k)], n_total, grid_size)
            ft_name = f"{exp_cfg['problem_prefix']}_F{F}_k{k}"
        elif pde == "navierstokes":
            raw_ft = DataUtils.generate_navierstokes(param_ft, n_total, grid_size)
            ft_name = f"{exp_cfg['problem_prefix']}{param_ft[0]}"
        elif pde == "heat":
            raw_ft = DataUtils.generate_heat(
                n_total, grid_size, include_advection=exp_cfg["finetune_params"].get("include_advection", True)
            )
            ft_name = f"{exp_cfg['problem_prefix']}_extended"
        elif pde == "reactdiff":
            raw_ft = DataUtils.generate_reactdiff(
                n_total, grid_size, include_advection=exp_cfg["finetune_params"].get("include_advection", True)
            )
            ft_name = f"{exp_cfg['problem_prefix']}_extended"
        else:
            raise ValueError(f"Unknown PDE type for finetuning: {pde}")
        ft_train, ft_val, ft_test = split_ds(raw_ft)

    ft_entry = (ft_name, ft_train, ft_val, ft_test)
    return pretrain_entries, ft_entry


def _param_list_from_cfg(exp: dict, stage: str) -> list:
    """Extract parameter list from experiment description for the given stage."""
    key = f"{stage}_params"
    if stage == "pretrain":
        if exp["pde"] == "burgers":
            return exp[key]["nu_list"]
        elif exp["pde"] == "grayscott":
            return exp[key]["params_list"]
        elif exp["pde"] == "navierstokes":
            return exp[key]["Re_list"]
    else:  # finetune
        if exp["pde"] == "burgers":
            return exp[key]["nu_list"]
        elif exp["pde"] == "grayscott":
            return exp[key]["params_list"]
        elif exp["pde"] == "navierstokes":
            return exp[key]["Re_list"]
    raise ValueError("Unsupported parameter extraction")


# ----------------------------------------------------------------------
# Single problem loader (for fine‑tuning / scratch where name is constant)
# ----------------------------------------------------------------------
class SingleProblemLoader:
    """
    Wraps a torch DataLoader and yields tuples of (problem_name, x, y)
    where the name is fixed (e.g., 'ft'). Compatible with Trainer's
    iteration interface.
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, name: str):
        self.loader = dataloader
        self.name = name

    def __iter__(self):
        for x, y in self.loader:
            yield self.name, x, y

    def __len__(self):
        return len(self.loader)


# ----------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------
def run_pretraining(
    model: ModelBase,
    pretrain_datasets: List[Tuple[str, Dataset, Dataset]],  # (name, train_ds, val_ds)
    config: Config,
) -> Tuple[Trainer, ModelBase]:
    """
    Pretrain the given model on multiple physics problems.
    Returns the Trainer (containing avg epoch time) and the trained model
    with best checkpoint loaded.
    """
    # Prepare loaders
    train_ds_list = [ds[1] for ds in pretrain_datasets]
    val_ds_list = [ds[2] for ds in pretrain_datasets]
    names = [ds[0] for ds in pretrain_datasets]
    pretrain_cfg = config.training_params["pretrain"]

    train_loader = MultiPhysicsLoader(
        train_ds_list,
        names,
        batch_size=pretrain_cfg["batch_size"],
        shuffle=True,
    )
    val_loader = MultiPhysicsLoader(
        val_ds_list,
        names,
        batch_size=pretrain_cfg["batch_size"],
        shuffle=False,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
        phase="pretrain",
    )
    trainer.train()
    return trainer, model


def run_finetune(
    pretrained_model: ModelBase,
    ft_name: str,
    ft_train_ds: Dataset,
    ft_val_ds: Dataset,
    in_channels: int,
    out_channels: int,
    config: Config,
) -> Tuple[Trainer, ModelBase]:
    """
    Fine‑tune the pretrained model by freezing the body, adding a new adapter,
    and training on the target dataset.
    """
    ft_cfg = config.training_params["finetune"]

    # Freeze body
    pretrained_model.freeze_body()

    # Add fine‑tune adapter
    pretrained_model.add_adapter(ft_name, in_channels, out_channels)

    # Create loaders
    train_loader = SingleProblemLoader(
        torch.utils.data.DataLoader(
            ft_train_ds,
            batch_size=ft_cfg["batch_size"],
            shuffle=True,
            num_workers=0,
        ),
        ft_name,
    )
    val_loader = SingleProblemLoader(
        torch.utils.data.DataLoader(
            ft_val_ds,
            batch_size=ft_cfg["batch_size"],
            shuffle=False,
            num_workers=0,
        ),
        ft_name,
    )

    trainer = Trainer(
        model=pretrained_model,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
        phase="finetune",
    )
    trainer.train()
    return trainer, pretrained_model


def run_scratch(
    model: ModelBase,
    ft_name: str,
    ft_train_ds: Dataset,
    ft_val_ds: Dataset,
    in_channels: int,
    out_channels: int,
    config: Config,
) -> Tuple[Trainer, ModelBase]:
    """
    Train a randomly initialised model from scratch on the target problem,
    as baseline.
    """
    scratch_cfg = config.training_params["scratch"]

    # Add single adapter
    model.add_adapter(ft_name, in_channels, out_channels)

    train_loader = SingleProblemLoader(
        torch.utils.data.DataLoader(
            ft_train_ds,
            batch_size=scratch_cfg["batch_size"],
            shuffle=True,
            num_workers=0,
        ),
        ft_name,
    )
    val_loader = SingleProblemLoader(
        torch.utils.data.DataLoader(
            ft_val_ds,
            batch_size=scratch_cfg["batch_size"],
            shuffle=False,
            num_workers=0,
        ),
        ft_name,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
        phase="scratch",
    )
    trainer.train()
    return trainer, model


# ----------------------------------------------------------------------
# Main experiment loop for one architecture + experiment combo
# ----------------------------------------------------------------------
def run_experiment(
    arch: str,
    exp_cfg: Dict[str, Any],
    config: Config,
) -> Dict[str, Any]:
    """
    Execute the full pretrain→finetune and scratch pipelines for a given
    architecture and experiment, returning collected metrics.
    """
    # Override architecture in config copy
    saved_arch = config.model_params["architecture"]
    config.model_params["architecture"] = arch

    # Prepare datasets
    try:
        pretrain_entries, (ft_name, ft_train, ft_val, ft_test) = prepare_datasets(exp_cfg, config)
    except Exception as e:
        print(f"Skipping experiment {exp_cfg['id']} for arch {arch} due to dataset error: {e}")
        config.model_params["architecture"] = saved_arch
        return {}

    # Determine channels and dimension
    dim = exp_cfg["dim"]
    if exp_cfg.get("pde") == "pdebench":
        # infer channels from first pretrain dataset
        first_ds = pretrain_entries[0][1]  # train dataset
        input_channels_pret = first_ds.inputs.shape[1]
        ft_input_channels = ft_train.inputs.shape[1]
        output_channels = ft_train.outputs.shape[1]
    else:
        input_channels_pret = exp_cfg.get("input_channels_pretrain", exp_cfg["input_channels"])
        ft_input_channels = exp_cfg.get("input_channels_finetune", exp_cfg["input_channels"])
        output_channels = exp_cfg["output_channels"]

    # ---------------------- Pretrain → Fine‑tune -----------------------
    # Build model for pretraining
    model = ModelBase(copy.deepcopy(config.model_params), dim)
    # Add pretrain adapters
    for name, train_ds, _, _ in pretrain_entries:
        in_ch = train_ds.inputs.shape[1]
        out_ch = train_ds.outputs.shape[1] if name != ft_name else output_channels
        model.add_adapter(name, in_ch, out_ch)

    # Pretrain
    try:
        trainer_pret, model = run_pretraining(model, pretrain_entries, config)
        pretrain_epoch_time = trainer_pret.avg_epoch_time
    except Exception as e:
        print(f"Pretraining failed for {arch}/{exp_cfg['id']}: {e}")
        config.model_params["architecture"] = saved_arch
        return {}

    # Fine‑tune
    try:
        trainer_ft, model_ft = run_finetune(
            model, ft_name, ft_train, ft_val, ft_input_channels, output_channels, config
        )
        ft_epoch_time = trainer_ft.avg_epoch_time
    except Exception as e:
        print(f"Finetuning failed for {arch}/{exp_cfg['id']}: {e}")
        config.model_params["architecture"] = saved_arch
        return {}

    # Evaluate fine‑tuned model on test set
    evaluator = Evaluator(model_ft, {ft_name: ft_test}, config)
    ft_metrics = evaluator.evaluate(ft_name)
    ft_n_params = sum(p.numel() for p in model_ft.parameters())

    # ---------------------- Scratch baseline -----------------------
    # Build fresh model
    model_scratch = ModelBase(copy.deepcopy(config.model_params), dim)
    model_scratch.add_adapter(ft_name, ft_input_channels, output_channels)

    scratch_epoch_time = 0.0
    scratch_metrics = {"mse": float("inf"), "nmae": float("inf")}
    try:
        trainer_scratch, model_scratch = run_scratch(
            model_scratch, ft_name, ft_train, ft_val, ft_input_channels, output_channels, config
        )
        scratch_epoch_time = trainer_scratch.avg_epoch_time
        evaluator_scratch = Evaluator(model_scratch, {ft_name: ft_test}, config)
        scratch_metrics = evaluator_scratch.evaluate(ft_name)
    except Exception as e:
        print(f"Scratch training failed for {arch}/{exp_cfg['id']}: {e}")

    # Restore original architecture
    config.model_params["architecture"] = saved_arch

    # Collect results
    result = {
        "arch": arch,
        "exp_id": exp_cfg["id"],
        "table": exp_cfg["table"],
        "pretrain_epoch_time": pretrain_epoch_time,
        "finetune": {
            "mse": ft_metrics["mse"],
            "nmae": ft_metrics["nmae"],
            "epoch_time": ft_epoch_time,
            "params": ft_n_params,
        },
        "scratch": {
            "mse": scratch_metrics["mse"],
            "nmae": scratch_metrics["nmae"],
            "epoch_time": scratch_epoch_time,
        },
    }
    return result


# ----------------------------------------------------------------------
# Result aggregation and table printing
# ----------------------------------------------------------------------
def print_table1(results: List[Dict]) -> None:
    """Aggregate results for Table 1 and print formatted table."""
    # Filter only table 1 entries
    t1 = [r for r in results if r.get("table") == 1 and r]
    if not t1:
        print("No results for Table 1.")
        return

    # Group by architecture
    from collections import defaultdict

    arch_data = defaultdict(list)
    for r in t1:
        arch_data[r["arch"]].append(r)

    print("\n" + "=" * 80)
    print("Table 1: Out‑of‑sample parameter values (average across 3 PDEs)")
    print("-" * 80)
    header = f"{'Model':<25}{'MSE':<15}{'NMAE (%)':<12}{'Avg. epoch (s)':<15}{'Param.'}"
    print(header)
    print("-" * 80)

    row_order = [
        "Mamba FNO (pretr.)",
        "Mamba FNO (scratch)",
        "Perc. (pretr.)",
        "Perc. (scratch)",
        "FNO (scratch)",
        "Swin‑v2 (p.+s.)",
        "CoDA‑NO (pretr.)",
        "CoDA‑NO (scratch)",
    ]
    row_data = {}

    for arch, entries in arch_data.items():
        # Average finetune metrics
        ft_mse = np.mean([e["finetune"]["mse"] for e in entries])
        ft_nmae = np.mean([e["finetune"]["nmae"] for e in entries])
        ft_epoch = np.mean([e["finetune"]["epoch_time"] for e in entries])
        ft_params = entries[0]["finetune"]["params"]  # order-of-magnitude constant
        # Average scratch metrics (if present)
        scr_mse = np.mean([e["scratch"]["mse"] for e in entries])
        scr_nmae = np.mean([e["scratch"]["nmae"] for e in entries])
        scr_epoch = np.mean([e["scratch"]["epoch_time"] for e in entries])

        # Map architecture to friendly names
        if arch == "mamba_fno":
            row_data["Mamba FNO (pretr.)"] = (ft_mse, ft_nmae, ft_epoch, ft_params)
            row_data["Mamba FNO (scratch)"] = (scr_mse, scr_nmae, scr_epoch, ft_params)
        elif arch == "perceiver_fno":
            row_data["Perc. (pretr.)"] = (ft_mse, ft_nmae, ft_epoch, ft_params)
            row_data["Perc. (scratch)"] = (scr_mse, scr_nmae, scr_epoch, ft_params)
        elif arch == "fno":
            # Only scratch for vanilla FNO (paper doesn't show pretrained FNO)
            row_data["FNO (scratch)"] = (scr_mse, scr_nmae, scr_epoch, ft_params)
        elif arch == "swin_v2":
            # Swin‑v2 reported only fine‑tuned
            row_data["Swin‑v2 (p.+s.)"] = (ft_mse, ft_nmae, ft_epoch, ft_params)
        elif arch == "coda_no":
            row_data["CoDA‑NO (pretr.)"] = (ft_mse, ft_nmae, ft_epoch, ft_params)
            row_data["CoDA‑NO (scratch)"] = (scr_mse, scr_nmae, scr_epoch, ft_params)

    # Print rows in order
    for row_name in row_order:
        if row_name in row_data:
            mse, nmae, epoch, params = row_data[row_name]
            # Format MSE in scientific notation
            mse_str = f"{mse:.3e}"
            nmae_pct = nmae * 100.0  # convert fraction to percent
            nmae_str = f"{nmae_pct:.4f}"
            epoch_str = f"{epoch:.2f}"
            params_str = f"≈ {params//1e6}e6" if params >= 1e6 else str(params)
            print(f"{row_name:<25}{mse_str:<15}{nmae_str:<12}{epoch_str:<15}{params_str}")
    print("=" * 80)


def print_table2(results: List[Dict]) -> None:
    """Aggregate results for Table 2 and print formatted table."""
    t2 = [r for r in results if r.get("table") == 2 and r]
    if not t2:
        print("No results for Table 2.")
        return

    from collections import defaultdict

    arch_data = defaultdict(list)
    for r in t2:
        arch_data[r["arch"]].append(r)

    print("\n" + "=" * 80)
    print("Table 2: Input extension & multi‑physics transfer (average across 3 experiments)")
    print("-" * 80)
    header = f"{'Model':<25}{'MSE':<15}{'NMAE (%)':<12}{'Avg. epoch (s)':<15}"
    print(header)
    print("-" * 80)

    row_order = [
        "Mamba FNO (pretr.)",
        "Mamba FNO (scratch)",
        "Perc. (pretr.)",
        "Perc. (scratch)",
        "FNO (scratch)",
        "Swin‑v2 (p.+s.)",
        "CoDA‑NO (pretr.)",
        "CoDA‑NO (scratch)",
    ]
    row_data = {}

    for arch, entries in arch_data.items():
        ft_mse = np.mean([e["finetune"]["mse"] for e in entries])
        ft_nmae = np.mean([e["finetune"]["nmae"] for e in entries])
        ft_epoch = np.mean([e["finetune"]["epoch_time"] for e in entries])
        scr_mse = np.mean([e["scratch"]["mse"] for e in entries])
        scr_nmae = np.mean([e["scratch"]["nmae"] for e in entries])
        scr_epoch = np.mean([e["scratch"]["epoch_time"] for e in entries])

        if arch == "mamba_fno":
            row_data["Mamba FNO (pretr.)"] = (ft_mse, ft_nmae, ft_epoch)
            row_data["Mamba FNO (scratch)"] = (scr_mse, scr_nmae, scr_epoch)
        elif arch == "perceiver_fno":
            row_data["Perc. (pretr.)"] = (ft_mse, ft_nmae, ft_epoch)
            row_data["Perc. (scratch)"] = (scr_mse, scr_nmae, scr_epoch)
        elif arch == "fno":
            row_data["FNO (scratch)"] = (scr_mse, scr_nmae, scr_epoch)
        elif arch == "swin_v2":
            row_data["Swin‑v2 (p.+s.)"] = (ft_mse, ft_nmae, ft_epoch)
        elif arch == "coda_no":
            row_data["CoDA‑NO (pretr.)"] = (ft_mse, ft_nmae, ft_epoch)
            row_data["CoDA‑NO (scratch)"] = (scr_mse, scr_nmae, scr_epoch)

    for row_name in row_order:
        if row_name in row_data:
            mse, nmae, epoch = row_data[row_name]
            mse_str = f"{mse:.3e}"
            nmae_pct = nmae * 100.0
            nmae_str = f"{nmae_pct:.4f}"
            epoch_str = f"{epoch:.2f}"
            print(f"{row_name:<25}{mse_str:<15}{nmae_str:<12}{epoch_str:<15}")
    print("=" * 80)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from 'Towards Universal Neural Operators through Multiphysics Pretraining'."
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--arch",
        type=str,
        nargs="*",
        default=None,
        help="Run only specific architectures (e.g., fno mamba_fno). Default: all.",
    )
    parser.add_argument(
        "--table",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only experiments for a specific table (1 or 2). Default: both.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    set_seeds(args.seed)

    # Load configuration
    config = Config.from_yaml(args.config)

    # Define experiment list
    experiments = build_experiment_list(config)

    # Filter experiments if table selected
    if args.table:
        experiments = [e for e in experiments if e["table"] == args.table]

    # Architectures to run
    all_architectures = ["fno", "mamba_fno", "perceiver_fno", "swin_v2", "coda_no"]
    if args.arch:
        architectures = [a for a in args.arch if a in all_architectures]
    else:
        architectures = all_architectures

    # Collect all results
    all_results = []
    for arch in architectures:
        print(f"\n=== Running architecture: {arch} ===")
        for exp in experiments:
            # Skip Swin‑v2 for 1D experiments (not supported)
            if arch == "swin_v2" and exp["dim"] == 1:
                print(f"  Skipping 1D experiment {exp['id']} for Swin‑v2 (not supported).")
                continue
            print(f"  Experiment: {exp['id']}")
            res = run_experiment(arch, exp, config)
            if res:
                all_results.append(res)
            # (Optionally) save intermediate results
            with open("results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Print tables
    if any(r["table"] == 1 for r in all_results):
        print_table1(all_results)
    if any(r["table"] == 2 for r in all_results):
        print_table2(all_results)


if __name__ == "__main__":
    main()
