from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return nn.functional.mse_loss(pred, target).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().mean().item()


def nmae(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Normalized Mean Absolute Error (equation 3 in the paper).
    NMAE = ||G_θ(a) - u||_1 / (max_G(u) - min_G(u) + ε)
    """
    abs_err = (pred - target).abs().mean().item()
    denom = (target.max() - target.min() + eps).item()
    return abs_err / denom


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Relative L2 error: ||pred - target||_2 / ||target||_2"""
    num = torch.norm(pred - target, p=2).item()
    den = torch.norm(target, p=2).item() + eps
    return num / den


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Compute all metrics for a batch of predictions."""
    return {
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "nmae": nmae(pred, target),
        "nmae_pct": nmae(pred, target) * 100.0,
        "rel_l2": relative_l2(pred, target),
    }


# ---------------------------------------------------------------------------
# Full dataset evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    physics_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a model on a full dataset split.

    Args:
        model: trained neural operator
        loader: DataLoader for the evaluation split
        device: compute device
        physics_name: for MultiPhysicsModel, specify which physics adapter to use
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)

        if physics_name is not None:
            pred = model(x, physics_name)
        else:
            pred = model(x)

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return compute_metrics(preds, targets)


# ---------------------------------------------------------------------------
# Comparison table generation (reproduces Tables 1 and 2)
# ---------------------------------------------------------------------------

def evaluate_all_models(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple models and return a comparison table.

    Args:
        models: dict mapping model_name → model
        test_loader: test DataLoader
        device: compute device

    Returns:
        dict mapping model_name → metrics dict
    """
    results = {}
    for name, model in models.items():
        model.to(device)
        metrics = evaluate_model(model, test_loader, device)
        results[name] = metrics
        print(
            f"{name:30s} | MSE: {metrics['mse']:.4e} | "
            f"NMAE: {metrics['nmae_pct']:.4f}%"
        )
    return results


def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print results in the format of Tables 1 and 2 from the paper."""
    header = f"{'Model':<30} {'MSE':>12} {'NMAE (%)':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['mse']:>12.4e} {metrics['nmae_pct']:>12.4f}")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Timing utilities (for Avg. epoch (s) column in the tables)
# ---------------------------------------------------------------------------

import time


def benchmark_epoch_time(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_warmup: int = 2,
    n_measure: int = 5,
) -> float:
    """
    Measure average epoch time in seconds.
    Matches the 'Avg. epoch (s)' column in Tables 1 and 2.
    """
    model.train()

    # Warmup
    for _ in range(n_warmup):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            break  # one batch per warmup epoch

    # Measure
    times = []
    for _ in range(n_measure):
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
        times.append(time.time() - t0)

    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Checkpoint loading utilities
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Experiment evaluation script
# ---------------------------------------------------------------------------

def run_evaluation(
    cfg: dict,
    device: torch.device,
) -> None:
    """
    Load all trained models and evaluate them, printing Tables 1 and 2.
    """
    import yaml
    from data import build_dataset, build_dataloader, get_n_in_out
    from model import build_model

    print("\n=== Evaluating all models ===\n")

    for exp_name, exp_cfg in cfg.get("experiments", {}).items():
        print(f"\n--- Experiment: {exp_name} ---")

        # Determine test dataset
        if "finetune" in exp_cfg:
            ds_cfg_key = exp_cfg["finetune"]["dataset"]
        elif "scratch" in exp_cfg:
            ds_cfg_key = exp_cfg["scratch"]["dataset"]
        else:
            continue

        ds_cfg = cfg["datasets"][ds_cfg_key]
        t_in = ds_cfg.get("t_in", 1)
        n_in, n_out = get_n_in_out(ds_cfg["type"], t_in)

        test_ds = build_dataset(
            ds_cfg["type"],
            ds_cfg["file_path"],
            split="test",
            t_in=t_in,
            n_samples=ds_cfg.get("n_samples"),
        )
        test_loader = build_dataloader(test_ds, batch_size=16, shuffle=False)

        results = {}

        # Evaluate each model variant
        for variant_name, variant_cfg in exp_cfg.items():
            if not isinstance(variant_cfg, dict) or "save_dir" not in variant_cfg:
                continue

            model_key = variant_cfg.get("model", "fno")
            model_cfg = cfg["models"][model_key]
            spatial_dim = ds_cfg.get("spatial_dim", 2)

            model = build_model(
                model_cfg["type"],
                n_in=n_in,
                n_out=n_out,
                spatial_dim=spatial_dim,
                **model_cfg.get("kwargs", {}),
            )

            ckpt_path = Path(variant_cfg["save_dir"]) / f"{model_key}_{variant_name}_best.pt"
            if ckpt_path.exists():
                model = load_model_from_checkpoint(model, str(ckpt_path), device)
                metrics = evaluate_model(model, test_loader, device)
                results[f"{model_key} ({variant_name})"] = metrics
            else:
                print(f"  Checkpoint not found: {ckpt_path}")

        if results:
            print_results_table(results)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Evaluate trained neural operators")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    run_evaluation(cfg, device)
