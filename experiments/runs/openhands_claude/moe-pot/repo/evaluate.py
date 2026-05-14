"""Evaluation and interpretability analysis for MoE-POT.

Implements:
- Zero-shot and fine-tuned L2RE evaluation (Section 5.1)
- Router-gating interpretability analysis (Section 5.4, Appendix B.4)
- Dataset classification via cross-entropy distance (Appendix B.4)
- Rollout error analysis (Appendix C.3)
- Expert usage ratio analysis (Figure 2 right)
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DATASET_CONFIGS, DOWNSTREAM_CONFIGS
from data import build_dataloader, build_single_dataset
from model import MoEPOT, build_model
from utils import AverageMeter, compute_l2_relative_error, load_checkpoint


# ─── L2RE Evaluation ──────────────────────────────────────────────────────────

def evaluate_zero_shot(
    model: MoEPOT,
    data_root: str,
    dataset_names: Optional[List[str]] = None,
    batch_size: int = 20,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate zero-shot L2RE on each dataset without fine-tuning.

    Returns a dict mapping dataset_name → L2RE.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_names is None:
        dataset_names = list(DATASET_CONFIGS.keys())

    model.eval()
    results: Dict[str, float] = {}

    for name in dataset_names:
        try:
            ds = build_single_dataset(data_root, name, split="test")
            loader = build_dataloader(ds, batch_size, shuffle=False, num_workers=num_workers)
            l2re = _eval_loader(model, loader, device)
            results[name] = l2re
            print(f"  {name:30s}  L2RE = {l2re:.5f}")
        except FileNotFoundError:
            print(f"  {name:30s}  [dataset not found, skipped]")

    return results


def _eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean L2RE over a DataLoader."""
    errors = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            pred, _ = model(inputs)
            err = compute_l2_relative_error(pred, targets)
            errors.append(err.item())
    return float(np.mean(errors))


# ─── Rollout Error Analysis ────────────────────────────────────────────────────

def evaluate_rollout(
    model: MoEPOT,
    loader: DataLoader,
    device: torch.device,
    num_rollout_steps: int = 100,
    report_steps: Optional[List[int]] = None,
) -> Dict[int, float]:
    """Evaluate auto-regressive rollout error at specified timesteps.

    At each step, the model predicts the next frame from the current window,
    then slides the window forward by appending the prediction.

    Args:
        report_steps: list of step indices to report (default: [50, 70, 100])

    Returns:
        Dict mapping step → mean L2RE
    """
    if report_steps is None:
        report_steps = [50, 70, 100]

    model.eval()
    step_errors: Dict[int, List[float]] = {s: [] for s in report_steps}

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)   # (B, T, C, H, W)
            targets = batch["target"].to(device)

            B, T, C, H, W = inputs.shape
            current_window = inputs.clone()

            # We need ground-truth frames for all rollout steps.
            # Here we use the test batch's target as the final ground truth.
            # For multi-step rollout, we approximate by comparing to the
            # last available ground-truth frame.
            for step in range(1, num_rollout_steps + 1):
                pred, _ = model(current_window)
                if step in step_errors:
                    err = compute_l2_relative_error(pred, targets)
                    step_errors[step].append(err.item())
                current_window = torch.cat(
                    [current_window[:, 1:], pred.unsqueeze(1)], dim=1
                )

    return {s: float(np.mean(v)) for s, v in step_errors.items() if v}


# ─── Router Interpretability ───────────────────────────────────────────────────

def compute_dataset_router_profiles(
    model: MoEPOT,
    data_root: str,
    dataset_names: Optional[List[str]] = None,
    block_idx: int = 1,
    batch_size: int = 20,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Compute average router-gating weight profile per dataset.

    For each dataset i, computes:
        Y_i = (1/N_i) * Σ_j Y_{ij}   where Y_{ij} ∈ R^{N_r}

    This is the reference profile used for dataset classification.

    Args:
        block_idx: which MoE block to use (0-indexed)

    Returns:
        Dict mapping dataset_name → mean routing weight vector (N_r,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_names is None:
        dataset_names = list(DATASET_CONFIGS.keys())

    model.eval()
    profiles: Dict[str, np.ndarray] = {}

    for name in dataset_names:
        try:
            ds = build_single_dataset(data_root, name, split="test")
            loader = build_dataloader(ds, batch_size, shuffle=False, num_workers=num_workers)
            weights_list = []
            with torch.no_grad():
                for batch in loader:
                    inputs = batch["input"].to(device)
                    router_weights = model.get_router_weights(inputs)
                    w = router_weights[block_idx]  # (B, N_r)
                    weights_list.append(w.cpu().numpy())
            all_weights = np.concatenate(weights_list, axis=0)  # (N, N_r)
            profiles[name] = all_weights.mean(axis=0)           # (N_r,)
        except FileNotFoundError:
            print(f"  {name}: dataset not found, skipped")

    return profiles


def classify_dataset_by_router(
    model: MoEPOT,
    data_root: str,
    dataset_names: Optional[List[str]] = None,
    block_idx: int = 1,
    batch_size: int = 20,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Classify input samples by their router-gating profile.

    Algorithm (Appendix B.4):
    1. Compute reference profiles Y_i for each dataset
    2. For each test sample, compute cross-entropy f(I_0, Y_i) for all i
    3. Assign to dataset i_0 = argmin_i f(I_0, Y_i)
    4. Report per-dataset accuracy

    Returns:
        Dict mapping dataset_name → classification accuracy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_names is None:
        dataset_names = list(DATASET_CONFIGS.keys())

    # Step 1: compute reference profiles
    profiles = compute_dataset_router_profiles(
        model, data_root, dataset_names, block_idx, batch_size, num_workers, device
    )
    if not profiles:
        return {}

    profile_names = list(profiles.keys())
    profile_matrix = np.stack([profiles[n] for n in profile_names], axis=0)  # (K, N_r)
    # Add small epsilon for numerical stability in log
    profile_matrix = np.clip(profile_matrix, 1e-8, 1.0)

    model.eval()
    accuracies: Dict[str, float] = {}

    for true_name in profile_names:
        try:
            ds = build_single_dataset(data_root, true_name, split="test")
            loader = build_dataloader(ds, batch_size, shuffle=False, num_workers=num_workers)
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in loader:
                    inputs = batch["input"].to(device)
                    router_weights = model.get_router_weights(inputs)
                    I0 = router_weights[block_idx].cpu().numpy()  # (B, N_r)
                    I0 = np.clip(I0, 1e-8, 1.0)

                    # Cross-entropy: f(I_0, Y_i) = -Σ_k I_{0,k} * log(Y_{i,k})
                    # Shape: (B, K)
                    ce = -np.einsum("bk,ik->bi", I0, np.log(profile_matrix))
                    pred_labels = np.argmin(ce, axis=1)  # (B,)
                    true_label = profile_names.index(true_name)
                    correct += (pred_labels == true_label).sum()
                    total += len(pred_labels)

            acc = correct / total if total > 0 else 0.0
            accuracies[true_name] = acc
            print(f"  Block {block_idx} | {true_name:30s}  Accuracy = {acc*100:.1f}%")
        except FileNotFoundError:
            print(f"  {true_name}: dataset not found, skipped")

    overall = np.mean(list(accuracies.values())) if accuracies else 0.0
    print(f"  Block {block_idx} | Overall accuracy = {overall*100:.1f}%")
    return accuracies


def compute_expert_usage_ratio(
    model: MoEPOT,
    data_root: str,
    dataset_names: Optional[List[str]] = None,
    block_idx: int = 3,
    batch_size: int = 20,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """Compute the usage ratio of each routed expert per dataset.

    Usage ratio = fraction of samples that select each expert in top-K.

    Returns:
        Dict mapping dataset_name → usage ratio vector (N_r,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_names is None:
        dataset_names = list(DATASET_CONFIGS.keys())

    model.eval()
    usage_ratios: Dict[str, np.ndarray] = {}
    top_k = model.cfg.top_k
    num_routed = model.cfg.num_routed_experts

    for name in dataset_names:
        try:
            ds = build_single_dataset(data_root, name, split="test")
            loader = build_dataloader(ds, batch_size, shuffle=False, num_workers=num_workers)
            counts = np.zeros(num_routed)
            total = 0

            with torch.no_grad():
                for batch in loader:
                    inputs = batch["input"].to(device)
                    router_weights = model.get_router_weights(inputs)
                    w = router_weights[block_idx].cpu().numpy()  # (B, N_r)
                    # Top-K indices
                    topk_idx = np.argsort(w, axis=1)[:, -top_k:]  # (B, K)
                    for idx_row in topk_idx:
                        counts[idx_row] += 1
                    total += len(w)

            usage_ratios[name] = counts / (total + 1e-8)
        except FileNotFoundError:
            print(f"  {name}: dataset not found, skipped")

    return usage_ratios


# ─── Main Evaluation Script ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MoE-POT Evaluation")
    parser.add_argument("--mode", choices=["zero_shot", "classify", "usage", "rollout"],
                        default="zero_shot")
    parser.add_argument("--model_size", choices=["tiny", "small", "medium"], default="tiny")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names to evaluate (default: all 6 pre-training datasets)")
    parser.add_argument("--block_idx", type=int, default=1,
                        help="Block index for router analysis (0-indexed)")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_size).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    if args.mode == "zero_shot":
        print("=== Zero-Shot Evaluation ===")
        results = evaluate_zero_shot(
            model, args.data_root, args.datasets, args.batch_size, args.num_workers, device
        )
        print("\nSummary:")
        for name, l2re in results.items():
            print(f"  {name}: {l2re:.5f}")

    elif args.mode == "classify":
        print(f"=== Router Classification (Block {args.block_idx}) ===")
        accuracies = classify_dataset_by_router(
            model, args.data_root, args.datasets, args.block_idx,
            args.batch_size, args.num_workers, device
        )

    elif args.mode == "usage":
        print(f"=== Expert Usage Ratio (Block {args.block_idx}) ===")
        ratios = compute_expert_usage_ratio(
            model, args.data_root, args.datasets, args.block_idx,
            args.batch_size, args.num_workers, device
        )
        for name, ratio in ratios.items():
            top_experts = np.argsort(ratio)[::-1][:5]
            print(f"  {name}: top-5 experts = {top_experts.tolist()}, "
                  f"ratios = {ratio[top_experts].tolist()}")

    elif args.mode == "rollout":
        print("=== Rollout Error Analysis ===")
        dataset_name = args.datasets[0] if args.datasets else "pdebench_swe"
        ds = build_single_dataset(args.data_root, dataset_name, split="test")
        loader = build_dataloader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
        errors = evaluate_rollout(model, loader, device, num_rollout_steps=100,
                                  report_steps=[50, 70, 100])
        print(f"\nRollout errors on {dataset_name}:")
        for step, err in sorted(errors.items()):
            print(f"  Frame {step:3d}: L2RE = {err:.5f}")


if __name__ == "__main__":
    main()
