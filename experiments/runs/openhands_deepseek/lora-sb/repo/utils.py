"""Utility functions for LoRA-SB experiments."""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import json
from datetime import datetime


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count total or trainable parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_param_count(count: int) -> str:
    """Format parameter count into human-readable string."""
    if count >= 1e9:
        return f"{count/1e9:.2f}B"
    elif count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    return str(count)


def compute_flops_estimate(
    model: nn.Module,
    seq_length: int,
    batch_size: int = 1,
) -> Dict[str, float]:
    """Estimate FLOPs and MACs for a forward pass.

    This is a rough approximation based on the number of
    linear layer operations. For exact counts, use a profiler.

    Args:
        model: The model.
        seq_length: Input sequence length.
        batch_size: Batch size.

    Returns:
        Dict with 'flops' and 'macs' estimates.
    """
    total_flops = 0
    total_params = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            in_f = module.in_features
            out_f = module.out_features
            total_params += in_f * out_f
            flops_per_token = 2 * in_f * out_f
            total_flops += flops_per_token * seq_length * batch_size

    return {
        "flops": total_flops * 2,
        "macs": total_flops,
        "params": total_params,
    }


def save_results(
    results: Dict[str, Any],
    output_path: str,
    experiment_name: str = "lora-sb",
):
    """Save experiment results to JSON file.

    Args:
        results: Dictionary of results.
        output_path: Directory to save results.
        experiment_name: Name of the experiment.
    """
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_path, filename)

    serializable = {}
    for k, v in results.items():
        if isinstance(v, (np.integer,)):
            serializable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def verify_orthonormality(
    B: torch.Tensor,
    A: torch.Tensor,
    atol: float = 1e-4,
) -> Dict[str, bool]:
    """Verify that B and A are orthonormal as required by LoRA-SB.

    Checks:
        B^T B ≈ I
        A A^T ≈ I

    Returns:
        Dict with 'b_orthonormal' and 'a_orthonormal' boolean flags.
    """
    B_T_B = B.T @ B
    A_A_T = A @ A.T
    eye_r = torch.eye(B.shape[1], device=B.device, dtype=B.dtype)

    b_check = torch.allclose(B_T_B, eye_r, atol=atol)
    a_check = torch.allclose(A_A_T, eye_r, atol=atol)

    return {
        "b_orthonormal": b_check,
        "a_orthonormal": a_check,
        "b_max_diff": float((B_T_B - eye_r).abs().max()),
        "a_max_diff": float((A_A_T - eye_r).abs().max()),
    }


def get_linear_module_names(
    model: nn.Module,
    target_modules: List[str],
) -> List[str]:
    """Get names of all nn.Linear modules matching target substrings.

    Args:
        model: The model.
        target_modules: List of substrings to match (e.g. ['query', 'value']).

    Returns:
        List of full module names.
    """
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in target_modules:
                if target in name.lower():
                    names.append(name)
                    break
    return names


def compute_init_overhead(
    num_samples: int,
    time_seconds: float,
    total_train_time: float,
) -> float:
    """Compute initialization overhead as percentage of total training time.

    Args:
        num_samples: Number of samples used for init.
        time_seconds: Time taken for initialization.
        total_train_time: Total training time in seconds.

    Returns:
        Overhead as percentage.
    """
    return 100.0 * time_seconds / total_train_time if total_train_time > 0 else 0.0


class AverageMeter:
    """Track the average and current value of a metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"
