"""
Utility functions shared across the codebase.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
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
        self.avg = self.sum / self.count


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy as percentage."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item() * 100.0


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy as percentage."""
    _, top_k_preds = logits.topk(k, dim=-1)
    correct = top_k_preds.eq(labels.unsqueeze(-1).expand_as(top_k_preds))
    return correct.any(dim=-1).float().mean().item() * 100.0


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list:
    """
    Separate parameters into decay and no-decay groups.
    Bias terms and normalization parameters should not have weight decay.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "norm" in name or "gamma" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def save_checkpoint(
    state: dict,
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        import shutil
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    strict: bool = True,
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    return checkpoint


def print_trainable_params(model: nn.Module) -> None:
    """Print trainable parameter names and counts."""
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            print(f"  {name}: {n:,}")
    print(f"Total trainable: {total:,} ({total / 1e6:.3f}M)")


def format_results_table(
    results: dict,
    methods: list,
    tasks: list,
) -> str:
    """Format results as a text table similar to Table 1 in the paper."""
    col_width = 8
    header = f"{'Method':<20}" + "".join(f"{t[:col_width]:>{col_width}}" for t in tasks)
    lines = [header, "-" * len(header)]

    for method in methods:
        row = f"{method:<20}"
        for task in tasks:
            acc = results.get(method, {}).get(task, 0.0)
            row += f"{acc:>{col_width}.1f}"
        lines.append(row)

    return "\n".join(lines)
