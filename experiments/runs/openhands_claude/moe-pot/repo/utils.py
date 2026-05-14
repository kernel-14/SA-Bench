"""Utility functions for MoE-POT."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class AverageMeter:
    """Tracks a running average of a scalar metric."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def compute_l2_relative_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the L2 relative error (L2RE) per sample, then average over batch.

    L2RE = ||pred - target||_2 / ||target||_2

    Args:
        pred:   (B, C, H, W)
        target: (B, C, H, W)

    Returns:
        scalar tensor — mean L2RE over the batch
    """
    diff_norm = (pred - target).flatten(1).norm(dim=1)
    target_norm = target.flatten(1).norm(dim=1).clamp(min=1e-8)
    return (diff_norm / target_norm).mean()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch: int = 0,
) -> None:
    """Save model (and optionally optimizer/scheduler) state to disk."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, str(path))
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """Load model state from a checkpoint file.

    Returns the epoch number stored in the checkpoint (0 if not present).
    """
    ckpt = torch.load(path, map_location="cpu")

    # Handle both raw state_dict and wrapped checkpoint dicts
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        epoch = ckpt.get("epoch", 0)
    else:
        model.load_state_dict(ckpt, strict=False)
        epoch = 0

    print(f"Loaded checkpoint from {path} (epoch {epoch})")
    return epoch


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
