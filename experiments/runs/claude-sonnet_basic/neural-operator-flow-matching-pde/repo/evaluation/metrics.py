"""
Evaluation metrics for PDE prediction.

Implements:
1. L2 Relative Error (L2RE): standard metric for PDE foundation models
2. Variance-normalized RMSE (VRMSE): as suggested by The Well benchmark

L2RE = ||y_pred - y_true||_2 / ||y_true||_2
VRMSE = RMSE / std(y_true)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union


def l2_relative_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute L2 Relative Error (L2RE).

    L2RE = ||pred - target||_2 / ||target||_2

    Args:
        pred: (B, ...) predicted values
        target: (B, ...) ground truth values
        reduction: 'mean', 'sum', or 'none'
        eps: small value to avoid division by zero

    Returns:
        l2re: scalar or per-sample L2RE
    """
    # Flatten spatial dimensions
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)

    # Per-sample L2RE
    numerator = torch.norm(pred_flat - target_flat, dim=-1)
    denominator = torch.norm(target_flat, dim=-1).clamp(min=eps)
    l2re = numerator / denominator

    if reduction == "mean":
        return l2re.mean()
    elif reduction == "sum":
        return l2re.sum()
    else:
        return l2re


def vrmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Variance-normalized RMSE (VRMSE).

    VRMSE = RMSE / std(target)
    where RMSE = sqrt(mean((pred - target)^2))

    As suggested by The Well benchmark (Ohana et al., 2025).

    Args:
        pred: (B, ...) predicted values
        target: (B, ...) ground truth values
        reduction: 'mean', 'sum', or 'none'
        eps: small value to avoid division by zero

    Returns:
        vrmse: scalar or per-sample VRMSE
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)

    # RMSE per sample
    mse = ((pred_flat - target_flat) ** 2).mean(dim=-1)
    rmse = torch.sqrt(mse)

    # Variance normalization: std of target
    target_std = target_flat.std(dim=-1).clamp(min=eps)
    vrmse_val = rmse / target_std

    if reduction == "mean":
        return vrmse_val.mean()
    elif reduction == "sum":
        return vrmse_val.sum()
    else:
        return vrmse_val


class PDEMetrics:
    """
    Computes and accumulates PDE prediction metrics.

    Tracks L2RE and VRMSE across batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.l2re_sum = 0.0
        self.vrmse_sum = 0.0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a new batch.

        Args:
            pred: (B, ...) predicted values
            target: (B, ...) ground truth values
        """
        B = pred.shape[0]

        l2re_batch = l2_relative_error(pred, target, reduction="none")
        vrmse_batch = vrmse(pred, target, reduction="none")

        self.l2re_sum += l2re_batch.sum().item()
        self.vrmse_sum += vrmse_batch.sum().item()
        self.count += B

    def compute(self):
        """
        Compute final metrics.

        Returns:
            dict with 'l2re' and 'vrmse' keys
        """
        if self.count == 0:
            return {"l2re": float("nan"), "vrmse": float("nan")}

        return {
            "l2re": self.l2re_sum / self.count,
            "vrmse": self.vrmse_sum / self.count,
        }


def compute_vorticity(u: torch.Tensor, v: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """
    Compute vorticity omega = dv/dx - du/dy using finite differences.

    Args:
        u: (B, H, W) x-velocity field
        v: (B, H, W) y-velocity field
        dx: grid spacing

    Returns:
        omega: (B, H, W) vorticity field
    """
    # Central differences
    dvdx = (torch.roll(v, -1, dims=-1) - torch.roll(v, 1, dims=-1)) / (2 * dx)
    dudy = (torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2)) / (2 * dx)
    return dvdx - dudy
