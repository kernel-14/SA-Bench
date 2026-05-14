"""Evaluation metrics for neural operators.

As used in the paper: Range-Normalized Mean Absolute Error (NMAE) and Mean Squared Error (MSE).

NMAE formula (Equation 3 in paper):
    NMAE(θ) = (1/|D_test|) * Σ_{(a,u)∈D_test} ||G_θ(a) - u||_{1,G} / (max_G u - min_G u + ε)
"""
import torch
import torch.nn as nn


class NMAE:
    """Range-Normalized Mean Absolute Error metric.

    Normalizes the L1 error by the range of the ground truth values over the domain G.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch, ..., output_channels) — model predictions
            target: (batch, ..., output_channels) — ground truth
        Returns:
            NMAE value (scalar)
        """
        # L1 error summed over spatial and channel dimensions
        l1_error = torch.abs(pred - target).sum(dim=tuple(range(1, pred.ndim)))

        # Range of target over all dimensions except batch
        flat_target = target.reshape(target.shape[0], -1)
        target_range = flat_target.max(dim=-1).values - flat_target.min(dim=-1).values + self.eps

        nmae = l1_error / (target_range * target.numel() / target.shape[0])
        return nmae.mean()


class MSE:
    """Mean Squared Error metric."""

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Compute both MSE and NMAE metrics.

    Args:
        pred: (batch, ..., output_channels) — model predictions
        target: (batch, ..., output_channels) — ground truth
    Returns:
        dict with 'mse' and 'nmae' values
    """
    mse = torch.mean((pred - target) ** 2).item()

    # NMAE computation
    l1_error = torch.abs(pred - target).mean(dim=tuple(range(1, pred.ndim)))
    flat_target = target.reshape(target.shape[0], -1)
    target_range = flat_target.max(dim=-1).values - flat_target.min(dim=-1).values + eps
    nmae = (l1_error / target_range).mean().item()

    # NMAE as percentage
    nmae_pct = nmae * 100.0

    return {
        'mse': mse,
        'nmae': nmae,
        'nmae_pct': nmae_pct,
    }


class NMALoss(nn.Module):
    """Range-Normalized Mean Absolute Error as a training loss."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_error = torch.abs(pred - target).sum(dim=tuple(range(1, pred.ndim)))
        flat_target = target.reshape(target.shape[0], -1)
        target_range = flat_target.max(dim=-1).values - flat_target.min(dim=-1).values + self.eps
        nmae = l1_error / (target_range * target.numel() / target.shape[0])
        return nmae.mean()
