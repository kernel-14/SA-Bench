"""
Metrics for evaluating neural operator performance.

From the paper:
NMAE(theta) = (1/|D_test|) * sum_{(a,u) in D_test} ||G_theta(a) - u||_{1,G} / (max_G u - min_G u + eps)

where ||.||_{1,G} is the L1 norm over the grid G.
"""

import torch
import numpy as np
from typing import Dict, Tuple


def compute_nmae(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute Normalized Mean Absolute Error (NMAE).
    
    NMAE = mean over samples of (||pred - target||_1 / (max(target) - min(target) + eps))
    
    Args:
        pred: Predicted values (batch, ...)
        target: True values (batch, ...)
        eps: Small constant for numerical stability
    
    Returns:
        NMAE value (as percentage if multiplied by 100)
    """
    batch_size = pred.shape[0]
    
    # Flatten spatial dimensions
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    
    # L1 norm per sample
    l1_norm = torch.abs(pred_flat - target_flat).mean(dim=-1)
    
    # Range normalization per sample
    target_range = target_flat.max(dim=-1).values - target_flat.min(dim=-1).values + eps
    
    # NMAE per sample
    nmae_per_sample = l1_norm / target_range
    
    return nmae_per_sample.mean().item()


def compute_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Compute Mean Squared Error (MSE).
    
    Args:
        pred: Predicted values (batch, ...)
        target: True values (batch, ...)
    
    Returns:
        MSE value
    """
    return torch.mean((pred - target) ** 2).item()


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute all metrics used in the paper.
    
    Args:
        pred: Predicted values (batch, ...)
        target: True values (batch, ...)
        eps: Small constant for numerical stability
    
    Returns:
        Dictionary with 'mse' and 'nmae' keys
    """
    return {
        "mse": compute_mse(pred, target),
        "nmae": compute_nmae(pred, target, eps),
    }


def compute_relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Compute relative L2 error (commonly used in FNO papers).
    
    Args:
        pred: Predicted values (batch, ...)
        target: True values (batch, ...)
    
    Returns:
        Relative L2 error
    """
    batch_size = pred.shape[0]
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    
    l2_error = torch.norm(pred_flat - target_flat, dim=-1)
    l2_target = torch.norm(target_flat, dim=-1) + eps
    
    return (l2_error / l2_target).mean().item()
