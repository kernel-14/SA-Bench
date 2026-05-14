"""Evaluation metrics for PDE foundation models.

Implements the two metrics used in the paper:
1. L2 Relative Error (L2RE): ||x - x_hat||_2 / ||x||_2
2. Variance-normalized RMSE (VRMSE): RMSE(x, x_hat) / std(x)

VRMSE is suggested by The Well benchmark [Ohana et al., 2025].
"""

import torch
from typing import Dict, Optional


def compute_l2re(pred: torch.Tensor, target: torch.Tensor,
                  dim: Optional[tuple] = None,
                  reduction: str = 'mean') -> torch.Tensor:
    """Compute L2 Relative Error.
    
    L2RE = ||pred - target||_2 / ||target||_2
    
    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        dim: Dimensions to compute norm over (default: all except batch)
        reduction: 'mean' or 'none'
        
    Returns:
        L2RE value(s)
    """
    if dim is None:
        dim = tuple(range(1, pred.dim()))
    
    error_norm = torch.norm(pred - target, p=2, dim=dim)
    target_norm = torch.norm(target, p=2, dim=dim)
    
    # Avoid division by zero
    l2re = error_norm / (target_norm + 1e-8)
    
    if reduction == 'mean':
        return l2re.mean()
    return l2re


def compute_vrmse(pred: torch.Tensor, target: torch.Tensor,
                   dim: Optional[tuple] = None,
                   reduction: str = 'mean') -> torch.Tensor:
    """Compute Variance-Normalized Root Mean Square Error.
    
    VRMSE = RMSE(pred, target) / std(target)
    
    Where:
        RMSE = sqrt(mean((pred - target)^2))
        std = standard deviation of target
    
    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        dim: Dimensions to compute statistics over (default: all except batch)
        reduction: 'mean' or 'none'
        
    Returns:
        VRMSE value(s)
    """
    if dim is None:
        dim = tuple(range(1, pred.dim()))
    
    # RMSE
    mse = torch.mean((pred - target) ** 2, dim=dim)
    rmse = torch.sqrt(mse + 1e-8)
    
    # Standard deviation of target
    if isinstance(dim, int):
        target_std = torch.std(target, dim=dim)
    else:
        # For multiple dims, compute std over all specified dims
        target_flat = target.flatten(1)
        target_std = torch.std(target_flat, dim=1)
    
    vrmse = rmse / (target_std + 1e-8)
    
    if reduction == 'mean':
        return vrmse.mean()
    return vrmse


def compute_both_metrics(pred: torch.Tensor, 
                          target: torch.Tensor,
                          dim: Optional[tuple] = None) -> Dict[str, float]:
    """Compute both L2RE and VRMSE.
    
    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        dim: Dimensions for metrics
        
    Returns:
        Dictionary with 'L2RE' and 'VRMSE'
    """
    l2re = compute_l2re(pred, target, dim=dim).item()
    vrmse = compute_vrmse(pred, target, dim=dim).item()
    
    return {'L2RE': l2re, 'VRMSE': vrmse}


def evaluate_long_rollout(predictions: list, 
                          ground_truth: list) -> Dict[str, list]:
    """Evaluate long-term rollout predictions.
    
    Args:
        predictions: List of predicted frames
        ground_truth: List of ground truth frames
        
    Returns:
        Dictionary with step-wise L2RE values
    """
    results = {'L2RE': []}
    
    for pred, gt in zip(predictions, ground_truth):
        l2re = compute_l2re(pred, gt).item()
        results['L2RE'].append(l2re)
    
    # Add average
    results['Average_L2RE'] = sum(results['L2RE']) / len(results['L2RE'])
    
    return results


def compute_ensemble_variance(ensemble: list) -> torch.Tensor:
    """Compute batch-wise variance of ensemble predictions.
    
    Used for Fig. 3 in the paper: ensemble variance as function of k3.
    
    Args:
        ensemble: List of ensemble member predictions
        
    Returns:
        Mean variance across ensemble members
    """
    stacked = torch.stack(ensemble, dim=0)  # (N_ensemble, B, ...)
    variance = torch.var(stacked, dim=0)  # (B, ...)
    return variance.mean().item()
