"""
Evaluation metrics for WDNO.

Implements:
- MSE (Mean Squared Error) for simulation tasks
- MAE (Mean Absolute Error)
- L-infinity error
- Control objective I for control tasks
- Relative L2 error
"""

import torch
import numpy as np
from typing import Union


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.
    
    Used as the primary metric for simulation tasks.
    Computed on entire state sequences excluding initial conditions.
    
    Args:
        pred: Predicted states (B, T, ...)
        target: Ground truth states (B, T, ...)
    
    Returns:
        MSE value
    """
    return ((pred - target) ** 2).mean().item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        pred: Predicted states
        target: Ground truth states
    
    Returns:
        MAE value
    """
    return (pred - target).abs().mean().item()


def compute_linf(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute L-infinity error (maximum absolute error).
    
    Args:
        pred: Predicted states
        target: Ground truth states
    
    Returns:
        L-infinity error
    """
    return (pred - target).abs().max().item()


def compute_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute relative L2 error.
    
    rel_l2 = ||pred - target||_2 / ||target||_2
    
    Args:
        pred: Predicted states
        target: Ground truth states
    
    Returns:
        Relative L2 error
    """
    return (torch.norm(pred - target) / torch.norm(target)).item()


def compute_burgers_control_objective(
    u_final: torch.Tensor,
    u_target: torch.Tensor,
    f: torch.Tensor,
    alpha: float = 0.00002,
    dx: float = 1.0 / 120,
    dt: float = 8.0 / 80,
) -> float:
    """
    Compute Burgers' equation control objective.
    
    I = integral_D |u(T, x) - u*(x)|^2 dx + alpha * integral_{[0,T]xD} |f(t, x)|^2 dt dx
    
    Args:
        u_final: Final state u(T, x) of shape (B, nx)
        u_target: Target state u*(x) of shape (B, nx)
        f: Control force of shape (B, nt, nx)
        alpha: Weight of energy cost (default: 0.00002 from paper)
        dx: Spatial grid spacing
        dt: Temporal grid spacing
    
    Returns:
        Control objective value
    """
    # State deviation term
    state_term = ((u_final - u_target) ** 2).sum(dim=-1) * dx
    
    # Energy term
    energy_term = (f ** 2).sum(dim=(-2, -1)) * dx * dt
    
    objective = (state_term + alpha * energy_term).mean().item()
    return objective


def compute_2d_control_objective(
    smoke_through_bucket: torch.Tensor,
) -> float:
    """
    Compute 2D incompressible fluid control objective.
    
    I = percentage of smoke NOT passing through the target bucket
    
    Args:
        smoke_through_bucket: Percentage of smoke through target bucket (B,)
    
    Returns:
        Control objective (1 - percentage through bucket)
    """
    return (1.0 - smoke_through_bucket).mean().item()


def evaluate_simulation(
    pred: torch.Tensor,
    target: torch.Tensor,
    exclude_initial: bool = True,
) -> dict:
    """
    Comprehensive evaluation for simulation tasks.
    
    Args:
        pred: Predicted states (B, T+1, ...) or (B, T, ...)
        target: Ground truth states (B, T+1, ...) or (B, T, ...)
        exclude_initial: Whether to exclude initial condition from evaluation
    
    Returns:
        Dictionary with MSE, MAE, L-inf, and relative L2 errors
    """
    if exclude_initial:
        pred = pred[:, 1:] if pred.shape[1] > target.shape[1] else pred
        target = target[:, 1:] if target.shape[1] > pred.shape[1] else target
    
    return {
        "mse": compute_mse(pred, target),
        "mae": compute_mae(pred, target),
        "linf": compute_linf(pred, target),
        "rel_l2": compute_relative_l2(pred, target),
    }


def evaluate_super_resolution(
    pred_list: list,
    target: torch.Tensor,
    interpolation_mode: str = "linear",
) -> list:
    """
    Evaluate zero-shot super-resolution at multiple levels.
    
    As described in the paper: "we interpolate the outcomes of each super-resolution
    step to the highest resolution level."
    
    Args:
        pred_list: List of predictions at different resolution levels
        target: Ground truth at highest resolution
        interpolation_mode: "linear" or "nearest"
    
    Returns:
        List of MSE values at each resolution level
    """
    mse_list = []
    target_size = target.shape[-2:]
    
    for pred in pred_list:
        if pred.shape[-2:] != target_size:
            # Interpolate to highest resolution
            if interpolation_mode == "linear":
                pred_interp = torch.nn.functional.interpolate(
                    pred, size=target_size, mode="bilinear", align_corners=False
                )
            else:
                pred_interp = torch.nn.functional.interpolate(
                    pred, size=target_size, mode="nearest"
                )
        else:
            pred_interp = pred
        
        mse = compute_mse(pred_interp, target)
        mse_list.append(mse)
    
    return mse_list
