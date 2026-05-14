"""
utils.py – Utility functions and constants for consistency model training.

This module provides:
- Constants for the EDM noise schedule and iCT training.
- The pseudo‑Huber loss (smooth L2 distance).
- The time‑step weighting used in the consistency loss.
- Checkpointing helpers (save/load both the trainable model and its EMA copy).

All functions are designed to be stateless and used by the trainer and evaluator.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional

# ----------------------------------------------------------------------
# Constants (mirroring the paper and config.yaml)
# ----------------------------------------------------------------------
SIGMA_DATA: float = 0.5         # σ_data, used in model parameterisation
SIGMA_MIN: float = 0.002        # σ_0, minimum noise level
SIGMA_MAX: float = 80.0         # σ_T, maximum noise level
RHO: float = 7.0                # exponent for noise schedule spacing

# Parameters for the log‑normal timestep sampling distribution
# (used in schedules.py, kept here for a single source of truth)
P_MEAN: float = -1.1
P_STD: float = 2.0

# Initial and final number of discrete timesteps for the exponential schedule
S0: int = 10
S1: int = 1280


# ----------------------------------------------------------------------
# Loss helpers
# ----------------------------------------------------------------------
def pseudo_huber_loss(x: Tensor, y: Tensor, c: float = 0.03) -> Tensor:
    """
    Pseudo‑Huber distance: sqrt(||x - y||² + c²) - c.

    This is the distance function used in the consistency loss. It provides
    a smoothed L2 distance that is well‑behaved for small differences.

    Args:
        x: Tensor of shape (B, C, H, W) or any shape, the prediction 
           (with stop‑gradient applied by the caller).
        y: Tensor of same shape as x, the target (requires gradient).
        c: Smoothing constant. For CIFAR‑10 32×32 the paper uses ~0.03.

    Returns:
        loss_per_sample: Tensor of shape (B,) containing the per‑sample distance.
    """
    diff = y - x
    # sum of squares over all dimensions except the batch dimension
    sq_norm = diff.pow(2).sum(dim=list(range(1, diff.ndim)))   # shape (B,)
    loss = torch.sqrt(sq_norm + c ** 2) - c
    return loss


def weighting(sigma_i: Tensor, sigma_ip1: Tensor) -> Tensor:
    """
    Compute the consistency loss weighting λ(σ_i) = 1 / (σ_{i+1} - σ_i).

    Args:
        sigma_i: Tensor of shape (B,) – current noise level σ_{t_i}.
        sigma_ip1: Tensor of same shape – next noise level σ_{t_{i+1}}.

    Returns:
        weights: Tensor of shape (B,) with elementwise weights.
    """
    # The schedule is strictly increasing, so no division by zero in normal use.
    return 1.0 / (sigma_ip1 - sigma_i)


# ----------------------------------------------------------------------
# Checkpointing (model + EMA model)
# ----------------------------------------------------------------------
def save_model(
    path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: Optional[int] = None,
) -> None:
    """
    Save the training state to a file.

    Args:
        path: File path (e.g., 'checkpoints/step_10000.pt').
        model: The trainable consistency model.
        ema_model: The exponential moving average copy of the model.
        optimizer: Optional optimizer state to resume training.
        step: Optional current training step for bookkeeping.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if step is not None:
        checkpoint['step'] = step

    torch.save(checkpoint, path)


def load_model(
    path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """
    Load training state from a checkpoint file.

    The state dictionaries are loaded in‑place into the provided
    model/ema_model/optimizer.

    Args:
        path: Path to the saved checkpoint.
        model: The trainable model to load weights into.
        ema_model: The EMA model to load weights into.
        optimizer: Optional optimizer to load state into.

    Returns:
        step: The training step at which the checkpoint was saved,
              or 0 if not stored.
    """
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    step = checkpoint.get('step', 0)
    return step
