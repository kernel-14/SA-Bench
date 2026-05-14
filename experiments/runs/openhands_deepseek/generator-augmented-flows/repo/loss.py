"""
Loss functions for consistency models.

- Consistency Distillation (CD) loss
- Consistency Training (CT) loss
- Generator-Augmented Consistency Training (GC) loss
- Joint IC/GC training loss with parameter mu
"""
from typing import Optional

import torch
import torch.nn as nn


def pseudo_huber_loss(x: torch.Tensor, y: torch.Tensor, c: float = 0.00054) -> torch.Tensor:
    """
    Pseudo-Huber loss: sqrt(c^2 + ||x - y||^2) - c
    As in Song & Dhariwal (2024).
    """
    diff = x - y
    diff_norm = diff.reshape(diff.shape[0], -1).norm(dim=1)
    return (diff_norm.pow(2) + c ** 2).sqrt() - c


def l2_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard L2/MSE loss averaged per-sample."""
    diff = x - y
    return diff.reshape(diff.shape[0], -1).pow(2).mean(dim=1)


def distance_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    loss_type: str = "pseudo_huber",
    c: float = 0.00054,
) -> torch.Tensor:
    """Distance function D(x, y) from the paper."""
    if loss_type == "pseudo_huber":
        return pseudo_huber_loss(x, y, c)
    elif loss_type == "l2":
        return l2_loss(x, y)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def weighting_fn(sigma_i: torch.Tensor, sigma_i_plus_1: torch.Tensor) -> torch.Tensor:
    """
    Weighting lambda(sigma_t_i) = 1 / (sigma_{i+1} - sigma_i)
    As in Song & Dhariwal (2024).
    """
    return 1.0 / (sigma_i_plus_1 - sigma_i)


def consistency_distillation_loss(
    f_theta: nn.Module,
    x_ti_plus_1: torch.Tensor,
    sigma_ti: torch.Tensor,
    sigma_ti_plus_1: torch.Tensor,
    velocity_field: callable,
    loss_type: str = "pseudo_huber",
) -> torch.Tensor:
    """
    Consistency Distillation loss (Equation 4):
    L_CD(theta) = E[ lambda(sigma_ti) * D( sg(f_theta(x_ti^Phi, sigma_ti)), f_theta(x_{t_i+1}, sigma_{t_i+1}) ) ]

    where x_ti^Phi = Phi(x_{t_i+1}, t_{i+1}) = x_{t_i+1} + (t_i - t_{i+1}) * v_{t_i+1}(x_{t_i+1})

    Args:
        f_theta: Consistency model
        x_ti_plus_1: Intermediate points at sigma_{t+1}
        sigma_ti: sigma at time i
        sigma_ti_plus_1: sigma at time i+1
        velocity_field: Function v(x, sigma) returning the velocity
        loss_type: Distance metric

    Returns:
        Scalar loss
    """
    # Euler step: x_ti^Phi = x_{t_i+1} + (sigma_ti - sigma_ti_plus_1) * v(x_{t_i+1})
    with torch.no_grad():
        v = velocity_field(x_ti_plus_1, sigma_ti_plus_1)
        delta_sigma = (sigma_ti - sigma_ti_plus_1).view(-1, 1, 1, 1)
        x_ti_phi = x_ti_plus_1 + delta_sigma * v

    # Consistency loss
    with torch.no_grad():
        target = f_theta(x_ti_phi, sigma_ti)

    pred = f_theta(x_ti_plus_1, sigma_ti_plus_1)

    d = distance_fn(target, pred, loss_type)
    w = weighting_fn(sigma_ti, sigma_ti_plus_1)

    return (w * d).mean()


def consistency_training_loss(
    f_theta: nn.Module,
    x_ti: torch.Tensor,
    x_ti_plus_1: torch.Tensor,
    sigma_ti: torch.Tensor,
    sigma_ti_plus_1: torch.Tensor,
    loss_type: str = "pseudo_huber",
) -> torch.Tensor:
    """
    Consistency Training loss (Equation 6):
    L_CT(theta) = E[ lambda(sigma_ti) * D( sg(f_theta(x_ti, sigma_ti)), f_theta(x_{t_i+1}, sigma_{t_i+1}) ) ]

    Uses the one-sample Monte Carlo estimate of the velocity field.
    """
    with torch.no_grad():
        target = f_theta(x_ti, sigma_ti)

    pred = f_theta(x_ti_plus_1, sigma_ti_plus_1)

    d = distance_fn(target, pred, loss_type)
    w = weighting_fn(sigma_ti, sigma_ti_plus_1)

    return (w * d).mean()


def gc_consistency_training_loss(
    f_theta: nn.Module,
    x_tilde_ti: torch.Tensor,
    x_tilde_ti_plus_1: torch.Tensor,
    sigma_ti: torch.Tensor,
    sigma_ti_plus_1: torch.Tensor,
    loss_type: str = "pseudo_huber",
) -> torch.Tensor:
    """
    Generator-Augmented Consistency Training loss (Equation 15):
    L_GC(theta) = E[ lambda(sigma_ti) * D( sg(f_theta(tilde{x}_ti, sigma_ti)), f_theta(tilde{x}_{t_i+1}, sigma_{t_i+1}) ) ]
    """
    with torch.no_grad():
        target = f_theta(x_tilde_ti, sigma_ti)

    pred = f_theta(x_tilde_ti_plus_1, sigma_ti_plus_1)

    d = distance_fn(target, pred, loss_type)
    w = weighting_fn(sigma_ti, sigma_ti_plus_1)

    return (w * d).mean()


def joint_gc_loss(
    f_theta: nn.Module,
    x_ti: torch.Tensor,
    x_ti_plus_1: torch.Tensor,
    sigma_ti: torch.Tensor,
    sigma_ti_plus_1: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "pseudo_huber",
) -> torch.Tensor:
    """
    Joint IC/GC training loss with parameter mu (Equation 17):
    L_{GC-mu}(theta) = mu * L_GC(theta) + (1 - mu) * L_CT(theta)

    This is computed by:
    1. For each sample: x_hat_ti_mixed = mask * x_hat_ti + (1 - mask) * x_star
    2. x_ti = x_hat_ti_mixed + sigma_ti * z
    3. x_ti_plus_1 = x_hat_ti_mixed + sigma_{t_i+1} * z
    4. Compute consistency loss on these pairs

    The pairs x_ti and x_ti_plus_1 should already be constructed using the GC/IC mixing.
    """
    with torch.no_grad():
        target = f_theta(x_ti, sigma_ti)

    pred = f_theta(x_ti_plus_1, sigma_ti_plus_1)

    d = distance_fn(target, pred, loss_type)
    w = weighting_fn(sigma_ti, sigma_ti_plus_1)

    return (w * d).mean()
