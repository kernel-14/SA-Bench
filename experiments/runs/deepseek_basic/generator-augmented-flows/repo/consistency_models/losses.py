"""
Loss functions for consistency model training.

Implements:
1. Consistency Distillation (CD) loss - Equation (4)
2. Consistency Training (CT) loss - Equation (6)
3. Generator-Augmented Consistency Training loss - Equation (15)
4. Joint learning loss - Equation (16)

All based on the paper.
"""

import torch
import torch.nn as nn
from typing import Optional
from .coupling import (
    get_intermediate_points,
    generator_augmented_coupling,
    batch_ot_coupling,
)


def consistency_distillation_loss(
    consistency_model: nn.Module,
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_i: torch.Tensor,
    sigma_next: torch.Tensor,
    score_model: nn.Module,
    lambda_weight: torch.Tensor,
    distance_fn: callable = None,
    use_edm: bool = True,
) -> torch.Tensor:
    """
    Consistency Distillation loss (Equation 4).

    L_CD(θ) = E[ λ(σ_i) * D(sg(f_θ(x_{t_i}^Φ, σ_i)), f_θ(x_{t_{i+1}}, σ_{i+1})) ]

    where x_{t_i}^Φ = x_{t_{i+1}} + (t_i - t_{i+1}) * v_{t_{i+1}}(x_{t_{i+1}})

    This requires a pre-trained score/velocity model.

    Args:
        consistency_model: f_θ
        x_star: Data samples [B, C, H, W]
        z: Noise samples [B, C, H, W]
        sigma_i: Current noise levels [B]
        sigma_next: Next noise levels [B]
        score_model: Pre-trained score/diffusion model providing v_t(x)
        lambda_weight: Weighting λ(σ_i) for each sample [B]
        distance_fn: Distance function D (default: pseudo-Huber)
        use_edm: Use EDM velocity field formulation

    Returns:
        Scalar loss
    """
    if distance_fn is None:
        distance_fn = pseudo_huber_loss

    # x_{t_{i+1}}
    x_next = get_intermediate_points(x_star, z, sigma_next)

    # Get velocity field from score model
    with torch.no_grad():
        if use_edm:
            # In EDM: v_t(x) = -t * score(x, t)
            # But we need the PF-ODE discretization:
            # x_{t_i}^Φ = x_{t_{i+1}} + (σ_i - σ_{i+1}) * v(x_{t_{i+1}}, σ_{i+1})
            v_t = score_model.get_velocity(x_next, sigma_next)
        else:
            v_t = score_model.get_velocity(x_next, sigma_next)

    # Euler step to get x_{t_i}^Φ
    # x_{t_i}^Φ = x_{t_{i+1}} + (σ_i - σ_{i+1}) * v_t
    sigma_i_expanded = sigma_i
    sigma_next_expanded = sigma_next
    while sigma_i_expanded.dim() < x_next.dim():
        sigma_i_expanded = sigma_i_expanded.unsqueeze(-1)
        sigma_next_expanded = sigma_next_expanded.unsqueeze(-1)

    x_phi = x_next + (sigma_i_expanded - sigma_next_expanded) * v_t

    # Consistency model outputs
    with torch.no_grad():
        f_phi = consistency_model(x_phi, sigma_i)
    f_next = consistency_model(x_next, sigma_next)

    # Distance
    dist = distance_fn(f_phi, f_next)

    # Apply weighting
    while lambda_weight.dim() < dist.dim():
        lambda_weight = lambda_weight.unsqueeze(-1)

    loss = (lambda_weight * dist).mean()
    return loss


def consistency_training_loss(
    consistency_model: nn.Module,
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_i: torch.Tensor,
    sigma_next: torch.Tensor,
    lambda_weight: torch.Tensor,
    distance_fn: callable = None,
    use_ot: bool = False,
    ot_solver: str = "sinkhorn",
) -> torch.Tensor:
    """
    Consistency Training loss (Equation 6).

    L_CT(θ) = E[ λ(σ_i) * D(sg(f_θ(x_{t_i}, σ_i)), f_θ(x_{t_{i+1}}, σ_{i+1})) ]

    Uses one-sample Monte Carlo estimate: x_{t_i} = x_* + σ_i * z
    instead of the proper x_{t_i}^Φ from CD.

    Args:
        consistency_model: f_θ
        x_star: Data samples [B, C, H, W]
        z: Noise samples [B, C, H, W]
        sigma_i: Current noise levels [B]
        sigma_next: Next noise levels [B]
        lambda_weight: Weighting λ(σ_i) [B]
        distance_fn: Distance function D
        use_ot: Use OT coupling
        ot_solver: OT solver type

    Returns:
        Scalar loss
    """
    if distance_fn is None:
        distance_fn = pseudo_huber_loss

    # OT coupling if requested
    if use_ot:
        x_star, z = batch_ot_coupling(x_star, z, ot_solver=ot_solver)

    # x_{t_i} = x_* + σ_i * z (one-sample Monte Carlo)
    x_ti = get_intermediate_points(x_star, z, sigma_i)
    x_next = get_intermediate_points(x_star, z, sigma_next)

    # Consistency model outputs with stop-gradient on first
    with torch.no_grad():
        f_ti = consistency_model(x_ti, sigma_i)
    f_next = consistency_model(x_next, sigma_next)

    # Distance
    dist = distance_fn(f_ti, f_next)

    # Apply weighting
    while lambda_weight.dim() < dist.dim():
        lambda_weight = lambda_weight.unsqueeze(-1)

    loss = (lambda_weight * dist).mean()
    return loss


def gc_consistency_loss(
    consistency_model: nn.Module,
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_i: torch.Tensor,
    sigma_next: torch.Tensor,
    lambda_weight: torch.Tensor,
    distance_fn: callable = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generator-Augmented Consistency Training loss (Equation 15).

    L_GC(θ) = E_q[ λ(σ_i) * D(sg(f_θ(tilde{x}_{t_i}, σ_i)), f_θ(tilde{x}_{t_{i+1}}, σ_{i+1})) ]

    where tilde{x} points are constructed via GC (Equations 13-14).

    Args:
        consistency_model: f_θ
        x_star: Data samples [B, C, H, W]
        z: Noise samples [B, C, H, W]
        sigma_i: Current noise levels [B]
        sigma_next: Next noise levels [B]
        lambda_weight: Weighting [B]
        distance_fn: Distance function
        mask: Joint learning mask [B], if provided mixes IC/GC

    Returns:
        Scalar loss
    """
    if distance_fn is None:
        distance_fn = pseudo_huber_loss

    # Construct GC intermediate points
    tilde_x_ti, tilde_x_next = generator_augmented_coupling(
        x_star, z, sigma_i, sigma_next, consistency_model, mask=mask
    )

    # Consistency model outputs with stop-gradient on first
    with torch.no_grad():
        f_ti = consistency_model(tilde_x_ti, sigma_i)
    f_next = consistency_model(tilde_x_next, sigma_next)

    # Distance
    dist = distance_fn(f_ti, f_next)

    # Apply weighting
    while lambda_weight.dim() < dist.dim():
        lambda_weight = lambda_weight.unsqueeze(-1)

    loss = (lambda_weight * dist).mean()
    return loss


def joint_gc_loss(
    consistency_model: nn.Module,
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_i: torch.Tensor,
    sigma_next: torch.Tensor,
    lambda_weight: torch.Tensor,
    mu: float = 0.5,
    distance_fn: callable = None,
) -> torch.Tensor:
    """
    Joint learning loss combining IC and GC (Equation 16).

    L_{GC-μ}(θ) = μ * L_GC(θ) + (1-μ) * L_CT(θ)

    Implemented by sampling a mask: each sample is GC with prob μ, IC with prob 1-μ.

    Args:
        consistency_model: f_θ
        x_star: Data samples [B, C, H, W]
        z: Noise samples [B, C, H, W]
        sigma_i: Current noise levels [B]
        sigma_next: Next noise levels [B]
        lambda_weight: Weighting [B]
        mu: Joint learning factor (probability of using GC)
        distance_fn: Distance function

    Returns:
        Scalar loss
    """
    if distance_fn is None:
        distance_fn = pseudo_huber_loss

    B = x_star.shape[0]

    # Create mask: Bernoulli(mu) for each sample
    mask = torch.bernoulli(torch.full((B,), mu, device=x_star.device))

    # Compute GC loss (with joint learning mask)
    loss = gc_consistency_loss(
        consistency_model, x_star, z, sigma_i, sigma_next,
        lambda_weight, distance_fn, mask=mask
    )
    return loss


def pseudo_huber_loss(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Pseudo-Huber loss (used as distance function D in consistency models).

    D(x, y) = sqrt(||x - y||^2 + c^2) - c

    This is a smooth approximation of the L1 norm and is used in
    Song and Dhariwal (2024) for consistency model training.

    Args:
        x: First tensor
        y: Second tensor
        c: Smoothing parameter

    Returns:
        Element-wise pseudo-Huber distance
    """
    diff_sq = ((x - y) ** 2).sum(dim=tuple(range(1, x.dim())), keepdim=True)
    return torch.sqrt(diff_sq + c ** 2) - c


def lpips_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    LPIPS-based distance function.
    Can be used as alternative to pseudo-Huber for perceptual quality.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        LPIPS distance
    """
    # Placeholder - requires lpips package
    diff_sq = ((x - y) ** 2).sum(dim=tuple(range(1, x.dim())), keepdim=True)
    return torch.sqrt(diff_sq + 1e-8)


def l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    L1 distance (used in some configurations for α=1).

    D(x, y) = ||x - y||_1
    """
    return (x - y).abs().sum(dim=tuple(range(1, x.dim())), keepdim=True)


def l2_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    L2 distance (used for α=2 in Theorem 1).

    D(x, y) = ||x - y||_2
    """
    return torch.sqrt(((x - y) ** 2).sum(dim=tuple(range(1, x.dim())), keepdim=True) + 1e-8)


def get_distance_fn(name: str = "pseudo_huber", **kwargs) -> callable:
    """
    Get a distance function by name.

    Args:
        name: "pseudo_huber", "l1", "l2", "lpips"
        **kwargs: Additional arguments to the distance function

    Returns:
        Distance function callable
    """
    if name == "pseudo_huber":
        c = kwargs.get("c", 1.0)
        return lambda x, y: pseudo_huber_loss(x, y, c=c)
    elif name == "l1":
        return l1_loss
    elif name == "l2":
        return l2_loss
    elif name == "lpips":
        return lpips_distance
    else:
        raise ValueError(f"Unknown distance function: {name}")
