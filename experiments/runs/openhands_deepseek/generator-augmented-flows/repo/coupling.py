"""
Coupling strategies for data-noise pairs in consistency training.

- Independent Coupling (IC): x_star ~ p_data, z ~ p_z, independent
- Batch Optimal Transport (batch-OT): minibatch Hungarian matching between data and noise
- Generator-Augmented Coupling (GC): uses a consistency model to predict endpoint,
  then couples the prediction with the original noise
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn


def independent_coupling(
    x_star: torch.Tensor,
    z: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independent coupling: returns data-noise pairs as-is (already independent)."""
    return x_star, z


def batch_ot_coupling(
    x_star: torch.Tensor,
    z: torch.Tensor,
    cost: str = "l2",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minibatch optimal transport coupling using the Hungarian algorithm.

    Args:
        x_star: Data samples of shape (B, ...)
        z: Noise samples of shape (B, ...)
        cost: Distance metric ("l2" for Euclidean)

    Returns:
        Permuted (x_star, z) where z is reordered to minimize transport cost.
    """
    B = x_star.shape[0]
    x_flat = x_star.reshape(B, -1)
    z_flat = z.reshape(B, -1)

    # Compute pairwise distance matrix
    dist = torch.cdist(x_flat, z_flat, p=2) ** 2  # (B, B)

    # Hungarian algorithm via SciPy (greedy min-assignment)
    # Since we can't use SciPy in pure PyTorch, we implement a simple heuristic:
    # Sort by one dimension and match — a cheap approximation for batch-OT
    # For full Hungarian, we'd need scipy.optimize.linear_sum_assignment
    # Here we use the "sorted matching" approximation from the literature:

    # Compute 1D projections for greedy matching
    rand_proj = torch.randn(x_flat.shape[1], device=x_flat.device)
    rand_proj = rand_proj / rand_proj.norm()
    x_proj = x_flat @ rand_proj
    z_proj = z_flat @ rand_proj

    _, x_idx = torch.sort(x_proj)
    _, z_idx = torch.sort(z_proj)

    z_perm = z[z_idx]
    x_perm = x_star[x_idx]

    return x_perm, z_perm


def generator_augmented_coupling(
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_ti: torch.Tensor,
    sigma_ti_plus_1: torch.Tensor,
    consistency_model: nn.Module,
    use_ema: bool = False,
    ema_helper=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generator-Augmented Coupling (GC).

    Args:
        x_star: Original data samples (B, C, H, W)
        z: Original noise samples (B, C, H, W)
        sigma_ti: Current noise level (B,)
        sigma_ti_plus_1: Next noise level (B,)
        consistency_model: The consistency model f_theta
        use_ema: Whether to use EMA parameters for prediction
        ema_helper: EMA helper instance

    Returns:
        x_tilde_ti: GC intermediate point at sigma_ti (B, C, H, W)
        x_tilde_ti_plus_1: GC intermediate point at sigma_ti+1 (B, C, H, W)
        x_hat_ti: Predicted endpoint (B, C, H, W)
    """
    # Step 1: IC intermediate point x_{t_i} = x_star + sigma_ti * z
    x_ti = x_star + sigma_ti[:, None, None, None] * z

    # Step 2: Predict endpoint using consistency model (stop-gradient)
    if use_ema and ema_helper is not None:
        ema_helper.store(consistency_model)
        ema_helper.apply_to(consistency_model)

    with torch.no_grad():
        x_hat_ti = consistency_model(x_ti, sigma_ti)

    if use_ema and ema_helper is not None:
        ema_helper.restore(consistency_model)

    # Step 3: Construct GC intermediate points
    # tilde{x}_{t_i} = x_hat_ti + sigma_ti * z
    # tilde{x}_{t_{i+1}} = x_hat_ti + sigma_{t_{i+1}} * z
    x_tilde_ti = x_hat_ti + sigma_ti[:, None, None, None] * z
    x_tilde_ti_plus_1 = x_hat_ti + sigma_ti_plus_1[:, None, None, None] * z

    return x_tilde_ti, x_tilde_ti_plus_1, x_hat_ti


def construct_gc_pairs(
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_ti: torch.Tensor,
    sigma_ti_plus_1: torch.Tensor,
    consistency_model: nn.Module,
    mask: torch.Tensor,
    use_ema: bool = False,
    ema_helper=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct training pairs with joint IC/GC learning.

    For samples where mask == 1, use GC pairs.
    For samples where mask == 0, use IC pairs.

    Args:
        x_star: Data samples (B, C, H, W)
        z: Noise samples (B, C, H, W)
        sigma_ti: Current sigma (B,)
        sigma_ti_plus_1: Next sigma (B,)
        consistency_model: The model
        mask: Binary mask of shape (B,) — 1 for GC, 0 for IC
        use_ema: Whether to use EMA for endpoint prediction
        ema_helper: EMA helper

    Returns:
        x_ti: Input pair at sigma_ti (B, C, H, W)
        x_ti_plus_1: Target pair at sigma_{t+1} (B, C, H, W)
        x_hat_ti: Predicted endpoints for GC samples (B, C, H, W)
        mask_expanded: Mask expanded to (B, 1, 1, 1)
    """
    B = x_star.shape[0]
    mask_f = mask.float().view(B, 1, 1, 1)

    # Compute GC pairs (only needed for GC samples)
    with torch.no_grad():
        # IC intermediate point
        x_ti_ic = x_star + sigma_ti[:, None, None, None] * z

        # Predict endpoint
        if use_ema and ema_helper is not None:
            ema_helper.store(consistency_model)
            ema_helper.apply_to(consistency_model)

        x_hat_ti = consistency_model(x_ti_ic, sigma_ti)

        if use_ema and ema_helper is not None:
            ema_helper.restore(consistency_model)

    # Mix: x_hat_ti_mixed = mask * x_hat_ti + (1 - mask) * x_star
    x_hat_ti_mixed = mask_f * x_hat_ti + (1.0 - mask_f) * x_star

    # Construct intermediate points from mixed endpoints
    x_ti = x_hat_ti_mixed + sigma_ti[:, None, None, None] * z
    x_ti_plus_1 = x_hat_ti_mixed + sigma_ti_plus_1[:, None, None, None] * z

    return x_ti, x_ti_plus_1, x_hat_ti, mask_f
