"""
Data-noise coupling strategies for consistency model training.

Implements three coupling strategies:
1. Independent Coupling (IC): q_I(x_*, z) = p_*(x_*) p_z(z)
2. Minibatch Optimal Transport Coupling (batch-OT): Hungarian/Sinkhorn
3. Generator-Augmented Coupling (GC): Predict endpoints via consistency model

Based on the paper Equations (13)-(14) and Section 3-4.
"""

import torch
import torch.nn as nn
from typing import Optional


def independent_coupling(
    x_star: torch.Tensor,
    z: torch.Tensor,
) -> tuple:
    """
    Standard independent coupling (IC).
    q_I(x_*, z) = p_*(x_*) p_z(z)
    Data and noise are already independent; returned as-is.
    """
    return x_star, z


def get_intermediate_points(
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Construct intermediate noisy points: x_t = x_* + sigma_t * z
    """
    sigma_expanded = sigma
    while sigma_expanded.dim() < x_star.dim():
        sigma_expanded = sigma_expanded.unsqueeze(-1)
    return x_star + sigma_expanded * z


def generator_augmented_coupling(
    x_star: torch.Tensor,
    z: torch.Tensor,
    sigma_i: torch.Tensor,
    sigma_next: torch.Tensor,
    consistency_model: nn.Module,
    mask: Optional[torch.Tensor] = None,
) -> tuple:
    """
    Generator-Augmented Coupling (GC) from Equations (13)-(14).

    Steps:
      (x_*, z) ~ q_I                       [IC sampling]
      x_{t_i} = x_* + sigma_{t_i} * z      [IC intermediate point]
      hat{x}_{t_i} = sg(f(x_{t_i}, sigma_{t_i}))  [endpoint prediction]
      tilde{x}_{t_i} = hat{x}_{t_i} + sigma_{t_i} * z
      tilde{x}_{t_{i+1}} = hat{x}_{t_i} + sigma_{t_{i+1}} * z

    Joint learning (mask):
      hat{x}_{t_i} = m * sg(f(...)) + (1-m) * x_*

    Args:
        x_star: Data samples [B, C, H, W]
        z: Noise samples [B, C, H, W]
        sigma_i: Current noise levels [B]
        sigma_next: Next noise levels [B]
        consistency_model: Consistency model f_theta
        mask: Binary mask [B] for joint learning

    Returns:
        (tilde_x_ti, tilde_x_next): GC intermediate points
    """
    with torch.no_grad():
        x_ti = get_intermediate_points(x_star, z, sigma_i)
        x_hat = consistency_model(x_ti, sigma_i)

        if mask is not None:
            mask_e = mask
            while mask_e.dim() < x_star.dim():
                mask_e = mask_e.unsqueeze(-1)
            x_hat = mask_e * x_hat + (1.0 - mask_e) * x_star

    tilde_x_ti = get_intermediate_points(x_hat, z, sigma_i)
    tilde_x_next = get_intermediate_points(x_hat, z, sigma_next)
    return tilde_x_ti, tilde_x_next


def batch_ot_coupling(
    x_star: torch.Tensor,
    z: torch.Tensor,
    ot_solver: str = "sinkhorn",
    sinkhorn_reg: float = 0.05,
    sinkhorn_iters: int = 100,
) -> tuple:
    """
    Minibatch Optimal Transport coupling.

    Finds permutation pi minimizing sum_i ||x_i - z_{pi(i)}||^2.

    Args:
        x_star: Data samples [B, ...]
        z: Noise samples [B, ...]
        ot_solver: "sinkhorn" or "hungarian"
        sinkhorn_reg: Sinkhorn regularization
        sinkhorn_iters: Sinkhorn iterations

    Returns:
        (x_star, z_permuted)
    """
    B = x_star.shape[0]
    x_flat = x_star.reshape(B, -1)
    z_flat = z.reshape(B, -1)

    cost = torch.cdist(x_flat, z_flat, p=2) ** 2

    if ot_solver == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
            cost_np = cost.detach().cpu().numpy()
            _, col_ind = linear_sum_assignment(cost_np)
            perm = torch.tensor(col_ind, device=x_star.device, dtype=torch.long)
            return x_star, z[perm]
        except ImportError:
            ot_solver = "sinkhorn"

    # Sinkhorn
    reg = sinkhorn_reg * cost.mean()
    K = torch.exp(-cost / reg)
    u = torch.ones(B, 1, device=x_star.device)
    v = torch.ones(B, 1, device=x_star.device)
    for _ in range(sinkhorn_iters):
        v = 1.0 / (B * K.T @ u + 1e-8)
        u = 1.0 / (B * K @ v + 1e-8)
    P = u * K * v.T
    z_permuted = (P @ z_flat).reshape(z.shape)
    return x_star, z_permuted


def compute_r_proxy(
    x_t: torch.Tensor,
    z: torch.Tensor,
    sigma_t: torch.Tensor,
    denoiser: nn.Module,
) -> torch.Tensor:
    """
    Compute proxy regularizer from Section 4.2.1:
      tilde{R}_t = E[||dot{x}_t - v_t(x_t)||^2]

    In EDM setting (sigma_t = t):
      dot{x}_t = z
      v_t(x_t) = (1/t) * (x_t - D(x_t, t))

    Returns:
        Scalar mean squared difference
    """
    with torch.no_grad():
        x_t_denoised = denoiser(x_t, sigma_t)
        sigma_e = sigma_t
        while sigma_e.dim() < x_t.dim():
            sigma_e = sigma_e.unsqueeze(-1)
        v_t = (x_t - x_t_denoised) / sigma_e
        diff = z - v_t
        return (diff ** 2).mean()
