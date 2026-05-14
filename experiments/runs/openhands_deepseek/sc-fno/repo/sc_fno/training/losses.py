"""Loss functions for SC-FNO training.

Loss terms (per Section 2):
- L_u: Data loss on solution paths (MSE)
- L_s: Sensitivity loss on Jacobians du/dp
- L_eq: Physics-informed equation loss (optional PINN regularizer)

The total loss is: L_total = c1 * L_u + c2 * L_s + c3 * L_eq
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def data_loss(
    u_pred: torch.Tensor, u_true: torch.Tensor
) -> torch.Tensor:
    """MSE loss on solution paths.

    L_u = (1/M) * Σ || û(x_j, t_j; p) - u(x_j, t_j; p) ||^2

    Args:
        u_pred: Predicted solution (B, *grid_dims, C)
        u_true: True solution (B, *grid_dims, C)
    Returns:
        Scalar loss.
    """
    return F.mse_loss(u_pred, u_true)


def sensitivity_loss(
    du_pred: Dict[str, torch.Tensor],
    du_true: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """MSE loss on parameter sensitivities (Jacobians).

    L_s = (1/M) * Σ || ∂û/∂p - ∂u/∂p ||^2

    For each parameter, computes MSE between predicted and true Jacobians.
    Aggregates across all parameters.

    Args:
        du_pred: Dict mapping param name -> predicted Jacobian.
        du_true: Dict mapping param name -> true Jacobian.
    Returns:
        Scalar loss (mean across parameters).
    """
    losses = []
    for name in du_pred:
        if name in du_true:
            losses.append(F.mse_loss(du_pred[name], du_true[name]))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def pde_residual(
    u_pred: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    params: Dict[str, torch.Tensor],
    eq_type: str,
) -> torch.Tensor:
    """Compute PDE residual for equation loss.

    L_PDE = (1/N) Σ |N[u(x_i, t_i); p]|^2

    Supports the PDEs used in the paper:
    - pde1: Wave equation
    - pde2: Burgers equation
    - pde3: Navier-Stokes

    Args:
        u_pred: Predicted solution.
        x: Spatial coordinate grid.
        t: Time coordinate grid.
        params: Dict of parameter tensors.
        eq_type: Type of PDE ("pde1", "pde2", "pde3", "pde4")
    Returns:
        Scalar residual loss.
    """
    if eq_type == "pde1":
        return _pde1_residual(u_pred, x, t, params)
    elif eq_type == "pde2":
        return _pde2_residual(u_pred, x, t, params)
    elif eq_type == "pde3":
        return _pde3_residual(u_pred, x, t, params)
    elif eq_type == "pde4":
        return _pde4_residual(u_pred, x, t, params)
    else:
        return torch.tensor(0.0)


def _pde1_residual(
    u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, params: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Residual for PDE1: ∂²u/∂t² = c² ∂²u/∂x² + α ∂u/∂t + β u + γ sin(ω u)."""
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    du_dt = (u[..., 2:, :] - u[..., :-2, :]) / (2 * dt)
    d2u_dt2 = (u[..., 2:, :] + u[..., :-2, :] - 2 * u[..., 1:-1, :]) / (dt**2)

    u_mid = u[..., 1:-1, :]
    d2u_dx2 = (
        u[..., 1:-1, 2:] + u[..., 1:-1, :-2] - 2 * u[..., 1:-1, 1:-1]
    ) / (dx**2)

    rhs = (
        params["c"] ** 2 * d2u_dx2
        + params["alpha"] * du_dt
        + params["beta"] * u_mid
        + params["gamma"] * torch.sin(params["omega"] * u_mid)
    )
    return F.mse_loss(d2u_dt2, rhs)


def _pde2_residual(
    u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, params: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Residual for PDE2: (1/π) ∂u/∂t + α u ∂u/∂x = γ ∂²u/∂x² + δ sin(ω t)."""
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    du_dt = (u[..., 2:, :] - u[..., :-2, :]) / (2 * dt)
    du_dx = (u[..., 1:-1, 2:] - u[..., 1:-1, :-2]) / (2 * dx)
    d2u_dx2 = (
        u[..., 1:-1, 2:] + u[..., 1:-1, :-2] - 2 * u[..., 1:-1, 1:-1]
    ) / (dx**2)

    u_mid = u[..., 1:-1, 1:-1]

    left = du_dt / torch.pi + params["alpha"] * u_mid * du_dx
    right = (
        params["gamma"] * d2u_dx2
        + params["delta"] * torch.sin(params["omega"] * t[1:-1].unsqueeze(-1))
    )
    return F.mse_loss(left, right)


def _pde3_residual(
    u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, params: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Residual for PDE3: Navier-Stokes (vorticity).
    Computed at final time only in the paper setting.
    """
    return torch.tensor(0.0)


def _pde4_residual(
    u: torch.Tensor, x: torch.Tensor, t: torch.Tensor, params: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Residual for PDE4: ∂u/∂t = ε ∂²u/∂x² + α u - β u³."""
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    du_dt = (u[..., 2:, :] - u[..., :-2, :]) / (2 * dt)
    d2u_dx2 = (
        u[..., 1:-1, 2:] + u[..., 1:-1, :-2] - 2 * u[..., 1:-1, 1:-1]
    ) / (dx**2)

    u_mid = u[..., 1:-1, 1:-1]

    rhs = (
        params["epsilon"] * d2u_dx2
        + params["alpha"] * u_mid
        - params["beta"] * u_mid**3
    )
    return F.mse_loss(du_dt, rhs)


def pde_loss(
    u_pred: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    params: Dict[str, torch.Tensor],
    eq_type: str,
) -> torch.Tensor:
    """Total PDE loss = L_PDE + α * (L_IC + L_BC)."""
    return pde_residual(u_pred, x, t, params, eq_type)


def ic_loss(
    u_pred: torch.Tensor, u_ic: torch.Tensor
) -> torch.Tensor:
    """Initial condition loss: ensure u(x, 0) matches IC."""
    return F.mse_loss(u_pred[..., 0, :], u_ic) if u_pred.dim() >= 3 else F.mse_loss(
        u_pred[..., :1, :], u_ic
    )


def bc_loss(
    u_pred: torch.Tensor,
) -> torch.Tensor:
    """Periodic boundary condition loss."""
    if u_pred.dim() < 3:
        return torch.tensor(0.0)
    return F.mse_loss(u_pred[..., :1], u_pred[..., -1:])
