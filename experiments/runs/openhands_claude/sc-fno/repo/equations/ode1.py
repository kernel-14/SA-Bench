"""
ODE1: Composite Harmonic Oscillator

  du/dt = α sin(απt) + β cos(βπt),   u(0) = sin(γπ)

Analytical solution:
  u(t) = -1/π cos(απt) + 1/π sin(βπt) + sin(γπ) + 1/π

Analytical sensitivities:
  ∂u/∂α = t sin(απt)
  ∂u/∂β = t cos(βπt)
  ∂u/∂γ = π cos(γπ)

Parameters: α ∈ [1,3], β ∈ [1,3], γ ∈ [0,1]
Domain: t ∈ [0,1], N=100 time steps, M=10 initial steps given
"""

import math
from typing import Dict, Tuple

import torch


class ODE1Solver:
    """Analytical solver for the Composite Harmonic Oscillator."""

    param_names = ["alpha", "beta", "gamma"]
    param_ranges = {"alpha": (1.0, 3.0), "beta": (1.0, 3.0), "gamma": (0.0, 1.0)}
    N = 100
    M = 10
    t_start = 0.0
    t_end = 1.0

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.t = torch.linspace(self.t_start, self.t_end, self.N, device=device)

    def solve(self, params: torch.Tensor) -> torch.Tensor:
        """
        Compute analytical solution.

        Args:
            params: (batch, 3) tensor [alpha, beta, gamma]

        Returns:
            u: (batch, N) solution trajectory
        """
        alpha = params[:, 0:1]  # (batch, 1)
        beta = params[:, 1:2]
        gamma = params[:, 2:3]
        t = self.t.unsqueeze(0)  # (1, N)

        u = (
            -1.0 / math.pi * torch.cos(alpha * math.pi * t)
            + 1.0 / math.pi * torch.sin(beta * math.pi * t)
            + torch.sin(gamma * math.pi)
            + 1.0 / math.pi
        )
        return u  # (batch, N)

    def jacobian(self, params: torch.Tensor) -> torch.Tensor:
        """
        Compute analytical Jacobian ∂u/∂p.

        Args:
            params: (batch, 3) tensor [alpha, beta, gamma]

        Returns:
            jac: (batch, N, 3) Jacobian tensor
        """
        alpha = params[:, 0:1]  # (batch, 1)
        beta = params[:, 1:2]
        gamma = params[:, 2:3]
        t = self.t.unsqueeze(0)  # (1, N)

        du_dalpha = t * torch.sin(alpha * math.pi * t)          # (batch, N)
        du_dbeta = t * torch.cos(beta * math.pi * t)            # (batch, N)
        du_dgamma = math.pi * torch.cos(gamma * math.pi) * torch.ones_like(t)  # (batch, N)

        jac = torch.stack([du_dalpha, du_dbeta, du_dgamma], dim=-1)  # (batch, N, 3)
        return jac

    def sample_params(self, n_samples: int) -> torch.Tensor:
        """Sample parameters uniformly from their ranges."""
        params = torch.zeros(n_samples, 3, device=self.device)
        for i, name in enumerate(self.param_names):
            lo, hi = self.param_ranges[name]
            params[:, i] = torch.rand(n_samples, device=self.device) * (hi - lo) + lo
        return params

    def generate_dataset(self, n_samples: int) -> Dict[str, torch.Tensor]:
        """
        Generate dataset with solution paths and Jacobians.

        Returns dict with keys: params, u, jacobian
        """
        params = self.sample_params(n_samples)
        u = self.solve(params)
        jac = self.jacobian(params)
        return {"params": params, "u": u, "jacobian": jac}

    def pinn_residual(self, u_pred: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Compute PINN residual: du/dt - (α sin(απt) + β cos(βπt)) = 0.

        Args:
            u_pred: (batch, N) predicted solution (requires_grad=True)
            params: (batch, 3) parameters

        Returns:
            residual: (batch, N-1) PDE residual at interior points
        """
        alpha = params[:, 0:1]
        beta = params[:, 1:2]
        dt = (self.t_end - self.t_start) / (self.N - 1)
        t = self.t.unsqueeze(0)

        rhs = alpha * torch.sin(alpha * math.pi * t) + beta * torch.cos(beta * math.pi * t)

        # Finite difference for du/dt
        du_dt = (u_pred[:, 1:] - u_pred[:, :-1]) / dt
        residual = du_dt - (rhs[:, :-1] + rhs[:, 1:]) / 2.0
        return residual
