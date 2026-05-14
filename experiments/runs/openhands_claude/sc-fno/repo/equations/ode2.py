"""
ODE2: Duffing Oscillator

  ẍ + δẋ + αt + βt³ = γ cos(ωt)

Rewritten as first-order system:
  dx/dt = v
  dv/dt = γ cos(ωt) - δv - αt - βt³

Initial conditions: x(0) = ε, v(0) = ζ

Parameters:
  α ∈ [0.02, 0.06], β ∈ [0.01, 0.03], γ ∈ [20, 60],
  δ ∈ [0.5, 1.5], ω ∈ [0.2, 0.6], ε ∈ [0.0, 0.2], ζ ∈ [0.0, 0.2]

Domain: t ∈ [0, 1], N=100 time steps, M=10 initial steps given
"""

import math
from typing import Dict

import torch


class ODE2Solver:
    """Numerical solver for the Duffing Oscillator using RK4."""

    param_names = ["alpha", "beta", "gamma", "delta", "omega", "epsilon", "zeta"]
    param_ranges = {
        "alpha": (0.02, 0.06),
        "beta": (0.01, 0.03),
        "gamma": (20.0, 60.0),
        "delta": (0.5, 1.5),
        "omega": (0.2, 0.6),
        "epsilon": (0.0, 0.2),
        "zeta": (0.0, 0.2),
    }
    N = 100
    M = 10
    t_start = 0.0
    t_end = 1.0

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.t = torch.linspace(self.t_start, self.t_end, self.N, device=device)
        self.dt = (self.t_end - self.t_start) / (self.N - 1)

    def _rhs(self, t: torch.Tensor, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        RHS of the Duffing system.

        Args:
            t: scalar time
            state: (batch, 2) [x, v]
            params: (batch, 7)

        Returns:
            dstate: (batch, 2)
        """
        x = state[:, 0]
        v = state[:, 1]
        alpha = params[:, 0]
        beta = params[:, 1]
        gamma = params[:, 2]
        delta = params[:, 3]
        omega = params[:, 4]

        dx_dt = v
        dv_dt = gamma * torch.cos(omega * t) - delta * v - alpha * t - beta * t ** 3

        return torch.stack([dx_dt, dv_dt], dim=1)

    def solve(self, params: torch.Tensor) -> torch.Tensor:
        """
        Solve Duffing ODE using RK4.

        Args:
            params: (batch, 7) [alpha, beta, gamma, delta, omega, epsilon, zeta]

        Returns:
            x: (batch, N) position trajectory
        """
        batch = params.shape[0]
        epsilon = params[:, 5]
        zeta = params[:, 6]

        state = torch.stack([epsilon, zeta], dim=1)  # (batch, 2)
        trajectory = [state[:, 0]]

        dt = self.dt
        for i in range(1, self.N):
            t_curr = self.t[i - 1]
            k1 = self._rhs(t_curr, state, params)
            k2 = self._rhs(t_curr + dt / 2, state + dt / 2 * k1, params)
            k3 = self._rhs(t_curr + dt / 2, state + dt / 2 * k2, params)
            k4 = self._rhs(t_curr + dt, state + dt * k3, params)
            state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(state[:, 0])

        return torch.stack(trajectory, dim=1)  # (batch, N)

    def jacobian_fd(self, params: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """
        Compute Jacobian ∂u/∂p via 4th-order central finite differences.

        Args:
            params: (batch, 7)
            eps: finite difference step size

        Returns:
            jac: (batch, N, 7)
        """
        batch, n_params = params.shape
        jac = torch.zeros(batch, self.N, n_params, device=self.device)

        for i in range(n_params):
            p_pp = params.clone()
            p_pp[:, i] += 2 * eps
            p_p = params.clone()
            p_p[:, i] += eps
            p_m = params.clone()
            p_m[:, i] -= eps
            p_mm = params.clone()
            p_mm[:, i] -= 2 * eps

            u_pp = self.solve(p_pp)
            u_p = self.solve(p_p)
            u_m = self.solve(p_m)
            u_mm = self.solve(p_mm)

            jac[:, :, i] = (-u_pp + 8 * u_p - 8 * u_m + u_mm) / (12 * eps)

        return jac

    def jacobian_ad(self, params: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian ∂u/∂p via automatic differentiation.

        Uses torch.autograd.functional.jacobian for efficiency.

        Returns:
            jac: (batch, N, n_params)
        """
        batch = params.shape[0]
        n_params = params.shape[1]
        jac = torch.zeros(batch, self.N, n_params, device=self.device)

        # Process each sample independently to get per-sample Jacobians
        for b in range(batch):
            p_b = params[b:b+1].detach().requires_grad_(True)  # (1, n_params)

            def solve_single(p):
                return self.solve(p)  # (1, N)

            # Compute Jacobian: output (1, N) w.r.t. input (1, n_params)
            jac_b = torch.autograd.functional.jacobian(
                solve_single, p_b, create_graph=False
            )  # (1, N, 1, n_params)
            jac[b] = jac_b[0, :, 0, :]  # (N, n_params)

        return jac

    def sample_params(self, n_samples: int) -> torch.Tensor:
        """Sample parameters uniformly from their ranges."""
        params = torch.zeros(n_samples, len(self.param_names), device=self.device)
        for i, name in enumerate(self.param_names):
            lo, hi = self.param_ranges[name]
            params[:, i] = torch.rand(n_samples, device=self.device) * (hi - lo) + lo
        return params

    def generate_dataset(self, n_samples: int, use_ad: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate dataset with solution paths and Jacobians.

        Returns dict with keys: params, u, jacobian
        """
        params = self.sample_params(n_samples)
        u = self.solve(params)
        if use_ad:
            jac = self.jacobian_ad(params)
        else:
            jac = self.jacobian_fd(params)
        return {"params": params, "u": u, "jacobian": jac}

    def pinn_residual(self, u_pred: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Compute PINN residual for the Duffing oscillator.

        Args:
            u_pred: (batch, N) predicted position
            params: (batch, 7)

        Returns:
            residual: (batch, N-2) residual at interior points
        """
        dt = self.dt
        alpha = params[:, 0:1]
        beta = params[:, 1:2]
        gamma = params[:, 2:3]
        delta = params[:, 3:4]
        omega = params[:, 4:5]
        t = self.t.unsqueeze(0)

        # Second derivative via finite differences
        d2u_dt2 = (u_pred[:, 2:] - 2 * u_pred[:, 1:-1] + u_pred[:, :-2]) / (dt ** 2)
        du_dt = (u_pred[:, 2:] - u_pred[:, :-2]) / (2 * dt)
        t_mid = t[:, 1:-1]

        rhs = gamma * torch.cos(omega * t_mid) - delta * du_dt - alpha * t_mid - beta * t_mid ** 3
        residual = d2u_dt2 - rhs
        return residual
