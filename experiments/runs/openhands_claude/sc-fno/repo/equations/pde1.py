"""
PDE1: Generalized Nonlinear Damped Wave Equation

  ∂²u/∂t² = c² ∂²u/∂x² + α ∂u/∂t + βu + γ sin(ωu)

Rewritten as first-order system:
  ∂u/∂t = v
  ∂v/∂t = c² ∂²u/∂x² + αv + βu + γ sin(ωu)

Initial conditions: u(x,0) = u0(x), ∂u/∂t(x,0) = u0'(x)
Boundary conditions: periodic

Parameters:
  c ∈ [0, 0.25], α ∈ [0, 0.1], β ∈ [0, 0.25], γ ∈ [0, 0.25], ω ∈ [0, 0.25]

Domain: x ∈ [0,1], t ∈ [0,1], Sx=20, N=30, M=5
"""

import math
from typing import Dict

import torch
import torch.nn.functional as F


class PDE1Solver:
    """Numerical solver for the Generalized Nonlinear Damped Wave Equation."""

    param_names = ["c", "alpha", "beta", "gamma", "omega"]
    param_ranges = {
        "c": (0.0, 0.25),
        "alpha": (0.0, 0.1),
        "beta": (0.0, 0.25),
        "gamma": (0.0, 0.25),
        "omega": (0.0, 0.25),
    }
    N = 30
    M = 5
    Sx = 20
    t_start = 0.0
    t_end = 1.0
    x_start = 0.0
    x_end = 1.0

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.t = torch.linspace(self.t_start, self.t_end, self.N, device=device)
        self.x = torch.linspace(self.x_start, self.x_end, self.Sx, device=device)
        self.dt = (self.t_end - self.t_start) / (self.N - 1)
        self.dx = (self.x_end - self.x_start) / (self.Sx - 1)

    def _laplacian_1d(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute ∂²u/∂x² with periodic boundary conditions.

        Args:
            u: (batch, Sx)

        Returns:
            d2u_dx2: (batch, Sx)
        """
        dx2 = self.dx ** 2
        u_left = torch.roll(u, 1, dims=-1)
        u_right = torch.roll(u, -1, dims=-1)
        return (u_left - 2 * u + u_right) / dx2

    def _rhs(self, u: torch.Tensor, v: torch.Tensor, params: torch.Tensor) -> tuple:
        """
        RHS of the wave equation system.

        Args:
            u: (batch, Sx) displacement
            v: (batch, Sx) velocity
            params: (batch, 5)

        Returns:
            (du_dt, dv_dt): each (batch, Sx)
        """
        c = params[:, 0:1]
        alpha = params[:, 1:2]
        beta = params[:, 2:3]
        gamma = params[:, 3:4]
        omega = params[:, 4:5]

        lap_u = self._laplacian_1d(u)
        du_dt = v
        dv_dt = c ** 2 * lap_u + alpha * v + beta * u + gamma * torch.sin(omega * u)
        return du_dt, dv_dt

    def _generate_ic(self, batch: int) -> tuple:
        """Generate random initial conditions."""
        # Random smooth initial displacement
        k = torch.randint(1, 4, (batch,), device=self.device).float()
        phi = torch.rand(batch, device=self.device) * 2 * math.pi
        x = self.x.unsqueeze(0)  # (1, Sx)
        u0 = 0.1 * torch.sin(k.unsqueeze(1) * math.pi * x + phi.unsqueeze(1))
        v0 = torch.zeros(batch, self.Sx, device=self.device)
        return u0, v0

    def solve(self, params: torch.Tensor, u0: torch.Tensor = None, v0: torch.Tensor = None) -> torch.Tensor:
        """
        Solve PDE1 using RK4.

        Args:
            params: (batch, 5)
            u0: (batch, Sx) initial displacement (generated if None)
            v0: (batch, Sx) initial velocity (generated if None)

        Returns:
            u_traj: (batch, Sx, N) solution trajectory
        """
        batch = params.shape[0]
        if u0 is None:
            u0, v0 = self._generate_ic(batch)

        u = u0.clone()
        v = v0.clone()
        trajectory = [u.clone()]

        dt = self.dt
        for _ in range(1, self.N):
            k1u, k1v = self._rhs(u, v, params)
            k2u, k2v = self._rhs(u + dt / 2 * k1u, v + dt / 2 * k1v, params)
            k3u, k3v = self._rhs(u + dt / 2 * k2u, v + dt / 2 * k2v, params)
            k4u, k4v = self._rhs(u + dt * k3u, v + dt * k3v, params)

            u = u + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
            v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            trajectory.append(u.clone())

        return torch.stack(trajectory, dim=2)  # (batch, Sx, N)

    def jacobian_fd(self, params: torch.Tensor, u0: torch.Tensor, v0: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """
        Compute Jacobian ∂u/∂p via 4th-order central finite differences.

        Returns:
            jac: (batch, Sx, N, n_params)
        """
        n_params = params.shape[1]
        batch = params.shape[0]
        jac = torch.zeros(batch, self.Sx, self.N, n_params, device=self.device)

        for i in range(n_params):
            p_pp = params.clone(); p_pp[:, i] += 2 * eps
            p_p = params.clone(); p_p[:, i] += eps
            p_m = params.clone(); p_m[:, i] -= eps
            p_mm = params.clone(); p_mm[:, i] -= 2 * eps

            u_pp = self.solve(p_pp, u0.clone(), v0.clone())
            u_p = self.solve(p_p, u0.clone(), v0.clone())
            u_m = self.solve(p_m, u0.clone(), v0.clone())
            u_mm = self.solve(p_mm, u0.clone(), v0.clone())

            jac[:, :, :, i] = (-u_pp + 8 * u_p - 8 * u_m + u_mm) / (12 * eps)

        return jac

    def jacobian_ad(self, params: torch.Tensor, u0: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian ∂u/∂p via automatic differentiation.

        Returns:
            jac: (batch, Sx, N, n_params)
        """
        batch = params.shape[0]
        n_params = params.shape[1]
        jac = torch.zeros(batch, self.Sx, self.N, n_params, device=self.device)

        for b in range(batch):
            p_b = params[b:b+1].detach().requires_grad_(True)
            u0_b = u0[b:b+1]
            v0_b = v0[b:b+1]

            def solve_single(p):
                return self.solve(p, u0_b, v0_b)  # (1, Sx, N)

            jac_b = torch.autograd.functional.jacobian(
                solve_single, p_b, create_graph=False
            )  # (1, Sx, N, 1, n_params)
            jac[b] = jac_b[0, :, :, 0, :]  # (Sx, N, n_params)

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

        Returns dict with keys: params, u, jacobian, u0, v0
        """
        params = self.sample_params(n_samples)
        u0, v0 = self._generate_ic(n_samples)
        u = self.solve(params, u0, v0)

        if use_ad:
            jac = self.jacobian_ad(params, u0, v0)
        else:
            jac = self.jacobian_fd(params, u0, v0)

        return {"params": params, "u": u, "jacobian": jac, "u0": u0, "v0": v0}

    def pinn_residual(self, u_pred: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Compute PINN residual for PDE1.

        Args:
            u_pred: (batch, Sx, N) predicted solution
            params: (batch, 5)

        Returns:
            residual: (batch, Sx, N-2) residual at interior time points
        """
        c = params[:, 0:1, None]
        alpha = params[:, 1:2, None]
        beta = params[:, 2:3, None]
        gamma = params[:, 3:4, None]
        omega = params[:, 4:5, None]

        dt = self.dt
        dx2 = self.dx ** 2

        # Second time derivative
        d2u_dt2 = (u_pred[:, :, 2:] - 2 * u_pred[:, :, 1:-1] + u_pred[:, :, :-2]) / dt ** 2
        du_dt = (u_pred[:, :, 2:] - u_pred[:, :, :-2]) / (2 * dt)
        u_mid = u_pred[:, :, 1:-1]

        # Spatial Laplacian with periodic BC
        u_left = torch.roll(u_mid, 1, dims=1)
        u_right = torch.roll(u_mid, -1, dims=1)
        lap_u = (u_left - 2 * u_mid + u_right) / dx2

        rhs = c ** 2 * lap_u + alpha * du_dt + beta * u_mid + gamma * torch.sin(omega * u_mid)
        return d2u_dt2 - rhs
