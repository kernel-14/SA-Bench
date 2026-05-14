"""
PDE4: Allen-Cahn Equation

  ∂u/∂t = ε ∂²u/∂x² + αu - βu³

Initial condition: u(x,0) = c tanh(ωx)
Boundary conditions: periodic

Parameters:
  c ∈ [0.1, 0.9], α ∈ [0.01, 1.0], β ∈ [0.01, 1.0], ω ∈ [5.0, 10.0], ε ∈ [0.01, 1.0]

Domain: x ∈ [0,1], t ∈ [0,1], Sx=40, N=30, M=5

Note: This PDE exhibits bifurcation behavior where small parameter changes
can cause abrupt phase transitions in solutions.
"""

import math
from typing import Dict, Optional

import torch


class PDE4Solver:
    """Numerical solver for the Allen-Cahn Equation."""

    param_names = ["c", "alpha", "beta", "omega", "epsilon"]
    param_ranges = {
        "c": (0.1, 0.9),
        "alpha": (0.01, 1.0),
        "beta": (0.01, 1.0),
        "omega": (5.0, 10.0),
        "epsilon": (0.01, 1.0),
    }
    N = 30
    M = 5
    Sx = 40
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

    def _generate_ic(self, params: torch.Tensor) -> torch.Tensor:
        """
        Generate initial condition u(x,0) = c tanh(ωx).

        Args:
            params: (batch, 5) [c, alpha, beta, omega, epsilon]

        Returns:
            u0: (batch, Sx)
        """
        c = params[:, 0:1]      # (batch, 1)
        omega = params[:, 3:4]  # (batch, 1)
        x = self.x.unsqueeze(0)  # (1, Sx)
        u0 = c * torch.tanh(omega * x)
        return u0  # (batch, Sx)

    def _rhs(self, u: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        RHS of Allen-Cahn equation.

        Args:
            u: (batch, Sx)
            params: (batch, 5)

        Returns:
            du_dt: (batch, Sx)
        """
        alpha = params[:, 1:2]    # (batch, 1)
        beta = params[:, 2:3]
        epsilon = params[:, 4:5]

        dx2 = self.dx ** 2
        u_left = torch.roll(u, 1, dims=-1)
        u_right = torch.roll(u, -1, dims=-1)
        d2u_dx2 = (u_left - 2 * u + u_right) / dx2

        du_dt = epsilon * d2u_dx2 + alpha * u - beta * u ** 3
        return du_dt

    def solve(self, params: torch.Tensor, u0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve Allen-Cahn PDE using RK4.

        Args:
            params: (batch, 5)
            u0: (batch, Sx) initial condition (generated from params if None)

        Returns:
            u_traj: (batch, Sx, N) solution trajectory
        """
        if u0 is None:
            u0 = self._generate_ic(params)

        u = u0.clone()
        trajectory = [u.clone()]

        dt = self.dt
        for _ in range(1, self.N):
            k1 = self._rhs(u, params)
            k2 = self._rhs(u + dt / 2 * k1, params)
            k3 = self._rhs(u + dt / 2 * k2, params)
            k4 = self._rhs(u + dt * k3, params)
            u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(u.clone())

        return torch.stack(trajectory, dim=2)  # (batch, Sx, N)

    def jacobian_fd(self, params: torch.Tensor, u0: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
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

            u_pp = self.solve(p_pp, u0.clone())
            u_p = self.solve(p_p, u0.clone())
            u_m = self.solve(p_m, u0.clone())
            u_mm = self.solve(p_mm, u0.clone())

            jac[:, :, :, i] = (-u_pp + 8 * u_p - 8 * u_m + u_mm) / (12 * eps)

        return jac

    def jacobian_ad(self, params: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
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

            def solve_single(p):
                return self.solve(p, u0_b)  # (1, Sx, N)

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

        Returns dict with keys: params, u, jacobian, u0
        """
        params = self.sample_params(n_samples)
        u0 = self._generate_ic(params)
        u = self.solve(params, u0)

        if use_ad:
            jac = self.jacobian_ad(params, u0)
        else:
            jac = self.jacobian_fd(params, u0)

        return {"params": params, "u": u, "jacobian": jac, "u0": u0}

    def pinn_residual(self, u_pred: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Compute PINN residual for Allen-Cahn equation.

        Args:
            u_pred: (batch, Sx, N) predicted solution
            params: (batch, 5)

        Returns:
            residual: (batch, Sx, N-2) residual at interior time points
        """
        alpha = params[:, 1:2, None]
        beta = params[:, 2:3, None]
        epsilon = params[:, 4:5, None]

        dt = self.dt
        dx2 = self.dx ** 2

        du_dt = (u_pred[:, :, 2:] - u_pred[:, :, :-2]) / (2 * dt)
        u_mid = u_pred[:, :, 1:-1]

        u_left = torch.roll(u_mid, 1, dims=1)
        u_right = torch.roll(u_mid, -1, dims=1)
        d2u_dx2 = (u_left - 2 * u_mid + u_right) / dx2

        rhs = epsilon * d2u_dx2 + alpha * u_mid - beta * u_mid ** 3
        return du_dt - rhs
