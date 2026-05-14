"""
PDE2: Forced Burgers' Equation

  (1/π) ∂u/∂t + α u ∂u/∂x = γ ∂²u/∂x² + δ sin(ωt)

Equivalently:
  ∂u/∂t = π(-α u ∂u/∂x + γ ∂²u/∂x² + δ sin(ωt))

Initial condition:
  u(x,0) = exp(-(x-x0)²/(2σ²)) + sin(0.5πx),  x0=0.5, σ=0.3

Boundary conditions: periodic, u(0,t) = u(1,t)

Parameters:
  α ∈ [0.1, 1.0], γ ∈ [0.025, 0.25], δ ∈ [0.1, 0.5], ω ∈ [0.01, 0.1]

Domain: x ∈ [0,1], t ∈ [0,π], Sx=40, N=30, M=5

Zoned variant: spatial domain divided into S=40 zones with per-zone α and δ,
plus global γ and ω → 2*40+2 = 82 parameters total.
"""

import math
from typing import Dict, Optional

import torch


class PDE2Solver:
    """Numerical solver for the Forced Burgers' Equation."""

    param_names = ["alpha", "gamma", "delta", "omega"]
    param_ranges = {
        "alpha": (0.1, 1.0),
        "gamma": (0.025, 0.25),
        "delta": (0.1, 0.5),
        "omega": (0.01, 0.1),
    }
    N = 30
    M = 5
    Sx = 40
    t_start = 0.0
    t_end = math.pi
    x_start = 0.0
    x_end = 1.0

    def __init__(self, device: torch.device = torch.device("cpu"), zoned: bool = False):
        self.device = device
        self.zoned = zoned
        self.t = torch.linspace(self.t_start, self.t_end, self.N, device=device)
        self.x = torch.linspace(self.x_start, self.x_end, self.Sx, device=device)
        self.dt = (self.t_end - self.t_start) / (self.N - 1)
        self.dx = (self.x_end - self.x_start) / (self.Sx - 1)

        if zoned:
            # 2*Sx + 2 parameters: alpha_i (Sx), delta_i (Sx), gamma, omega
            self.n_params = 2 * self.Sx + 2
            self.param_names_zoned = (
                [f"alpha_{i}" for i in range(self.Sx)]
                + [f"delta_{i}" for i in range(self.Sx)]
                + ["gamma", "omega"]
            )

    def _generate_ic(self, batch: int) -> torch.Tensor:
        """Generate initial condition u(x,0)."""
        x = self.x.unsqueeze(0)  # (1, Sx)
        x0 = 0.5
        sigma = 0.3
        u0 = torch.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + torch.sin(0.5 * math.pi * x)
        return u0.expand(batch, -1).clone()  # (batch, Sx)

    def _rhs(self, t_val: float, u: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        RHS of Burgers' equation.

        Args:
            t_val: current time (scalar)
            u: (batch, Sx) current solution
            params: (batch, 4) or (batch, 2*Sx+2) for zoned

        Returns:
            du_dt: (batch, Sx)
        """
        dx = self.dx

        if self.zoned:
            alpha = params[:, : self.Sx]          # (batch, Sx)
            delta = params[:, self.Sx : 2 * self.Sx]  # (batch, Sx)
            gamma = params[:, -2:-1]               # (batch, 1)
            omega = params[:, -1:]                 # (batch, 1)
        else:
            alpha = params[:, 0:1]  # (batch, 1)
            gamma = params[:, 1:2]
            delta = params[:, 2:3]
            omega = params[:, 3:4]

        # Upwind scheme for advection term (periodic BC)
        u_left = torch.roll(u, 1, dims=-1)
        u_right = torch.roll(u, -1, dims=-1)

        # Central difference for advection
        du_dx = (u_right - u_left) / (2 * dx)

        # Second derivative for diffusion
        d2u_dx2 = (u_left - 2 * u + u_right) / (dx ** 2)

        forcing = delta * torch.sin(omega * t_val)
        du_dt = math.pi * (-alpha * u * du_dx + gamma * d2u_dx2 + forcing)
        return du_dt

    def solve(self, params: torch.Tensor, u0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve PDE2 using RK4.

        Args:
            params: (batch, 4) or (batch, 2*Sx+2) for zoned
            u0: (batch, Sx) initial condition (generated if None)

        Returns:
            u_traj: (batch, Sx, N) solution trajectory
        """
        batch = params.shape[0]
        if u0 is None:
            u0 = self._generate_ic(batch)

        u = u0.clone()
        trajectory = [u.clone()]

        dt = self.dt
        for i in range(1, self.N):
            t_curr = self.t[i - 1].item()
            t_mid = t_curr + dt / 2
            t_next = t_curr + dt

            k1 = self._rhs(t_curr, u, params)
            k2 = self._rhs(t_mid, u + dt / 2 * k1, params)
            k3 = self._rhs(t_mid, u + dt / 2 * k2, params)
            k4 = self._rhs(t_next, u + dt * k3, params)

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
        if self.zoned:
            params = torch.zeros(n_samples, self.n_params, device=self.device)
            alpha_lo, alpha_hi = self.param_ranges["alpha"]
            delta_lo, delta_hi = self.param_ranges["delta"]
            gamma_lo, gamma_hi = self.param_ranges["gamma"]
            omega_lo, omega_hi = self.param_ranges["omega"]

            params[:, : self.Sx] = (
                torch.rand(n_samples, self.Sx, device=self.device) * (alpha_hi - alpha_lo) + alpha_lo
            )
            params[:, self.Sx : 2 * self.Sx] = (
                torch.rand(n_samples, self.Sx, device=self.device) * (delta_hi - delta_lo) + delta_lo
            )
            params[:, -2] = torch.rand(n_samples, device=self.device) * (gamma_hi - gamma_lo) + gamma_lo
            params[:, -1] = torch.rand(n_samples, device=self.device) * (omega_hi - omega_lo) + omega_lo
        else:
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
        u0 = self._generate_ic(n_samples)
        u = self.solve(params, u0)

        if use_ad:
            jac = self.jacobian_ad(params, u0)
        else:
            jac = self.jacobian_fd(params, u0)

        return {"params": params, "u": u, "jacobian": jac, "u0": u0}

    def pinn_residual(self, u_pred: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Compute PINN residual for PDE2.

        Args:
            u_pred: (batch, Sx, N) predicted solution
            params: (batch, 4)

        Returns:
            residual: (batch, Sx, N-2) residual at interior time points
        """
        alpha = params[:, 0:1, None]
        gamma = params[:, 1:2, None]
        delta = params[:, 2:3, None]
        omega = params[:, 3:4, None]

        dt = self.dt
        dx = self.dx
        t = self.t.unsqueeze(0).unsqueeze(0)  # (1, 1, N)

        du_dt = (u_pred[:, :, 2:] - u_pred[:, :, :-2]) / (2 * dt)
        u_mid = u_pred[:, :, 1:-1]

        u_left = torch.roll(u_mid, 1, dims=1)
        u_right = torch.roll(u_mid, -1, dims=1)
        du_dx = (u_right - u_left) / (2 * dx)
        d2u_dx2 = (u_left - 2 * u_mid + u_right) / (dx ** 2)

        t_mid = t[:, :, 1:-1]
        forcing = delta * torch.sin(omega * t_mid)

        lhs = du_dt / math.pi
        rhs = -alpha * u_mid * du_dx + gamma * d2u_dx2 + forcing
        return lhs - rhs
