```python
# solver.py
# ============================================================================
# Purpose: Provide differentiable and finite‑difference solvers for the ODEs and
#          PDEs studied in the SC‑FNO paper. Each solver implements the abstract
#          Solver interface and returns both solution paths and full Jacobians
#          (∂u/∂p) computed via automatic differentiation.
# ============================================================================

import math
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np
import torch
from torchdifq import odeint


# ---------------------------------------------------------------------------
# Helper spatial operators for PDE solvers (1D periodic domains)
# ---------------------------------------------------------------------------

def spatial_gradient_1d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Second‑order central difference on a periodic domain.

    Args:
        u: 1D tensor (..., S) on a uniform grid.
        dx: grid spacing.

    Returns:
        Tensor of same shape as u, containing ∂u/∂x.
    """
    u_right = torch.roll(u, shifts=-1, dims=-1)
    u_left = torch.roll(u, shifts=1, dims=-1)
    return (u_right - u_left) / (2.0 * dx)


def spatial_laplacian_1d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Second‑order central difference for ∂²u/∂x², periodic.

    Args:
        u: 1D tensor (..., S).
        dx: grid spacing.

    Returns:
        Tensor of same shape as u.
    """
    u_right = torch.roll(u, shifts=-1, dims=-1)
    u_left = torch.roll(u, shifts=1, dims=-1)
    return (u_right - 2.0 * u + u_left) / (dx * dx)


# ---------------------------------------------------------------------------
# Helper functions for Navier‑Stokes solver (2D periodic, spectral)
# ---------------------------------------------------------------------------

def _fft_poisson_solver(omega: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Solve ∇² ψ = -ω on a 2D periodic domain using the Fourier method.

    Args:
        omega: (..., Ny, Nx) tensor, vorticity.
        dx, dy: grid spacings.

    Returns:
        psi: stream function, same shape as omega.
    """
    Ny, Nx = omega.shape[-2], omega.shape[-1]
    # Wavenumbers kx, ky (scaled)
    kx = torch.fft.fftfreq(Nx, d=dx) * 2.0 * math.pi
    ky = torch.fft.fftfreq(Ny, d=dy) * 2.0 * math.pi
    kx = kx.to(omega.device)
    ky = ky.to(omega.device)

    # k_sq = kx² + ky² (broadcast over 2D grid)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='xy')
    k_sq = kx_grid ** 2 + ky_grid ** 2
    k_sq[0, 0] = 1.0  # avoid division by zero for DC mode

    omega_hat = torch.fft.fft2(omega)
    psi_hat = omega_hat / k_sq  # no negative sign? Actually ∇² ψ = -ω, so -ω_hat = -k_sq * psi_hat => psi_hat = ω_hat / k_sq.
    # Zero out DC mode
    psi_hat[..., 0, 0] = 0.0
    psi = torch.fft.ifft2(psi_hat).real
    return psi


def _fourier_gradient(omega: torch.Tensor, dx: float, dy: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ∂ω/∂x and ∂ω/∂y using spectral differentiation.

    Args:
        omega: (..., Ny, Nx) tensor.
        dx, dy: grid spacings.

    Returns:
        omega_x, omega_y: same shape as omega.
    """
    Ny, Nx = omega.shape[-2], omega.shape[-1]
    kx = torch.fft.fftfreq(Nx, d=dx) * 2.0 * math.pi
    ky = torch.fft.fftfreq(Ny, d=dy) * 2.0 * math.pi
    kx = kx.to(omega.device)
    ky = ky.to(omega.device)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='xy')

    omega_hat = torch.fft.fft2(omega)
    omega_x_hat = omega_hat * (1j * kx_grid)
    omega_y_hat = omega_hat * (1j * ky_grid)
    omega_x = torch.fft.ifft2(omega_x_hat).real
    omega_y = torch.fft.ifft2(omega_y_hat).real
    return omega_x, omega_y


def _fourier_laplacian(omega: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Compute ∇²ω using spectral method.

    Args:
        omega: (..., Ny, Nx) tensor.

    Returns:
        laplacian: same shape as omega.
    """
    Ny, Nx = omega.shape[-2], omega.shape[-1]
    kx = torch.fft.fftfreq(Nx, d=dx) * 2.0 * math.pi
    ky = torch.fft.fftfreq(Ny, d=dy) * 2.0 * math.pi
    kx = kx.to(omega.device)
    ky = ky.to(omega.device)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='xy')
    k_sq = kx_grid ** 2 + ky_grid ** 2

    omega_hat = torch.fft.fft2(omega)
    lap_hat = -omega_hat * k_sq  # Fourier transform of Laplacian: -k² * f_hat
    laplacian = torch.fft.ifft2(lap_hat).real
    return laplacian


# ---------------------------------------------------------------------------
# Abstract Solver interface
# ---------------------------------------------------------------------------

class Solver(ABC):
    """Abstract base class for all ODE/PDE solvers.

    Each concrete solver must implement `solve` (returning solution `u` and
    Jacobian `J`) and `compute_jacobian` (computing Jacobian from a graph).
    """

    @abstractmethod
    def solve(
        self,
        p: torch.Tensor,
        u0: Optional[torch.Tensor],
        t: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate the dynamical system and return the solution and its Jacobian.

        Args:
            p:      Parameter tensor of shape (..., P).
            u0:     Initial state tensor. Can be `None` for some equations where
                    the initial condition is derived from `p`. When provided,
                    shape should be (..., state_dim) or (..., *spatial_dims, state_dim).
            t:      1D tensor of time points.
            grid:   Optional spatial grid coordinates (e.g., 1D linspace, or 2D meshgrid).

        Returns:
            (u, J) where
                u : solution       (..., *spatial_dims, len(t), state_dim)
                J : Jacobian ∂u/∂p (..., P,        *spatial_dims, len(t), state_dim)
        """
        ...

    @abstractmethod
    def compute_jacobian(self, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """AD‑based Jacobian from an existing graph.

        Args:
            u: Solution tensor with computational graph.
            p: Parameters with respect to which Jacobian is taken.

        Returns:
            J: Jacobian of u with respect to p, shape (..., P, *u_shape[1:]).
        """
        ...

    def analytical_solution(self, p: torch.Tensor, t: torch.Tensor) -> Optional[torch.Tensor]:
        """Optional: exact analytical solution (only for validation)."""
        return None


# ---------------------------------------------------------------------------
# Concrete solvers
# ---------------------------------------------------------------------------

class HarmonicOscillatorSolver(Solver):
    """Composite harmonic oscillator (ODE1)."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-7, method: str = 'dopri5'):
        self.rtol = rtol
        self.atol = atol
        self.method = method

    # ------------------------------------------------------------------
    # Private: integration without gradient tracking (used by FD solver)
    # ------------------------------------------------------------------
    def _forward(
        self,
        p: torch.Tensor,
        u0: Optional[torch.Tensor],
        t: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate the ODE and return the solution without building a grad graph."""
        B = p.shape[0]
        u_list = []
        for i in range(B):
            p_i = p[i]  # (3,)
            # initial condition
            if u0 is not None:
                u0_i = u0[i]  # (state_dim,)
            else:
                gamma = p_i[2]
                u0_i = torch.sin(gamma * math.pi).unsqueeze(0)  # (1,)

            def rhs(t_val, state):
                alpha = p_i[0]
                beta = p_i[1]
                du = alpha * torch.sin(alpha * math.pi * t_val) + beta * torch.cos(beta * math.pi * t_val)
                return du.unsqueeze(0)   # shape (1,)

            # Use odeint without create_graph to avoid building graph
            u_i = odeint(rhs, u0_i, t, rtol=self.rtol, atol=self.atol, method=self.method,
                         options=dict(create_graph=False))
            u_i = u_i.transpose(0, 1)   # (1, len(t))
            u_list.append(u_i)

        u = torch.stack(u_list, dim=0)   # (B, 1, len(t))  -> state_dim=1, we keep dims: (B, 1, len(t))
        return u

    # ------------------------------------------------------------------
    # solve with Jacobian computation via functional Jacobian
    # ------------------------------------------------------------------
    def solve(
        self,
        p: torch.Tensor,
        u0: Optional[torch.Tensor],
        t: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = p.shape[0]
        u_list, J_list = [], []
        for i in range(B):
            p_i = p[i].clone().requires_grad_(True)   # enable grad
            if u0 is not None:
                u0_i = u0[i].clone()
            else:
                gamma = p_i[2]
                u0_i = torch.sin(gamma * math.pi).unsqueeze(0)

            # define a function that maps p_i -> u
            def _single_solve(pp):
                def rhs(t_val, state):
                    alpha = pp[0]
                    beta = pp[1]
                    du = alpha * torch.sin(alpha * math.pi * t_val) + beta * torch.cos(beta * math.pi * t_val)
                    return du.unsqueeze(0)
                # u0_i is fixed (could be recomputed from pp if pp changes gamma)
                # but note: if pp changes, initial condition might change (gamma). So we need to recompute u0_i
                # We'll recompute u0_i from pp.
                gamma_v = pp[2]
                _u0 = torch.sin(gamma_v * math.pi).unsqueeze(0)
                u_sol = odeint(rhs, _u0, t, rtol=self.rtol, atol=self.atol, method=self.method,
                               options=dict(create_graph=True))
                return u_sol.transpose(0, 1).squeeze(0)   # (1, len(t)) -> (len(t),) or (len(t), 1)? We'll squeeze to (len(t),)
            # compute Jacobian using torch.autograd.functional.jacobian
            J_i = torch.autograd.functional.jacobian(_single_solve, p_i, vectorize=True)
            # J_i shape: output_shape (len(t), 1) + input_shape (3,) -> (len(t), 1, 3). We want (3, 1, len(t))
            # Actually _single_solve returns (len(t),) if we squeezed. We'll keep dimension explicit.
            # Better: let _single_solve return (len(t), 1) tensor.
            # We'll adjust _single_solve to return (len(t), 1).
            # Re-define:
            def _single_solve(pp):
                gamma_v = pp[2]
                _u0 = torch.sin(gamma_v * math.pi).unsqueeze(0)
                def rhs(t_val, state):
                    alpha = pp[0]
                    beta = pp[1]
                    du = alpha * torch.sin(alpha * math.pi * t_val) + beta * torch.cos(beta * math.pi * t_val)
                    return du.unsqueeze(0)
                u_sol = odeint(rhs, _u0, t, rtol=self.rtol, atol=self.atol, method=self.method,
                               options=dict(create_graph=True))
                return u_sol.transpose(0, 1)  # (1, len(t)) -> keep as (1, len(t))

            # Recompute J_i with new _single_solve
            J_i = torch.autograd.functional.jacobian(_single_solve, p_i, vectorize=True)
            # J_i shape: (1, len(t), 3). We want (3, 1, len(t))
            J_i = J_i.permute(2, 0, 1)  # (3, 1, len(t))

            # Also obtain u
            u_i = _single_solve(p_i)  # (1, len(t))
            u_i = u_i.detach()  # we already have Jacobian
            u_list.append(u_i)
            J_list.append(J_i)

        u = torch.stack(u_list, dim=0)      # (B, 1, len(t))
        J = torch.stack(J_list, dim=0)      # (B, 3, 1, len(t))
        return u, J

    # fallback compute_jacobian from graph (not used in training, but required by interface)
    def compute_jacobian(self, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # This is a naive implementation for completeness; not used.
        J = []
        for i in range(u.shape[0]):
            J_i = torch.autograd.grad(u[i].sum(), p[i], create_graph=False, retain_graph=False)[0]
            J.append(J_i)
        return torch.stack(J, dim=0).unsqueeze(2)  # dummy shape

    # analytical solution provided in the paper
    def analytical_solution(self, p: torch.Tensor, t: torch.Tensor) -> Optional[torch.Tensor]:
        B = p.shape[0]
        u = []
        for i in range(B):
            alpha, beta, gamma = p[i][0], p[i][1], p[i][2]
            t_vals = t
            u_i = -1/math.pi * torch.cos(alpha * math.pi * t_vals) \
                  + 1/math.pi * torch.sin(beta * math.pi * t_vals) \
                  + torch.sin(gamma * math.pi) \
                  + 1/math.pi
            u.append(u_i.unsqueeze(0))
        return torch.stack(u, dim=0)  # (B, len(t))


class DuffingSolver(Solver):
    """Duffing oscillator (ODE2)."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-7, method: str = 'dopri5'):
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def _forward(
        self,
        p: torch.Tensor,
        u0: Optional[torch.Tensor],
        t: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = p.shape[0]
        u_list = []
        for i in range(B):
            p_i = p[i]  # (7,): alpha, beta, gamma, delta, omega, epsilon, zeta
            if u0 is not None:
                x0, v0 = u0[i][0], u0[i][1]
            else:
                x0 = p_i[5]   # epsilon
                v0 = p_i[6]   # zeta

            def rhs(t_val, state):
                x, v = state[0], state[1]
                alpha, beta, gamma, delta, omega = p_i[0], p_i[1], p_i[2], p_i[3], p_i[4]
                dx = v
                dv = -delta * v - alpha * t_val - beta * t_val**3 + gamma * torch.cos(omega * t_val)
                return torch.stack([dx, dv])

            u0_i = torch.tensor([x0, v0], device=p.device)
            u_i = odeint(rhs, u0_i, t, rtol=self.rtol, atol=self.atol, method=self.method,
                         options=dict(create_graph=False))
            u_i = u_i.transpose(0, 1)  # (2, len(t))
            u_list.append(u_i)

        return torch.stack(u_list, dim=0)  # (B, 2, len(t))

    def solve(
        self,
        p: torch.Tensor,
        u0: Optional[torch.Tensor],
        t: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = p.shape[0]
        u_list, J_list = [], []
        for i in range(B):
            p_i = p[i].clone().requires_grad_(True)

            def _single_solve(pp):
                # recompute x0, v0 if they depend on pp
                x0 = pp[5]
                v0 = pp[6]
                u0_i = torch.stack([x0, v0])
                def rhs(t_val, state):
                    x, v = state[0], state[1]
                    alpha, beta, gamma, delta, omega = pp[0], pp[1], pp[2], pp[3], pp[4]
                    dx = v
                    dv = -delta * v - alpha * t_val - beta * t_val**3 + gamma * torch.cos(omega * t_val)
                    return torch.stack([dx, dv])
                u_sol = odeint(rhs, u0_i, t, rtol=self.rtol, atol=self.atol, method=self.method,
                               options=dict(create_graph=True))
                return u_sol.transpose(0, 1)  # (2, len(t))

            J_i = torch.autograd.functional.jacobian(_single_solve, p_i, vectorize=True)
            # J_i shape: (2, len(t), 7). We want (7, 2, len(t))
            J_i = J_i.permute(2, 0, 1).contiguous()

            u_i = _single_solve(p_i).detach()
            u_list.append(u_i)
            J_list.append(J_i)

        u = torch.stack(u_list, dim=0)   # (B, 2, len(t))
        J = torch.stack(J_list, dim=0)   # (B, 7, 2, len(t))
        return u, J

    def compute_jacobian(self, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # not used, fallback
        return torch.zeros_like(u).unsqueeze(1).repeat(1, p.shape[-1], 1, 1)


class DampedWaveSolver(Solver):
    """Generalised Nonlinear Damped Wave Equation (PDE1)."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-7, method: str = 'dopri5'):
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def _forward(
        self,
        p: torch.Tensor,
        u0: torch.Tensor,
        t: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        # u0 must be provided: shape (B, 2, Sx) because state is [U, V]
        B = p.shape[0]
        Sx = grid.numel()
        dx = (grid[-1] - grid[0]) / (Sx - 1)
        u_list = []
        for i in range(B):
            p_i = p[i]  # (5,) c, alpha, beta, gamma, omega
            u0_i = u0[i]  # (2, Sx)

            def rhs(t_val, state):
                U = state[0]   # (Sx,)
                V = state[1]
                c, a, b, g, o = p_i[0], p_i[1], p_i[2], p_i[3], p_i[4]
                dU = V
                dV = c**2 * spatial_laplacian_1d(U, dx) + a * V + b * U + g * torch.sin(o * U)
                return torch.stack([dU, dV])

            u0_i_t = u0_i
            u_i = odeint(rhs, u0_i_t, t, rtol=self.rtol, atol=self.atol, method=self.method,
                         options=dict(create_graph=False))
            # u_i shape (len(t), 2, Sx) -> (2, Sx, len(t))
            u_i = u_i.permute(1, 2, 0)
            u_list.append(u_i)
        return torch.stack(u_list, dim=0)  # (B, 2, Sx, len(t))

    def solve(
        self,
        p: torch.Tensor,
        u0: torch.Tensor,
        t: torch.Tensor,
        grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = p.shape[0]
        Sx = grid.numel()
        dx = (grid[-1] - grid[0]) / (Sx - 1)
        u_list, J_list = [], []
        for i in range(B):
            p_i = p[i].clone().requires_grad_(True)
            u0_i = u0[i]  # (2, Sx)

            def _single_solve(pp):
                c, a, b, g, o = pp[0], pp[1], pp[2], pp[3], pp[4]
                def rhs(t_val, state):
                    U = state[0]
                    V = state[1]
                    dU = V
                    dV = c**2 * spatial_laplacian_1d(U, dx) + a * V + b * U + g * torch.sin(o * U)
                    return torch.stack([dU, dV])
                u_sol = odeint(rhs, u0_i, t, rtol=self.rtol, atol=self.atol, method=self.method,
                               options=dict(create_graph=True))
                return u_sol.permute(1, 2, 0)  # (2, Sx, len(t))

            J_i = torch.autograd.functional.jacobian(_single_solve, p_i, vectorize=True)
            # output shape: (2, Sx, len(t)), input shape: (5,)
            # J_i shape: (2, Sx, len(t), 5) -> we want (5, 2, Sx, len(t))
            J_i = J_i.permute(3, 0, 1, 2).contiguous()

            u_i = _single_solve(p_i).detach()
            u_list.append(u_i)
            J_list.append(J_i)

        u = torch.stack(u_list, dim=0)   # (B, 2, Sx, len(t))
        J = torch.stack(J_list, dim=0)   # (B, 5, 2, Sx, len(t))
        return u, J

    def compute_jacobian(self, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # placeholder
        return torch.zeros_like(u).unsqueeze(1).repeat(1, p.shape[-1], 1, 1, 1)


class BurgersSolver(Solver):
    """Forced Burgers' equation (PDE2)."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-7, method: str = 'dopri5'):
        self.rtol = rtol
        self.atol = atol
        self.method = method

    def _forward(
        self,
        p: torch.Tensor,
        u0: torch.Tensor,
        t: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        B = p.shape[0]
        Sx = grid.numel()
        dx = (grid[-1] - grid[0]) / (Sx - 1)
        u_list = []
        for i in range(B):
            p_i = p[i]  # (4,): alpha, gamma, delta, omega
            u0_i = u0[i]  # (Sx,)

            def rhs(t_val, state):
                alpha, gamma, delta, omega = p_i[0], p_i[1], p_i[2], p_i[3]
                # spatial derivatives
                u_x = spatial_gradient_1d(state, dx)
                u_xx = spatial_laplacian_1d(state, dx)
                # RHS: -π α u u_x + π γ u_xx + π δ sin(ω t)
                dudt = -math.pi * alpha * state * u_x + math.pi * gamma * u_xx + math.pi * delta * torch.sin(omega * t_val)
                return dudt

            u_i = odeint(rhs, u0_i, t, rtol=self.rtol, atol=self.atol, method=self.method,
                         options=dict(create_graph=False))
            u_i = u_i.transpose(0, 1)  # (Sx, len(t))
            u_list.append(u_i)

        return torch.stack(u_list, dim=0)  # (B, Sx, len(t))

    def solve(
        self,
        p: torch.Tensor,
        u0: torch.Tensor,
        t: torch.Tensor,
        grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = p.shape[0]
        Sx = grid.numel()
        dx = (grid[-1] - grid[0]) / (Sx - 1)
        u_list, J_list = [], []
        for i in range(B):
            p_i = p[i].clone().requires_grad_(True)
            u0_i = u0[i]

            def _single_solve(pp):
                alpha, gamma, delta, omega = pp[0], pp[1], pp[2], pp[3]
                def rhs(t_val, state):
                    u_x = spatial_gradient_1d(state, dx)
                    u_xx = spatial_laplacian_1d(state, dx)
                    dudt = -math.pi * alpha * state * u_x + math.pi * gamma * u_xx + math.pi * delta * torch.sin(omega * t_val)
                    return dudt
                u_sol = odeint(rhs, u0_i, t, rtol=self.rtol, atol=self.atol, method=self.method,
                               options=dict(create_graph=True))
                return u_sol.transpose(0, 1)  # (Sx, len(t))

            J_i = torch.autograd.functional.jacobian(_single_solve, p_i, vectorize=True)
            # J_i shape: (Sx, len(t), 4) -> (4, Sx, len(t))
            J_i = J_i.permute(2, 0, 1).contiguous()

            u_i = _single_solve(p_i).detach()
            u_list.append(u_i)
            J_list.append(J_i)

        u = torch.stack(u_list, dim=0)   # (B, Sx, len(t))
        J = torch.stack(J_list, dim=0)   # (B, 4, Sx, len(t))
        return u, J

    def compute_jacobian(self, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(u).unsqueeze(1).repeat(1, p.shape[-1], 1, 1)


class BurgersZonedSolver(Solver):
    """Forced Burgers' equation with zonal parameters (PDE2‑zoned, 82 parameters)."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-7, method: str = 'dopri5',
                 num_zones: int = 40):
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.num_zones = num_zones

    def _forward(
        self,
        p: torch.Tensor,
        u0: torch.Tensor,
        t: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        B = p.shape[0]
        Sx = grid.numel()
        dx = (grid[-1] - grid[0]) / (Sx - 1)
        u_list = []
        for i in range(B):
            p_i = p[i]   # (82,)
            alpha_zonal = p_i[:self.num_zones]   # (Sx,)
            delta_zonal = p_i[self.num_zones:2*self.num_zones]
            gamma = p_i[-2]
            omega = p_i[-1]
            # Create spatial fields of alpha and delta by repeating zone values.
            # Since we have num_zones == Sx, each zone corresponds to exactly one grid point.
            # That means each grid point has its own alpha and delta.
            alpha_field = alpha_zonal   # already (Sx,)
            delta_field = delta_zonal
            u0_i = u0[i]

            def rhs(t_val, state):
                u_x = spatial_gradient_1d(state, dx)
                u_xx = spatial_laplacian_1d(state, dx)
                dudt = -math.pi * alpha_field * state * u_x + math.pi * gamma * u_xx + math.pi * delta_field * torch.sin(omega * t_val)
                return dudt

            u_i = odeint(rhs, u0_i, t, rtol=self.rtol, atol=self.atol, method=self.method,
                         options=dict(create_graph=False))
            u_i = u_i.transpose(0, 1)  # (Sx, len(t))
            u_list.append(u_i)

        return torch.stack(u_list, dim=0)

    def solve(
        self,
        p: torch.Tensor,
        u0: torch.Tensor,
        t: torch.Tensor,
        grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = p.shape[0]
        Sx = grid.numel()
        dx = (grid[-1] - grid[0]) / (Sx - 1)
        u_list, J_list = [], []
        for i in range(B):
            p_i = p[i].clone().requires_grad_(True)
            u0_i = u0[i]

            def _single_solve(pp):
                alpha_z = pp[:self.num_zones]
                delta_z = pp[self.num_zones:2*self.num_zones]
                gamma = pp[-2]
                omega = pp[-1]
                alpha_f = alpha_z
                delta_f = delta_z

                def rhs(t_val, state):
                    u_x = spatial_gradient_1d(state, dx)
                    u_xx = spatial_laplacian_1d(state, dx)
                    dudt = -math.pi * alpha_f * state * u_x + math.pi * gamma * u_xx + math.pi * delta_f * torch.sin(omega * t_val)
                    return dudt
                u_sol = odeint(rhs, u0_i, t, rtol=self.rtol, atol=self.atol, method=self.method,
                               options=dict(create_graph=True))
                return u_sol.transpose(0, 1)  # (Sx, len(t))

            # Jacobian computation might be heavy due to 82 inputs. Vectorize may help.
            J_i = torch.autograd.functional.jacobian(_single_solve, p_i, vectorize=True)
            # J_i shape: (Sx, len(t), 82) -> (82, Sx, len(t))
            J_i = J_i.permute(2, 0, 1).contiguous()

            u_i = _single_solve(p_i).detach()
            u_list.append(u_i)
            J_list.append(J_i)

        u = torch.stack(u_list, dim=0)
        J = torch.stack(J_list, dim=0)
        return u, J

    def compute_jacobian(self, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(u).unsqueeze(1).repeat(1, p.shape[-1], 1, 1)


class NavierStokesSolver(Solver):
    """Stream function‑vorticity formulation of Navier‑Stokes equations (PDE3)."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-7, method: str = 'dopri5',
                 Re: float = 1000.0):
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.Re = Re

    def _forward(
        self,
        p: torch.Tensor,
        u0: Optional[torch.Tensor],
        t: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        B = p.shape[0]
        Ny, Nx = grid.shape[0], grid.shape[1]   # grid is 2D meshgrid (Ny, Nx)?
        # grid is a tensor of shape (2, Ny, Nx) maybe. We'll assume xy_grid from config.
        # For simplicity, we'll extract coordinates from grid[0,:,0] for x, grid[0,0,:] for y.
        # Actually grid can be a tuple of (X, Y) or a stacked tensor. We'll accept a tensor of shape (2, Ny, Nx) where grid[0] is X, grid[1] is Y.
        if grid.dim() == 3 and grid.shape[0] == 2:
            X = grid[0]  # (Ny, Nx)
            Y = grid[1]
        else:
            raise ValueError("Grid for NavierStokesSolver must be (2, Ny, Nx) meshgrid.")

        dx = (X[0, -1] - X[0, 0]) / (Nx - 1)
        dy = (Y[-1, 0] - Y[0, 0]) / (Ny - 1)

        u_list = []
        for i in range(B):
            p_i = p[i]  # (2,): alpha, beta
            if u0 is not None:
                omega0 = u0[i]  # (Ny, Nx)
            else:
                alpha, beta = p_i[0], p_i[1]
                omega0 = torch.sin(alpha * X) * torch.cos(beta * Y) \
                         + torch.cos(alpha * Y) * torch.sin(beta * X) \
                         + torch.sin(alpha * X + beta * Y) * torch.cos(alpha * Y - beta * X)

            def rhs(t_val, omega):
                # omega: (Ny, Nx)
                psi = _fft_poisson_solver(omega, dx, dy)
                # psi already real
                # compute spatial derivatives
                omega_x, omega_y = _fourier_gradient(omega, dx, dy)
                psi_x, psi_y = _fourier_gradient(psi, dx, dy)
                laplacian_omega = _fourier_laplacian(omega, dx, dy)

                # advection term: psi_y * omega_x - psi_x * omega_y
                adv = psi_y * omega_x - psi_x * omega_y
                domega_dt = -adv + (1.0 / self.Re) * laplacian_omega
                return domega_dt

            # Reshape omega0 to flat for odeint? odeint expects state as 1D tensor. We'll flatten.
            omega0_flat = omega0.flatten()
            def rhs_flat(t_val, state):
                omega = state.view(Ny, Nx)
                domega = rhs(t_val, omega)
                return domega.flatten()

            omega_final_flat = odeint(rhs_flat, omega0_flat, t[-1:], rtol=self.rtol, atol=self.atol, method=self.method,
                                      options=dict(create_graph=False))[-1]  # (1, Ny*Nx) -> we take last time step
            omega_final = omega_final_flat.view(Ny, Nx)  # (Ny, Nx)
            u_list.append(omega_final.unsqueeze(0))  # (1, Ny, Nx) -> state_dim=1?
        u = torch.stack(u_list, dim=0)  # (B, 1, Ny, Nx)  -> we'll keep as (B, Ny, Nx) unsqueezed? We'll add a state_dim dimension: (B, 1, Ny, Nx)
        u = u.unsqueeze(1)  # (B, 1, Ny, Nx)  (state_dim=1)
        return u

    def solve(
        self,
        p: torch.Tensor,
        u0: Optional[torch.Tensor],
        t: torch.Tensor,           # only final time is used; typically t = [0, 3]
        grid: torch.Tensor,        # shape (2, Ny, Nx)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = p.shape[0]
        Ny, Nx = grid.shape[1], grid.shape[2]
        dx = (grid[0, 0, -1] - grid[0, 0, 0]) / (Nx - 1)
        dy = (grid[1, -1, 0] - grid[1, 0, 0]) / (Ny - 1)

        u_list, J_list = [], []
        t_solve = t[-1:]  # only final time
        for i in range(B):
            p_i = p[i].clone().requires_grad_(True)
            def _single_solve(pp):
                alpha, beta = pp[0], pp[1]
                # initial condition from pp
                omega0 = torch.sin(alpha * grid[0]) * torch.cos(beta * grid[1]) \
                         + torch.cos(alpha * grid[1]) * torch.sin(beta * grid[0]) \
                         + torch.sin(alpha * grid[0] + beta * grid[1]) * torch.cos(alpha * grid[1] - beta * grid[0])
                # flatten
                omega0_flat = omega0.flatten()
                def rhs_flat(t_val, state):
                    omega = state.view(Ny, Nx)
                    psi = _fft_poisson_solver(omega, dx, dy)
                    omega_x, omega_y = _fourier_gradient(omega, dx, dy)
                    psi_x, psi_y = _fourier_gradient(psi, dx, dy)
                    laplacian_omega = _fourier_laplacian(omega, dx, dy)
                    adv = psi_y * omega_x - psi_x * omega_y
                    domega_dt = -adv + (1.0 / self.Re) * laplacian_omega
                    return domega_dt.flatten()
                omega_final_flat = odeint(rhs_flat, omega0_flat, t_solve,
                                         rtol=self.rtol, atol=self.atol, method=self.method,
                                         options=dict(create_graph=True))[-1]
                return omega_final_flat.view(Ny, Nx)  # (Ny, Nx)

            J_i = torch.autograd.functional.jacobian(_single_solve, p_i, vectorize=True)
            # J_i shape: (Ny, Nx, 2) -> (2, Ny, Nx)
            J_i = J_i.permute(2, 0, 1).contiguous()
            # Add state_dim=1
            J_i = J_i.unsqueeze(2)  # (2, Ny, Nx, 1)? We'll keep