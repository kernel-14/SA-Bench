"""
PDE3: Stream Function-Vorticity Formulation of the Navier-Stokes Equations

  ∂ω/∂t + ψ_y ∂ω/∂x - ψ_x ∂ω/∂y = (1/Re)(∂²ω/∂x² + ∂²ω/∂y²)
  ∂²ψ/∂x² + ∂²ψ/∂y² = -ω

Initial condition:
  ω(x,y,0) = sin(αx)cos(βy) + cos(αy)sin(βx) + sin(αx+βy)cos(αy-βx)

Boundary conditions: periodic in both x and y

Parameters: α ∈ [π, 5π], β ∈ [π, 5π]
Re = 1000

Domain: x,y ∈ [0,1], t ∈ [0,3], Sx=Sy=64, M=1 (initial condition only)
Goal: predict vorticity at t=3s
"""

import math
from typing import Dict, Optional

import torch


class PDE3Solver:
    """Pseudo-spectral solver for the Navier-Stokes equations (vorticity-stream function)."""

    param_names = ["alpha", "beta"]
    param_ranges = {
        "alpha": (math.pi, 5 * math.pi),
        "beta": (math.pi, 5 * math.pi),
    }
    N = 30
    M = 1
    Sx = 64
    Sy = 64
    t_start = 0.0
    t_end = 3.0
    Re = 1000.0

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.t = torch.linspace(self.t_start, self.t_end, self.N, device=device)
        self.dt = (self.t_end - self.t_start) / (self.N - 1)

        # Wavenumbers for pseudo-spectral method
        kx = torch.fft.fftfreq(self.Sx, d=1.0 / self.Sx).to(device)
        ky = torch.fft.fftfreq(self.Sy, d=1.0 / self.Sy).to(device)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        self.KX = KX
        self.KY = KY
        self.K2 = KX ** 2 + KY ** 2
        self.K2[0, 0] = 1.0  # avoid division by zero

        # Dealiasing mask (2/3 rule)
        kmax_x = self.Sx // 3
        kmax_y = self.Sy // 3
        self.dealias = ((torch.abs(KX) < kmax_x) & (torch.abs(KY) < kmax_y)).float()

    def _generate_ic(self, params: torch.Tensor) -> torch.Tensor:
        """
        Generate initial vorticity from parameters.

        Args:
            params: (batch, 2) [alpha, beta]

        Returns:
            omega0: (batch, Sx, Sy)
        """
        alpha = params[:, 0:1, None]  # (batch, 1, 1)
        beta = params[:, 1:2, None]

        x = torch.linspace(0, 1, self.Sx, device=self.device)
        y = torch.linspace(0, 1, self.Sy, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        X = X.unsqueeze(0)  # (1, Sx, Sy)
        Y = Y.unsqueeze(0)

        omega0 = (
            torch.sin(alpha * X) * torch.cos(beta * Y)
            + torch.cos(alpha * Y) * torch.sin(beta * X)
            + torch.sin(alpha * X + beta * Y) * torch.cos(alpha * Y - beta * X)
        )
        return omega0  # (batch, Sx, Sy)

    def _stream_function(self, omega_hat: torch.Tensor) -> torch.Tensor:
        """Solve Poisson equation for stream function in Fourier space."""
        return -omega_hat / self.K2.unsqueeze(0)

    def _rhs_spectral(self, omega_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute RHS of vorticity equation in Fourier space.

        Args:
            omega_hat: (batch, Sx, Sy) complex Fourier coefficients

        Returns:
            domega_hat_dt: (batch, Sx, Sy) complex
        """
        KX = self.KX.unsqueeze(0)
        KY = self.KY.unsqueeze(0)
        K2 = self.K2.unsqueeze(0)
        dealias = self.dealias.unsqueeze(0)

        psi_hat = self._stream_function(omega_hat)

        # Velocity field: u = ∂ψ/∂y, v = -∂ψ/∂x
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat

        # Vorticity gradients
        domega_dx_hat = 1j * KX * omega_hat
        domega_dy_hat = 1j * KY * omega_hat

        # Transform to physical space
        u = torch.fft.ifft2(u_hat * dealias).real
        v = torch.fft.ifft2(v_hat * dealias).real
        domega_dx = torch.fft.ifft2(domega_dx_hat * dealias).real
        domega_dy = torch.fft.ifft2(domega_dy_hat * dealias).real

        # Nonlinear advection term
        advection = u * domega_dx + v * domega_dy
        advection_hat = torch.fft.fft2(advection) * dealias

        # Diffusion term
        diffusion_hat = -(1.0 / self.Re) * K2 * omega_hat

        return -advection_hat + diffusion_hat

    def solve(self, params: torch.Tensor, omega0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve NS equations using pseudo-spectral method with RK4.

        Args:
            params: (batch, 2)
            omega0: (batch, Sx, Sy) initial vorticity (generated from params if None)

        Returns:
            omega_final: (batch, Sx, Sy) vorticity at t=T_end
        """
        if omega0 is None:
            omega0 = self._generate_ic(params)

        omega_hat = torch.fft.fft2(omega0)
        dt = self.dt

        for _ in range(self.N - 1):
            k1 = self._rhs_spectral(omega_hat)
            k2 = self._rhs_spectral(omega_hat + dt / 2 * k1)
            k3 = self._rhs_spectral(omega_hat + dt / 2 * k2)
            k4 = self._rhs_spectral(omega_hat + dt * k3)
            omega_hat = omega_hat + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return torch.fft.ifft2(omega_hat).real  # (batch, Sx, Sy)

    def solve_trajectory(self, params: torch.Tensor, omega0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Solve NS equations and return full trajectory.

        Returns:
            omega_traj: (batch, Sx, Sy, N)
        """
        if omega0 is None:
            omega0 = self._generate_ic(params)

        omega_hat = torch.fft.fft2(omega0)
        trajectory = [torch.fft.ifft2(omega_hat).real]
        dt = self.dt

        for _ in range(self.N - 1):
            k1 = self._rhs_spectral(omega_hat)
            k2 = self._rhs_spectral(omega_hat + dt / 2 * k1)
            k3 = self._rhs_spectral(omega_hat + dt / 2 * k2)
            k4 = self._rhs_spectral(omega_hat + dt * k3)
            omega_hat = omega_hat + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(torch.fft.ifft2(omega_hat).real)

        return torch.stack(trajectory, dim=-1)  # (batch, Sx, Sy, N)

    def jacobian_fd(self, params: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """
        Compute Jacobian ∂ω_final/∂p via 4th-order central finite differences.

        Returns:
            jac: (batch, Sx, Sy, 2)
        """
        batch, n_params = params.shape
        jac = torch.zeros(batch, self.Sx, self.Sy, n_params, device=self.device)

        for i in range(n_params):
            p_pp = params.clone(); p_pp[:, i] += 2 * eps
            p_p = params.clone(); p_p[:, i] += eps
            p_m = params.clone(); p_m[:, i] -= eps
            p_mm = params.clone(); p_mm[:, i] -= 2 * eps

            u_pp = self.solve(p_pp)
            u_p = self.solve(p_p)
            u_m = self.solve(p_m)
            u_mm = self.solve(p_mm)

            jac[:, :, :, i] = (-u_pp + 8 * u_p - 8 * u_m + u_mm) / (12 * eps)

        return jac

    def jacobian_ad(self, params: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian ∂ω_final/∂p via automatic differentiation.

        Returns:
            jac: (batch, Sx, Sy, 2)
        """
        batch = params.shape[0]
        n_params = params.shape[1]
        jac = torch.zeros(batch, self.Sx, self.Sy, n_params, device=self.device)

        for b in range(batch):
            p_b = params[b:b+1].detach().requires_grad_(True)

            def solve_single(p):
                return self.solve(p)  # (1, Sx, Sy)

            jac_b = torch.autograd.functional.jacobian(
                solve_single, p_b, create_graph=False
            )  # (1, Sx, Sy, 1, n_params)
            jac[b] = jac_b[0, :, :, 0, :]  # (Sx, Sy, n_params)

        return jac

    def sample_params(self, n_samples: int) -> torch.Tensor:
        """Sample parameters uniformly from their ranges."""
        params = torch.zeros(n_samples, 2, device=self.device)
        for i, name in enumerate(self.param_names):
            lo, hi = self.param_ranges[name]
            params[:, i] = torch.rand(n_samples, device=self.device) * (hi - lo) + lo
        return params

    def generate_dataset(self, n_samples: int, use_ad: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate dataset with solution paths and Jacobians.

        Returns dict with keys: params, u (final vorticity), jacobian, omega0
        """
        params = self.sample_params(n_samples)
        omega0 = self._generate_ic(params)
        omega_final = self.solve(params, omega0)

        if use_ad:
            jac = self.jacobian_ad(params)
        else:
            jac = self.jacobian_fd(params)

        return {"params": params, "u": omega_final, "jacobian": jac, "omega0": omega0}
