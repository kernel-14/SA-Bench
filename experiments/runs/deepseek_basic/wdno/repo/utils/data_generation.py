"""
Data generation for PDE systems used in WDNO.

Implements the data generation procedures described in the paper appendices:
- 1D Burgers' equation (Appendix F)
- 1D Advection equation (PDEBench)
- 1D Compressible Navier-Stokes (PDEBench, Appendix G)
- 2D Incompressible fluid (Appendix H)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ─── 1D Burgers' Equation Data Generation (Appendix F.2) ───

def generate_burgers_initial_condition(
    n_points: int = 120,
    batch_size: int = 1,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate initial condition for 1D Burgers' equation.

    u(0,x) = Σ_{i=1}^{2} a_i exp(-(x-b_i)²/(2σ_i²))

    Parameters from Appendix F.2:
        a_1 ~ U(0, 2), a_2 ~ U(-2, 0)
        b_1 ~ U(0.2, 0.4), b_2 ~ U(0.6, 0.8)
        σ_1 ~ U(0.05, 0.15), σ_2 ~ U(0.05, 0.15)

    Returns:
        Initial condition u_0 of shape (B, n_points)
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.linspace(0, 1, n_points)

    # Sample parameters
    a1 = torch.rand(batch_size) * 2.0  # U(0, 2)
    a2 = torch.rand(batch_size) * (-2.0)  # U(-2, 0)
    b1 = 0.2 + torch.rand(batch_size) * 0.2  # U(0.2, 0.4)
    b2 = 0.6 + torch.rand(batch_size) * 0.2  # U(0.6, 0.8)
    sigma1 = 0.05 + torch.rand(batch_size) * 0.10  # U(0.05, 0.15)
    sigma2 = 0.05 + torch.rand(batch_size) * 0.10  # U(0.05, 0.15)

    # Compute u_0
    u0 = torch.zeros(batch_size, n_points)
    for i in range(batch_size):
        g1 = a1[i] * torch.exp(-(x - b1[i])**2 / (2 * sigma1[i]**2))
        g2 = a2[i] * torch.exp(-(x - b2[i])**2 / (2 * sigma2[i]**2))
        u0[i] = g1 + g2

    return u0


def generate_burgers_control(
    n_time: int = 80,
    n_space: int = 120,
    batch_size: int = 1,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate control force for 1D Burgers' equation.

    f(t,x) = Σ_{i=1}^{8} a_i exp(-(x-b_{1,i})²/(2σ_{1,i}²)) exp(-(t-b_{2,i})²/(2σ_{2,i}²))

    Parameters from Appendix F.2:
        b_{1,i} ~ U(0, 1)
        b_{2,i} ~ U(0, 1) — scaled time
        σ_{1,i} ~ U(0.1, 0.4)
        σ_{2,i} ~ U(0.1, 0.4)
        a_1 ~ U(-1.5, 1.5)
        For i≥2: a_i ~ U(-1.5, 1.5) or 0 with equal probability

    Returns:
        Control force f of shape (B, n_time, n_space)
    """
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.linspace(0, 1, n_space)
    t = torch.linspace(0, 1, n_time)  # Normalized time
    X, T = torch.meshgrid(x, t, indexing='ij')  # (n_space, n_time)

    f = torch.zeros(batch_size, n_time, n_space)

    for b in range(batch_size):
        for i in range(8):
            b1 = torch.rand(1).item()  # U(0, 1)
            b2 = torch.rand(1).item()  # U(0, 1)
            sigma1 = 0.1 + torch.rand(1).item() * 0.3  # U(0.1, 0.4)
            sigma2 = 0.1 + torch.rand(1).item() * 0.3  # U(0.1, 0.4)

            if i == 0:
                a = -1.5 + torch.rand(1).item() * 3.0  # U(-1.5, 1.5)
            else:
                if torch.rand(1).item() < 0.5:
                    a = 0.0
                else:
                    a = -1.5 + torch.rand(1).item() * 3.0  # U(-1.5, 1.5)

            spatial = torch.exp(-(X - b1)**2 / (2 * sigma1**2))
            temporal = torch.exp(-(T - b2)**2 / (2 * sigma2**2))
            f[b] += a * (spatial * temporal).T  # (n_time, n_space)

    return f


def solve_burgers_fdm(
    u0: torch.Tensor,
    f: torch.Tensor,
    nu: float = 0.01,
    T: float = 8.0,
    n_time: int = 80,
    n_space: int = 120,
    dt_solver: Optional[float] = None,
    n_time_solver: Optional[int] = None,
) -> torch.Tensor:
    """
    Solve 1D Burgers' equation using finite difference method.

    ∂u/∂t = -u·∂u/∂x + ν·∂²u/∂x² + f(t,x)

    The solver uses a high-resolution internal grid (as described in Appendix F.2):
    - Space: 120×16 internal grid points, time: 4800×16 internal steps
    - Then downsampled 16× before saving

    For simplicity, we implement a basic FDM scheme.

    Args:
        u0: Initial condition (B, n_space)
        f: Control force (B, n_time, n_space)
        nu: Diffusion coefficient (0.01)
        T: Total time
        n_time: Number of output time steps (80)
        n_space: Number of spatial points (120)
        dt_solver: Internal solver time step
        n_time_solver: Internal solver time steps

    Returns:
        Solution u (B, n_time+1, n_space) — includes initial condition
    """
    batch_size = u0.shape[0]
    dx = 1.0 / (n_space - 1)

    if dt_solver is None:
        # Use internal high-resolution time stepping
        dt_solver = T / (n_time * 16)  # 16× finer than output
        n_time_solver = n_time * 16

    u = torch.zeros(batch_size, n_time + 1, n_space)
    u[:, 0, :] = u0

    u_current = u0.clone()
    dt = dt_solver
    solver_steps_per_output = n_time_solver // n_time

    for t_out in range(1, n_time + 1):
        for _ in range(solver_steps_per_output):
            # Compute spatial derivatives
            u_next = u_current.clone()

            # First derivative: ∂u/∂x (central difference)
            for i in range(1, n_space - 1):
                dudx = (u_current[:, i + 1] - u_current[:, i - 1]) / (2 * dx)
                d2udx2 = (u_current[:, i + 1] - 2 * u_current[:, i] + u_current[:, i - 1]) / (dx ** 2)

                # Burgers' equation RHS: -u·∂u/∂x + ν·∂²u/∂x² + f
                rhs = -u_current[:, i] * dudx + nu * d2udx2

                # Index into f at the nearest output time
                f_idx = min(t_out - 1, f.shape[1] - 1)
                rhs = rhs + f[:, f_idx, i]

                u_next[:, i] = u_current[:, i] + dt * rhs

            # Dirichlet boundary: u=0
            u_next[:, 0] = 0
            u_next[:, -1] = 0

            u_current = u_next

        u[:, t_out, :] = u_current

    return u


# ─── 1D Compressible Navier-Stokes Data Loading ───

def load_cfd_shock_data(
    filepath: str = None,
    n_time: int = 81,
    n_space: int = 120,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load 1D compressible Navier-Stokes data from PDEBench.

    The dataset '1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5'
    contains shock-tube simulations with:
    - η = 10^{-8}, ζ = 10^{-8} (very small viscosity)
    - Piecewise constant initial conditions
    - Variables: density ρ, velocity v, pressure p

    Returns:
        data: Tensor of shape (N, 3, 81, 120) or similar for (ρ, v, p)
        initial_conditions: Initial condition data
    """
    import h5py

    if filepath is None:
        raise ValueError("Must provide path to PDEBench CFD dataset")

    with h5py.File(filepath, 'r') as f:
        # PDEBench format: 'density', 'velocity', 'pressure' with shape (N, T, X)
        density = torch.tensor(f['density'][:])
        velocity = torch.tensor(f['velocity'][:]) if 'velocity' in f else None
        pressure = torch.tensor(f['pressure'][:]) if 'pressure' in f else None

    # Stack as channels: (N, 3, T, X)
    data = torch.stack([density, velocity, pressure], dim=1)

    # Extract initial condition
    initial = data[:, :, 0:1, :]  # (N, 3, 1, X)

    return data, initial


# ─── ERA5 Data Loading ───

def prepare_era5_data(
    temperature_data: torch.Tensor,
    input_steps: int = 12,
    output_steps: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare ERA5 temperature data for the forecasting task.

    Task: Predict next 20 hours given past 12 hours.

    Args:
        temperature_data: Full temperature data
        input_steps: Number of input time steps (12)
        output_steps: Number of output time steps (20)

    Returns:
        inputs: (N, input_steps, H, W)
        targets: (N, output_steps, H, W)
    """
    N = temperature_data.shape[0] - input_steps - output_steps + 1
    inputs = torch.zeros(N, input_steps, *temperature_data.shape[2:])
    targets = torch.zeros(N, output_steps, *temperature_data.shape[2:])

    for i in range(N):
        inputs[i] = temperature_data[i:i + input_steps]
        targets[i] = temperature_data[i + input_steps:i + input_steps + output_steps]

    return inputs, targets


# ─── Objective Functions ───

def burgers_control_objective(
    u_T: torch.Tensor,
    f: torch.Tensor,
    u_target: torch.Tensor,
    alpha: float = 0.00002,
) -> torch.Tensor:
    """
    Control objective for 1D Burgers' equation (Eq. 6).

    I = ∫_D |u(T,x) - u*(x)|² dx + α ∫_{[0,T]×D} |f(t,x)|² dt dx

    Args:
        u_T: Final state u(T,x) — (B, n_space)
        f: Control force — (B, n_time, n_space)
        u_target: Target state u*(x) — (B, n_space)
        alpha: Energy penalty weight

    Returns:
        Objective value I
    """
    state_loss = torch.mean((u_T - u_target) ** 2, dim=-1)  # (B,)
    energy_loss = torch.mean(f ** 2, dim=(-1, -2))  # (B,)
    return state_loss + alpha * energy_loss


def fluid_control_objective(
    smoke_percentage: torch.Tensor,
) -> torch.Tensor:
    """
    Control objective for 2D incompressible fluid (Section 4.4).

    I = percentage of smoke NOT passing through the target bucket.
    We want to minimize this (maximize smoke through bucket).

    Args:
        smoke_percentage: Fraction of smoke through target bucket (B,)

    Returns:
        Objective value (lower is better)
    """
    return 1.0 - smoke_percentage  # minimize uncollected smoke
