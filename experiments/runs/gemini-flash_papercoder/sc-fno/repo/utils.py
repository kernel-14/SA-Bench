import os
import random
import yaml
import numpy as np
import torch
import math
from typing import Callable, Dict, Optional, Tuple, Any

# Ensure torchdiffeq is installed and available
try:
    import torchdiffeq
except ImportError:
    raise ImportError("Please install torchdiffeq: pip install torchdiffeq")


def seed_everything(seed: int) -> None:
    """
    Sets random seeds for reproducibility across torch, numpy, and random modules.

    Args:
        seed: The integer seed value to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """
    Returns the available computational device ('cuda' if GPU is available, else 'cpu').

    Returns:
        A string indicating the device.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_path: str) -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path: The file path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(state: Dict[str, Any], filepath: str) -> None:
    """
    Saves model checkpoints.

    Args:
        state: A dictionary containing the objects to be saved (e.g., model.state_dict(),
               optimizer.state_dict(), epoch, loss).
        filepath: The full path including the filename where the checkpoint should be saved.
    """
    dirname = os.path.dirname(filepath)
    if dirname:  # Create directory if it exists and is not empty string
        os.makedirs(dirname, exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[str] = None
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """
    Loads model and optimizer states from a checkpoint file.

    Args:
        filepath: The path to the checkpoint file.
        model: The model instance to load the state into.
        optimizer: The optimizer instance to load the state into (optional).
        device: The device ("cuda" or "cpu") to map the loaded tensors to.

    Returns:
        A tuple containing the loaded model, optimizer (if provided), and other metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {filepath}")
    # Return model, optimizer, and any other metadata (e.g., epoch, best_loss)
    meta_data = {k: v for k, v in checkpoint.items() if k not in ['model_state_dict', 'optimizer_state_dict']}
    return model, optimizer, meta_data


def compute_r2(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes the R-squared (R²) metric.

    Args:
        y_pred: The predicted values.
        y_true: The true (ground truth) values.

    Returns:
        The R² value as a float.
    """
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    ss_tot = torch.sum((y_true_flat - y_true_flat.mean())**2)
    ss_res = torch.sum((y_true_flat - y_pred_flat)**2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0 # Perfect prediction of a constant if ss_res is also zero
    
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def compute_relative_l2(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Computes the Relative L2 error.

    Args:
        y_pred: The predicted values.
        y_true: The true (ground truth) values.

    Returns:
        The relative L2 error as a float.
    """
    numerator = torch.norm(y_pred - y_true)
    denominator = torch.norm(y_true)

    if denominator == 0:
        return 0.0 if numerator == 0 else float('inf')
    
    relative_l2 = numerator / denominator
    return relative_l2.item()


def normalize_data(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a tensor using a given mean and standard deviation.

    Args:
        data: The tensor data to be normalized.
        mean: The mean tensor for normalization.
        std: The standard deviation tensor for normalization.

    Returns:
        The normalized data tensor.
    """
    return (data - mean) / (std + 1e-8)  # Add epsilon for numerical stability


def denormalize_data(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes a tensor back to its original scale using a given mean and standard deviation.

    Args:
        data: The normalized tensor data.
        mean: The mean tensor used for normalization.
        std: The standard deviation tensor used for normalization.

    Returns:
        The denormalized data tensor.
    """
    return (data * (std + 1e-8)) + mean


# Helper functions for spatial derivatives (periodic boundary conditions assumed)
def _compute_dx_1d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Compute first spatial derivative in 1D with periodic BCs using central difference."""
    return (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)) / (2 * dx)

def _compute_dxx_1d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Compute second spatial derivative in 1D with periodic BCs using central difference."""
    return (torch.roll(u, shifts=-1, dims=-1) - 2 * u + torch.roll(u, shifts=1, dims=-1)) / (dx**2)

def _compute_dx_2d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Compute first spatial derivative in x-direction in 2D with periodic BCs."""
    return (torch.roll(u, shifts=-1, dims=-2) - torch.roll(u, shifts=1, dims=-2)) / (2 * dx)

def _compute_dy_2d(u: torch.Tensor, dy: float) -> torch.Tensor:
    """Compute first spatial derivative in y-direction in 2D with periodic BCs."""
    return (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)) / (2 * dy)

def _compute_dxx_2d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Compute second spatial derivative in x-direction in 2D with periodic BCs."""
    return (torch.roll(u, shifts=-1, dims=-2) - 2 * u + torch.roll(u, shifts=1, dims=-2)) / (dx**2)

def _compute_dyy_2d(u: torch.Tensor, dy: float) -> torch.Tensor:
    """Compute second spatial derivative in y-direction in 2D with periodic BCs."""
    return (torch.roll(u, shifts=-1, dims=-1) - 2 * u + torch.roll(u, shifts=1, dims=-1)) / (dy**2)


def get_pde_equation_function(pde_id: str) -> Callable[..., Any]:
    """
    Returns the Python function representing the RHS of the PDE, parameterized for `torchdiffeq`.
    The returned function computes d(state)/dt given the current time, state, and parameters.
    It expects `params` to be a dict of torch.Tensor, `spatial_grid` to be torch.Tensor.

    Args:
        pde_id: Identifier for the PDE (e.g., "PDE1", "PDE2").

    Returns:
        A callable function `f(t: torch.Tensor, u_state: torch.Tensor, params: Dict[str, torch.Tensor],
                             spatial_grid_x: torch.Tensor, spatial_grid_y: Optional[torch.Tensor] = None) -> torch.Tensor`
        that computes the RHS of the PDE.
    """

    if pde_id == "PDE1":
        # PDE1: Generalized Nonlinear Damped Wave Equation
        # d^2u/dt^2 = c^2 d^2u/dx^2 + alpha du/dt + beta u + gamma sin(omega u)
        # State for torchdiffeq: [u, du/dt]
        # So, u_state = [u_val, u_t_val]
        # RHS will return [du_dt, d2u_dt2]
        def rhs_pde1(t: torch.Tensor, u_state_flat: torch.Tensor, params: Dict[str, torch.Tensor], spatial_grid_x: torch.Tensor, spatial_grid_y: Optional[torch.Tensor] = None) -> torch.Tensor:
            S_x = spatial_grid_x.shape[0]
            dx = spatial_grid_x[1] - spatial_grid_x[0]

            # u_state_flat contains [u, du/dt] flattened
            # Reshape from (2 * S_x,) to (2, S_x)
            u_state = u_state_flat.view(2, S_x)
            u_val = u_state[0]
            u_t_val = u_state[1]

            c = params['c']
            alpha_param = params['alpha'] # Renamed to avoid conflict with `alpha` in params
            beta_param = params['beta']   # Renamed
            gamma_param = params['gamma'] # Renamed
            omega_param = params['omega'] # Renamed

            # Compute spatial derivatives
            u_xx = _compute_dxx_1d(u_val, dx)

            # Compute d2u/dt2
            d2u_dt2 = c**2 * u_xx + alpha_param * u_t_val + beta_param * u_val + gamma_param * torch.sin(omega_param * u_val)
            
            # Return new state [du/dt, d2u/dt2]
            return torch.cat([u_t_val, d2u_dt2], dim=0)

        return rhs_pde1

    elif pde_id == "PDE2" or pde_id == "PDE2_Zoned":
        # PDE2: Forced Burgers’ Equation
        # (1/pi) du/dt + alpha u du/dx = gamma d2u/dx2 + delta sin(omega t)
        # Rearrange for du/dt:
        # du/dt = pi * (gamma d2u/dx2 - alpha u du/dx + delta sin(omega t))
        def rhs_pde2(t: torch.Tensor, u_flat: torch.Tensor, params: Dict[str, torch.Tensor], spatial_grid_x: torch.Tensor, spatial_grid_y: Optional[torch.Tensor] = None) -> torch.Tensor:
            S_x = spatial_grid_x.shape[0]
            dx = spatial_grid_x[1] - spatial_grid_x[0]

            u_val = u_flat.view(S_x)

            # Extract parameters. Handle zoned parameters if applicable.
            if pde_id == "PDE2_Zoned":
                # params will contain 'alpha_zones' and 'delta_zones' as tensors
                alpha_val = params['alpha_zones'].view(S_x) # Assume alpha for each spatial point
                delta_val = params['delta_zones'].view(S_x) # Assume delta for each spatial point
                gamma_val = params['global_gamma']
                omega_val = params['global_omega']
            else: # PDE2
                alpha_val = params['alpha']
                gamma_val = params['gamma']
                delta_val = params['delta']
                omega_val = params['omega']

            # Compute spatial derivatives
            u_x = _compute_dx_1d(u_val, dx)
            u_xx = _compute_dxx_1d(u_val, dx)

            # Compute du/dt
            du_dt = math.pi * (gamma_val * u_xx - alpha_val * u_val * u_x + delta_val * torch.sin(omega_val * t))
            
            return du_dt

        return rhs_pde2
    
    elif pde_id == "PDE3":
        # PDE3: Stream Function-Vorticity Formulation of the Navier-Stokes Equations
        # 1. d_omega/dt + psi_y d_omega/dx - psi_x d_omega/dy = (1/Re) * (d2_omega/dx2 + d2_omega/dy2)
        # 2. d2_psi/dx2 + d2_psi/dy2 = -omega (Poisson equation for stream function psi)
        # State for torchdiffeq: omega_flat
        def rhs_pde3(t: torch.Tensor, omega_flat: torch.Tensor, params: Dict[str, torch.Tensor], spatial_grid_x: torch.Tensor, spatial_grid_y: torch.Tensor) -> torch.Tensor:
            S_x = spatial_grid_x.shape[0]
            S_y = spatial_grid_y.shape[0]
            dx = spatial_grid_x[1] - spatial_grid_x[0]
            dy = spatial_grid_y[1] - spatial_grid_y[0]

            omega = omega_flat.view(S_x, S_y)
            Re = params['Re'] # Reynolds number (fixed parameter)

            # Solve Poisson equation d2_psi/dx2 + d2_psi/dy2 = -omega for psi
            # Using spectral method for periodic boundary conditions
            # Fourier transform (forward)
            omega_hat = torch.fft.fft2(omega)
            
            # Create k-space grid for derivatives
            kx = 2 * math.pi * torch.fft.fftfreq(S_x, d=dx).to(omega.device)
            ky = 2 * math.pi * torch.fft.fftfreq(S_y, d=dy).to(omega.device)
            kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='ij')

            # Laplacian in Fourier space: -(kx^2 + ky^2) * psi_hat = -omega_hat
            # psi_hat = omega_hat / (kx^2 + ky^2)
            # Handle zero frequency (DC component) to avoid division by zero.
            # The DC component of psi is arbitrary, usually set to 0.
            denominator = -(kx_grid**2 + ky_grid**2)
            # Set denominator for DC component (k=0,0) to a large value or 1 to make psi_hat=0
            denominator[0, 0] = 1.0 # Avoid division by zero, ensures psi_hat[0,0] becomes 0
            
            psi_hat = omega_hat / denominator
            psi_hat[0,0] = 0.0 # Enforce zero mean for psi

            psi = torch.fft.ifft2(psi_hat).real # Inverse Fourier transform to get psi in real space

            # Compute derivatives of psi
            psi_x = _compute_dx_2d(psi, dx)
            psi_y = _compute_dy_2d(psi, dy)

            # Compute derivatives of omega
            omega_x = _compute_dx_2d(omega, dx)
            omega_y = _compute_dy_2d(omega, dy)
            omega_xx = _compute_dxx_2d(omega, dx)
            omega_yy = _compute_dyy_2d(omega, dy)

            # Compute d_omega/dt
            convection_term = psi_y * omega_x - psi_x * omega_y
            diffusion_term = (1 / Re) * (omega_xx + omega_yy)
            
            d_omega_dt = -convection_term + diffusion_term
            
            return d_omega_dt.flatten()

        return rhs_pde3

    elif pde_id == "PDE4":
        # PDE4: Allen-Cahn equation
        # du/dt = epsilon d2u/dx2 + alpha u - beta u^3
        def rhs_pde4(t: torch.Tensor, u_flat: torch.Tensor, params: Dict[str, torch.Tensor], spatial_grid_x: torch.Tensor, spatial_grid_y: Optional[torch.Tensor] = None) -> torch.Tensor:
            S_x = spatial_grid_x.shape[0]
            dx = spatial_grid_x[1] - spatial_grid_x[0]

            u_val = u_flat.view(S_x)

            epsilon = params['epsilon']
            alpha_param = params['alpha'] # Renamed
            beta_param = params['beta']   # Renamed

            # Compute spatial derivatives
            u_xx = _compute_dxx_1d(u_val, dx)

            # Compute du/dt
            du_dt = epsilon * u_xx + alpha_param * u_val - beta_param * (u_val**3)
            
            return du_dt

        return rhs_pde4

    else:
        raise ValueError(f"Unknown PDE ID: {pde_id}")


def get_ode_equation_function(ode_id: str) -> Callable[..., Any]:
    """
    Returns the Python function representing the RHS of the ODE, parameterized for `torchdiffeq`.
    The returned function computes d(state)/dt given the current time, state, and parameters.
    It expects `params` to be a dict of torch.Tensor.

    Args:
        ode_id: Identifier for the ODE (e.g., "ODE1", "ODE2").

    Returns:
        A callable function `f(t: torch.Tensor, u_state: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor`
        that computes the RHS of the ODE.
    """
    if ode_id == "ODE1":
        # ODE1: Composite Harmonic Oscillator
        # du/dt = alpha sin(alpha pi t) + beta cos(beta pi t)
        # Note: gamma only affects initial condition u(0) = sin(gamma pi), not the dynamics.
        def rhs_ode1(t: torch.Tensor, u_state: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
            alpha = params['alpha']
            beta = params['beta']
            
            # u_state is the current value of u. For this ODE, du/dt does not depend on u.
            # It's good practice to keep u_state in the signature as odeint expects it.
            du_dt = alpha * torch.sin(alpha * math.pi * t) + beta * torch.cos(beta * math.pi * t)
            return du_dt.unsqueeze(0) if du_dt.ndim == 0 else du_dt # Ensure it's a tensor (batch_size, 1) or (1,)


        return rhs_ode1

    elif ode_id == "ODE2":
        # ODE2: Duffing Oscillator Equation
        # d^2x/dt^2 + delta dx/dt + alpha x + beta x^3 = gamma cos(omega t)
        # Convert to 1st order system:
        # u1 = x, u2 = dx/dt
        # du1/dt = u2
        # du2/dt = -delta u2 - alpha u1 - beta u1^3 + gamma cos(omega t)
        def rhs_ode2(t: torch.Tensor, u_state: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
            # u_state is a tensor of shape (batch_size, 2) or (2,) for [x, dx/dt]
            x_val = u_state[..., 0]
            dx_dt_val = u_state[..., 1]

            delta_param = params['delta']
            alpha_param = params['alpha'] # Renamed
            beta_param = params['beta']   # Renamed
            gamma_param = params['gamma'] # Renamed
            omega_param = params['omega'] # Renamed

            du1_dt = dx_dt_val
            du2_dt = -delta_param * dx_dt_val - alpha_param * x_val - beta_param * (x_val**3) + gamma_param * torch.cos(omega_param * t)
            
            return torch.stack([du1_dt, du2_dt], dim=-1)

        return rhs_ode2

    else:
        raise ValueError(f"Unknown ODE ID: {ode_id}")

