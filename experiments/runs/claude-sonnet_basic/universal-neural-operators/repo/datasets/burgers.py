"""
Burgers' equation dataset generator.

Burgers' equation: du/dt + u * du/dx = nu * d^2u/dx^2
where nu is the viscosity coefficient.

For the out-of-sample parameter values scenario, we train on one set of nu values
and fine-tune on different nu values.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List


def solve_burgers_1d(
    u0: np.ndarray,
    nu: float,
    nx: int = 256,
    nt: int = 100,
    dt: float = 0.001,
    dx: float = None,
) -> np.ndarray:
    """
    Solve 1D Burgers' equation using pseudo-spectral method.
    
    du/dt + u * du/dx = nu * d^2u/dx^2
    
    Args:
        u0: Initial condition (nx,)
        nu: Viscosity coefficient
        nx: Number of spatial points
        nt: Number of time steps
        dt: Time step size
        dx: Spatial step size (computed from nx if None)
    
    Returns:
        u: Solution array (nt+1, nx)
    """
    if dx is None:
        dx = 2 * np.pi / nx

    u = np.zeros((nt + 1, nx))
    u[0] = u0

    # Wavenumbers for spectral differentiation
    k = np.fft.rfftfreq(nx, d=1.0 / nx)

    for t in range(nt):
        u_hat = np.fft.rfft(u[t])
        # Nonlinear term: u * du/dx (dealiased)
        du_dx = np.fft.irfft(1j * k * u_hat, n=nx)
        nonlinear = u[t] * du_dx
        # Diffusion term in spectral space
        diffusion = -nu * k ** 2 * u_hat
        # Time integration (Euler)
        nonlinear_hat = np.fft.rfft(nonlinear)
        u_hat_new = u_hat + dt * (diffusion - nonlinear_hat)
        u[t + 1] = np.fft.irfft(u_hat_new, n=nx)

    return u


def generate_burgers_data(
    n_samples: int = 1000,
    nx: int = 256,
    nt: int = 100,
    dt: float = 0.001,
    nu_range: Tuple[float, float] = (0.001, 0.1),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Burgers' equation dataset.
    
    Args:
        n_samples: Number of samples
        nx: Spatial resolution
        nt: Number of time steps
        dt: Time step
        nu_range: Range of viscosity values
        seed: Random seed
    
    Returns:
        inputs: (n_samples, 2, nx) - [initial condition, viscosity field]
        outputs: (n_samples, 1, nx) - final state
        nus: (n_samples,) - viscosity values used
    """
    rng = np.random.RandomState(seed)
    
    inputs = np.zeros((n_samples, 2, nx))
    outputs = np.zeros((n_samples, 1, nx))
    nus = np.zeros(n_samples)
    
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    
    for i in range(n_samples):
        # Random viscosity
        nu = rng.uniform(*nu_range)
        nus[i] = nu
        
        # Random initial condition (sum of sinusoids)
        n_modes = rng.randint(1, 6)
        u0 = np.zeros(nx)
        for _ in range(n_modes):
            k = rng.randint(1, 5)
            amp = rng.uniform(-1, 1)
            phase = rng.uniform(0, 2 * np.pi)
            u0 += amp * np.sin(k * x + phase)
        
        # Solve
        u = solve_burgers_1d(u0, nu, nx, nt, dt)
        
        # Input: initial condition + constant viscosity field
        inputs[i, 0] = u0
        inputs[i, 1] = nu * np.ones(nx)
        
        # Output: final state
        outputs[i, 0] = u[-1]
    
    return inputs, outputs, nus


class BurgersDataset(Dataset):
    """
    Dataset for Burgers' equation.
    
    Supports out-of-sample parameter values scenario where pretraining
    and fine-tuning use different viscosity ranges.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        nx: int = 256,
        nt: int = 100,
        dt: float = 0.001,
        nu_range: Tuple[float, float] = (0.001, 0.1),
        seed: int = 42,
        data: Optional[Tuple] = None,
    ):
        if data is not None:
            self.inputs, self.outputs, self.nus = data
        else:
            self.inputs, self.outputs, self.nus = generate_burgers_data(
                n_samples, nx, nt, dt, nu_range, seed
            )
        
        self.inputs = torch.FloatTensor(self.inputs)
        self.outputs = torch.FloatTensor(self.outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    @property
    def n_input(self):
        return self.inputs.shape[1]

    @property
    def n_output(self):
        return self.outputs.shape[1]
