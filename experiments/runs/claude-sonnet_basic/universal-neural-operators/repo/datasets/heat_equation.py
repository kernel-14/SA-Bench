"""
Heat equation dataset generator.

Heat equation: du/dt = alpha * Laplacian(u)
Extended with convection: du/dt = alpha * Laplacian(u) + v * du/dx

For the input function set extension scenario, we:
1. Pretrain on heat equation (inputs: u0, alpha)
2. Fine-tune on heat equation with convection (inputs: u0, alpha, v)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


def solve_heat_1d(
    u0: np.ndarray,
    alpha: float,
    nt: int = 100,
    dt: float = 0.001,
    convection: float = 0.0,
) -> np.ndarray:
    """
    Solve 1D heat equation (optionally with convection) using spectral method.
    
    du/dt = alpha * d^2u/dx^2 + convection * du/dx
    
    Args:
        u0: Initial condition (nx,)
        alpha: Thermal diffusivity
        nt: Number of time steps
        dt: Time step
        convection: Convection velocity (0 for pure heat equation)
    
    Returns:
        u: Final state (nx,)
    """
    nx = len(u0)
    k = np.fft.rfftfreq(nx, d=1.0/nx)
    
    u = u0.copy()
    
    for _ in range(nt):
        u_hat = np.fft.rfft(u)
        # Diffusion
        diffusion = -alpha * k**2 * u_hat
        # Convection
        conv = 1j * convection * k * u_hat
        u_hat_new = u_hat + dt * (diffusion + conv)
        u = np.fft.irfft(u_hat_new, n=nx)
    
    return u


def generate_heat_data(
    n_samples: int = 1000,
    nx: int = 256,
    nt: int = 100,
    dt: float = 0.001,
    alpha_range: Tuple[float, float] = (0.01, 0.1),
    convection_range: Optional[Tuple[float, float]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate heat equation dataset.
    
    If convection_range is None, generates pure heat equation data.
    If convection_range is provided, generates heat+convection data.
    
    Returns:
        inputs: (n_samples, n_in, nx) where n_in=2 (no convection) or 3 (with convection)
        outputs: (n_samples, 1, nx)
    """
    rng = np.random.RandomState(seed)
    
    n_in = 2 if convection_range is None else 3
    inputs = np.zeros((n_samples, n_in, nx))
    outputs = np.zeros((n_samples, 1, nx))
    
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    
    for i in range(n_samples):
        alpha = rng.uniform(*alpha_range)
        
        # Random initial condition
        n_modes = rng.randint(1, 6)
        u0 = np.zeros(nx)
        for _ in range(n_modes):
            k = rng.randint(1, 5)
            amp = rng.uniform(-1, 1)
            phase = rng.uniform(0, 2*np.pi)
            u0 += amp * np.sin(k * x + phase)
        
        if convection_range is not None:
            v = rng.uniform(*convection_range)
            u_final = solve_heat_1d(u0, alpha, nt, dt, convection=v)
            inputs[i, 0] = u0
            inputs[i, 1] = alpha * np.ones(nx)
            inputs[i, 2] = v * np.ones(nx)
        else:
            u_final = solve_heat_1d(u0, alpha, nt, dt, convection=0.0)
            inputs[i, 0] = u0
            inputs[i, 1] = alpha * np.ones(nx)
        
        outputs[i, 0] = u_final
    
    return inputs, outputs


class HeatEquationDataset(Dataset):
    """
    Dataset for heat equation (with optional convection).
    
    Supports the input function set extension scenario:
    - Pretrain on heat equation (n_input=2: u0, alpha)
    - Fine-tune on heat+convection (n_input=3: u0, alpha, v)
    """

    def __init__(
        self,
        n_samples: int = 1000,
        nx: int = 256,
        nt: int = 100,
        dt: float = 0.001,
        alpha_range: Tuple[float, float] = (0.01, 0.1),
        convection_range: Optional[Tuple[float, float]] = None,
        seed: int = 42,
        data: Optional[Tuple] = None,
    ):
        if data is not None:
            self.inputs, self.outputs = data
        else:
            self.inputs, self.outputs = generate_heat_data(
                n_samples, nx, nt, dt, alpha_range, convection_range, seed
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
