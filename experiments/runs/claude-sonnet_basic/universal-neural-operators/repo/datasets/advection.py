"""
Advection equation dataset generator.

1D advection equation: du/dt + c * du/dx = 0
where c is the advection velocity.

Used in the multi-physics pretraining scenario (PDEBench).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


def solve_advection_1d(
    u0: np.ndarray,
    c: float,
    nt: int = 100,
    dt: float = 0.001,
) -> np.ndarray:
    """
    Solve 1D advection equation using spectral method.
    
    du/dt + c * du/dx = 0
    
    Args:
        u0: Initial condition (nx,)
        c: Advection velocity
        nt: Number of time steps
        dt: Time step
    
    Returns:
        u: Final state (nx,)
    """
    nx = len(u0)
    k = np.fft.rfftfreq(nx, d=1.0/nx)
    
    u = u0.copy()
    
    for _ in range(nt):
        u_hat = np.fft.rfft(u)
        # Advection in spectral space
        u_hat_new = u_hat * np.exp(-1j * c * k * dt)
        u = np.fft.irfft(u_hat_new, n=nx)
    
    return u


def generate_advection_data(
    n_samples: int = 1000,
    nx: int = 256,
    nt: int = 100,
    dt: float = 0.001,
    c_range: Tuple[float, float] = (-2.0, 2.0),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate advection equation dataset.
    
    Returns:
        inputs: (n_samples, 2, nx) - [initial condition, velocity field]
        outputs: (n_samples, 1, nx) - final state
    """
    rng = np.random.RandomState(seed)
    
    inputs = np.zeros((n_samples, 2, nx))
    outputs = np.zeros((n_samples, 1, nx))
    
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    
    for i in range(n_samples):
        c = rng.uniform(*c_range)
        
        # Random initial condition
        n_modes = rng.randint(1, 6)
        u0 = np.zeros(nx)
        for _ in range(n_modes):
            k = rng.randint(1, 5)
            amp = rng.uniform(-1, 1)
            phase = rng.uniform(0, 2*np.pi)
            u0 += amp * np.sin(k * x + phase)
        
        u_final = solve_advection_1d(u0, c, nt, dt)
        
        inputs[i, 0] = u0
        inputs[i, 1] = c * np.ones(nx)
        outputs[i, 0] = u_final
    
    return inputs, outputs


class AdvectionDataset(Dataset):
    """Dataset for 1D advection equation."""

    def __init__(
        self,
        n_samples: int = 1000,
        nx: int = 256,
        nt: int = 100,
        dt: float = 0.001,
        c_range: Tuple[float, float] = (-2.0, 2.0),
        seed: int = 42,
        data: Optional[Tuple] = None,
    ):
        if data is not None:
            self.inputs, self.outputs = data
        else:
            self.inputs, self.outputs = generate_advection_data(
                n_samples, nx, nt, dt, c_range, seed
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
