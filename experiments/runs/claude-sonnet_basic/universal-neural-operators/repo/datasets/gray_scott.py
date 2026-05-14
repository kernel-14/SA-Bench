"""
Gray-Scott reaction-diffusion model dataset generator.

The Gray-Scott model describes a reaction-diffusion system:
du/dt = Du * Laplacian(u) - u*v^2 + F*(1-u)
dv/dt = Dv * Laplacian(v) + u*v^2 - (F+k)*v

where:
- u, v are concentrations of two chemical species
- Du, Dv are diffusion coefficients
- F is the feed rate
- k is the kill rate

For the out-of-sample parameter values scenario, we train on one set of (F, k) values
and fine-tune on different (F, k) values.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


def solve_gray_scott_2d(
    u0: np.ndarray,
    v0: np.ndarray,
    Du: float = 0.16,
    Dv: float = 0.08,
    F: float = 0.035,
    k: float = 0.065,
    nt: int = 1000,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve 2D Gray-Scott reaction-diffusion system using finite differences.
    
    Args:
        u0, v0: Initial conditions (nx, ny)
        Du, Dv: Diffusion coefficients
        F: Feed rate
        k: Kill rate
        nt: Number of time steps
        dt: Time step
    
    Returns:
        u, v: Final states (nx, ny)
    """
    nx, ny = u0.shape
    dx = 1.0
    
    u = u0.copy()
    v = v0.copy()
    
    for _ in range(nt):
        # Laplacian with periodic boundary conditions
        lap_u = (
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
        ) / dx**2
        
        lap_v = (
            np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
            np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4 * v
        ) / dx**2
        
        uvv = u * v * v
        
        u_new = u + dt * (Du * lap_u - uvv + F * (1 - u))
        v_new = v + dt * (Dv * lap_v + uvv - (F + k) * v)
        
        u = np.clip(u_new, 0, 1)
        v = np.clip(v_new, 0, 1)
    
    return u, v


def generate_gray_scott_data(
    n_samples: int = 200,
    nx: int = 64,
    ny: int = 64,
    nt: int = 500,
    dt: float = 1.0,
    F_range: Tuple[float, float] = (0.02, 0.06),
    k_range: Tuple[float, float] = (0.05, 0.07),
    Du: float = 0.16,
    Dv: float = 0.08,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gray-Scott reaction-diffusion dataset.
    
    Returns:
        inputs: (n_samples, 4, nx, ny) - [u0, v0, F_field, k_field]
        outputs: (n_samples, 2, nx, ny) - [u_final, v_final]
    """
    rng = np.random.RandomState(seed)
    
    inputs = np.zeros((n_samples, 4, nx, ny))
    outputs = np.zeros((n_samples, 2, nx, ny))
    
    for i in range(n_samples):
        # Random parameters
        F = rng.uniform(*F_range)
        k = rng.uniform(*k_range)
        
        # Initial conditions: uniform with small random perturbations
        u0 = np.ones((nx, ny)) + 0.05 * rng.randn(nx, ny)
        v0 = np.zeros((nx, ny)) + 0.05 * rng.randn(nx, ny)
        
        # Add a seed region
        cx, cy = nx // 2, ny // 2
        r = nx // 8
        u0[cx-r:cx+r, cy-r:cy+r] = 0.5 + 0.1 * rng.randn(2*r, 2*r)
        v0[cx-r:cx+r, cy-r:cy+r] = 0.25 + 0.1 * rng.randn(2*r, 2*r)
        
        u0 = np.clip(u0, 0, 1)
        v0 = np.clip(v0, 0, 1)
        
        # Solve
        u_final, v_final = solve_gray_scott_2d(u0, v0, Du, Dv, F, k, nt, dt)
        
        # Inputs: initial conditions + parameter fields
        inputs[i, 0] = u0
        inputs[i, 1] = v0
        inputs[i, 2] = F * np.ones((nx, ny))
        inputs[i, 3] = k * np.ones((nx, ny))
        
        # Outputs: final states
        outputs[i, 0] = u_final
        outputs[i, 1] = v_final
    
    return inputs, outputs


class GrayScottDataset(Dataset):
    """
    Dataset for Gray-Scott reaction-diffusion model.
    """

    def __init__(
        self,
        n_samples: int = 200,
        nx: int = 64,
        ny: int = 64,
        nt: int = 500,
        dt: float = 1.0,
        F_range: Tuple[float, float] = (0.02, 0.06),
        k_range: Tuple[float, float] = (0.05, 0.07),
        seed: int = 42,
        data: Optional[Tuple] = None,
    ):
        if data is not None:
            self.inputs, self.outputs = data
        else:
            self.inputs, self.outputs = generate_gray_scott_data(
                n_samples, nx, ny, nt, dt, F_range, k_range, seed=seed
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
