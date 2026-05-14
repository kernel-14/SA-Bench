"""
Reaction-diffusion equation dataset generator.

Standard reaction-diffusion:
du/dt = D * Laplacian(u) + R(u)

Extended with advection:
du/dt = D * Laplacian(u) + R(u) + v * du/dx

For the input function set extension scenario, we:
1. Pretrain on reaction-diffusion (inputs: u0, D, r)
2. Fine-tune on reaction-diffusion with advection (inputs: u0, D, r, v)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


def solve_reaction_diffusion_1d(
    u0: np.ndarray,
    D: float,
    r: float,
    nt: int = 100,
    dt: float = 0.001,
    advection: float = 0.0,
) -> np.ndarray:
    """
    Solve 1D reaction-diffusion equation (optionally with advection).
    
    du/dt = D * d^2u/dx^2 + r * u * (1 - u) + advection * du/dx
    
    Uses operator splitting: spectral for diffusion/advection, explicit for reaction.
    
    Args:
        u0: Initial condition (nx,)
        D: Diffusion coefficient
        r: Reaction rate
        nt: Number of time steps
        dt: Time step
        advection: Advection velocity
    
    Returns:
        u: Final state (nx,)
    """
    nx = len(u0)
    k = np.fft.rfftfreq(nx, d=1.0/nx)
    
    u = u0.copy()
    
    for _ in range(nt):
        # Reaction step (explicit)
        reaction = r * u * (1 - u)
        u_half = u + 0.5 * dt * reaction
        
        # Diffusion + advection step (spectral)
        u_hat = np.fft.rfft(u_half)
        diffusion = -D * k**2 * u_hat
        adv = 1j * advection * k * u_hat
        u_hat_new = u_hat + dt * (diffusion + adv)
        u_new = np.fft.irfft(u_hat_new, n=nx)
        
        # Second reaction half-step
        reaction2 = r * u_new * (1 - u_new)
        u = u_new + 0.5 * dt * reaction2
        u = np.clip(u, 0, 1)
    
    return u


def generate_reaction_diffusion_data(
    n_samples: int = 1000,
    nx: int = 256,
    nt: int = 100,
    dt: float = 0.001,
    D_range: Tuple[float, float] = (0.001, 0.05),
    r_range: Tuple[float, float] = (0.1, 2.0),
    advection_range: Optional[Tuple[float, float]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate reaction-diffusion dataset.
    
    Returns:
        inputs: (n_samples, n_in, nx) where n_in=3 (no advection) or 4 (with advection)
        outputs: (n_samples, 1, nx)
    """
    rng = np.random.RandomState(seed)
    
    n_in = 3 if advection_range is None else 4
    inputs = np.zeros((n_samples, n_in, nx))
    outputs = np.zeros((n_samples, 1, nx))
    
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    
    for i in range(n_samples):
        D = rng.uniform(*D_range)
        r = rng.uniform(*r_range)
        
        # Random initial condition (smooth, between 0 and 1)
        n_modes = rng.randint(1, 4)
        u0 = np.zeros(nx)
        for _ in range(n_modes):
            k = rng.randint(1, 4)
            amp = rng.uniform(0, 0.5)
            phase = rng.uniform(0, 2*np.pi)
            u0 += amp * np.sin(k * x + phase)
        u0 = 0.5 + 0.4 * u0 / (np.abs(u0).max() + 1e-8)
        u0 = np.clip(u0, 0, 1)
        
        if advection_range is not None:
            v = rng.uniform(*advection_range)
            u_final = solve_reaction_diffusion_1d(u0, D, r, nt, dt, advection=v)
            inputs[i, 0] = u0
            inputs[i, 1] = D * np.ones(nx)
            inputs[i, 2] = r * np.ones(nx)
            inputs[i, 3] = v * np.ones(nx)
        else:
            u_final = solve_reaction_diffusion_1d(u0, D, r, nt, dt, advection=0.0)
            inputs[i, 0] = u0
            inputs[i, 1] = D * np.ones(nx)
            inputs[i, 2] = r * np.ones(nx)
        
        outputs[i, 0] = u_final
    
    return inputs, outputs


class ReactionDiffusionDataset(Dataset):
    """
    Dataset for reaction-diffusion equation (with optional advection).
    
    Supports the input function set extension scenario:
    - Pretrain on reaction-diffusion (n_input=3: u0, D, r)
    - Fine-tune on reaction-diffusion+advection (n_input=4: u0, D, r, v)
    """

    def __init__(
        self,
        n_samples: int = 1000,
        nx: int = 256,
        nt: int = 100,
        dt: float = 0.001,
        D_range: Tuple[float, float] = (0.001, 0.05),
        r_range: Tuple[float, float] = (0.1, 2.0),
        advection_range: Optional[Tuple[float, float]] = None,
        seed: int = 42,
        data: Optional[Tuple] = None,
    ):
        if data is not None:
            self.inputs, self.outputs = data
        else:
            self.inputs, self.outputs = generate_reaction_diffusion_data(
                n_samples, nx, nt, dt, D_range, r_range, advection_range, seed
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
