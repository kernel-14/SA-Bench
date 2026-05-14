"""
Navier-Stokes equations dataset generator.

2D incompressible Navier-Stokes equations in vorticity form:
dw/dt + u * dw/dx + v * dw/dy = (1/Re) * Laplacian(w) + f
where w is vorticity, (u, v) is velocity, Re is Reynolds number, f is forcing.

For the out-of-sample parameter values scenario, we train on one set of Re values
and fine-tune on different Re values.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


def solve_navier_stokes_2d(
    w0: np.ndarray,
    Re: float = 1000.0,
    nt: int = 50,
    dt: float = 0.001,
    forcing: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve 2D incompressible Navier-Stokes in vorticity form using pseudo-spectral method.
    
    dw/dt + J(psi, w) = (1/Re) * Laplacian(w) + f
    
    Args:
        w0: Initial vorticity (nx, ny)
        Re: Reynolds number
        nt: Number of time steps
        dt: Time step
        forcing: External forcing (nx, ny), None for no forcing
    
    Returns:
        w: Final vorticity (nx, ny)
    """
    nx, ny = w0.shape
    
    # Wavenumbers
    kx = np.fft.fftfreq(nx, d=1.0/nx)
    ky = np.fft.fftfreq(ny, d=1.0/ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # Avoid division by zero
    
    w = w0.copy()
    
    for _ in range(nt):
        w_hat = np.fft.fft2(w)
        
        # Stream function: psi_hat = -w_hat / K2
        psi_hat = -w_hat / K2
        
        # Velocity: u = dpsi/dy, v = -dpsi/dx
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat
        
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))
        
        # Vorticity gradients
        dw_dx = np.real(np.fft.ifft2(1j * KX * w_hat))
        dw_dy = np.real(np.fft.ifft2(1j * KY * w_hat))
        
        # Nonlinear term (Jacobian)
        nonlinear = u * dw_dx + v * dw_dy
        
        # Diffusion in spectral space
        diffusion_hat = -(1.0 / Re) * K2 * w_hat
        
        # Time integration
        nonlinear_hat = np.fft.fft2(nonlinear)
        
        if forcing is not None:
            forcing_hat = np.fft.fft2(forcing)
            w_hat_new = w_hat + dt * (diffusion_hat - nonlinear_hat + forcing_hat)
        else:
            w_hat_new = w_hat + dt * (diffusion_hat - nonlinear_hat)
        
        w = np.real(np.fft.ifft2(w_hat_new))
    
    return w


def generate_navier_stokes_data(
    n_samples: int = 200,
    nx: int = 64,
    ny: int = 64,
    nt: int = 50,
    dt: float = 0.001,
    Re_range: Tuple[float, float] = (500.0, 2000.0),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Navier-Stokes dataset.
    
    Returns:
        inputs: (n_samples, 2, nx, ny) - [initial vorticity, Re field]
        outputs: (n_samples, 1, nx, ny) - [final vorticity]
    """
    rng = np.random.RandomState(seed)
    
    inputs = np.zeros((n_samples, 2, nx, ny))
    outputs = np.zeros((n_samples, 1, nx, ny))
    
    # Fixed forcing (Kolmogorov forcing)
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    forcing = 0.1 * np.sin(4 * Y)
    
    for i in range(n_samples):
        # Random Reynolds number
        Re = rng.uniform(*Re_range)
        
        # Random initial vorticity (sum of Fourier modes)
        w0 = np.zeros((nx, ny))
        for k in range(1, 5):
            for l in range(1, 5):
                amp = rng.uniform(-0.5, 0.5)
                phase = rng.uniform(0, 2*np.pi)
                w0 += amp * np.sin(k * X + l * Y + phase)
        
        # Solve
        w_final = solve_navier_stokes_2d(w0, Re, nt, dt, forcing)
        
        # Inputs: initial vorticity + Re field
        inputs[i, 0] = w0
        inputs[i, 1] = Re / 2000.0 * np.ones((nx, ny))  # Normalized Re
        
        # Output: final vorticity
        outputs[i, 0] = w_final
    
    return inputs, outputs


class NavierStokesDataset(Dataset):
    """
    Dataset for 2D Navier-Stokes equations.
    """

    def __init__(
        self,
        n_samples: int = 200,
        nx: int = 64,
        ny: int = 64,
        nt: int = 50,
        dt: float = 0.001,
        Re_range: Tuple[float, float] = (500.0, 2000.0),
        seed: int = 42,
        data: Optional[Tuple] = None,
    ):
        if data is not None:
            self.inputs, self.outputs = data
        else:
            self.inputs, self.outputs = generate_navier_stokes_data(
                n_samples, nx, ny, nt, dt, Re_range, seed
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
