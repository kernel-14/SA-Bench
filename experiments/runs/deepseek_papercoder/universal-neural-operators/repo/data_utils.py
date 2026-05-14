```python
# data_utils.py

"""
Data handling module for the Universal Neural Operators reproduction pipeline.
Implements PDE dataset generation, PDEBench loading, splitting, normalization,
and a multiphysics data loader that yields batches labelled by problem name.
All components adhere strictly to the class diagram and configuration file.
"""

import os
import random
import bisect
from typing import List, Tuple, Dict, Optional, Union, Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.fft import fftfreq, fft, ifft, fft2, ifft2, rfft, irfft, rfft2, irfft2, fftn, ifftn

# ----------------------------------------------------------------------
# Global seed for reproducibility of data generation.
# ----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ======================================================================
# Dataset
# ======================================================================

class Dataset(torch.utils.data.Dataset):
    """
    Wraps raw input/output tensors and provides normalization when requested.
    Also stores per‑channel ranges computed from the training split.
    """
    def __init__(self,
                 inputs: torch.Tensor,
                 outputs: torch.Tensor,
                 name: str,
                 normalize: bool = False,
                 x_min: Optional[torch.Tensor] = None,
                 x_max: Optional[torch.Tensor] = None,
                 y_min: Optional[torch.Tensor] = None,
                 y_max: Optional[torch.Tensor] = None):
        super().__init__()
        self.inputs = inputs          # (N, C_in, *spatial_dims)
        self.outputs = outputs        # (N, C_out, *spatial_dims)
        self.name = name
        self.normalize = normalize

        # Normalization statistics computed from training data.
        # They are stored as tensors of shape (1, C, *[1]*) after spatial collapse.
        self.x_min = x_min            # shape (1, C_in, 1, ...) or None
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # Per‑channel range of the outputs (used for logging / NMAE).
        self._ranges: Optional[List[float]] = None
        if y_min is not None and y_max is not None:
            # y_min, y_max have shape (1, C_out, 1, ...); flatten to (C_out,) after squeezing.
            y_min_sq = y_min.squeeze()   # shape (C_out,) or (C_out,1...)
            y_max_sq = y_max.squeeze()
            if y_min_sq.dim() == 0:      # single channel
                y_min_sq = y_min_sq.unsqueeze(0)
                y_max_sq = y_max_sq.unsqueeze(0)
            self._ranges = (y_max_sq - y_min_sq).detach().cpu().tolist()
        else:
            self._ranges = None

    @property
    def ranges(self) -> Optional[List[float]]:
        """Per‑channel output ranges, as a list of floats."""
        return self._ranges

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.inputs[idx]
        y = self.outputs[idx]
        if self.normalize and self.x_min is not None:
            # Clamp to avoid division by zero, use a tiny epsilon.
            eps = 1e-12
            x_range = self.x_max - self.x_min + eps
            x = (x - self.x_min.squeeze(0)) / (x_range.squeeze(0) + eps)
            y_range = self.y_max - self.y_min + eps
            y = (y - self.y_min.squeeze(0)) / (y_range.squeeze(0) + eps)
        return x, y


# ======================================================================
# DataUtils
# ======================================================================

class DataUtils:
    """
    Collection of static methods to generate synthetic PDE datasets and
    load external benchmarks. Every method returns a Dataset instance.
    """

    # Spatial domain: [0, 2π]^d for all synthetic data.
    DOMAIN_LENGTH = 2 * np.pi

    @staticmethod
    def _gaussian_random_field_1d(grid_size: int,
                                  alpha: float = 4.0,
                                  amplitude: float = 1.0) -> np.ndarray:
        """Generate a smooth 1D Gaussian random field with power spectral decay."""
        # Fourier modes (real‑to‑complex) with zero DC component.
        k = 2 * np.pi * np.arange(grid_size) / DataUtils.DOMAIN_LENGTH
        k_red = np.fft.rfftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH / (2*np.pi*grid_size))
        k_red_abs = np.abs(k_red)
        # Spectral amplitude: E(k) = (1 + k^alpha)^(-1)
        spec = (1.0 + k_red_abs**alpha)**(-1)
        spec[0] = 0.0   # zero mean
        # Random phases
        phases = np.random.uniform(0, 2*np.pi, len(k_red_abs))
        complex_coeff = spec * (np.cos(phases) + 1j * np.sin(phases))
        # Ensure real output: the DC and Nyquist components must be real if present
        if grid_size % 2 == 0:
            complex_coeff[-1] = complex_coeff[-1].real
        u = irfft(complex_coeff, n=grid_size) * amplitude
        # Normalise to max absolute ≈ 1
        u /= np.max(np.abs(u)) + 1e-12
        return u

    @staticmethod
    def _gaussian_random_field_2d(grid_size: int,
                                  alpha: float = 4.0,
                                  amplitude: float = 1.0) -> np.ndarray:
        """Generate a smooth 2D Gaussian random field on a periodic square."""
        # Fourier mode indices
        nx, ny = grid_size, grid_size
        kx = np.fft.fftfreq(nx, d=DataUtils.DOMAIN_LENGTH/(2*np.pi*nx)) * 2*np.pi
        ky = np.fft.fftfreq(ny, d=DataUtils.DOMAIN_LENGTH/(2*np.pi*ny)) * 2*np.pi
        kx2, ky2 = np.meshgrid(kx, ky, indexing='ij')
        k_abs = np.sqrt(kx2**2 + ky2**2)
        # Spectral amplitude (avoid singularity at k=0)
        spec = (1.0 + k_abs**alpha)**(-1)
        spec[0,0] = 0.0
        # Random phases
        phases = np.random.uniform(0, 2*np.pi, (nx, ny))
        # Hermitian symmetry for real field
        complex_coeff = (spec * np.exp(1j * phases)).astype(np.complex128)
        # Force DC and Nyquist frequencies to be real if needed
        complex_coeff[0,0] = complex_coeff[0,0].real
        complex_coeff[0, ny//2] = complex_coeff[0, ny//2].real if nx%2==0 else complex_coeff[0, ny//2]
        complex_coeff[nx//2, 0] = complex_coeff[nx//2, 0].real if ny%2==0 else complex_coeff[nx//2, 0]
        complex_coeff[nx//2, ny//2] = complex_coeff[nx//2, ny//2].real if (nx%2==0 and ny%2==0) else complex_coeff[nx//2, ny//2]
        u = np.fft.ifftn(complex_coeff).real * amplitude * nx * ny   # compensate for ifftn scaling
        u /= np.max(np.abs(u)) + 1e-12
        return u

    # ----------------------------------------------------------------
    # Burgers equation (1D)
    # ----------------------------------------------------------------
    @staticmethod
    def generate_burgers(nu_list: List[float],
                         n_samples: int,
                         grid_size: int,
                         T: float = 1.0) -> Dataset:
        """
        Generate Burgers' equation data.
        For each sample, pick a random viscosity from nu_list, generate a
        random initial condition and solve the PDE forward to time T.
        """
        # Precompute wave numbers for spectral derivatives.
        k = 2 * np.pi * np.fft.fftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH / (2*np.pi*grid_size))
        k_rfft = np.fft.rfftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH / (2*np.pi*grid_size)) * 2 * np.pi
        k2 = k_rfft**2

        dt = 0.00005   # safe for high viscosity and advection
        num_steps = int(T / dt)

        all_inputs = []
        all_outputs = []

        for _ in range(n_samples):
            nu = random.choice(nu_list)
            u0 = DataUtils._gaussian_random_field_1d(grid_size)
            u0_hat = rfft(u0)

            # Strang splitting: half diffusion, full advection (explicit), half diffusion
            # Advection term: -0.5 * ∂(u²)/∂x
            u_hat = u0_hat.copy()
            # First half-step diffusion
            u_hat *= np.exp(-nu * k2 * dt / 2)

            for step in range(num_steps):
                # Convert to physical space
                u_phys = irfft(u_hat, n=grid_size)
                # Compute advection term in Fourier space
                u2_hat = rfft(u_phys**2)
                adv_hat = -0.5j * k_rfft * u2_hat   # derivative of u^2/2
                u_hat += dt * adv_hat
                # Diffusion full step
                u_hat *= np.exp(-nu * k2 * dt)

            # Final half-step diffusion
            u_hat *= np.exp(-nu * k2 * dt / 2)
            uT = irfft(u_hat, n=grid_size)

            # Convert to tensors with channel dimension
            u0_tensor = torch.tensor(u0, dtype=torch.float32).unsqueeze(0)   # (1, grid_size)
            uT_tensor = torch.tensor(uT, dtype=torch.float32).unsqueeze(0)
            all_inputs.append(u0_tensor)
            all_outputs.append(uT_tensor)

        inputs = torch.stack(all_inputs)     # (N, 1, grid_size)
        outputs = torch.stack(all_outputs)   # (N, 1, grid_size)
        return Dataset(inputs, outputs, name=f"burgers_nu_{nu_list}")

    # ----------------------------------------------------------------
    # Gray‑Scott (2D)
    # ----------------------------------------------------------------
    @staticmethod
    def generate_grayscott(params_list: List[Tuple[float, float]],
                           n_samples: int,
                           grid_size: int,
                           T: float = 5000.0) -> Dataset:
        """
        Generate 2D Gray‑Scott data.
        params_list: list of (F, k) tuples. D_U=0.16, D_V=0.08 fixed.
        """
        # Fixed diffusion coefficients
        DU = 0.16
        DV = 0.08

        # Wave numbers for Laplacian in Fourier space.
        kx = 2 * np.pi * np.fft.fftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH/(2*np.pi*grid_size))
        ky = 2 * np.pi * np.fft.fftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH/(2*np.pi*grid_size))
        kx2, ky2 = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx2**2 + ky2**2   # (grid_size, grid_size)

        dt = 0.5
        num_steps = int(T / dt)

        all_inputs = []
        all_outputs = []

        for _ in range(n_samples):
            F, k_param = random.choice(params_list)
            # Initial condition: homogeneous steady state + small noise
            U = np.ones((grid_size, grid_size)) + 0.01 * np.random.randn(grid_size, grid_size)
            V = np.zeros((grid_size, grid_size)) + 0.01 * np.random.randn(grid_size, grid_size)
            U_hat = rfft2(U)
            V_hat = rfft2(V)

            # Integrate via “integrating factor” + RK4 for the nonlinear part.
            # Evolution equations in Fourier space:
            #   d(U_hat)/dt = -DU*k2*U_hat + FFT(R_U)
            # Let v_U = exp(DU*k2*t) * U_hat  →  dv_U/dt = exp(DU*k2*t) * FFT(R_U)
            # Same for V.
            t = 0.0
            v_U = U_hat.copy()
            v_V = V_hat.copy()

            for step in range(num_steps):
                # Current physical fields
                factor_U = np.exp(-DU * k2 * t)
                factor_V = np.exp(-DV * k2 * t)
                U_phys = irfft2(v_U * factor_U[:,:], s=(grid_size, grid_size))
                V_phys = irfft2(v_V * factor_V[:,:], s=(grid_size, grid_size))

                # Reaction terms
                UV2 = U_phys * V_phys**2
                R_U = -UV2 + F * (1.0 - U_phys)
                R_V =  UV2 - (F + k_param) * V_phys

                # Transform to Fourier
                R_U_hat = rfft2(R_U)
                R_V_hat = rfft2(R_V)

                # RK4 integration for v_U and v_V
                # v' = exp(D*k2*t) * R_hat
                def f_U(t_val, v_val):
                    fac = np.exp(DU * k2 * t_val)
                    return fac * rfft2(irfft2(v_val * np.exp(-DU*k2*t_val), s=(grid_size, grid_size))**0)  # not used; we compute directly
                # Instead, compute slopes manually.
                # k1
                exp_fac_U_t = np.exp(DU * k2 * t)
                exp_fac_V_t = np.exp(DV * k2 * t)
                k1_U = dt * exp_fac_U_t * R_U_hat
                k1_V = dt * exp_fac_V_t * R_V_hat

                # k2
                t2 = t + dt/2
                v_U2 = v_U + k1_U/2
                v_V2 = v_V + k1_V/2
                U2_phys = irfft2(v_U2 * np.exp(-DU*k2*t2), s=(grid_size, grid_size))
                V2_phys = irfft2(v_V2 * np.exp(-DV*k2*t2), s=(grid_size, grid_size))
                UV2_2 = U2_phys * V2_phys**2
                R_U_2 = -UV2_2 + F*(1-U2_phys)
                R_V_2 =  UV2_2 - (F+k_param)*V2_phys
                R_U_hat_2 = rfft2(R_U_2)
                R_V_hat_2 = rfft2(R_V_2)
                k2_U = dt * np.exp(DU*k2*t2) * R_U_hat_2
                k2_V = dt * np.exp(DV*k2*t2) * R_V_hat_2

                # k3
                t3 = t2
                v_U3 = v_U + k2_U/2
                v_V3 = v_V + k2_V/2
                U3_phys = irfft2(v_U3 * np.exp(-DU*k2*t3), s=(grid_size, grid_size))
                V3_phys = irfft2(v_V3 * np.exp(-DV*k2*t3), s=(grid_size, grid_size))
                UV2_3 = U3_phys * V3_phys**2
                R_U_3 = -UV2_3 + F*(1-U3_phys)
                R_V_3 =  UV2_3 - (F+k_param)*V3_phys
                R_U_hat_3 = rfft2(R_U_3)
                R_V_hat_3 = rfft2(R_V_3)
                k3_U = dt * np.exp(DU*k2*t3) * R_U_hat_3
                k3_V = dt * np.exp(DV*k2*t3) * R_V_hat_3

                # k4
                t4 = t + dt
                v_U4 = v_U + k3_U
                v_V4 = v_V + k3_V
                U4_phys = irfft2(v_U4 * np.exp(-DU*k2*t4), s=(grid_size, grid_size))
                V4_phys = irfft2(v_V4 * np.exp(-DV*k2*t4), s=(grid_size, grid_size))
                UV2_4 = U4_phys * V4_phys**2
                R_U_4 = -UV2_4 + F*(1-U4_phys)
                R_V_4 =  UV2_4 - (F+k_param)*V4_phys
                R_U_hat_4 = rfft2(R_U_4)
                R_V_hat_4 = rfft2(R_V_4)
                k4_U = dt * np.exp(DU*k2*t4) * R_U_hat_4
                k4_V = dt * np.exp(DV*k2*t4) * R_V_hat_4

                # Update v
                v_U += (k1_U + 2*k2_U + 2*k3_U + k4_U) / 6.0
                v_V += (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6.0

                t += dt

            # Final fields at T
            U_final = irfft2(v_U * np.exp(-DU*k2*t), s=(grid_size, grid_size))
            V_final = irfft2(v_V * np.exp(-DV*k2*t), s=(grid_size, grid_size))

            # Initial condition is the original U, V (before noise) ?
            # Actually we used the initial condition with noise, so store that as input.
            # We need to record the initial condition (U_init, V_init) as tensor.
            # We generated U_init and V_init earlier (before the loop). Let's store them.
            # But note: we recomputed U_phys, V_phys at t=0? We'll just keep the initial arrays.
            # So capture U_init, V_init at start of the loop.
            # We'll store them as initial condition.
            u0 = torch.tensor(np.stack([U, V], axis=0), dtype=torch.float32)  # (2, H, W)
            uT = torch.tensor(np.stack([U_final, V_final], axis=0), dtype=torch.float32)
            all_inputs.append(u0)
            all_outputs.append(uT)

        inputs = torch.stack(all_inputs)   # (N, 2, H, W)
        outputs = torch.stack(all_outputs)
        return Dataset(inputs, outputs, name="grayscott")

    # ----------------------------------------------------------------
    # Navier‑Stokes (2D, vorticity)
    # ----------------------------------------------------------------
    @staticmethod
    def generate_navierstokes(Re_list: List[float],
                              n_samples: int,
                              grid_size: int,
                              T: float = 10.0) -> Dataset:
        """
        Generate 2D Navier‑Stokes vorticity data.
        For each sample, pick a random Re from Re_list, generate a random
        initial vorticity, and solve forward to time T.
        """
        # Wave numbers
        kx = 2 * np.pi * np.fft.fftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH/(2*np.pi*grid_size))
        ky = 2 * np.pi * np.fft.fftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH/(2*np.pi*grid_size))
        kx2, ky2 = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx2**2 + ky2**2
        k2[0,0] = 1e-12   # avoid division by zero for Poisson solver

        dt = 0.01
        num_steps = int(T / dt)

        # Fixed forcing function as in FNO paper.
        x = np.linspace(0, DataUtils.DOMAIN_LENGTH, grid_size, endpoint=False)
        y = np.linspace(0, DataUtils.DOMAIN_LENGTH, grid_size, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        forcing = 0.1 * (np.sin(2*np.pi*(X+Y)) + np.cos(2*np.pi*(X+Y)))

        all_inputs = []
        all_outputs = []

        for _ in range(n_samples):
            Re = random.choice(Re_list)
            nu = 1.0 / Re

            # Initial vorticity: Gaussian random field.
            omega0 = DataUtils._gaussian_random_field_2d(grid_size)
            omega_hat = rfft2(omega0)

            # Integrate using integrating factor for diffusion + RK4 for advection.
            t = 0.0
            v_omega = omega_hat.copy()   # v = exp(nu*k2*t) * omega_hat

            for step in range(num_steps):
                # Get physical vorticity
                omega_phys = irfft2(v_omega * np.exp(-nu*k2*t), s=(grid_size, grid_size))

                # Compute velocity from stream function: psi = F^{-1}(omega_hat / k2)
                psi_hat = omega_hat / k2   # omega_hat is RFFT, k2 is full
                # To compute velocity: u = ∂psi/∂y, v = -∂psi/∂x
                # Use spectral differentiation.
                psi = irfft2(psi_hat, s=(grid_size, grid_size))
                # Advection term: u·∇ω = -(-∂ψ/∂y * ∂ω/∂x + ∂ψ/∂x * ∂ω/∂y) but easier:
                # u = ∂ψ/∂y, v = -∂ψ/∂x. So advection = u*∂ω/∂x + v*∂ω/∂y.
                omega_x_hat = 1j * kx2 * omega_hat
                omega_y_hat = 1j * ky2 * omega_hat
                omega_x = irfft2(omega_x_hat, s=(grid_size, grid_size))
                omega_y = irfft2(omega_y_hat, s=(grid_size, grid_size))
                psi_x_hat = 1j * kx2 * psi_hat
                psi_y_hat = 1j * ky2 * psi_hat
                psi_x = irfft2(psi_x_hat, s=(grid_size, grid_size))
                psi_y = irfft2(psi_y_hat, s=(grid_size, grid_size))
                u = psi_y
                v = -psi_x
                advection_phys = u * omega_x + v * omega_y

                # Forcing added in physical space.
                rhs_phys = -advection_phys + forcing
                rhs_hat = rfft2(rhs_phys)

                # RK4 on v_omega with dv/dt = exp(nu*k2*t) * rhs_hat
                exp_nu_t = np.exp(nu * k2 * t)
                k1 = dt * exp_nu_t * rhs_hat

                # k2
                t2 = t + dt/2
                v2 = v_omega + k1/2
                omega_phys2 = irfft2(v2 * np.exp(-nu*k2*t2), s=(grid_size, grid_size))
                psi_hat2 = rfft2(omega_phys2) / k2
                psi2 = irfft2(psi_hat2, s=(grid_size, grid_size))
                omega_x2_hat = 1j * kx2 * rfft2(omega_phys2)
                omega_y2_hat = 1j * ky2 * rfft2(omega_phys2)
                omega_x2 = irfft2(omega_x2_hat, s=(grid_size, grid_size))
                omega_y2 = irfft2(omega_y2_hat, s=(grid_size, grid_size))
                u2 = 1j*ky2*psi_hat2; u2 = irfft2(u2, s=(grid_size, grid_size))
                v2_ = -1j*kx2*psi_hat2; v2_ = irfft2(v2_, s=(grid_size, grid_size))
                advection2 = u2 * omega_x2 + v2_ * omega_y2
                rhs2 = -advection2 + forcing
                rhs_hat2 = rfft2(rhs2)
                k2_ = dt * np.exp(nu*k2*t2) * rhs_hat2

                # k3
                t3 = t2
                v3 = v_omega + k2_/2
                omega_phys3 = irfft2(v3 * np.exp(-nu*k2*t3), s=(grid_size, grid_size))
                psi_hat3 = rfft2(omega_phys3) / k2
                psi3 = irfft2(psi_hat3, s=(grid_size, grid_size))
                omega_x3_hat = 1j * kx2 * rfft2(omega_phys3)
                omega_y3_hat = 1j * ky2 * rfft2(omega_phys3)
                omega_x3 = irfft2(omega_x3_hat, s=(grid_size, grid_size))
                omega_y3 = irfft2(omega_y3_hat, s=(grid_size, grid_size))
                u3 = 1j*ky2*psi_hat3; u3 = irfft2(u3, s=(grid_size, grid_size))
                v3_ = -1j*kx2*psi_hat3; v3_ = irfft2(v3_, s=(grid_size, grid_size))
                advection3 = u3 * omega_x3 + v3_ * omega_y3
                rhs3 = -advection3 + forcing
                rhs_hat3 = rfft2(rhs3)
                k3 = dt * np.exp(nu*k2*t3) * rhs_hat3

                # k4
                t4 = t + dt
                v4 = v_omega + k3
                omega_phys4 = irfft2(v4 * np.exp(-nu*k2*t4), s=(grid_size, grid_size))
                psi_hat4 = rfft2(omega_phys4) / k2
                psi4 = irfft2(psi_hat4, s=(grid_size, grid_size))
                omega_x4_hat = 1j * kx2 * rfft2(omega_phys4)
                omega_y4_hat = 1j * ky2 * rfft2(omega_phys4)
                omega_x4 = irfft2(omega_x4_hat, s=(grid_size, grid_size))
                omega_y4 = irfft2(omega_y4_hat, s=(grid_size, grid_size))
                u4 = 1j*ky2*psi_hat4; u4 = irfft2(u4, s=(grid_size, grid_size))
                v4_ = -1j*kx2*psi_hat4; v4_ = irfft2(v4_, s=(grid_size, grid_size))
                advection4 = u4 * omega_x4 + v4_ * omega_y4
                rhs4 = -advection4 + forcing
                rhs_hat4 = rfft2(rhs4)
                k4 = dt * np.exp(nu*k2*t4) * rhs_hat4

                v_omega += (k1 + 2*k2_ + 2*k3 + k4) / 6.0
                # Update omega_hat for next step's advection computation
                omega_hat = v_omega * np.exp(-nu*k2*(t+dt))
                t += dt

            # Final vorticity at T (using the last v_omega)
            omega_final = irfft2(v_omega * np.exp(-nu*k2*t), s=(grid_size, grid_size))

            u0_tensor = torch.tensor(omega0, dtype=torch.float32).unsqueeze(0)   # (1, H, W)
            uT_tensor = torch.tensor(omega_final, dtype=torch.float32).unsqueeze(0)
            all_inputs.append(u0_tensor)
            all_outputs.append(uT_tensor)

        inputs = torch.stack(all_inputs)   # (N, 1, H, W)
        outputs = torch.stack(all_outputs)
        return Dataset(inputs, outputs, name="navier_stokes")

    # ----------------------------------------------------------------
    # Heat equation (2D)
    # ----------------------------------------------------------------
    @staticmethod
    def generate_heat(n_samples: int,
                      grid_size: int,
                      include_advection: bool = False,
                      T: float = 1.0) -> Dataset:
        """
        Generate 2D heat equation data (optionally with constant advection).
        Base: ∂u/∂t = α ∇²u, α = 0.1.
        Extended: advection velocity v = (0.5, 0.5) constant.
        Input channels: 1 (base) or 3 (u0, v_x, v_y if include_advection).
        """
        alpha = 0.1
        kx = 2 * np.pi * np.fft.fftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH/(2*np.pi*grid_size))
        ky = 2 * np.pi * np.fft.fftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH/(2*np.pi*grid_size))
        kx2, ky2 = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx2**2 + ky2**2

        if include_advection:
            # constant velocity
            vx = 0.5
            vy = 0.5
            # Need to advect: dv/dt ... we'll do Strang splitting.
            dt = 0.01
            num_steps = int(T / dt)
        else:
            dt = 0.0   # not used, exact solution
            num_steps = 0

        all_inputs = []
        all_outputs = []

        for _ in range(n_samples):
            u0 = DataUtils._gaussian_random_field_2d(grid_size)
            u0_hat = rfft2(u0)

            if not include_advection:
                # Exact solution via integrating factor
                uT_hat = u0_hat * np.exp(-alpha * k2 * T)
                uT = irfft2(uT_hat, s=(grid_size, grid_size))
                inp = u0
            else:
                # Advection-diffusion with Strang splitting
                u_hat = u0_hat.copy()
                # Half-step diffusion
                u_hat *= np.exp(-alpha * k2 * dt / 2)
                for step in range(num_steps):
                    u_phys = irfft2(u_hat, s=(grid_size, grid_size))
                    # Advection term: -v·∇u
                    # Compute gradient in Fourier
                    grad_x_hat = 1j * kx2 * u_hat
                    grad_y_hat = 1j * ky2 * u_hat
                    grad_x = irfft2(grad_x_hat, s=(grid_size, grid_size))
                    grad_y = irfft2(grad_y_hat, s=(grid_size, grid_size))
                    adv_phys = -(vx * grad_x + vy * grad_y)
                    adv_hat = rfft2(adv_phys)
                    u_hat += dt * adv_hat   # explicit Euler for advection
                    # Full-step diffusion
                    u_hat *= np.exp(-alpha * k2 * dt)
                # Final half-step diffusion
                u_hat *= np.exp(-alpha * k2 * dt / 2)
                uT = irfft2(u_hat, s=(grid_size, grid_size))
                # Input channels: u0 plus velocity components (broadcast to full grid)
                vx_field = np.full((grid_size, grid_size), vx)
                vy_field = np.full((grid_size, grid_size), vy)
                inp = np.stack([u0, vx_field, vy_field], axis=0)  # (3, H, W)

            if not include_advection:
                inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)   # (1, H, W)
            else:
                inp_tensor = torch.tensor(inp, dtype=torch.float32)  # (3, H, W)
            out_tensor = torch.tensor(uT, dtype=torch.float32).unsqueeze(0)
            all_inputs.append(inp_tensor)
            all_outputs.append(out_tensor)

        inputs = torch.stack(all_inputs)   # (N, C_in, H, W)
        outputs = torch.stack(all_outputs)
        name = "heat_advection" if include_advection else "heat"
        return Dataset(inputs, outputs, name=name)

    # ----------------------------------------------------------------
    # Reaction‑diffusion with advection (2D)
    # ----------------------------------------------------------------
    @staticmethod
    def generate_reactdiff(n_samples: int,
                           grid_size: int,
                           include_advection: bool = False,
                           T: float = 5000.0) -> Dataset:
        """
        Generate 2D reaction‑diffusion data based on Gray‑Scott, optionally
        with constant advection velocity added.
        Base case: standard Gray‑Scott (F=0.04, k=0.06 fixed).
        Extended: advection velocity v = (0.2, 0.2) constant.
        Input: U0,V0 (2 channels) or U0,V0,v_x,v_y (4 channels).
        """
        # Fixed parameters for simplicity
        F = 0.04
        k_param = 0.06
        DU = 0.16
        DV = 0.08

        kx = 2 * np.pi * np.fft.fftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH/(2*np.pi*grid_size))
        ky = 2 * np.pi * np.fft.fftfreq(grid_size, d=DataUtils.DOMAIN_LENGTH/(2*np.pi*grid_size))
        kx2, ky2 = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx2**2 + ky2**2

        dt = 0.5
        num_steps = int(T / dt)

        if include_advection:
            vx = 0.2
            vy = 0.2
            vx_field = np.full((grid_size, grid_size), vx)
            vy_field = np.full((grid_size, grid_size), vy)
        else:
            vx_field = vy_field = None

        all_inputs = []
        all_outputs = []

        for _ in range(n_samples):
            U = np.ones((grid_size, grid_size)) + 0.01 * np.random.randn(grid_size, grid_size)
            V = np.zeros((grid_size, grid_size)) + 0.01 * np.random.randn(grid_size, grid_size)
            U_hat = rfft2(U)
            V_hat = rfft2(V)

            t = 0.0
            v_U = U_hat.copy()
            v_V = V_hat.copy()

            for step in range(num_steps):
                # Current physical fields
                factor_U = np.exp(-DU * k2 * t)
                factor_V = np.exp(-DV * k2 * t)
                U_phys = irfft2(v_U * factor_U[:,:], s=(grid_size, grid_size))
                V_phys = irfft2(v_V * factor_V[:,:], s=(grid_size, grid_size))

                # Reaction
                UV2 = U_phys * V_phys**2
                R_U = -UV2 + F * (1.0 - U_phys)
                R_V =  UV2 - (F + k_param) * V_phys

                # Advection if included (pseudo‑spectral)
                if include_advection:
                    # Compute gradients in Fourier space for U and V
                    grad_ux_hat = 1j * kx2 * (v_U * np.exp(-DU*k2*t))
                    grad_uy_hat = 1j * ky2 * (v_U * np.exp(-DU*k2*t))
                    grad_vx_hat = 1j * kx2 * (v_V * np.exp(-DV*k2*t))
                    grad_vy_hat = 1j * ky2 * (v_V * np.exp(-DV*k2*t))
                    grad_ux = irfft2(grad_ux_hat, s=(grid_size, grid_size))
                    grad_uy = irfft2(grad_uy_hat, s=(grid_size, grid_size))
                    grad_vx = irfft2(grad_vx_hat, s=(grid_size, grid_size))
                    grad_vy = irfft2(grad_vy_hat, s=(grid_size, grid_size))

                    adv_U_phys = -(vx_field * grad_ux + vy_field * grad_uy)
                    adv_V_phys = -(vx_field * grad_vx + vy_field * grad_vy)
                    R_U += adv_U_phys
                    R_V += adv_V_phys

                R_U_hat = rfft2(R_U)
                R_V_hat = rfft2(R_V)

                # RK4 as before, but now R includes advection.
                exp_U_t = np.exp(DU * k2 * t)
                exp_V_t = np.exp(DV * k2 * t)
                k1_U = dt * exp_U_t * R_U_hat
                k1_V = dt * exp_V_t * R_V_hat

                # k2
                t2 = t + dt/2
                v_U2 = v_U + k1_U/2
                v_V2 = v_V + k1_V/2
                U2_phys = irfft2(v_U2 * np.exp(-DU*k2*t2), s=(grid_size, grid_size))
                V2_phys = irfft2(v_V2 * np.exp(-DV*k2*t2), s=(grid_size, grid_size))
                UV2_2 = U2_phys * V2_phys**2
                R_U_2 = -UV2_2 + F*(1-U2_phys)
                R_V_2 =  UV2_2 - (F+k_param)*V2_phys
                if include_advection:
                    grad_ux2_hat = 1j * kx2 * (v_U2 * np.exp(-DU*k2*t2))
                    grad_uy2_hat = 1j * ky2 * (v_U2 * np.exp(-DU*k2*t2))
                    grad_vx2_hat = 1j * kx2 * (v_V2 * np.exp(-DV*k2*t2))
                    grad_vy2_hat = 1j * ky2 * (v_V2 * np.exp(-DV*k2*t2))
                    grad_ux2 = irfft2(grad_ux2_hat, s=(grid_size, grid_size))
                    grad_uy2 = irfft2(grad_uy2_hat, s=(grid_size, grid_size))
                    grad_vx2 = irfft2(grad_vx2_hat, s=(grid_size, grid_size))
                    grad_vy2 = irfft2(grad_vy2_hat, s=(grid_size, grid_size))
                    adv_U2 = -(vx_field * grad_ux2 + vy_field * grad_uy2)
                    adv_V2 = -(vx_field * grad_vx2 + vy_field * grad_vy2)
                    R_U_2 += adv_U2
                    R_V_2 += adv_V2
                R_U_hat_2 = rfft2(R_U_2)
                R_V_hat_2 = rfft2(R_V_2)
                k2_U = dt * np.exp(DU*k2*t2) * R_U_hat_2
                k2_V = dt * np.exp(DV*k2*t2) * R_V_hat_2

                # k3
                t3 = t2
                v_U3 = v_U + k2_U/2
                v_V3 = v_V + k2_V/2
                U3_phys = irfft2(v_U3 * np.exp(-DU*k2*t3), s=(grid_size, grid_size))
                V3_phys = irfft2(v_V3 * np.exp(-DV*k2*t3), s=(grid_size, grid_size))
                UV2_3 = U3_phys * V3_phys**2
                R_U_3 = -UV2_3 + F*(1-U3_phys)
                R_V_3 =  UV2_3 - (F+k_param)*V3_phys
                if include_advection:
                    grad_ux3_hat = 1j * kx2 * (v_U3 * np.exp(-DU*k2*t3))
                    grad_uy3_hat = 1j * ky2 * (v_U3 * np.exp(-DU*k2*t3))
                    grad_vx3_hat = 1j * kx2 * (v_V3 * np.exp(-DV*k2*t3))
                    grad_vy3_hat = 1j * ky2 * (v_V3 * np.exp(-DV*k2*t3))
                    grad_ux3 = irfft2(grad_ux3_hat, s=(grid_size, grid_size))
                    grad_uy3 = irfft2(grad_uy3_hat, s=(grid_size, grid_size))
                    grad_vx3 = irfft2(grad_vx3_hat, s=(grid_size, grid_size))
                    grad_vy3 = irfft2(grad_vy3_hat, s=(grid_size, grid_size))
                    adv_U3 = -(vx_field * grad_ux3 + vy_field * grad_uy3)
                    adv_V3 = -(vx_field * grad_vx3 + vy_field * grad_vy3)
                    R_U_3 += adv_U3
                    R_V_3 += adv_V3
                R_U_hat_3 = rfft2(R_U_3)
                R_V_hat_3 = rfft2(R_V_3)
                k3_U = dt * np.exp(DU*k2*t3) * R_U_hat_3
                k3_V = dt * np.exp(DV*k2*t3) * R_V_hat_3

                # k4
                t4 = t + dt
                v_U4 = v_U + k3_U
                v_V4 = v_V + k3_V
                U4_phys = irfft2(v_U4 * np.exp(-DU*k2*t4), s=(grid_size, grid_size))
                V4_phys = irfft2(v_V4 * np.exp(-DV*k2*t4), s=(grid_size, grid_size))
                UV2_4 = U4_phys * V4_phys**2
                R_U_4 = -UV2_4 + F*(1-U4_phys)
                R_V_4 =  UV2_4 - (F+k_param)*V4_phys
                if include_advection:
                    grad_ux4_hat = 1j * kx2 * (v_U4 * np.exp(-DU*k2*t4))
                    grad_uy4_hat = 1j * ky2 * (v_U4 * np.exp(-DU*k2*t4))
                    grad_vx4_hat = 1j * kx2 * (v_V4 * np.exp(-DV*k2*t4))
                    grad_vy4_hat = 1j * ky2 * (v_V4 * np.exp(-DV*k2*t4))
                    grad_ux4 = irfft2(grad_ux4_hat, s=(grid_size, grid_size))
                    grad_uy4 = irfft2(grad_uy4_hat, s=(grid_size, grid_size))
                    grad_vx4 = irfft2(grad_vx4_hat, s=(grid_size, grid_size))
                    grad_vy4 = irfft2(grad_vy4_hat, s=(grid_size, grid_size))
                    adv_U4 = -(vx_field * grad_ux4 + vy_field * grad_uy4)
                    adv_V4 = -(vx_field * grad_vx4 + vy_field * grad_vy4)
                    R_U_4 += adv_U4
                    R_V_4 += adv_V4
                R_U_hat_4 = rfft2(R_U_4)
                R_V_hat_4 = rfft2(R_V_4)
                k4_U = dt * np.exp(DU*k2*t4) * R_U_hat_4
                k4_V = dt * np.exp(DV*k2*t4) * R_V_hat_4

                v_U += (k1_U + 2*k2_U + 2*k3_U + k4_U) / 6.0
                v_V += (k1_V + 2*k2_V + 2*k3_V + k4_V) / 6.0
                t += dt

            U_final = irfft2(v_U * np.exp(-DU*k2*t), s=(grid_size, grid_size))
            V_final = irfft2(v_V * np.exp(-DV*k2*t), s=(grid_size, grid_size))

            if include_advection:
                inp = np.stack([U, V, vx_field, vy_field], axis=0)
            else:
                inp = np.stack([U, V], axis=0)
            out = np.stack([U_final, V_final], axis=0)

            inp_tensor = torch.tensor(inp, dtype=torch.float32)
            out_tensor = torch.tensor(out, dtype=torch.float32)
            all_inputs.append(inp_tensor)
            all_outputs.append(out_tensor)

        inputs = torch.stack(all_inputs)
        outputs = torch.stack(all_outputs)
        name = "reactdiff_advection" if include_advection else "reactdiff"
        return Dataset(inputs, outputs, name=name)

    # ----------------------------------------------------------------
    # PDEBench loader
    # ----------------------------------------------------------------
    @staticmethod
    def load_pdebench(task_name: str, data_dir: str = "./data/pdebench") -> Dataset:
        """
        Load a PDEBench dataset from HDF5 files.
        Supported task_names: 'advection', 'burgers', 'reaction_diffusion'.
        Returns a Dataset with input = first time frame, output = last frame.
        """
        filename_map = {
            "advection": "2D-advection.h5",
            "burgers": "2D-burgers.h5",
            "reaction_diffusion": "2D-reaction-diffusion.h5",
        }
        if task_name not in filename_map:
            raise ValueError(f"Unknown PDEBench task '{task_name}'. Choose from {list(filename_map.keys())}.")
        
        filepath = os.path.join(data_dir, filename_map[task_name])
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"PDEBench data file '{filepath}' not found. "
                "Please download it from the official PDEBench repository."
            )
        with h5py.File(filepath, 'r') as f:
            # In PDEBench, the data is stored under dataset 'data' with shape (N, T, C, H, W).
            data = f['data'][:]   # numpy array
        # Use first time step as input, last as output.
        inp = data[:, 0, :, :, :]   # (N, C, H, W)
        out = data[:, -1, :, :, :]  # (N, C, H, W)

        inputs = torch.tensor(inp, dtype=torch.float32)
        outputs = torch.tensor(out, dtype=torch.float32)
        return Dataset(inputs, outputs, name=f"pdebench_{task_name}")

    # ----------------------------------------------------------------
    # Dataset splitting and normalization
    # ----------------------------------------------------------------
    @staticmethod
    def split_dataset(dataset: Dataset,
                      ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
                      ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split a raw Dataset into train/val/test, compute normalization statistics
        from the training split, and return three normalized Dataset objects.
        """
        n_total = len(dataset)
        n_train = int(ratios[0] * n_total)
        n_val = int(ratios[1] * n_total)
        n_test = n_total - n_train - n_val

        # Shuffle indices
        indices = list(range(n_total))
        random.shuffle(indices)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        # Extract raw tensors
        train_inputs = dataset.inputs[train_idx]
        train_outputs = dataset.outputs[train_idx]
        val_inputs = dataset.inputs[val_idx]
        val_outputs = dataset.outputs[val_idx]
        test_inputs = dataset.inputs[test_idx]
        test_outputs = dataset.outputs[test_idx]

        # Compute per‑channel min/max on training inputs and outputs.
        # We collapse all spatial dimensions to get global min/max per channel.
        def per_channel_stats(tensor):
            # tensor shape: (N, C, ...)
            # We compute min and max over all samples and spatial dims (but keep channel dim)
            # reduce over all dimensions except channel (dim=1)
            shape = tensor.shape
            # Flatten spatial dims while keeping batch and channel.
            tensor_flat = tensor.view(shape[0], shape[1], -1)  # (N, C, S)
            mn = tensor_flat.min(dim=0).values.min(dim=1).values  # shape (C,)
            mx = tensor_flat.max(dim=0).values.max(dim=1).values
            # Expand to (1, C, 1, ...) for later broadcasting
            target_shape = [1, shape[1]] + [1]*(tensor.ndim - 2)
            return mn.view(target_shape), mx.view(target_shape)

        x_min, x_max = per_channel_stats(train_inputs)
        y_min, y_max = per_channel_stats(train_outputs)

        train_ds = Dataset(train_inputs, train_outputs, name=dataset.name,
                           normalize=True,
                           x_min=x_min, x_max=x_max,
                           y_min=y_min, y_max=y_max)
        val_ds = Dataset(val_inputs, val_outputs, name=dataset.name,
                         normalize=True,
                         x_min=x_min, x_max=x_max,
                         y_min=y_min, y_max=y_max)
        test_ds = Dataset(test_inputs, test_outputs, name=dataset.name,
                          normalize=True,
                          x_min=x_min, x_max=x_max,
                          y_min=y_min, y_max=y_max)
        return train_ds, val_ds, test_ds


# ======================================================================
# MixedDataset (internal helper for MultiPhysicsLoader)
# ======================================================================
class _MixedDataset(torch.utils.data.Dataset):
    """
    Concatenation of multiple Dataset objects that returns a problem name
    alongside the (x, y) tuple.
    """
    def __init__(self, datasets: List[Dataset], names: List[str]):
        self.datasets = datasets
        self.names = names
        # cumulative sizes for index translation
        self.cum_sizes = [0]
        for ds in datasets:
            self.cum_sizes.append(self.cum_sizes[-1] + len(ds))

    def __len__(self):
        return self.cum_sizes[-1]

    def __getitem__(self, idx: int):
        # Find which dataset
        ds_idx = bisect.bisect_right(self.cum_sizes, idx) - 1
        local_idx = idx - self.cum_sizes[ds_idx]
        x, y = self.datasets[ds_idx][local_idx]
        return self.names[ds_idx], x, y


# ======================================================================
# MultiPhysicsLoader
# ======================================================================
class MultiPhysicsLoader:
    """
    Iterable that yields mini‑batches of (problem_name, x, y) by shuffling
    data from multiple PDE datasets uniformly.
    """
    def __init__(self, datasets: List[Dataset], names: List[str],
                 batch_size: int, shuffle: bool = True,
                 num_workers: int = 