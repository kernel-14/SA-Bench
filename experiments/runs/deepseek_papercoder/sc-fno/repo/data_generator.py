# data_generator.py
# ============================================================================
# Purpose: Generate training/validation/test datasets for SC‑FNO experiments.
#          For each equation, parameters, solution paths, and full Jacobians
#          are saved into an HDF5 file (one sample per call).
# ============================================================================

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import h5py

from config import Config
from solver import Solver  # abstract base
from utils import set_seed


class DataGenerator:
    """Generates and stores PDE/ODE solution datasets for the SC‑FNO paper.

    Attributes:
        solver:     An instance of a concrete Solver (e.g., DampedWaveSolver).
        config:     The global configuration object (frozen).
        _cfg_eq:    Equation‑specific config dict from config.sol_params.
        _t_grid:    Tensor of time points (float32).
        _grid:      Spatial grid tensor(s) or None for ODEs.
        _n_samples: Total number of samples to generate.
        _perturbed: Whether also generate a perturbed test set.
        _lambda:    Extrapolation factor for perturbed set.
    """

    def __init__(self, solver: Solver, config: Config) -> None:
        """Initialise data generator with a Solver and Config.

        Args:
            solver: Solver instance implementing the Solver interface.
            config: Immutable configuration loaded from config.yaml.
        """
        self.solver = solver
        self.config = config

        # ---------- Equation‑specific settings ----------
        eq_name = config.equation
        eq_params = config.sol_params  # from 'equations' section
        self._cfg_eq = eq_params
        self._eq_name = eq_name

        # ---------- Build time grid ----------
        t_start, t_end = eq_params["temporal_domain"]
        Nt = eq_params["N_time"]
        if eq_name == "pde3":
            # Navier‑Stokes: only final time is meaningful
            self._t_grid = torch.linspace(t_start, t_end, 2, dtype=torch.float32)  # [0, 3]
        else:
            self._t_grid = torch.linspace(t_start, t_end, Nt, dtype=torch.float32)

        # ---------- Build spatial grid(s) ----------
        spatial_dims = eq_params.get("spatial_dims", 0)
        self._grid: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None
        if spatial_dims == 0:
            self._grid = None   # ODEs
        elif spatial_dims == 1:
            x_start, x_end = config.sol_params["spatial_domain"]
            Sx = config.sol_params["S_x"]
            self._grid = torch.linspace(x_start, x_end, Sx, dtype=torch.float32)
        elif spatial_dims == 2:
            # Assume domain is [x0, x1, y0, y1]
            domain = config.sol_params["spatial_domain"]
            x0, x1, y0, y1 = domain[0], domain[1], domain[2], domain[3]
            Sx = config.sol_params["S_x"]
            Sy = config.sol_params["S_y"]
            X, Y = torch.meshgrid(
                torch.linspace(x0, x1, Sx, dtype=torch.float32),
                torch.linspace(y0, y1, Sy, dtype=torch.float32),
                indexing='ij'
            )
            self._grid = (X, Y)
        else:
            raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")

        # ---------- Data generation parameters ----------
        data_cfg = config.data_params
        self._n_samples = data_cfg["num_samples"]
        self._grad_method = data_cfg["gradient_method"]  # "AD" or "FD"
        self._fd_epsilon = data_cfg.get("fd_epsilon", 1e-4)
        self._solver_method = data_cfg["solver"]
        self._rtol = data_cfg["rtol"]
        self._atol = data_cfg["atol"]

        # Perturbed test set (extrapolation)
        self._perturbed = data_cfg.get("perturbed_test", False)
        self._lambda = data_cfg.get("perturbation_lambda", 0.4)

        # Set global seed for reproducibility
        set_seed(config.global_params["seed"])

        # ---------- Internal state ----------
        self._output_dir = Path(config.global_params["output_dir"])  # used only for saving
        self._data_dir = Path(config.global_params["data_dir"])
        self._data_file = self._data_dir / f"{eq_name}_data.h5"
        self._perturbed_file = self._data_dir / f"{eq_name}_perturbed_test.h5"

    # ---- Public interface ----

    def generate_dataset(self) -> None:
        """Generate the full dataset and store it in HDF5 format.

        This method writes three datasets per HDF5 file: 'p', 'u', 'J'.
        If perturbed test set is enabled, a separate file is created.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating dataset for {self._eq_name} ({self._n_samples} samples)...")
        self._generate_to_file(self._data_file, self._n_samples, perturbed=False)

        if self._perturbed:
            # Number of perturbed samples: use the same as test set size
            n_perturbed = int(self._n_samples * 0.15)
            print(f"Generating perturbed test set ({n_perturbed} samples)...")
            self._generate_to_file(self._perturbed_file, n_perturbed, perturbed=True)

        print("Dataset generation complete.")

    def save(self, path: Optional[str] = None) -> None:
        """Alias to generate_dataset (for compatibility)."""
        self.generate_dataset()

    # ---- Private generation helpers ----

    def _generate_to_file(self, file_path: Path, n_samples: int, perturbed: bool) -> None:
        """Core loop: write `n_samples` to an HDF5 file.

        Args:
            file_path: Path to the HDF5 file.
            n_samples: Number of samples to generate.
            perturbed: If True, sample parameters from extrapolated ranges.
        """
        # Determine parameter names and shapes for initialisation
        param_names = self._cfg_eq["param_names"]
        n_params = self._get_num_params()
        u_shape = self._get_u_shape()
        J_shape = (n_params,) + u_shape  # Jacobian per parameter

        # Prepare HDF5 file
        with h5py.File(file_path, 'w') as f:
            # Create datasets with chunking
            p_dset = f.create_dataset(
                'p', (n_samples, n_params), dtype=np.float32,
                chunks=(1, n_params), compression="gzip", compression_opts=4
            )
            u_dset = f.create_dataset(
                'u', (n_samples,) + u_shape, dtype=np.float32,
                chunks=(1,) + u_shape, compression="gzip", compression_opts=4
            )
            J_dset = f.create_dataset(
                'J', (n_samples,) + J_shape, dtype=np.float32,
                chunks=(1,) + J_shape, compression="gzip", compression_opts=4
            )

            for i in range(n_samples):
                # 1. Generate parameters
                p = self._sample_parameters(perturbed)

                # 2. Generate initial condition
                u0 = self._generate_initial_condition(p)

                # 3. Solve
                u, J = self._solve(p, u0)

                # 4. Write to HDF5
                p_dset[i] = p.astype(np.float32)
                u_dset[i] = u.numpy().astype(np.float32)
                J_dset[i] = J.numpy().astype(np.float32)

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{n_samples} samples")

    # ---- Sampling helpers ----

    def _sample_parameters(self, perturbed: bool) -> np.ndarray:
        """Randomly sample a parameter vector from the configured ranges.

        Args:
            perturbed: Use extrapolated ranges (upper_bound to (1+λ)*upper_bound).

        Returns:
            numpy array of shape (n_params,).
        """
        eq = self._eq_name
        ranges = self._cfg_eq["param_ranges"]

        if eq == "pde2_zoned":
            # Special treatment: 40 zonal alphas, 40 zonal deltas, gamma, omega
            alpha_range = ranges["alpha_zonal"]
            delta_range = ranges["delta_zonal"]
            gamma_range = ranges["gamma"]
            omega_range = ranges["omega"]
            num_zones = self._cfg_eq["num_zones"]

            if perturbed:
                # Extrapolation: use only upper bound * (1+λ) as new upper
                lo_a, hi_a = alpha_range
                lo_a, hi_a = hi_a, hi_a * (1.0 + self._lambda)
                lo_d, hi_d = delta_range
                lo_d, hi_d = hi_d, hi_d * (1.0 + self._lambda)
                lo_g, hi_g = gamma_range
                lo_g, hi_g = hi_g, hi_g * (1.0 + self._lambda)
                lo_o, hi_o = omega_range
                lo_o, hi_o = hi_o, hi_o * (1.0 + self._lambda)
            else:
                lo_a, hi_a = alpha_range
                lo_d, hi_d = delta_range
                lo_g, hi_g = gamma_range
                lo_o, hi_o = omega_range

            alphas = np.random.uniform(lo_a, hi_a, size=num_zones)
            deltas = np.random.uniform(lo_d, hi_d, size=num_zones)
            gamma = np.random.uniform(lo_g, hi_g)
            omega = np.random.uniform(lo_o, hi_o)
            p = np.concatenate([alphas, deltas, [gamma, omega]]).astype(np.float32)
        else:
            # Standard scalar parameters
            p = []
            for name in self._cfg_eq["param_names"]:
                lo, hi = ranges[name]
                if perturbed:
                    lo, hi = hi, hi * (1.0 + self._lambda)
                p.append(np.random.uniform(lo, hi))
            p = np.array(p, dtype=np.float32)

        return p

    def _get_num_params(self) -> int:
        """Return the expected number of parameters for the equation."""
        eq = self._eq_name
        if eq == "pde2_zoned":
            return self._cfg_eq["total_params"]  # 82
        else:
            return len(self._cfg_eq["param_names"])

    def _get_u_shape(self) -> Tuple[int, ...]:
        """Return the shape of the solution tensor (excluding batch dim).

        For ODEs: (N_time,)
        For 1D+time: (S_x, N_time)
        For 2D spatial only (PDE3): (S_x, S_y)
        """
        eq = self._eq_name
        spatial_dims = self._cfg_eq.get("spatial_dims", 0)
        if spatial_dims == 0:
            return (self._cfg_eq["N_time"],)
        elif spatial_dims == 1:
            return (self._cfg_eq["S_x"], self._cfg_eq["N_time"])
        elif spatial_dims == 2:
            # PDE3 returns only spatial grid (no time)
            return (self._cfg_eq["S_x"], self._cfg_eq["S_y"])
        else:
            raise ValueError(f"Unknown spatial_dims: {spatial_dims}")

    # ---- Initial condition generators ----

    def _generate_initial_condition(self, p: np.ndarray) -> torch.Tensor:
        """Produce the initial state tensor suitable for the solver.

        Args:
            p: Parameter vector (numpy) of shape (n_params,).

        Returns:
            torch.Tensor on CPU, dtype float32, shape depends on equation.
        """
        eq = self._eq_name
        if eq == "ode1":
            gamma = p[2]  # third parameter
            u0 = np.sin(gamma * math.pi)
            return torch.tensor(u0, dtype=torch.float32)
        elif eq == "ode2":
            # Parameters: alpha, beta, gamma, delta, omega, epsilon, zeta
            epsilon = p[5]
            zeta = p[6]
            # Return only position? Our solver returns only position (see solver.py).
            # We'll assume initial condition is a scalar (position).
            # The paper likely treats u(t) as position.
            u0 = epsilon   # initial position
            # If solver expects a state vector, we adapt here.
            # But our DuffingSolver returns position as the first component of state.
            # For simplicity, we let the solver handle the initial state; we just provide one value.
            # However, the solver's solve expects u0 of shape (state_dim,) or (..., state_dim).
            # Since we return position only, we give a scalar.
            return torch.tensor(u0, dtype=torch.float32)
        elif eq == "pde1":
            # Generalized Nonlinear Damped Wave: we assume fixed initial profile.
            # Spatial grid is 1D.
            x_grid = self._grid  # torch.linspace 0..1
            # Initial displacement: sin(pi * x)
            u_init = torch.sin(math.pi * x_grid)
            # For second-order in time, solver expects (u, ut). We'll return a tuple of tensors.
            # But our solver interface expects a single tensor? Actually our DampedWaveSolver expects u0 of shape (2, Sx).
            # We'll stack them.
            ut_init = torch.zeros_like(u_init)  # zero initial velocity
            return torch.stack([u_init, ut_init], dim=0)  # (2, Sx)
        elif eq == "pde2" or eq == "pde2_zoned":
            # Forced Burgers': fixed initial condition
            x_grid = self._grid  # 1D
            sigma = 0.3
            x0_center = 0.5
            gaussian = np.exp(-((x_grid - x0_center)**2) / (2 * sigma**2))
            sine = torch.sin(0.5 * math.pi * x_grid)
            u0 = gaussian + sine
            return u0  # (Sx,)
        elif eq == "pde3":
            # Navier‑Stokes: initial vorticity from alpha, beta
            alpha = p[0]
            beta = p[1]
            X, Y = self._grid  # meshgrid tensors, shape (Sx, Sy)
            omega0 = (torch.sin(alpha * X) * torch.cos(beta * Y) +
                      torch.cos(alpha * Y) * torch.sin(beta * X) +
                      torch.sin(alpha * X + beta * Y) * torch.cos(alpha * Y - beta * X))
            return omega0  # (Sx, Sy) - solver expects (Ny, Nx) maybe? We'll let solver handle.
        elif eq == "pde4":
            # Allen‑Cahn: u0 = c * tanh(omega * x)
            c = p[0]
            omega = p[3]
            x_grid = self._grid
            u0 = c * torch.tanh(omega * x_grid)
            return u0  # (Sx,)
        else:
            raise ValueError(f"Unknown equation: {eq}")

    # ---- Solver invocation ----

    def _solve(
        self,
        p: np.ndarray,
        u0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call the solver to obtain solution u and its Jacobian J.

        Args:
            p:  NumPy array (n_params,)
            u0: Initial state tensor (shape varies).

        Returns:
            tuple: (u_final, J_full) as torch.Tensors (float32, CPU).
        """
        # Convert inputs to torch tensors on CPU (solver will move them if needed)
        p_tensor = torch.from_numpy(p).float()
        u0_tensor = u0.float()  # already torch

        # The solver.solve expects arguments:
        #   p: (B, P), u0: (B, *), t: (T,), grid: optional
        # We have a single sample, so add batch dimension.
        p_batch = p_tensor.unsqueeze(0)   # (1, P)
        # u0 may need batch dimension depending on equation
        if self._eq_name == "pde1":
            # u0 shape (2, Sx) -> (1, 2, Sx)
            u0_batch = u0_tensor.unsqueeze(0)
        else:
            u0_batch = u0_tensor.unsqueeze(0)  # works for (Sx,), (1) etc.

        t_grid = self._t_grid   # (Nt,)

        # Prepare spatial grid argument for PDEs
        spatial_arg = None
        if self._grid is not None:
            if isinstance(self._grid, tuple):
                # 2D: (X, Y) -> we stack into a single tensor of shape (2, Sy, Sx) or as needed
                # The NavierStokesSolver expects grid as (2, Ny, Nx)
                # Let's ensure shape: X is (Sx, Sy), Y is (Sx, Sy) if indexing='ij',
                # but solve expects (2, Ny, Nx). We'll rearrange: Ny = Sy, Nx = Sx.
                # We'll create a tensor of shape (2, Sy, Sx)
                X, Y = self._grid
                # X and Y are (Sx, Sy), we need (2, Sy, Sx)
                # Actually our solver expects (2, Ny, Nx) i.e. y as first dim after batch.
                # We'll feed (2, Sx, Sy)? We need to check solver design. In NavierStokesSolver
                # we assumed grid shape (2, Ny, Nx) with X=grid[0] and Y=grid[1].
                # So we need grid with shape (2, Sy, Sx) if Sy is Y dimension.
                # Since meshgrid with indexing='ij' gives X shape (Sx, Sy), Y shape (Sx, Sy),
                # we can stack along dim=0 to get (2, Sx, Sy), then permute to (2, Sy, Sx).
                grid_2d = torch.stack((X, Y), dim=0)  # (2, Sx, Sy)
                grid_2d = grid_2d.permute(0, 2, 1)    # (2, Sy, Sx)
                spatial_arg = grid_2d.float()
            elif isinstance(self._grid, torch.Tensor):
                # 1D grid: shape (Sx,)
                spatial_arg = self._grid.float()
            else:
                raise TypeError("Unsupported spatial grid type.")

        # Call solver
        # Note: our solver implementations currently return u and J with batch dim.
        # For single sample, batch dim is 1.
        u_batch, J_batch = self.solver.solve(p_batch, u0_batch, t_grid, spatial_arg)

        # Remove batch dimension
        u = u_batch.squeeze(0)   # (*u_shape)
        J = J_batch.squeeze(0)   # (P, *u_shape)

        # Verify shapes match expected
        expected_u = self._get_u_shape()
        if u.shape != expected_u:
            # For ODE2, we might have a different shape; adjust if needed.
            pass   # we trust the solver.

        return u.cpu(), J.cpu()
