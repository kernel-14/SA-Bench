import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft, fft2, ifft2, rfft, irfft
from typing import Dict, Any, List, Tuple, Optional, Union

from config import Config
from utils import normalize_data, denormalize_data, set_seed

# PDE Bench data will be simulated locally, as external loading is not defined.
# For PDEBench, we use similar generation logic to other PDEs but with parameters
# that might correspond to PDEBench problem types.

class DatasetManager:
    """
    Manages data generation, loading, preprocessing, and DataLoader creation for all PDE scenarios.
    """

    def __init__(self, config: Config):
        """
        Initializes the DatasetManager with the global configuration.

        Args:
            config (Config): The global configuration object.
        """
        if not isinstance(config, Config):
            raise TypeError(f"Expected config to be an instance of Config, got {type(config)}")

        self.config = config
        self.device = torch.device(self.config.device)
        
        self.base_dir = self.config.data_settings.get('base_dir', 'data/')
        self.spatial_resolution = self.config.data_settings.get('spatial_resolution', 64)
        self.temporal_resolution = self.config.data_settings.get('temporal_resolution', 20)
        self.time_steps = self.config.data_settings.get('time_steps', self.temporal_resolution) # Assuming time_steps = temporal_resolution
        
        self.train_ratio = self.config.data_settings.get('train_ratio', 0.8)
        self.val_ratio = self.config.data_settings.get('val_ratio', 0.1)
        self.test_ratio = self.config.data_settings.get('test_ratio', 0.1)
        
        self.pde_configs = self.config.pde_configs

        self.generated_data_dir = os.path.join(self.base_dir, "generated_pde_data")
        os.makedirs(self.generated_data_dir, exist_ok=True)

        self.data_cache: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor], float, float]] = {}
        self.dataset_min_max_vals: Dict[str, Tuple[float, float]] = {} # Stores (min_y, max_y) for each dataset key

    def _get_spatial_coords(self, domain: Union[List[float], List[List[float]]], resolution: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generates spatial coordinates for 1D or 2D domains.

        Args:
            domain (Union[List[float], List[List[float]]]): Spatial domain.
                                                                e.g., [0.0, 1.0] for 1D,
                                                                [[0.0, 1.0], [0.0, 1.0]] for 2D.
            resolution (int): Spatial resolution per dimension.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: x_coords, and y_coords (None for 1D).
        """
        if isinstance(domain[0], list): # 2D domain
            x_min, x_max = domain[0]
            y_min, y_max = domain[1]
            x_coords = np.linspace(x_min, x_max, resolution)
            y_coords = np.linspace(y_min, y_max, resolution)
            return x_coords, y_coords
        else: # 1D domain
            x_min, x_max = domain
            x_coords = np.linspace(x_min, x_max, resolution)
            return x_coords, None

    def _get_temporal_coords(self, t_span: List[float], time_steps: int) -> np.ndarray:
        """
        Generates temporal coordinates.

        Args:
            t_span (List[float]): Time span [t_start, t_end].
            time_steps (int): Number of time steps.

        Returns:
            np.ndarray: Array of time coordinates.
        """
        return np.linspace(t_span[0], t_span[1], time_steps)

    def _generate_initial_condition(self, ic_config: Dict[str, Any],
                                    x_coords: np.ndarray, y_coords: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generates initial conditions (u0) based on configuration.

        Args:
            ic_config (Dict[str, Any]): Configuration for initial condition generation.
            x_coords (np.ndarray): Spatial coordinates for x-axis.
            y_coords (Optional[np.ndarray]): Spatial coordinates for y-axis (if 2D).

        Returns:
            np.ndarray: Initial condition array.
        """
        ic_type = ic_config.get('type', 'random_perturbation')
        
        if y_coords is None: # 1D
            if ic_type == 'sine_wave':
                amplitude = ic_config.get('amplitude', 1.0)
                frequency = ic_config.get('frequency', 2.0 * np.pi)
                phase = ic_config.get('phase', 0.0)
                u0 = amplitude * np.sin(frequency * x_coords + phase)
            elif ic_type == 'gaussian_bump':
                amplitude = ic_config.get('amplitude', 1.0)
                mean = ic_config.get('mean', 0.5 * (x_coords.min() + x_coords.max()))
                sigma = ic_config.get('sigma', 0.1)
                u0 = amplitude * np.exp(-((x_coords - mean)**2) / (2 * sigma**2))
            elif ic_type == 'random_perturbation':
                base_value = ic_config.get('base_value', 0.5)
                noise_level = ic_config.get('noise_level', 0.05)
                u0 = base_value + noise_level * (2 * np.random.rand(*x_coords.shape) - 1)
            else:
                u0 = np.zeros_like(x_coords)
            return u0
        else: # 2D
            X, Y = np.meshgrid(x_coords, y_coords)
            if ic_type == 'gaussian_bump':
                amplitude = ic_config.get('amplitude', 1.0)
                mean_x = ic_config.get('mean_x', 0.5 * (x_coords.min() + x_coords.max()))
                mean_y = ic_config.get('mean_y', 0.5 * (y_coords.min() + y_coords.max()))
                sigma = ic_config.get('sigma', 0.1)
                u0 = amplitude * np.exp(-((X - mean_x)**2 + (Y - mean_y)**2) / (2 * sigma**2))
            elif ic_type == 'random_spots':
                base_value = ic_config.get('base_value', 0.5)
                spot_level = ic_config.get('spot_level', 0.2)
                u0 = base_value + spot_level * (2 * np.random.rand(*X.shape) - 1)
            elif ic_type == 'random_vortex':
                u0 = np.random.rand(*X.shape) - 0.5
            else:
                u0 = np.zeros_like(X)
            return u0


    # --- PDE Solvers (Simplified implementations for data generation) ---

    def _solve_burgers_1d(self, u0: np.ndarray, x_coords: np.ndarray, t_coords: np.ndarray, nu: float) -> np.ndarray:
        """
        Solves 1D Burgers' equation using a pseudo-spectral method.
        u_t + u * u_x = nu * u_xx
        Periodic boundary conditions assumed.

        Args:
            u0 (np.ndarray): Initial condition, shape (Nx,).
            x_coords (np.ndarray): Spatial grid, shape (Nx,).
            t_coords (np.ndarray): Time grid, shape (Nt,).
            nu (float): Viscosity parameter.

        Returns:
            np.ndarray: Solution u(x,t), shape (Nx, Nt).
        """
        Nx = len(x_coords)
        L = x_coords.max() - x_coords.min()
        k = 2 * np.pi / L * np.fft.fftfreq(Nx, d=x_coords[1]-x_coords[0]) # Wavenumbers

        def rhs(t, u_hat):
            u = np.real(ifft(u_hat))
            u_x = np.real(ifft(1j * k * u_hat))
            u_xx_hat = -k**2 * u_hat
            non_linear_term_hat = fft(-u * u_x)
            return non_linear_term_hat + nu * u_xx_hat

        u0_hat = fft(u0)
        sol = solve_ivp(rhs, [t_coords.min(), t_coords.max()], u0_hat,
                        t_eval=t_coords, method='RK45', rtol=1e-5, atol=1e-8)
        
        if not sol.success:
            print(f"Burgers 1D solver failed: {sol.message}")
            # Fallback to initial condition if solver fails
            return np.tile(u0[:, np.newaxis], (1, len(t_coords)))

        return np.real(ifft(sol.y, axis=0)) # Shape (Nx, Nt)


    def _solve_heat_1d(self, u0: np.ndarray, x_coords: np.ndarray, t_coords: np.ndarray, alpha: float, convection_coeff: float = 0.0) -> np.ndarray:
        """
        Solves 1D Heat/Advection-Diffusion equation using an explicit finite difference method.
        u_t = alpha * u_xx - c * u_x
        Dirichlet boundary conditions (u=0) assumed for simplicity.

        Args:
            u0 (np.ndarray): Initial condition, shape (Nx,).
            x_coords (np.ndarray): Spatial grid, shape (Nx,).
            t_coords (np.ndarray): Time grid, shape (Nt,).
            alpha (float): Diffusion coefficient.
            convection_coeff (float): Convection coefficient (0 for pure heat).

        Returns:
            np.ndarray: Solution u(x,t), shape (Nx, Nt).
        """
        Nx = len(x_coords)
        dx = x_coords[1] - x_coords[0]
        dt = t_coords[1] - t_coords[0]
        Nt = len(t_coords)

        u = np.zeros((Nx, Nt))
        u[:, 0] = u0

        # Stability criterion for explicit FTCS
        if dt > 0.5 * dx**2 / alpha and alpha > 1e-9:
             print(f"Warning: dt ({dt}) might be too large for stability with alpha ({alpha}) and dx ({dx}). CFL = {alpha * dt / dx**2}")
        if convection_coeff != 0 and dt > dx / abs(convection_coeff):
             print(f"Warning: dt ({dt}) might be too large for advection stability with c ({convection_coeff}) and dx ({dx}). CFL = {abs(convection_coeff) * dt / dx}")

        for n in range(Nt - 1):
            u_prev = u[:, n].copy()
            
            # Boundary conditions (Dirichlet u=0 for simplicity, adjust if needed)
            u_prev[0] = 0.0
            u_prev[-1] = 0.0

            # Diffusion term (u_xx)
            diff_term = alpha * (np.roll(u_prev, 1) - 2 * u_prev + np.roll(u_prev, -1)) / dx**2
            
            # Convection term (u_x) - using central difference for now, could be upwind
            conv_term = -convection_coeff * (np.roll(u_prev, -1) - np.roll(u_prev, 1)) / (2 * dx)

            u[:, n+1] = u_prev + dt * (diff_term + conv_term)
            
            # Enforce boundary conditions for next step
            u[0, n+1] = 0.0
            u[-1, n+1] = 0.0

        return u

    def _solve_reaction_diffusion_2d(self, u0: np.ndarray, v0: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray,
                                    t_coords: np.ndarray, Du: float, Dv: float, F: float, k: float,
                                    advection_u: float = 0.0, advection_v: float = 0.0) -> np.ndarray:
        """
        Solves 2D Gray-Scott like Reaction-Diffusion system using explicit finite differences.
        u_t = Du * Laplacian(u) - u*v*v + F*(1-u) - adv_u * u_x
        v_t = Dv * Laplacian(v) + u*v*v - (F+k)*v - adv_v * v_y

        Args:
            u0, v0 (np.ndarray): Initial conditions for u and v, shape (Ny, Nx).
            x_coords, y_coords (np.ndarray): Spatial grids.
            t_coords (np.ndarray): Time grid.
            Du, Dv (float): Diffusion coefficients for u and v.
            F, k (float): Reaction parameters (feed rate, kill rate).
            advection_u, advection_v (float): Advection velocities for u and v (0 for pure RD).

        Returns:
            np.ndarray: Solution for u(x,y,t), shape (Ny, Nx, Nt).
                        (We return only u for simplicity, as per typical scalar output).
        """
        Ny, Nx = len(y_coords), len(x_coords)
        dx = x_coords[1] - x_coords[0]
        dy = y_coords[1] - y_coords[0]
        dt = t_coords[1] - t_coords[0]
        Nt = len(t_coords)

        u_sol = np.zeros((Ny, Nx, Nt))
        v_sol = np.zeros((Ny, Nx, Nt))
        u = u0.copy()
        v = v0.copy()

        # Stability check for explicit Euler (simplified)
        max_diff = max(Du, Dv)
        if dt > 0.25 * min(dx, dy)**2 / max_diff and max_diff > 1e-9:
             print(f"Warning: dt ({dt}) might be too large for 2D RD stability with Du={Du}, Dv={Dv}, dx={dx}, dy={dy}. "
                   f"CFL = {max_diff * dt / min(dx, dy)**2}")

        for n in range(Nt):
            u_sol[:, :, n] = u
            v_sol[:, :, n] = v

            if n < Nt - 1:
                # Calculate Laplacian using 5-point stencil
                laplacian_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 2 * u) / dy**2 + \
                              (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 2 * u) / dx**2
                laplacian_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) - 2 * v) / dy**2 + \
                              (np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 2 * v) / dx**2

                # Advection terms (upwind for stability if coefficients are non-zero)
                adv_ux = np.zeros_like(u)
                adv_uy = np.zeros_like(u)
                adv_vx = np.zeros_like(v)
                adv_vy = np.zeros_like(v)

                if advection_u > 0:
                    adv_ux = advection_u * (u - np.roll(u, 1, axis=1)) / dx
                elif advection_u < 0:
                    adv_ux = advection_u * (np.roll(u, -1, axis=1) - u) / dx
                
                if advection_v > 0:
                    adv_vy = advection_v * (v - np.roll(v, 1, axis=0)) / dy
                elif advection_v < 0:
                    adv_vy = advection_v * (np.roll(v, -1, axis=0) - v) / dy

                # Reaction terms
                reaction_u = -u * v**2 + F * (1 - u)
                reaction_v = u * v**2 - (F + k) * v

                # Update equations
                u_new = u + dt * (Du * laplacian_u + reaction_u - adv_ux - adv_uy)
                v_new = v + dt * (Dv * laplacian_v + reaction_v - adv_vx - adv_vy)
                
                u, v = u_new, v_new

                # Boundary conditions: Assume periodic for simplicity (due to roll)
                # If non-periodic, explicit boundary handling needed
        
        # For simplicity, returning only u solution
        return u_sol
    
    def _solve_navier_stokes_2d_simplified(self, u0: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, 
                                            t_coords: np.ndarray, nu: float, # nu is viscosity/diffusion
                                            advection_vel_x: float = 0.1, advection_vel_y: float = 0.0) -> np.ndarray:
        """
        Simplified 2D Navier-Stokes (Advection-Diffusion of a scalar).
        This does NOT solve full Navier-Stokes, but a scalar advection-diffusion problem
        which is a component often related to flow.
        phi_t + (u_vel * phi_x + v_vel * phi_y) = nu * (phi_xx + phi_yy)
        Assumes constant advection_vel_x and advection_vel_y.

        Args:
            u0 (np.ndarray): Initial scalar field (phi), shape (Ny, Nx).
            x_coords, y_coords (np.ndarray): Spatial grids.
            t_coords (np.ndarray): Time grid.
            nu (float): Viscosity/diffusion coefficient.
            advection_vel_x, advection_vel_y (float): Constant advection velocities.

        Returns:
            np.ndarray: Solution phi(x,y,t), shape (Ny, Nx, Nt).
        """
        Ny, Nx = len(y_coords), len(x_coords)
        dx = x_coords[1] - x_coords[0]
        dy = y_coords[1] - y_coords[0]
        dt = t_coords[1] - t_coords[0]
        Nt = len(t_coords)

        phi_sol = np.zeros((Ny, Nx, Nt))
        phi = u0.copy()

        # Stability check
        if nu > 1e-9 and dt > 0.25 * min(dx, dy)**2 / nu:
            print(f"Warning: dt ({dt}) might be too large for 2D Advection-Diffusion stability with nu={nu}. "
                  f"CFL Diffusion = {nu * dt / min(dx, dy)**2}")
        if (abs(advection_vel_x) + abs(advection_vel_y)) > 1e-9 and dt > min(dx, dy) / (abs(advection_vel_x) + abs(advection_vel_y)):
            print(f"Warning: dt ({dt}) might be too large for 2D Advection stability with vel_x={advection_vel_x}, vel_y={advection_vel_y}. "
                  f"CFL Advection = {(abs(advection_vel_x) + abs(advection_vel_y)) * dt / min(dx, dy)}")

        for n in range(Nt):
            phi_sol[:, :, n] = phi
            if n < Nt - 1:
                # Calculate Laplacian
                laplacian_phi = (np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) - 2 * phi) / dy**2 + \
                                (np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 2 * phi) / dx**2

                # Advection terms (using upwind scheme for stability)
                adv_x_term = np.zeros_like(phi)
                if advection_vel_x > 0:
                    adv_x_term = advection_vel_x * (phi - np.roll(phi, 1, axis=1)) / dx
                elif advection_vel_x < 0:
                    adv_x_term = advection_vel_x * (np.roll(phi, -1, axis=1) - phi) / dx
                
                adv_y_term = np.zeros_like(phi)
                if advection_vel_y > 0:
                    adv_y_term = advection_vel_y * (phi - np.roll(phi, 1, axis=0)) / dy
                elif advection_vel_y < 0:
                    adv_y_term = advection_vel_y * (np.roll(phi, -1, axis=0) - phi) / dy

                phi_new = phi + dt * (nu * laplacian_phi - adv_x_term - adv_y_term)
                phi = phi_new
        return phi_sol

    def _generate_pde_data(self, pde_config: Dict[str, Any], is_pretrain: bool) -> Tuple[List[torch.Tensor], List[torch.Tensor], float, float]:
        """
        Generates solutions for a single PDE problem based on its configuration.

        Args:
            pde_config (Dict[str, Any]): Configuration for the specific PDE.
            is_pretrain (bool): True if generating for pre-training, False for fine-tuning/scratch.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], float, float]:
                - List of input feature tensors.
                - List of output solution tensors.
                - Global min value of all solutions.
                - Global max value of all solutions.
        """
        equation_type = pde_config.get('equation_type')
        domain = pde_config.get('domain')
        t_span = pde_config.get('t_span')
        ic_config = pde_config.get('initial_condition_config', {})
        param_ranges = pde_config.get('param_ranges', {})

        num_samples_key = 'num_samples_pretrain' if is_pretrain else 'num_samples_finetune'
        num_samples = pde_config.get(num_samples_key, 100) # Default to 100 if not specified

        x_coords, y_coords = self._get_spatial_coords(domain, self.spatial_resolution)
        t_coords = self._get_temporal_coords(t_span, self.time_steps)

        is_2d = y_coords is not None
        
        all_input_features_X: List[torch.Tensor] = []
        all_solution_Y: List[torch.Tensor] = []
        min_y_global = float('inf')
        max_y_global = float('-inf')

        # To handle parameters that are constant (single value) vs ranges (pretrain/finetune)
        def _get_param_value(param_key: str):
            param_def = param_ranges.get(param_key)
            if isinstance(param_def, dict): # Has pretrain/finetune ranges
                range_key = 'pretrain' if is_pretrain else 'finetune'
                min_val, max_val = param_def.get(range_key, [param_def.get('default', 0.1), param_def.get('default', 0.1)])
                return np.random.uniform(min_val, max_val)
            elif isinstance(param_def, (float, int)): # Constant value
                return float(param_def)
            return None # Should not happen if config is well-formed

        for i in range(num_samples):
            current_params: Dict[str, float] = {}
            for param_name in param_ranges:
                current_params[param_name] = _get_param_value(param_name)

            u0_field = self._generate_initial_condition(ic_config, x_coords, y_coords)
            
            solution_grid: Optional[np.ndarray] = None
            if equation_type == 'Burgers':
                nu = current_params.get('viscosity', 0.01)
                solution_grid = self._solve_burgers_1d(u0_field, x_coords, t_coords, nu)
            elif equation_type == 'Heat':
                alpha = current_params.get('diffusion_coeff', 0.1)
                solution_grid = self._solve_heat_1d(u0_field, x_coords, t_coords, alpha)
            elif equation_type == 'Heat_Convection':
                alpha = current_params.get('diffusion_coeff', 0.1)
                convection_coeff = current_params.get('convection_coeff', 0.5)
                solution_grid = self._solve_heat_1d(u0_field, x_coords, t_coords, alpha, convection_coeff)
            elif equation_type == 'GrayScott' or equation_type == 'ReactionDiffusion':
                Du = current_params.get('D_u', 0.001)
                Dv = current_params.get('D_v', 0.0005)
                F = current_params.get('F', 0.04) # F for Feed rate
                k = current_params.get('k', 0.06) # k for Kill rate
                
                # For Gray-Scott, initial condition is usually u=1, v=0 with noise spots.
                # Here u0_field acts as base, let's generate v0 similarly or with inverse relation.
                v0_field = self._generate_initial_condition(ic_config, x_coords, y_coords) # Can reuse same config for v
                
                solution_grid = self._solve_reaction_diffusion_2d(u0_field, v0_field, x_coords, y_coords, t_coords, Du, Dv, F, k)
            elif equation_type == 'ReactionDiffusion_Advection':
                Du = current_params.get('D_u', 0.001)
                Dv = current_params.get('D_v', 0.0005)
                F = current_params.get('F', 0.04)
                k = current_params.get('k', 0.06)
                advection_velocity = current_params.get('advection_velocity', 0.1)

                v0_field = self._generate_initial_condition(ic_config, x_coords, y_coords)
                solution_grid = self._solve_reaction_diffusion_2d(u0_field, v0_field, x_coords, y_coords, t_coords,
                                                                 Du, Dv, F, k, advection_velocity, advection_velocity)
            elif equation_type == 'NavierStokes': # Simplified 2D Navier-Stokes
                nu = current_params.get('reynolds_number', 100.0) # Using Reynolds inverse as nu for convenience
                nu = 1.0 / nu if nu != 0 else 0.01 # Map Reynolds to viscosity
                # For simplified NS, can also sample advection_vel from a range.
                advection_vel_x = current_params.get('advection_vel_x', 0.1) # Example, could be sampled
                advection_vel_y = current_params.get('advection_vel_y', 0.0) # Example, could be sampled
                solution_grid = self._solve_navier_stokes_2d_simplified(u0_field, x_coords, y_coords, t_coords, nu,
                                                                        advection_vel_x, advection_vel_y)
            elif equation_type.startswith('PDEBench_'):
                # Placeholder: If PDEBench data loading mechanism were implemented, it would go here.
                # For now, simulate similar PDEs or return dummy data.
                print(f"Warning: PDEBench data loading not implemented, simulating dummy data for {equation_type}")
                if is_2d:
                    solution_grid = np.random.rand(self.spatial_resolution, self.spatial_resolution, self.time_steps) * 10
                else:
                    solution_grid = np.random.rand(self.spatial_resolution, self.time_steps) * 10
            else:
                raise ValueError(f"Unknown equation type: {equation_type}")

            if solution_grid is None:
                raise ValueError(f"Solution grid was not generated for {equation_type}")

            # Prepare input features (X) and output solutions (Y)
            if not is_2d: # 1D PDE: solution_grid (Nx, Nt)
                Nx = self.spatial_resolution
                Nt = self.time_steps
                
                # Expand x_coords and t_coords to match (Nx * Nt) for concatenation
                x_flat = np.repeat(x_coords, Nt) # (Nx * Nt,)
                t_flat = np.tile(t_coords, Nx)    # (Nx * Nt,)
                
                # Initial condition: map u0(x) to all time steps
                u0_flat_per_xt = np.repeat(u0_field, Nt) # (Nx * Nt,)

                # Parameters: repeat for every (x,t) point
                param_values = np.array(list(current_params.values())) # (num_params,)
                param_values_flat = np.tile(param_values, (Nx * Nt, 1)) # (Nx * Nt, num_params)

                # Concatenate all input features
                # Input features: (x_coord, t_coord, initial_condition_at_x, param1, param2, ...)
                x_tensor_sample = torch.tensor(np.stack([x_flat, t_flat, u0_flat_per_xt], axis=-1), dtype=torch.float32)
                x_tensor_sample = torch.cat([x_tensor_sample, torch.tensor(param_values_flat, dtype=torch.float32)], dim=-1)
                
                # Output solution: (Nx * Nt, 1)
                y_tensor_sample = torch.tensor(solution_grid.flatten()[:, np.newaxis], dtype=torch.float32)
            else: # 2D PDE: solution_grid (Ny, Nx, Nt)
                Ny, Nx = self.spatial_resolution, self.spatial_resolution
                Nt = self.time_steps
                
                # Create meshgrid for (X, Y) over all time steps
                X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords) # (Ny, Nx)
                X_flat = np.tile(X_mesh.flatten(), Nt) # (Ny*Nx*Nt,)
                Y_flat = np.tile(Y_mesh.flatten(), Nt) # (Ny*Nx*Nt,)
                t_flat = np.repeat(t_coords, Ny * Nx) # (Ny*Nx*Nt,)

                # Initial condition: u0(x,y) at t=0, repeated over time
                u0_flat_per_xyt = np.tile(u0_field.flatten(), Nt) # (Ny*Nx*Nt,)
                
                # Parameters: repeat for every (x,y,t) point
                param_values = np.array(list(current_params.values())) # (num_params,)
                param_values_flat = np.tile(param_values, (Ny * Nx * Nt, 1)) # (Ny*Nx*Nt, num_params)

                # Concatenate all input features
                # Input features: (x_coord, y_coord, t_coord, initial_condition_at_xy, param1, param2, ...)
                x_tensor_sample = torch.tensor(np.stack([X_flat, Y_flat, t_flat, u0_flat_per_xyt], axis=-1), dtype=torch.float32)
                x_tensor_sample = torch.cat([x_tensor_sample, torch.tensor(param_values_flat, dtype=torch.float32)], dim=-1)

                # Output solution: (Ny * Nx * Nt, 1)
                y_tensor_sample = torch.tensor(solution_grid.flatten()[:, np.newaxis], dtype=torch.float32)

            all_input_features_X.append(x_tensor_sample)
            all_solution_Y.append(y_tensor_sample)
            min_y_global = min(min_y_global, y_tensor_sample.min().item())
            max_y_global = max(max_y_global, y_tensor_sample.max().item())

        return all_input_features_X, all_solution_Y, min_y_global, max_y_global


    def load_multiphysics_pretrain_data(self) -> Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor], float, float]]:
        """
        Loads/generates data for all PDEs specified for the pre-training phase.

        Returns:
            Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor], float, float]]:
                A dictionary where keys are PDE names and values are tuples of
                (input_data_list, output_data_list, global_min_y, global_max_y).
        """
        multiphysics_datasets: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor], float, float]] = {}
        
        # Dynamically determine input_dim and output_dim from the first generated dataset
        first_pde_processed = False
        first_input_dim = None
        first_output_dim = None

        for pde_name, pde_config in self.pde_configs.items():
            if 'num_samples_pretrain' in pde_config and pde_config['num_samples_pretrain'] > 0:
                print(f"Generating/Loading pre-training data for {pde_name}...")
                cache_key = f"{pde_name}_pretrain"
                cache_file = os.path.join(self.generated_data_dir, f"{cache_key}.pt")

                if cache_key in self.data_cache:
                    pde_data = self.data_cache[cache_key]
                elif os.path.exists(cache_file):
                    print(f"Loading {cache_key} from cache...")
                    pde_data = torch.load(cache_file, map_location='cpu')
                    self.data_cache[cache_key] = pde_data
                else:
                    print(f"Generating {cache_key} data...")
                    # Generate data for pre-training scenario, with params from pretrain ranges
                    pde_data = self._generate_pde_data(pde_config, is_pretrain=True)
                    torch.save(pde_data, cache_file)
                    self.data_cache[cache_key] = pde_data
                
                multiphysics_datasets[pde_name] = pde_data

                if not first_pde_processed:
                    # Input feature tensor shape is (num_points, input_dim)
                    # Output feature tensor shape is (num_points, output_dim)
                    first_input_dim = pde_data[0][0].shape[-1]
                    first_output_dim = pde_data[1][0].shape[-1]
                    first_pde_processed = True
        
        if first_pde_processed:
            # Update the config with dynamically determined input_dim and output_dim
            # This is crucial for initializing adapters later
            self.config.model_settings['input_dim'] = first_input_dim
            self.config.model_settings['output_dim'] = first_output_dim
            print(f"Dynamically set model input_dim={first_input_dim}, output_dim={first_output_dim}")
        else:
            raise RuntimeError("No pre-training PDE data was generated/loaded. Check config.yaml for 'num_samples_pretrain'.")

        return multiphysics_datasets

    def load_finetuning_data(self, pde_name: str, scenario_type: str) -> Tuple[List[torch.Tensor], List[torch.Tensor], float, float]:
        """
        Loads/generates data for a specific PDE intended for either fine-tuning or scratch training.

        Args:
            pde_name (str): The name of the PDE to load/generate data for.
            scenario_type (str): "finetune" or "scratch".

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], float, float]:
                - List of input feature tensors.
                - List of output solution tensors.
                - Global min value of all solutions.
                - Global max value of all solutions.
        """
        if scenario_type not in ["finetune", "scratch"]:
            raise ValueError(f"scenario_type must be 'finetune' or 'scratch', got {scenario_type}")

        pde_config = self.pde_configs.get(pde_name)
        if pde_config is None:
            raise ValueError(f"PDE '{pde_name}' not found in config.pde_configs.")

        print(f"Generating/Loading {scenario_type} data for {pde_name}...")
        cache_key = f"{pde_name}_{scenario_type}"
        cache_file = os.path.join(self.generated_data_dir, f"{cache_key}.pt")

        if cache_key in self.data_cache:
            pde_data = self.data_cache[cache_key]
        elif os.path.exists(cache_file):
            print(f"Loading {cache_key} from cache...")
            pde_data = torch.load(cache_file, map_location='cpu')
            self.data_cache[cache_key] = pde_data
        else:
            print(f"Generating {cache_key} data...")
            # For fine-tuning/scratch, use finetune-specific ranges
            pde_data = self._generate_pde_data(pde_config, is_pretrain=False)
            torch.save(pde_data, cache_file)
            self.data_cache[cache_key] = pde_data
        
        # Ensure that input/output dimensions are set if this is the *first* data loaded
        if self.config.model_settings.get('input_dim') is None:
             self.config.model_settings['input_dim'] = pde_data[0][0].shape[-1]
             self.config.model_settings['output_dim'] = pde_data[1][0].shape[-1]
             print(f"Dynamically set model input_dim={self.config.model_settings['input_dim']}, output_dim={self.config.model_settings['output_dim']}")

        return pde_data

    def get_dataloaders(self, data_x_list: List[torch.Tensor], data_y_list: List[torch.Tensor],
                        dataset_key: str, batch_size: int, shuffle: bool,
                        train_ratio: Optional[float] = None, val_ratio: Optional[float] = None,
                        test_ratio: Optional[float] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Splits a given dataset into training, validation, and test sets and wraps them in PyTorch DataLoader objects.
        Normalizes the output data (data_y).

        Args:
            data_x_list (List[torch.Tensor]): List of input feature tensors for all samples.
            data_y_list (List[torch.Tensor]): List of output solution tensors for all samples.
            dataset_key (str): A unique string key for this dataset (e.g., "burgers_finetune").
                               Used to store min/max values for denormalization later.
            batch_size (int): Batch size for DataLoaders.
            shuffle (bool): Whether to shuffle the training data.
            train_ratio (Optional[float]): Training set ratio. Defaults to self.train_ratio.
            val_ratio (Optional[float]): Validation set ratio. Defaults to self.val_ratio.
            test_ratio (Optional[float]): Test set ratio. Defaults to self.test_ratio.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
        """
        if not all(isinstance(t, torch.Tensor) for t in data_x_list):
            raise TypeError("All elements in data_x_list must be torch.Tensor.")
        if not all(isinstance(t, torch.Tensor) for t in data_y_list):
            raise TypeError("All elements in data_y_list must be torch.Tensor.")
        if not isinstance(dataset_key, str) or not dataset_key:
            raise ValueError("dataset_key must be a non-empty string.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be a boolean, got {type(shuffle)}")

        train_ratio = train_ratio if train_ratio is not None else self.train_ratio
        val_ratio = val_ratio if val_ratio is not None else self.val_ratio
        test_ratio = test_ratio if test_ratio is not None else self.test_ratio

        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Data split ratios must sum to 1.0, but got {train_ratio + val_ratio + test_ratio}")

        # Concatenate list of tensors into single tensors
        full_data_x = torch.cat(data_x_list, dim=0).to(self.device)
        full_data_y = torch.cat(data_y_list, dim=0).to(self.device)

        # Determine min/max for normalization based on the entire dataset
        min_y = full_data_y.min().item()
        max_y = full_data_y.max().item()

        # Store these for denormalization in evaluation
        self.dataset_min_max_vals[dataset_key] = (min_y, max_y)
        
        # Normalize data_y
        normalized_data_y = normalize_data(full_data_y, min_y, max_y)

        full_dataset = TensorDataset(full_data_x, normalized_data_y)

        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size # Ensure all samples are accounted for

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Created DataLoaders for {dataset_key}: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        return train_loader, val_loader, test_loader

# Example Usage (for local testing, not part of the main project structure)
if __name__ == "__main__":
    from config import Config
    import yaml

    # Create a dummy config.yaml for testing
    dummy_config_content = """
    experiment:
      name: "test_data_manager"
      seed: 42
      device: "cpu"

    data:
      base_dir: "temp_test_data/"
      spatial_resolution: 32
      temporal_resolution: 10
      train_ratio: 0.7
      val_ratio: 0.15
      test_ratio: 0.15
      pde_configs:
        burgers_test_pretrain:
          equation_type: "Burgers"
          domain: [0.0, 1.0]
          t_span: [0.0, 1.0]
          initial_condition_config:
            type: "sine_wave"
            amplitude: 1.0
            frequency: 6.28
          param_ranges:
            viscosity:
              pretrain: [0.01, 0.05]
              finetune: [0.005, 0.009]
          num_samples_pretrain: 5
          num_samples_finetune: 0 # Not used for this specific entry

        heat_convection_test_finetune:
          equation_type: "Heat_Convection"
          domain: [0.0, 1.0]
          t_span: [0.0, 1.0]
          initial_condition_config:
            type: "gaussian_bump"
            amplitude: 1.0
            mean: 0.25
            sigma: 0.05
          param_ranges:
            diffusion_coeff: 0.05
            convection_coeff:
              pretrain: [0.0, 0.0] # Not used here, just for completeness
              finetune: [0.5, 1.5]
          num_samples_pretrain: 0
          num_samples_finetune: 3

        gray_scott_test_pretrain:
          equation_type: "ReactionDiffusion"
          domain: [[0.0, 1.0], [0.0, 1.0]]
          t_span: [0.0, 5.0]
          initial_condition_config:
            type: "random_spots"
            base_value: 0.5
            spot_level: 0.2
          param_ranges:
            D_u:
              pretrain: [0.0001, 0.0005]
            D_v:
              pretrain: [0.00005, 0.0001]
            F:
              pretrain: [0.03, 0.05]
            k:
              pretrain: [0.05, 0.07]
          num_samples_pretrain: 2

    model:
      input_dim: null # Will be set dynamically
      output_dim: null # Will be set dynamically
      hidden_dim: 64

    training:
      batch_size: 16
    """
    config_file_path = "temp_test_config_dm.yaml"
    with open(config_file_path, "w", encoding='utf-8') as f:
        f.write(dummy_config_content)

    try:
        set_seed(42)
        cfg = Config.load_config(config_file_path)
        dm = DatasetManager(cfg)

        print("\n--- Testing load_multiphysics_pretrain_data ---")
        pretrain_data = dm.load_multiphysics_pretrain_data()
        print(f"Loaded {len(pretrain_data)} pre-training PDEs.")
        for pde_name, (x_list, y_list, min_y, max_y) in pretrain_data.items():
            print(f"  {pde_name}: {len(x_list)} samples. First X shape: {x_list[0].shape}, First Y shape: {y_list[0].shape}, Min/Max Y: {min_y:.4f}/{max_y:.4f}")
        print(f"Config input_dim after pretrain: {cfg.model_settings.get('input_dim')}")
        print(f"Config output_dim after pretrain: {cfg.model_settings.get('output_dim')}")

        print("\n--- Testing get_dataloaders for a pretrain PDE ---")
        burgers_x, burgers_y, burgers_min_y, burgers_max_y = pretrain_data['burgers_test_pretrain']
        train_loader, val_loader, test_loader = dm.get_dataloaders(
            burgers_x, burgers_y, "burgers_test_pretrain_dataset", dm.config.training_settings['batch_size'], True
        )
        print(f"Burgers Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        # Check denormalization values
        print(f"Burgers dataset min/max for normalization: {dm.dataset_min_max_vals['burgers_test_pretrain_dataset']}")
        
        # Test a batch
        for batch_x, batch_y in train_loader:
            print(f"  Train Batch X shape: {batch_x.shape}, Y shape: {batch_y.shape}")
            break


        print("\n--- Testing load_finetuning_data ---")
        ft_x, ft_y, ft_min_y, ft_max_y = dm.load_finetuning_data('heat_convection_test_finetune', 'finetune')
        print(f"Loaded finetuning data for heat_convection_test_finetune: {len(ft_x)} samples.")
        print(f"  First X shape: {ft_x[0].shape}, First Y shape: {ft_y[0].shape}, Min/Max Y: {ft_min_y:.4f}/{ft_max_y:.4f}")
        
        print("\n--- Testing get_dataloaders for a finetune PDE ---")
        train_loader_ft, val_loader_ft, test_loader_ft = dm.get_dataloaders(
            ft_x, ft_y, "heat_convection_test_finetune_dataset", dm.config.training_settings['batch_size'], True
        )
        print(f"Heat_Convection Finetune Train batches: {len(train_loader_ft)}, Val batches: {len(val_loader_ft)}, Test batches: {len(test_loader_ft)}")
        print(f"Heat_Convection dataset min/max for normalization: {dm.dataset_min_max_vals['heat_convection_test_finetune_dataset']}")


    except (FileNotFoundError, yaml.YAMLError, ValueError, TypeError) as e:
        print(f"Error during DatasetManager test: {e}")
    finally:
        # Clean up dummy config file and data directory
        if os.path.exists(config_file_path):
            os.remove(config_file_path)
        if os.path.exists(dm.generated_data_dir):
            import shutil
            shutil.rmtree(dm.generated_data_dir)

