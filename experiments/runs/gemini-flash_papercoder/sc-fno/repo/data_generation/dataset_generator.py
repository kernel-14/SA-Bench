## data_generation/dataset_generator.py

import os
import random
import numpy as np
import torch
import math
from typing import Callable, Dict, Tuple, List, Any, Optional

# Relative imports from the project structure
from config import Config
from data_generation.ode_solver import ODESolver
from data_generation.pde_solver import PDESolver
from utils import get_ode_equation_function, get_pde_equation_function, get_device, seed_everything

class CustomDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset to hold generated PDE/ODE solutions and their Jacobians.
    """
    def __init__(self, data: Dict[str, List[torch.Tensor]]):
        self.data = data
        self.length = len(data['fno_input_encoder_data'])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {key: self.data[key][idx] for key in self.data}
        return sample

class DatasetGenerator:
    """
    Orchestrates the generation of full datasets for ODEs and PDEs,
    including solutions (u_true) and their parameter sensitivities (du_true_dp).
    Manages data splitting, caching, and preparing FNO-specific inputs/outputs.
    """

    def __init__(self, config: Config, ode_solver: ODESolver, pde_solver: PDESolver):
        """
        Initializes the DatasetGenerator.

        Args:
            config (Config): The configuration object containing global and equation-specific settings.
            ode_solver (ODESolver): An instance of the ODE solver for generating ODE data.
            pde_solver (PDESolver): An instance of the PDE solver for generating PDE data.
        """
        self.config = config
        self.ode_solver = ode_solver
        self.pde_solver = pde_solver
        self.device = get_device()

        # Set random seed for reproducibility of parameter sampling
        seed_everything(self.config.get("experiment.seed"))

        # Create dataset save path if it doesn't exist
        os.makedirs(self.config.get("dataset_generation.dataset_save_path", "./data"), exist_ok=True)

    def _sample_parameters(self, eq_id: str, num_samples: int) -> Dict[str, torch.Tensor]:
        """
        Samples parameter values uniformly from their defined ranges for a given equation.
        Handles both scalar and zoned parameters.

        Args:
            eq_id (str): Identifier for the specific ODE/PDE (e.g., "ODE1", "PDE2").
            num_samples (int): The total number of parameter sets to sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are parameter names
                                     and values are torch.Tensor of sampled values,
                                     shape (num_samples,) for scalar params, or (num_samples, num_zones) for zoned params.
        """
        eq_config = self.config.get(f"equations.{eq_id}")
        if eq_config is None:
            raise ValueError(f"Equation configuration not found for ID: {eq_id}")

        param_ranges = eq_config['parameters']
        sampled_params: Dict[str, torch.Tensor] = {}

        if eq_id == "PDE2_Zoned":
            # Handle global parameters
            for p_name in ['global_gamma', 'global_omega']:
                lower, upper = param_ranges[p_name]
                sampled_params[p_name] = torch.rand(num_samples, device=self.device) * (upper - lower) + lower
            
            # Handle zoned parameters
            num_zones = param_ranges['num_zones']
            for p_name_base in ['alpha', 'delta']:
                range_key = f"{p_name_base}_zone_range"
                lower, upper = param_ranges[range_key]
                # Each sample needs 'num_zones' values for this parameter
                sampled_params[f'{p_name_base}_zones'] = torch.rand(num_samples, num_zones, device=self.device) * (upper - lower) + lower
        else:
            # Handle all other scalar parameters
            for p_name, (lower, upper) in param_ranges.items():
                sampled_params[p_name] = torch.rand(num_samples, device=self.device) * (upper - lower) + lower
        
        return sampled_params

    def generate_sample(self, 
                        eq_type: str, 
                        eq_id: str, 
                        param_values_for_sample: Dict[str, Any], # float, List[float], or torch.Tensor
                        solver_type: str) -> Dict[str, torch.Tensor]:
        """
        Generates a single ground truth solution `u_true` and its Jacobian `du_true_dp`
        for a specific set of parameter values, along with formatted FNO inputs and targets.

        Args:
            eq_type (str): "ODE" or "PDE".
            eq_id (str): Specific equation identifier (e.g., "ODE1", "PDE1").
            param_values_for_sample (Dict[str, Any]): A dictionary containing the *scalar* value
                                                       for each parameter for this specific sample.
                                                       For zoned PDE2, `alpha_zones` and `delta_zones` will be 1D tensors.
            solver_type (str): "AD" or "FD" for ground truth generation.

        Returns:
            Dict[str, torch.Tensor]: Contains the full solution, Jacobian, FNO inputs/targets,
                                     and coordinate grids.
        """
        eq_config = self.config.get(f"equations.{eq_id}")
        if eq_config is None:
            raise ValueError(f"Equation configuration not found for ID: {eq_id}")

        # --- 1. Prepare Parameters for Solvers ---
        solver_params: Dict[str, torch.Tensor] = {}
        p_params_for_fno_ad_list = [] # List to collect all scalar parameters for FNO input
        
        # Consolidate actual parameter values into torch.Tensors for the solver
        for p_name, p_val in param_values_for_sample.items():
            if isinstance(p_val, (float, int)):
                param_tensor = torch.tensor(p_val, dtype=torch.float32, device=self.device)
            elif isinstance(p_val, (list, np.ndarray)):
                param_tensor = torch.tensor(p_val, dtype=torch.float32, device=self.device)
            elif isinstance(p_val, torch.Tensor):
                param_tensor = p_val.to(self.device)
            else:
                raise TypeError(f"Unsupported parameter type for {p_name}: {type(p_val)}")
            
            # For AD solver, params passed to `solve_and_derive_gradients` need `requires_grad=True`
            if solver_type == 'AD':
                solver_params[p_name] = param_tensor.clone().detach().requires_grad_(True)
            else:
                solver_params[p_name] = param_tensor.clone().detach() # Detach for FD as no grad needed from solver params

            # For FNO's `p_params_for_fno_ad`, we need a flattened tensor of all *actual* parameter values.
            # This tensor will later have `requires_grad=True` set during FNO training if needed for L_s.
            p_params_for_fno_ad_list.append(param_tensor.view(-1))
        
        p_params_for_fno_ad = torch.cat(p_params_for_fno_ad_list).detach()


        # --- 2. Prepare Grids and Initial Conditions (u0) ---
        t_discr_N = eq_config['time_discretization_N']
        t_max = self.config.get(f"equations.{eq_id}.t_max", 1.0) # Default t_max for general ODE/PDEs
        t_span = torch.linspace(0.0, t_max, t_discr_N, dtype=torch.float32, device=self.device)
        
        x_coords_full, y_coords_full = None, None
        u0: torch.Tensor
        
        if eq_type == "ODE":
            ode_equation_fn = get_ode_equation_function(eq_id)
            if eq_id == "ODE1":
                gamma = solver_params['gamma']
                u0 = torch.sin(gamma * math.pi).unsqueeze(0) # u(0) = sin(gamma pi)
            elif eq_id == "ODE2":
                epsilon = solver_params['epsilon']
                zeta = solver_params['zeta']
                u0 = torch.stack([epsilon, zeta]) # u0 = [x(0), dx/dt(0)]
            else:
                raise ValueError(f"Unsupported ODE ID for data generation: {eq_id}")
            
        elif eq_type == "PDE":
            pde_equation_fn = get_pde_equation_function(eq_id)
            S_x = eq_config['spatial_discretization_S_x']
            x_min = self.config.get(f"equations.{eq_id}.x_min", 0.0) # Default x_min
            x_max = self.config.get(f"equations.{eq_id}.x_max", 1.0) # Default x_max
            x_coords_full = torch.linspace(x_min, x_max, S_x, dtype=torch.float32, device=self.device)

            if eq_config.get('spatial_discretization_S_y') is not None:
                S_y = eq_config['spatial_discretization_S_y']
                y_min = self.config.get(f"equations.{eq_id}.y_min", 0.0) # Default y_min
                y_max = self.config.get(f"equations.{eq_id}.y_max", 1.0) # Default y_max
                y_coords_full = torch.linspace(y_min, y_max, S_y, dtype=torch.float32, device=self.device)

            if eq_id == "PDE1":
                # u(x,0) = u0, du/dt(x,0) = u0'
                # Assumption: simple initial conditions based on typical wave equation setups.
                # E.g., a sine wave for u0 and zero initial velocity.
                u0_val = torch.sin(math.pi * x_coords_full) if S_x > 1 else torch.tensor([0.0], device=self.device)
                u0_prime_val = torch.zeros_like(x_coords_full) if S_x > 1 else torch.tensor([0.0], device=self.device)
                u0 = torch.stack([u0_val, u0_prime_val]) # Initial state [u(x,0), u_t(x,0)]
            elif eq_id == "PDE2" or eq_id == "PDE2_Zoned":
                x0 = self.config.get(f"equations.{eq_id}.initial_condition_x0", 0.5)
                sigma = self.config.get(f"equations.{eq_id}.initial_condition_sigma", 0.3)
                u0 = torch.exp(-((x_coords_full - x0)**2) / (2 * sigma**2)) + torch.sin(0.5 * math.pi * x_coords_full)
                # Adjust t_span for Burgers' Equation, it's [0, pi]
                t_span = torch.linspace(0.0, math.pi, t_discr_N, dtype=torch.float32, device=self.device)
            elif eq_id == "PDE3":
                alpha = solver_params['alpha']
                beta = solver_params['beta']
                X, Y = torch.meshgrid(x_coords_full, y_coords_full, indexing='ij')
                u0 = (torch.sin(alpha * X) * torch.cos(beta * Y) +
                      torch.cos(alpha * Y) * torch.sin(beta * X) +
                      torch.sin(alpha * X + beta * Y) * torch.cos(alpha * Y - beta * X))
                t_max = self.config.get(f"equations.{eq_id}.output_time_point", 3.0)
                t_span = torch.tensor([0.0, t_max], dtype=torch.float32, device=self.device) # Only interested in IC and final time
            elif eq_id == "PDE4":
                c_param = solver_params['c']
                omega_param = solver_params['omega']
                u0 = c_param * torch.tanh(omega_param * x_coords_full)
            else:
                raise ValueError(f"Unsupported PDE ID for data generation: {eq_id}")
        else:
            raise ValueError(f"Unsupported equation type: {eq_type}")

        # --- 3. Call Numerical Solver ---
        u_true_all: torch.Tensor
        du_true_dp_all: torch.Tensor

        fd_epsilon = self.config.get("dataset_generation.fd_epsilon")

        if eq_type == "ODE":
            u_true_all, du_true_dp_all = self.ode_solver.solve_and_derive_gradients(
                equation_fn=ode_equation_fn,
                u0=u0,
                t_span=t_span,
                params=solver_params,
                grad_method=solver_type,
                fd_epsilon=fd_epsilon
            )
            # ODE solution: (N_t, state_dim). Jacobian: (N_t, state_dim, N_params)
            # For ODEs, spatial coords are typically not relevant for FNO input per se,
            # but FNO architecture might expect an 'x' dimension. We'll handle this by unsqueezing or treating state_dim as spatial.
            if eq_id == "ODE1": # u(t) is scalar, state_dim=1
                u_true_all = u_true_all.unsqueeze(-1) # (N_t, 1)
                du_true_dp_all = du_true_dp_all.unsqueeze(-2) # (N_t, 1, N_params)
            elif eq_id == "ODE2": # u(t) is a vector [x, dx/dt], state_dim=2
                # Keep u_true_all as (N_t, 2), du_true_dp_all as (N_t, 2, N_params)
                pass

        elif eq_type == "PDE":
            u_true_all, du_true_dp_all = self.pde_solver.solve_and_derive_gradients(
                equation_fn=pde_equation_fn,
                u0=u0,
                x_span=x_coords_full,
                t_span=t_span,
                params=solver_params,
                grad_method=solver_type,
                equation_id=eq_id
            )
            # PDE solution u_true_all: (N_t, S_x) or (N_t, S_x, S_y) or (N_t, 2, S_x) for PDE1
            # Jacobian du_true_dp_all: (N_t, S_x, N_params) or (N_t, S_x, S_y, N_params) or (N_t, 2, S_x, N_params)
        
        else:
            raise ValueError(f"Unsupported equation type: {eq_type}")

        # --- 4. Prepare FNO-Specific Inputs/Outputs ---
        M_input_timesteps = eq_config['M_input_timesteps']
        fno_coord_channels = eq_config['fno_coord_channels']

        # Determine shapes for FNO input concatenation
        if eq_type == "ODE":
            # For ODEs, FNO input needs initial conditions (u_0 to u_M-1), t-coords, and parameters.
            # We treat the state_dim as a "feature" or "spatial" dimension for consistency with FNO structure.
            # E.g., for ODE1, (N_t, 1) -> (N_t, 1, 1) after reshaping for FNO input's 'spatial' dim.
            # For ODE2, (N_t, 2) -> (N_t, 2, 1) after reshaping.
            # Let's simplify and assume FNO expects input (batch, time_steps, spatial_dims, features).
            # For ODEs, spatial_dims can be 1, and features will combine u, t, p.
            
            u_input_fno = u_true_all[:M_input_timesteps] # (M, state_dim) or (M, 1)
            fno_target_u = u_true_all[M_input_timesteps:] # (N_t - M, state_dim) or (N_t - M, 1)
            
            fno_target_du_dp = du_true_dp_all[M_input_timesteps:] # (N_t - M, state_dim, N_params) or (N_t - M, 1, N_params)

            # Create coords for FNO input
            t_coords_input = t_span[:M_input_timesteps] # (M,)
            
            # Broadcast coords and params to match u_input_fno dimensions
            # u_input_fno: (M, state_dim)
            # t_coords_input: (M,) -> (M, 1)
            # p_params_for_fno_ad: (N_params,) -> (1, N_params)
            
            # Combine u_input_fno, t_coords_input (repeated for state_dim), p_params_for_fno_ad (repeated for M, state_dim)
            
            # The general FNO input is typically (Batch, time_steps, spatial_x, spatial_y, features)
            # For ODEs, we can treat the state_dim as a 'spatial_x' dimension for FNO processing if needed.
            # Let's assume input_fno_data will be (M_input_timesteps, N_state_vars, num_channels)
            # where num_channels = 1 (for u_value) + 1 (for time) + N_params
            
            # Reshape for FNO input
            u_input_fno_reshaped = u_input_fno.unsqueeze(-1) # (M, state_dim, 1)
            t_coords_input_reshaped = t_coords_input.view(M_input_timesteps, 1, 1).expand(-1, u_input_fno.shape[1], -1) # (M, state_dim, 1)
            p_params_for_fno_ad_reshaped = p_params_for_fno_ad.view(1, 1, -1).expand(M_input_timesteps, u_input_fno.shape[1], -1) # (M, state_dim, N_params)
            
            fno_input_encoder_data = torch.cat([u_input_fno_reshaped, t_coords_input_reshaped, p_params_for_fno_ad_reshaped], dim=-1) # (M, state_dim, 1 + 1 + N_params)
            # For ODEs, it's possible the "x-mode" in config represents the state_dim, or is ignored.
            # We treat state_dim as the "spatial" dimension.
            
            # Targets just need to be (N_t-M, state_dim) or (N_t-M, 1)
            
            # Reshape for FNO output if state_dim > 1 (e.g. ODE2)
            if eq_id == "ODE2": # FNO predicts the state directly, (N_t-M, state_dim)
                fno_target_u_final = fno_target_u 
                fno_target_du_dp_final = fno_target_du_dp
            else: # ODE1, FNO predicts a scalar, (N_t-M, 1)
                fno_target_u_final = fno_target_u.squeeze(-1) if fno_target_u.dim() > 1 else fno_target_u
                fno_target_du_dp_final = fno_target_du_dp.squeeze(-2) if fno_target_du_dp.dim() > 2 else fno_target_du_dp # (N_t-M, N_params)


        elif eq_type == "PDE":
            if eq_id == "PDE3": # maps IC to final state
                u_input_fno = u_true_all[0:1] # Initial condition (1, S_x, S_y)
                fno_target_u = u_true_all[1:2] # Final state (1, S_x, S_y)
                fno_target_du_dp = du_true_dp_all[1:2] # Final Jacobian (1, S_x, S_y, N_params)
                
                # FNO input: u_input_fno (initial vorticity), spatial coords (X, Y), parameters (P)
                X_mesh, Y_mesh = torch.meshgrid(x_coords_full, y_coords_full, indexing='ij') # (S_x, S_y)
                # Expand to (1, S_x, S_y, 1) for feature concatenation
                X_mesh_exp = X_mesh.unsqueeze(0).unsqueeze(-1)
                Y_mesh_exp = Y_mesh.unsqueeze(0).unsqueeze(-1)
                u_input_fno_exp = u_input_fno.unsqueeze(-1) # (1, S_x, S_y, 1)

                p_params_for_fno_ad_exp = p_params_for_fno_ad.view(1, 1, 1, -1).expand(1, X_mesh.shape[0], Y_mesh.shape[1], -1)
                
                fno_input_encoder_data = torch.cat([u_input_fno_exp, X_mesh_exp, Y_mesh_exp, p_params_for_fno_ad_exp], dim=-1)
                # (1, S_x, S_y, 1 + 1 + 1 + N_params)
                # Output dimensions (N_t-M, S_x, S_y) if 2D spatial.
                fno_target_u_final = fno_target_u.squeeze(0) # (S_x, S_y)
                fno_target_du_dp_final = fno_target_du_dp.squeeze(0) # (S_x, S_y, N_params)

            else: # PDE1, PDE2, PDE4 (output N_t-M timesteps)
                u_input_fno = u_true_all[:M_input_timesteps] # (M, S_x) or (M, 2, S_x)
                fno_target_u = u_true_all[M_input_timesteps:] # (N_t - M, S_x) or (N_t - M, 2, S_x)
                fno_target_du_dp = du_true_dp_all[M_input_timesteps:] # (N_t - M, S_x, N_params) or (N_t - M, 2, S_x, N_params)

                # Coords for FNO input
                t_coords_input = t_span[:M_input_timesteps] # (M,)
                
                # Broadcast coords and params to match u_input_fno dimensions
                # Assuming FNO input is (M, S_x, Features)
                # Features = u (1 or state_dim) + t (1) + x (1) + p (N_params)
                
                # Expand t_coords_input to (M, 1, 1)
                t_coords_input_exp = t_coords_input.view(M_input_timesteps, 1, 1).expand(-1, x_coords_full.shape[0], -1)
                # Expand x_coords_full to (1, S_x, 1)
                x_coords_full_exp = x_coords_full.view(1, x_coords_full.shape[0], 1).expand(M_input_timesteps, -1, -1)
                # Expand p_params_for_fno_ad to (1, 1, N_params)
                p_params_for_fno_ad_exp = p_params_for_fno_ad.view(1, 1, -1).expand(M_input_timesteps, x_coords_full.shape[0], -1)
                
                # For PDE1, u_input_fno is (M, 2, S_x). We need to concatenate features along last dim.
                # Let's flatten state_dim for features. So (M, S_x, 2)
                if eq_id == "PDE1":
                    u_input_fno_exp = u_input_fno.permute(0, 2, 1) # (M, S_x, 2)
                    fno_input_encoder_data = torch.cat([u_input_fno_exp, t_coords_input_exp, x_coords_full_exp, p_params_for_fno_ad_exp], dim=-1)
                    # (M, S_x, 2 + 1 + 1 + N_params)
                    fno_target_u_final = fno_target_u.permute(0, 2, 1) # (N_t-M, S_x, 2)
                    fno_target_du_dp_final = fno_target_du_dp.permute(0, 2, 1, 3) # (N_t-M, S_x, 2, N_params)
                else: # PDE2, PDE4, PDE2_Zoned, u_input_fno is (M, S_x)
                    u_input_fno_exp = u_input_fno.unsqueeze(-1) # (M, S_x, 1)
                    fno_input_encoder_data = torch.cat([u_input_fno_exp, t_coords_input_exp, x_coords_full_exp, p_params_for_fno_ad_exp], dim=-1)
                    # (M, S_x, 1 + 1 + 1 + N_params)
                    fno_target_u_final = fno_target_u.unsqueeze(-1) if fno_target_u.dim() == 2 else fno_target_u # (N_t-M, S_x, 1)
                    fno_target_du_dp_final = fno_target_du_dp.unsqueeze(-2) if fno_target_du_dp.dim() == 3 else fno_target_du_dp # (N_t-M, S_x, 1, N_params)
                

        return {
            'u_true_all': u_true_all.detach(),
            'du_true_dp_all': du_true_dp_all.detach(),
            'fno_input_encoder_data': fno_input_encoder_data.detach(),
            'fno_params_for_ad': p_params_for_fno_ad.detach(), # This is the tensor to differentiate w.r.t in FNO
            'fno_target_u': fno_target_u_final.detach(),
            'fno_target_du_dp': fno_target_du_dp_final.detach(),
            't_coords_full': t_span.detach(),
            'x_coords_full': x_coords_full.detach() if x_coords_full is not None else None,
            'y_coords_full': y_coords_full.detach() if y_coords_full is not None else None,
        }

    def generate_dataset(self, eq_type: str, eq_id: str, num_samples: int, solver_type: str) -> Dict[str, CustomDataset]:
        """
        Generates a complete dataset (train, validation, test splits) by calling `generate_sample`
        multiple times, manages saving/loading, and returns `CustomDataset` objects.

        Args:
            eq_type (str): "ODE" or "PDE".
            eq_id (str): Specific equation identifier.
            num_samples (int): Total number of samples for the full dataset.
            solver_type (str): "AD" or "FD" for ground truth generation.

        Returns:
            Dict[str, CustomDataset]: Contains 'train', 'val', 'test' CustomDataset objects.
        """
        dataset_save_path = self.config.get("dataset_generation.dataset_save_path", "./data")
        filename = f"{eq_id}_{num_samples}_{solver_type}.pt"
        filepath = os.path.join(dataset_save_path, filename)

        if os.path.exists(filepath):
            print(f"Loading cached dataset from {filepath}")
            loaded_data = torch.load(filepath, map_location=self.device)
            # Convert loaded lists to CustomDataset instances
            datasets = {
                'train': CustomDataset(loaded_data['train']),
                'val': CustomDataset(loaded_data['val']),
                'test': CustomDataset(loaded_data['test']),
            }
            return datasets

        print(f"Generating new dataset for {eq_id} with {num_samples} samples using {solver_type} solver...")

        # --- 1. Sample All Parameters ---
        all_sampled_params = self._sample_parameters(eq_id, num_samples)

        # --- 2. Split Parameters for Train/Val/Test ---
        train_split = self.config.get("dataset_generation.train_split", 0.7)
        val_split = self.config.get("dataset_generation.val_split", 0.15)
        
        # Ensure splits sum to 1 or less
        if train_split + val_split > 1.0:
            raise ValueError("Train and validation splits sum to more than 1.0.")
        test_split = 1.0 - train_split - val_split if self.config.get("dataset_generation.test_split") is None else self.config.get("dataset_generation.test_split")

        # Shuffle indices for random splitting
        indices = torch.randperm(num_samples, device=self.device).tolist()
        
        num_train = int(train_split * num_samples)
        num_val = int(val_split * num_samples)
        # num_test = num_samples - num_train - num_val # Use this if test_split is not explicitly set or to ensure exact counts

        train_indices = indices[:num_train]
        val_indices = indices[num_train : num_train + num_val]
        test_indices = indices[num_train + num_val:]

        split_params: Dict[str, Dict[str, List[Any]]] = {
            'train': {p_name: [p_vals[i].cpu().item() if p_vals[i].numel() == 1 else p_vals[i].cpu().tolist() for i in train_indices] for p_name, p_vals in all_sampled_params.items()},
            'val': {p_name: [p_vals[i].cpu().item() if p_vals[i].numel() == 1 else p_vals[i].cpu().tolist() for i in val_indices] for p_name, p_vals in all_sampled_params.items()},
            'test': {p_name: [p_vals[i].cpu().item() if p_vals[i].numel() == 1 else p_vals[i].cpu().tolist() for i in test_indices] for p_name, p_vals in all_sampled_params.items()},
        }
        
        # Consolidate parameters per sample for the generator.
        # This is a list of dicts, where each dict is param_values_for_sample for one sample.
        train_param_dicts = [{p_name: split_params['train'][p_name][i] for p_name in all_sampled_params.keys()} for i in range(len(train_indices))]
        val_param_dicts = [{p_name: split_params['val'][p_name][i] for p_name in all_sampled_params.keys()} for i in range(len(val_indices))]
        test_param_dicts = [{p_name: split_params['test'][p_name][i] for p_name in all_sampled_params.keys()} for i in range(len(test_indices))]


        # --- 3. Generate Data for Each Split ---
        aggregated_data: Dict[str, Dict[str, List[torch.Tensor]]] = {
            'train': {
                'u_true_all': [], 'du_true_dp_all': [], 'fno_input_encoder_data': [],
                'fno_params_for_ad': [], 'fno_target_u': [], 'fno_target_du_dp': [],
                't_coords_full': [], 'x_coords_full': [], 'y_coords_full': []
            },
            'val': {
                'u_true_all': [], 'du_true_dp_all': [], 'fno_input_encoder_data': [],
                'fno_params_for_ad': [], 'fno_target_u': [], 'fno_target_du_dp': [],
                't_coords_full': [], 'x_coords_full': [], 'y_coords_full': []
            },
            'test': {
                'u_true_all': [], 'du_true_dp_all': [], 'fno_input_encoder_data': [],
                'fno_params_for_ad': [], 'fno_target_u': [], 'fno_target_du_dp': [],
                't_coords_full': [], 'x_coords_full': [], 'y_coords_full': []
            },
        }

        from tqdm import tqdm # For progress bar
        print("Generating training data...")
        for p_values in tqdm(train_param_dicts, desc="Generating Train Data"):
            sample = self.generate_sample(eq_type, eq_id, p_values, solver_type)
            for key, val in sample.items():
                if val is not None:
                    aggregated_data['train'][key].append(val)
        
        print("Generating validation data...")
        for p_values in tqdm(val_param_dicts, desc="Generating Val Data"):
            sample = self.generate_sample(eq_type, eq_id, p_values, solver_type)
            for key, val in sample.items():
                if val is not None:
                    aggregated_data['val'][key].append(val)

        print("Generating test data...")
        for p_values in tqdm(test_param_dicts, desc="Generating Test Data"):
            sample = self.generate_sample(eq_type, eq_id, p_values, solver_type)
            for key, val in sample.items():
                if val is not None:
                    aggregated_data['test'][key].append(val)

        # --- 4. Instantiate Datasets ---
        datasets = {
            'train': CustomDataset(aggregated_data['train']),
            'val': CustomDataset(aggregated_data['val']),
            'test': CustomDataset(aggregated_data['test']),
        }

        # --- 5. Save Dataset ---
        print(f"Saving generated dataset to {filepath}")
        torch.save({
            'train': aggregated_data['train'],
            'val': aggregated_data['val'],
            'test': aggregated_data['test'],
        }, filepath)
        
        print("Dataset generation complete.")
        return datasets

