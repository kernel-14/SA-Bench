
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py # Assuming PDEBench data is HDF5
from scipy.ndimage import zoom # For downsampling / interpolation
from typing import Tuple, Optional, Callable

from config import Config

class PDEDataset(Dataset):
    def __init__(self, config: Config, data_path: str, is_train: bool = True, resolution_factor: int = 1):
        """
        Generic PDE dataset. Loads data from HDF5 files or generates dummy data.
        Assumes data is stored as (num_samples, timesteps, spatial_x, [spatial_y], channels).
        For 1D PDEs, spatial_y might be 1 or absent.
        For 2D PDEs, spatial_y is present.
        """
        self.config = config
        self.data_path = data_path
        self.is_train = is_train
        self.resolution_factor = resolution_factor # For multi-resolution training
        self.is_3d_spatial = (config.pde_type in ["2d_fluid", "era5"]) # Spatial dimension includes (H, W) or (D, H, W)
        
        # Load or generate dummy data
        self.data_u, self.data_a, self.data_f, self.data_target_u = self._load_or_generate_data()

        # Apply downsampling if resolution_factor > 1 (for multi-resolution training)
        if self.resolution_factor > 1:
            self.data_u = self._downsample(self.data_u, self.resolution_factor)
            self.data_a = self._downsample(self.data_a, self.resolution_factor)
            if self.data_f is not None:
                self.data_f = self._downsample(self.data_f, self.resolution_factor)
            if self.data_target_u is not None:
                self.data_target_u = self._downsample(self.data_target_u, self.resolution_factor)
    
    def _load_or_generate_data(self):
        # Placeholder for actual data loading from HDF5
        # For now, generate dummy data based on config.pde_type
        print(f"Loading/Generating data for {self.config.pde_type}...")
        
        # Determine shapes based on PDE type
        if self.config.pde_type == "1d_burgers":
            # u: (B, T, S, C), a (u0): (B, 1, S, C), f: (B, T-1, S, C), target_u: (B, 1, S, C)
            num_samples = 40000 if self.is_train else 50 # As per paper F.2
            timesteps = self.config.data_res_h # 81
            spatial_res = self.config.data_res_w # 120
            data_u = torch.randn(num_samples, timesteps, spatial_res, self.config.raw_input_channels_x, dtype=torch.float32)
            # For 1D Burgers, a is u0 (1 channel), f is (T-1) time steps (1 channel)
            # When combined, cond has 2 channels.
            data_a = torch.randn(num_samples, 1, spatial_res, self.config.data_channels, dtype=torch.float32) # u0
            data_f = torch.randn(num_samples, timesteps - 1, spatial_res, self.config.data_channels, dtype=torch.float32) # f
            data_target_u = torch.randn(num_samples, 1, spatial_res, self.config.data_channels, dtype=torch.float32) # u_T
            
        elif self.config.pde_type == "1d_advection":
            # u: (B, T, S, C), a (u0): (B, 1, S, C)
            num_samples = 10000 if self.is_train else 100 # Example
            timesteps = self.config.data_res_h # 80
            spatial_res = self.config.data_res_w # 120
            channels = self.config.data_channels # 1

            data_u = torch.randn(num_samples, timesteps, spatial_res, self.config.raw_input_channels_x, dtype=torch.float32)
            data_a = torch.randn(num_samples, 1, spatial_res, self.config.raw_input_channels_cond, dtype=torch.float32) # u0
            data_f = None
            data_target_u = None

        elif self.config.pde_type == "1d_navier_stokes":
            # u: (B, T, S, C), a (u0): (B, 1, S, C)
            num_samples = 40000 if self.is_train else 50 # Assuming similar to Burgers
            timesteps = self.config.data_res_h # 81
            spatial_res = self.config.data_res_w # 120
            channels = self.config.data_channels # 1 (rho, v, p for Navier-Stokes, but paper says 1 channel output)

            data_u = torch.randn(num_samples, timesteps, spatial_res, self.config.raw_input_channels_x, dtype=torch.float32)
            data_a = torch.randn(num_samples, 1, spatial_res, self.config.raw_input_channels_cond, dtype=torch.float32) # u0
            data_f = None
            data_target_u = None

        elif self.config.pde_type == "2d_fluid":
            # u: (B, D, H, W, C), a (initial density): (B, 1, H, W, C_density), f: (B, D, H_force, W_force, C_force)
            # control can only be exercised out of the frame.
            # target_u (percentage of smoke): (B, 1, 1, 1, 1)
            num_samples = 10000 if self.is_train else 50 # Example
            timesteps = self.config.data_res_h # 32
            spatial_x = self.config.data_res_w # 64
            spatial_y = self.config.data_res_d # 64
            channels_u_raw = self.config.raw_input_channels_x # 3 (density, vx, vy)
            channels_cond_raw = self.config.raw_input_channels_cond # 2 (initial density + force for sim, or initial density + target for control)
            
            data_u = torch.randn(num_samples, timesteps, spatial_x, spatial_y, channels_u_raw, dtype=torch.float32)
            # a_data holds initial density (1ch), f_data holds control (1ch) or target_u_data holds target smoke (1ch)
            # For 2D fluid, condition_data is initial density (1ch) AND force (1ch) for simulation
            # OR initial density (1ch) AND target smoke (1ch) for control.
            # The `PDE_Solver_Mock` expects `u0` and `f`.
            # For PDEDataset, `data_a` is initial_density (B,1,H,W,1), `data_f` is force (B,T,H,W,1)
            # `data_target_u` is target smoke (B,1,1,1,1)
            data_a = torch.randn(num_samples, 1, spatial_x, spatial_y, 1, dtype=torch.float32) # Initial density field (1 channel)
            data_f = torch.randn(num_samples, timesteps, spatial_x, spatial_y, 1, dtype=torch.float32) # Control sequences (1 channel)
            data_target_u = torch.randn(num_samples, 1, 1, 1, 1, dtype=torch.float32) # Percentage of smoke (1 channel)
        
        elif self.config.pde_type == "era5":
            # u: (B, T, H, W, C), a (previous 12 hours state): (B, 12, H, W, C)
            num_samples = 5000 if self.is_train else 100 # Example
            predict_timesteps = self.config.data_res_h # 20
            past_timesteps = 12
            spatial_x = 720 # Example for 0.25 deg lat-lon
            spatial_y = 1440 # Example
            channels = self.config.data_channels # 1 (temperature)

            data_u = torch.randn(num_samples, predict_timesteps, spatial_x, spatial_y, self.config.raw_input_channels_x, dtype=torch.float32)
            data_a = torch.randn(num_samples, past_timesteps, spatial_x, spatial_y, self.config.raw_input_channels_cond, dtype=torch.float32)
            data_f = None
            data_target_u = None

        else:
            raise ValueError(f"Unknown PDE type: {self.config.pde_type}")
        
        # Save to HDF5 if not already present (only for demonstration/reproducibility setup)
        # In a real scenario, this would load from existing files.
        # if not os.path.exists(self.data_path):
        #     with h5py.File(self.data_path, 'w') as f:
        #         f.create_dataset('u', data=data_u.numpy())
        #         f.create_dataset('a', data=data_a.numpy())
        #         if data_f is not None:
        #             f.create_dataset('f', data=data_f.numpy())
        #         if data_target_u is not None:
        #             f.create_dataset('target_u', data=data_target_u.numpy())
        #     print(f"Dummy data saved to {self.data_path}")
            
        # Actual loading logic (if data_path exists)
        # try:
        #     with h5py.File(self.data_path, 'r') as f:
        #         data_u = torch.from_numpy(f['u'][()])
        #         data_a = torch.from_numpy(f['a'][()])
        #         data_f = torch.from_numpy(f['f'][()]) if 'f' in f else None
        #         data_target_u = torch.from_numpy(f['target_u'][()]) if 'target_u' in f else None
        #     print(f"Data loaded from {self.data_path}")
        # except FileNotFoundError:
        #     print(f"Warning: Data file {self.data_path} not found. Using dummy data.")

        return data_u, data_a, data_f, data_target_u
    
    def _downsample(self, data: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Downsamples the spatial and temporal dimensions of the data.
        Assumes data is (B, T, S_x, [S_y], C)
        """
        if data is None:
            return None

        # Determine zoom factors for (T, S_x, S_y) or (T, S_x)
        if self.is_3d_spatial:
            # (B, D, H, W, C) -> downsample D, H, W
            # zoom_factor = (1 / factor, 1 / factor, 1 / factor, 1) # if C is last
            original_shape = data.shape
            target_shape = list(original_shape)
            target_shape[1] = max(1, original_shape[1] // factor) # Time/Depth
            target_shape[2] = max(1, original_shape[2] // factor) # Spatial_X
            target_shape[3] = max(1, original_shape[3] // factor) # Spatial_Y
            
            downsampled_data = torch.zeros(target_shape, dtype=data.dtype, device=data.device)
            for i in range(original_shape[0]): # Iterate batch
                for c in range(original_shape[-1]): # Iterate channels
                    downsampled_data[i, ..., c] = torch.from_numpy(zoom(data[i, ..., c].cpu().numpy(), 
                                                                         (target_shape[1]/original_shape[1], 
                                                                          target_shape[2]/original_shape[2], 
                                                                          target_shape[3]/original_shape[3]), 
                                                                         order=1)) # Linear interpolation
            return downsampled_data
        else:
            # (B, T, S_x, C) -> downsample T, S_x
            original_shape = data.shape
            target_shape = list(original_shape)
            target_shape[1] = max(1, original_shape[1] // factor) # Time
            target_shape[2] = max(1, original_shape[2] // factor) # Spatial_X

            downsampled_data = torch.zeros(target_shape, dtype=data.dtype, device=data.device)
            for i in range(original_shape[0]): # Iterate batch
                for c in range(original_shape[-1]): # Iterate channels
                    downsampled_data[i, ..., c] = torch.from_numpy(zoom(data[i, ..., c].cpu().numpy(), 
                                                                         (target_shape[1]/original_shape[1], 
                                                                          target_shape[2]/original_shape[2]), 
                                                                         order=1)) # Linear interpolation
            return downsampled_data


    def __len__(self):
        return len(self.data_u)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        u_data = self.data_u[idx]
        a_data = self.data_a[idx]
        f_data = self.data_f[idx] if self.data_f is not None else None
        target_u_data = self.data_target_u[idx] if self.data_target_u is not None else None
        
        # Ensure 'a' and 'target_u' have the same number of spatial dimensions as 'u'
        # if u_data.dim() == 3 and a_data.dim() == 2: # (T, S, C) vs (S, C)
        #     a_data = a_data.unsqueeze(0) # (1, S, C)
        # if u_data.dim() == 4 and a_data.dim() == 3: # (T, S_x, S_y, C) vs (S_x, S_y, C)
        #     a_data = a_data.unsqueeze(0) # (1, S_x, S_y, C)

        return u_data, a_data, f_data, target_u_data

def get_dataloader(config: Config, is_train: bool, resolution_factor: int = 1) -> DataLoader:
    dataset = PDEDataset(config, os.path.join(config.output_dir, f"pde_data_{config.pde_type}.h5"), is_train, resolution_factor)
    return DataLoader(dataset, batch_size=config.train_batch_size, shuffle=is_train, num_workers=4, pin_memory=True)

# --- Objective function for control tasks ---
# This would typically involve a separate PDE solver to simulate u from f.
# For now, this is a placeholder that might return a dummy loss.
# The actual solver would be complex and outside the scope of *just* reproducing WDNO.
# The paper mentions using "ground-truth solver" or "surrogate model" to compute u for I.

class PDE_Solver_Mock(nn.Module):
    """
    A mock PDE solver that would simulate u given initial conditions and control force f.
    In a real scenario, this would be a high-fidelity numerical solver or a learned surrogate model.
    For now, it returns dummy output for u(T,x).
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.is_3d_spatial = (config.pde_type in ["2d_fluid", "era5"])

    def forward(self, u0: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        u0: initial condition (B, ..., C)
        f: control force (B, ..., C)
        Returns u_T: final state (B, ..., C) or (B, 1, ..., C)
        """
        # This is a very simplified mock. A real solver would be complex.
        # For control, we are often interested in u(T,x).
        
        # Determine the output shape based on the input f
        if self.is_3d_spatial:
            # For 2D fluid, f is (B, D, H, W, C_f), u0 is (B, 1, H, W, C_u0)
            # We need to return u(T,x) which would be (B, 1, H, W, C_u)
            # Simplified: just return a tensor of the appropriate target shape
            # Example: target_u for 2D fluid is (B, 1, 1, 1, 1) (percentage of smoke)
            # So, if this mock solver is used for the *state* u, it would return
            # u(T,x) as (B, 1, H, W, C_u)
            
            # For 2D fluid, u is (B, D, H, W, C_u). u(T,x) is last time step.
            # Let's say this mock returns u_T, the last timestep of the state.
            
            # The control objective for 2D fluid is `percentage of smoke not passing through the target bucket`.
            # This is a scalar per sample.
            # So this mock function should return the relevant `u` output for objective calculation.
            # Let's return a dummy scalar value for `objective_I` in `guidance_func`.
            # The actual calculation for `I` involves simulating `u` and then calculating
            # the percentage. This mock simplifies the entire simulation.
            
            # If we were to mock the output of `u(T,x)` (state at final time),
            # it would be something like:
            # B, D, H, W, C_f = f.shape
            # output_H, output_W, output_C = H, W, self.config.data_channels # Assuming u has 3 channels for 2D fluid
            # return torch.randn(B, 1, output_H, output_W, output_C, device=u0.device)
            return torch.randn(u0.shape[0], 1, 1, 1, 1, device=u0.device) # Mock scalar percentage for 2D fluid
        else:
            # For 1D Burgers, f is (B, T-1, S, C), u0 is (B, 1, S, C)
            # We need to return u(T,x) which is (B, 1, S, C)
            B, T_f, S_f, C_f = f.shape
            output_S = self.config.data_res_w
            output_C = self.config.data_channels
            return torch.randn(B, 1, output_S, output_C, device=u0.device) # Mock u_T

def create_guidance_objective(config: Config, pde_solver: PDE_Solver_Mock, target_u_val: torch.Tensor, u0_val: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates the guidance objective function for control tasks.
    The guidance function takes wavelet coefficients of f_hat (W_f_hat)
    and returns a scalar objective I for each sample in the batch.
    """
    def guidance_objective_func(W_f_hat: torch.Tensor) -> torch.Tensor:
        # W_f_hat is wavelet coefficients of f. First, inverse DWT to get f.
        # This requires an IDWT instance from the WDNO model.
        # Since this function is passed to WDNO.sample, it needs an IDWT object.
        # For simplicity in this example, let's assume `f_hat` is the output of the IDWT.
        # In a real setup, WDNO model's _apply_idwt would be used here.
        
        # Mock inverse wavelet transform for f_hat
        # The shape of W_f_hat is (B, C_total_wavelet, H_out, W_out) or (B, C_total_wavelet, D_out, H_out, W_out)
        # We need to convert it back to (B, T, S, C) or (B, D, H, W, C)
        # For now, let's assume f_hat is already the original space f, as a placeholder.
        
        # If the objective function needs actual `u` from `f`, we need a solver.
        # The PDE_Solver_Mock is used here.
        
        # Placeholder for IDWT (needs WDNO's IDWT)
        # Assuming W_f_hat's shape implies the original data shape after IDWT.
        # For 1D Burgers, W_f_hat is (B, C*4, T_wt, S_wt), need to reconstruct to (B, T, S, C)
        # For 2D Fluid, W_f_hat is (B, C*8, T_wt, S_wt_x, S_wt_y), need to reconstruct to (B, T, S_x, S_y, C)
        
        # Mock reconstruction of f from W_f_hat
        if config.pde_type == "1d_burgers":
            # Assuming `_apply_idwt` from WDNO model is accessible or re-implemented here.
            # For now, a simple placeholder. Shape of f is (B, T-1, S, C)
            f_recon = torch.randn(W_f_hat.shape[0], config.data_res_h - 1, config.data_res_w, config.data_channels, device=W_f_hat.device, requires_grad=True)
            
            # Simulate u from u0 and f
            u_T_pred = pde_solver(u0_val, f_recon) # u_T_pred is (B, 1, S, C)

            # Calculate objective I
            # I = int_D |u(T, x) - u*(x)|^2 dx + alpha int_[0,T]xD |f(t, x)|^2 dt dx
            # target_u_val is u*(x) at final time T: (B, 1, S, C)
            
            # L2 norm for spatial difference at final time
            term1 = ((u_T_pred - target_u_val)**2).mean(dim=[-1, -2, -3]) # Mean over S, T, C
            
            # L2 norm for control force (mean over T, S, C)
            # The paper's Eq 7 means sum over the spatial domain, then over time.
            # So mean over (T, S, C)
            term2 = (f_recon**2).mean(dim=[-1, -2, -3])
            
            objective_I = term1 + config.guidance_weight * term2
            
        elif config.pde_type == "2d_fluid":
            # Mock reconstruction of f from W_f_hat
            # Shape of f is (B, D, H, W, C_f)
            f_recon = torch.randn(W_f_hat.shape[0], config.data_res_h, config.data_res_w, config.data_res_d, 1, device=W_f_hat.device, requires_grad=True)
            
            # Simulate u (or relevant output like smoke percentage) from u0 and f
            # pde_solver for 2D fluid mock returns (B, 1, 1, 1, 1) (percentage)
            smoke_percentage_not_passed = pde_solver(u0_val, f_recon) # This is I directly

            # The objective I is defined as "percentage of smoke not passing through the target bucket"
            # So, smoke_percentage_not_passed *is* the objective value.
            objective_I = smoke_percentage_not_passed.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1) # To (B,)
            
        else:
            raise ValueError("Guidance objective not implemented for this PDE type.")

        return objective_I # Returns a scalar or batch of scalars for objective I

    return guidance_objective_func

