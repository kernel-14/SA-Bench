"""
Data preparation utilities for FNO training.

Prepares input tensors by combining spatial/temporal coordinates,
initial conditions, and parameters into the format expected by FNO.

Section 2.4: "The SC-FNO architecture processes parameters τ(p) alongside 
spatial coordinates and initial conditions through the lifting layer as 
function inputs. This layer reshapes and repeats parameters to match the 
problem's spatial-temporal dimensions, then concatenates them with other 
inputs before neural network processing."
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def prepare_input_tensor_ode(u_initial, t_grid, params, M=10):
    """
    Prepare input tensor for ODE FNO.
    
    Input format:
    - First M time steps of u (known history)
    - Parameters p (repeated across time)
    - Time coordinate t
    
    The operator maps: u[0:M] ∪ p -> u[M:N]
    
    Args:
        u_initial: (batch, n_steps, 1) - full solution
        t_grid: (n_steps,) - time grid
        params: (batch, n_params) - parameters
        M: number of initial time steps given as input
    
    Returns:
        input: (batch, n_steps - M, input_dim) - input to FNO
        target: (batch, n_steps - M, 1) - target u[M:N]
    """
    batch_size = u_initial.shape[0]
    n_steps = u_initial.shape[1]
    n_params = params.shape[1]
    
    # Target: u at times M to N-1
    target = u_initial[:, M:, :]  # (batch, n_steps-M, 1)
    n_target_steps = n_steps - M
    
    # Input features:
    # - Time coordinate for each output step
    # - Parameters (repeated)
    # - For ODEs, we might also include the last known value or summary of known history
    t_future = t_grid[M:]  # (n_target_steps,)
    
    # Build input tensor
    input_tensor = torch.zeros(batch_size, n_target_steps, n_params + 1, device=u_initial.device)
    input_tensor[:, :, 0] = t_future.unsqueeze(0)  # time coordinate
    input_tensor[:, :, 1:] = params.unsqueeze(1).repeat(1, n_target_steps, 1)  # params
    
    return input_tensor, target


def prepare_input_tensor_pde(u_initial, t_grid, x_grid, params, M=5):
    """
    Prepare input tensor for PDE FNO.
    
    Input format:
    - First M time steps of u (known history)
    - Parameters p (broadcast across spatial and temporal dimensions)
    - Spatial coordinates x
    - Time coordinates t
    
    The operator maps: u[:, 0:M] ∪ p -> u[:, M:N]
    
    Args:
        u_initial: (batch, n_t, n_x, 1) - full solution
        t_grid: (n_t,) - time grid
        x_grid: (n_x,) - spatial grid
        params: (batch, n_params) - parameters
        M: number of initial time steps given as input
    
    Returns:
        input: (batch, n_t-M, n_x, input_dim)
        target: (batch, n_t-M, n_x, 1)
    """
    batch_size, n_t, n_x, _ = u_initial.shape
    n_params = params.shape[1]
    
    target = u_initial[:, M:, :, :]  # (batch, n_t-M, n_x, 1)
    n_target_steps = n_t - M
    
    t_future = t_grid[M:]  # (n_target_steps,)
    
    # Input features per (t, x) point: t, x, p_0, ..., p_{k-1}
    n_features = 2 + n_params  # t, x, and all params
    input_tensor = torch.zeros(batch_size, n_target_steps, n_x, n_features, device=u_initial.device)
    
    # Time coordinate
    input_tensor[:, :, :, 0] = t_future.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, n_x)
    
    # Spatial coordinate
    input_tensor[:, :, :, 1] = x_grid.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_target_steps, 1)
    
    # Parameters (broadcast)
    input_tensor[:, :, :, 2:] = params.unsqueeze(1).unsqueeze(1).repeat(1, n_target_steps, n_x, 1)
    
    return input_tensor, target


def prepare_input_tensor_pde3(omega0, x_grid, y_grid, params):
    """
    Prepare input tensor for PDE3 (Navier-Stokes).
    Maps initial vorticity + params -> final vorticity at t=3.
    
    Args:
        omega0: (batch, n_x, n_y, 1) - initial vorticity
        x_grid, y_grid: spatial grids
        params: (batch, n_params)
    
    Returns:
        input: (batch, n_x, n_y, input_dim)
        target: (batch, n_x, n_y, 1) - final vorticity
    """
    batch_size, n_x, n_y, _ = omega0.shape
    n_params = params.shape[1]
    
    # Input: initial vorticity, x, y, params
    n_features = 1 + 2 + n_params
    input_tensor = torch.zeros(batch_size, n_x, n_y, n_features, device=omega0.device)
    
    # Initial vorticity
    input_tensor[:, :, :, 0] = omega0[:, :, :, 0]
    
    # Coordinates
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    input_tensor[:, :, :, 1] = X.unsqueeze(0)
    input_tensor[:, :, :, 2] = Y.unsqueeze(0)
    
    # Parameters
    input_tensor[:, :, :, 3:] = params.unsqueeze(1).unsqueeze(1).repeat(1, n_x, n_y, 1)
    
    return input_tensor


def prepare_dataloaders(data, train_ratio=0.7, val_ratio=0.15, batch_size=4, 
                        case='PDE1', M=5):
    """
    Prepare train/val/test dataloaders.
    
    The validation and test sets contain parameter values not encountered during training
    (ensured by random split, as parameters are randomly generated).
    
    Args:
        data: dict from generate_dataset()
        train_ratio, val_ratio: data split ratios
        batch_size: batch size for training
        case: which case (determines input preparation)
        M: number of initial timesteps to use as input
    
    Returns:
        train_loader, val_loader, test_loader
    """
    u_true = data['u_true']
    jac_true = data['jac_true']
    params = data['params']
    t_grid = data['t_grid']
    x_grid = data.get('x_grid', None)
    
    n_samples = u_true.shape[0]
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Shuffle indices
    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    loaders = {}
    for split, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        u_split = u_true[idx]
        jac_split = jac_true[idx]
        params_split = params[idx]
        
        if case in ['ODE1', 'ODE2']:
            input_tensor, target = prepare_input_tensor_ode(u_split, t_grid, params_split, M=M)
            n_target_t = input_tensor.shape[1]
            jac_target = jac_split[:, M:, :, :].reshape(len(idx), n_target_t, -1)
        elif case == 'PDE3':
            input_tensor = prepare_input_tensor_pde3(u_split, data['x_grid'], data['y_grid'], params_split)
            target = u_split  # Same shape
            jac_target = jac_split.reshape(len(idx), -1)
        else:
            input_tensor, target = prepare_input_tensor_pde(u_split, t_grid, x_grid, params_split, M=M)
            n_target_t = input_tensor.shape[1]
            n_x = input_tensor.shape[2]
            n_params = jac_split.shape[-1]
            jac_target = jac_split[:, M:, :, :, :]
        
        dataset = TensorDataset(input_tensor, target, jac_target, params_split)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        loaders[split] = loader
    
    return loaders['train'], loaders['val'], loaders['test']
