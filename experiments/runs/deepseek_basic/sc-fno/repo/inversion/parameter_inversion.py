"""
Parameter inversion using trained FNO/SC-FNO surrogate models.

Implements both single-parameter and multi-parameter inversion 
via gradient-based optimization (backpropagation through the surrogate).

Section 3.1: "We then used backpropagation to optimize the parameter 
by minimizing the discrepancy between the synthetic data and PDE solutions."

The inversion experiments evaluate:
- Single-parameter inversion (α in PDE1)
- Multi-parameter inversion (all parameters simultaneously)
- Comparison across FNO, SC-FNO, FNO-PINN, and other operators
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


def single_parameter_inversion(model, target_u, param_init, param_idx,
                               input_template, true_param=None,
                               n_iterations=500, lr=0.01,
                               device='cpu', verbose=True):
    """
    Invert a single parameter from observed solution.
    
    Uses gradient descent to find p that minimizes ||FNO(input(p)) - target_u||².
    
    Args:
        model: Trained FNO/SC-FNO surrogate
        target_u: Observed solution (ground truth from synthetic data)
        param_init: Initial guess for the parameter (tensor, requires_grad)
        param_idx: Index of the parameter in the input tensor to optimize
        input_template: Template input tensor (will be modified with optimized param)
        true_param: Ground truth parameter value (for monitoring)
        n_iterations: Number of optimization iterations
        lr: Learning rate for optimization
        device: torch device
        verbose: Print progress
    
    Returns:
        optimized_param: Final parameter estimate
        history: dict with 'param_history', 'loss_history'
    """
    model.eval()
    param = param_init.clone().detach().to(device).requires_grad_(True)
    optimizer = optim.Adam([param], lr=lr)
    mse_loss = nn.MSELoss()
    
    history = {'param_history': [], 'loss_history': []}
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # Build input with current parameter estimate
        x = input_template.clone().to(device)
        # Fill in the parameter value at the correct position
        x[..., param_idx + 2] = param  # +2 offset for t, x coordinates
        
        u_pred = model(x)
        loss = mse_loss(u_pred, target_u.to(device))
        loss.backward()
        optimizer.step()
        
        history['param_history'].append(param.item())
        history['loss_history'].append(loss.item())
        
        if verbose and iteration % 100 == 0:
            true_str = f", true={true_param:.4f}" if true_param is not None else ""
            print(f"  Iter {iteration}: param={param.item():.4f}, loss={loss.item():.6f}{true_str}")
    
    return param.detach(), history


def multi_parameter_inversion(model, target_u, param_init, param_indices,
                             input_template, true_params=None,
                             n_iterations=1000, lr=0.01,
                             device='cpu', verbose=True):
    """
    Simultaneously invert multiple parameters.
    
    Args:
        model: Trained surrogate
        target_u: Observed solution
        param_init: Initial guess for all parameters (tensor of shape (n_params,))
        param_indices: List of parameter indices in input tensor
        input_template: Template input tensor
        true_params: Ground truth parameter values
        n_iterations: Number of optimization iterations
        lr: Learning rate
        device: torch device
        verbose: Print progress
    
    Returns:
        optimized_params: Final parameter estimates
        history: dict with loss history and parameter histories
    """
    model.eval()
    n_params = len(param_indices)
    params = param_init.clone().detach().to(device).requires_grad_(True)
    optimizer = optim.Adam([params], lr=lr)
    mse_loss = nn.MSELoss()
    
    history = {'loss_history': [], 'param_history': [[] for _ in range(n_params)]}
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # Build input: copy template and insert all parameters
        x = input_template.clone().to(device)
        for j, idx in enumerate(param_indices):
            x[..., idx + 2] = params[j]  # +2 offset for t, x
        
        u_pred = model(x)
        loss = mse_loss(u_pred, target_u.to(device))
        loss.backward()
        optimizer.step()
        
        history['loss_history'].append(loss.item())
        for j in range(n_params):
            history['param_history'][j].append(params[j].item())
        
        if verbose and iteration % 200 == 0:
            param_str = ', '.join([f"p{j}={params[j].item():.4f}" for j in range(n_params)])
            print(f"  Iter {iteration}: {param_str}, loss={loss.item():.6f}")
    
    return params.detach(), history


def inversion_experiment(model, test_data, case, model_name='FNO',
                         device='cpu'):
    """
    Run full inversion experiment as described in Section 3.1.
    
    For PDE1:
    - Single-parameter: invert α only (treat others as known)
    - Multi-parameter: invert all 5 parameters simultaneously
    
    For PDE2:
    - Multi-parameter: invert all 4 parameters simultaneously
    
    Args:
        model: Trained surrogate model
        test_data: dict with test split data
        case: 'PDE1' or 'PDE2'
        model_name: Name for reporting
        device: torch device
    
    Returns:
        results: dict with R² and relative L² for each parameter
    """
    from utils.metrics import compute_r2, relative_l2_error
    
    model.eval()
    results = {}
    
    if case == 'PDE1':
        param_names = ['c', 'alpha', 'beta', 'gamma', 'omega']
        # Single-parameter inversion for alpha (index 1)
        print(f"\n{'='*60}")
        print(f"{model_name} - Single-Parameter Inversion (alpha) for {case}")
        print(f"{'='*60}")
        
        alpha_true_list = []
        alpha_pred_list = []
        
        for i in tqdm(range(min(50, len(test_data)))):
            # Get one test sample
            sample = test_data[i]
            input_tensor, u_true, _, params = sample
            
            # Extract true alpha
            alpha_true = params[1].item()
            alpha_true_list.append(alpha_true)
            
            # Initial guess (random)
            alpha_init = torch.tensor([0.05], requires_grad=True)
            
            # Prepare input template with known values for other parameters
            input_template = input_tensor.unsqueeze(0).clone()
            # Set all param slots to known values except alpha
            for j in range(len(param_names)):
                if j != 1:  # not alpha
                    input_template[..., j + 2] = params[j]
            
            # Run inversion
            opt_alpha, _ = single_parameter_inversion(
                model, u_true.unsqueeze(0), alpha_init, 1,
                input_template, true_param=alpha_true,
                n_iterations=300, lr=0.01, device=device, verbose=False
            )
            
            alpha_pred_list.append(opt_alpha.item())
        
        # Compute metrics
        alpha_true_t = torch.tensor(alpha_true_list)
        alpha_pred_t = torch.tensor(alpha_pred_list)
        results['alpha_R2'] = compute_r2(alpha_true_t, alpha_pred_t)
        results['alpha_relL2'] = relative_l2_error(alpha_true_t, alpha_pred_t)
        
        print(f"\nSingle-parameter inversion results for alpha:")
        print(f"  R² = {results['alpha_R2']:.4f}")
        print(f"  Relative L² = {results['alpha_relL2']:.4f}")
        
        # Multi-parameter inversion
        print(f"\n{'='*60}")
        print(f"{model_name} - Multi-Parameter Inversion for {case}")
        print(f"{'='*60}")
        
        all_true = [[] for _ in param_names]
        all_pred = [[] for _ in param_names]
        
        for i in tqdm(range(min(50, len(test_data)))):
            sample = test_data[i]
            input_tensor, u_true, _, params = sample
            
            for j in range(len(param_names)):
                all_true[j].append(params[j].item())
            
            # Initial guess (middle of range)
            init_vals = torch.tensor([0.125, 0.05, 0.125, 0.125, 0.125], requires_grad=True)
            
            input_template = input_tensor.unsqueeze(0).clone()
            
            opt_params, _ = multi_parameter_inversion(
                model, u_true.unsqueeze(0), init_vals,
                list(range(len(param_names))), input_template,
                true_params=params,
                n_iterations=500, lr=0.01, device=device, verbose=False
            )
            
            for j in range(len(param_names)):
                all_pred[j].append(opt_params[j].item())
        
        for j, name in enumerate(param_names):
            true_t = torch.tensor(all_true[j])
            pred_t = torch.tensor(all_pred[j])
            results[f'{name}_R2'] = compute_r2(true_t, pred_t)
            results[f'{name}_relL2'] = relative_l2_error(true_t, pred_t)
            print(f"  {name}: R² = {results[f'{name}_R2']:.4f}, Rel L² = {results[f'{name}_relL2']:.4f}")
    
    elif case == 'PDE2':
        param_names = ['alpha', 'gamma', 'delta', 'omega']
        
        print(f"\n{'='*60}")
        print(f"{model_name} - Multi-Parameter Inversion for {case}")
        print(f"{'='*60}")
        
        all_true = [[] for _ in param_names]
        all_pred = [[] for _ in param_names]
        
        for i in tqdm(range(min(50, len(test_data)))):
            sample = test_data[i]
            input_tensor, u_true, _, params = sample
            
            for j in range(len(param_names)):
                all_true[j].append(params[j].item())
            
            init_vals = torch.tensor([0.55, 0.1375, 0.3, 0.055], requires_grad=True)
            input_template = input_tensor.unsqueeze(0).clone()
            
            opt_params, _ = multi_parameter_inversion(
                model, u_true.unsqueeze(0), init_vals,
                list(range(len(param_names))), input_template,
                true_params=params,
                n_iterations=500, lr=0.01, device=device, verbose=False
            )
            
            for j in range(len(param_names)):
                all_pred[j].append(opt_params[j].item())
        
        for j, name in enumerate(param_names):
            true_t = torch.tensor(all_true[j])
            pred_t = torch.tensor(all_pred[j])
            results[f'{name}_R2'] = compute_r2(true_t, pred_t)
            results[f'{name}_relL2'] = relative_l2_error(true_t, pred_t)
            print(f"  {name}: R² = {results[f'{name}_R2']:.4f}, Rel L² = {results[f'{name}_relL2']:.4f}")
    
    return results
