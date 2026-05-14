"""
Main SC-FNO experiment script.
Implements the core SC-FNO training with proper sensitivity loss computation.

This is the central implementation of the paper:
"Sensitivity-Constrained Fourier Neural Operators for Forward and Inverse Problems
in Parametric Differential Equations"

Key contributions:
1. Sensitivity loss L_s = (1/M) * sum ||d(u_hat)/dp - du/dp||^2
2. Training with both solution paths and their Jacobians
3. Efficient sampling of spatial-temporal points for Jacobian computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fno import FNO1d, FNO2d, FNO3d


def relative_l2_loss(pred, target):
    """Relative L2 loss: ||pred - target|| / ||target||"""
    return torch.norm(pred - target) / (torch.norm(target) + 1e-8)


def r2_score_torch(pred, target):
    """R^2 score."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    ss_res = torch.sum((pred_flat - target_flat) ** 2)
    ss_tot = torch.sum((target_flat - target_flat.mean()) ** 2)
    return (1 - ss_res / (ss_tot + 1e-8)).item()


class SCFNO(nn.Module):
    """
    Sensitivity-Constrained Fourier Neural Operator.
    
    Wraps an FNO model and provides methods for computing sensitivity loss.
    The key feature is that parameters p are passed as inputs alongside
    initial conditions and spatial/temporal coordinates.
    
    Architecture:
    - Parameters p are repeated/broadcast to match spatial-temporal dimensions
    - Concatenated with initial conditions and coordinates
    - Processed through standard FNO layers
    
    The sensitivity loss is computed by:
    1. Forward pass to get u_pred
    2. Automatic differentiation to get d(u_pred)/dp
    3. Compare with pre-computed true Jacobians du/dp
    """

    def __init__(self, fno_model, n_params, input_dim):
        """
        Args:
            fno_model: Base FNO model
            n_params: Number of physical parameters
            input_dim: Total input dimension (including params)
        """
        super().__init__()
        self.fno = fno_model
        self.n_params = n_params
        self.input_dim = input_dim

    def forward(self, x):
        """Forward pass through FNO."""
        return self.fno(x)

    def forward_with_params(self, x_base, params):
        """
        Forward pass where params are explicitly tracked for gradient computation.
        
        Args:
            x_base: Base input without params (batch, ..., input_dim - n_params)
            params: Parameters (batch, n_params)
        
        Returns:
            u_pred: Predicted solution
        """
        # Broadcast params to match spatial-temporal dimensions
        # x_base shape: (batch, T, d) for 1D or (batch, S, T, d) for 2D
        if x_base.dim() == 3:
            # 1D case: (batch, T, d)
            T = x_base.shape[1]
            params_expanded = params.unsqueeze(1).expand(-1, T, -1)
        elif x_base.dim() == 4:
            # 2D case: (batch, S, T, d)
            S, T = x_base.shape[1], x_base.shape[2]
            params_expanded = params.unsqueeze(1).unsqueeze(2).expand(-1, S, T, -1)
        elif x_base.dim() == 5:
            # 3D case: (batch, Sx, Sy, T, d)
            Sx, Sy, T = x_base.shape[1], x_base.shape[2], x_base.shape[3]
            params_expanded = params.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, Sx, Sy, T, -1)
        else:
            raise ValueError(f"Unsupported input dimension: {x_base.dim()}")

        # Concatenate base input with params
        x = torch.cat([x_base, params_expanded], dim=-1)
        return self.fno(x)


def compute_jacobian_per_param(model, x_base, params, sample_idx, device="cpu"):
    """
    Compute Jacobian d(u_pred[sample_idx])/d(params) using automatic differentiation.
    
    This is the correct implementation: for each parameter p_i, we compute
    the gradient of each sampled output point w.r.t. p_i.
    
    The key insight: since params are broadcast to all spatial-temporal points,
    d(u[j])/d(p_i) can be computed by taking the gradient of u[j] w.r.t. params[:, i].
    
    Args:
        model: SCFNO model
        x_base: Base input (batch, ..., d)
        params: Parameters (batch, n_params) - must require grad
        sample_idx: Indices of output points to compute Jacobian at
        device: Computation device
    
    Returns:
        jacobian: Tensor of shape (batch, len(sample_idx), n_params)
        u_pred: Full predicted output
    """
    batch_size = params.shape[0]
    n_params = params.shape[1]
    n_sample = len(sample_idx)

    # Forward pass
    u_pred = model.forward_with_params(x_base, params)
    
    # Flatten output
    u_flat = u_pred.reshape(batch_size, -1)  # (batch, S*T*...)
    u_sampled = u_flat[:, sample_idx]  # (batch, n_sample)

    # Compute Jacobian: d(u_sampled)/d(params)
    # For each parameter p_i, compute gradient of sum(u_sampled) w.r.t. params
    # This gives us d(sum(u_sampled))/d(params[:, i]) = sum_j d(u_sampled[j])/d(params[:, i])
    # 
    # To get the full Jacobian, we need to compute for each output point separately.
    # For efficiency, we use the fact that params are shared across the batch:
    # d(u_sampled[b, j])/d(params[b, i]) is what we want.
    
    jacobian = torch.zeros(batch_size, n_sample, n_params, device=device)
    
    # Compute Jacobian: for each sampled output point j,
    # compute gradient of u_sampled[:, j] w.r.t. all params at once
    for j in range(n_sample):
        grad = torch.autograd.grad(
            outputs=u_sampled[:, j].sum(),
            inputs=params,
            create_graph=model.training,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        if grad is not None:
            # grad shape: (batch, n_params)
            jacobian[:, j, :] = grad
    
    return jacobian, u_pred


def compute_jacobian_efficient(model, x_base, params, n_sample_points=None, device="cpu"):
    """
    Efficiently compute Jacobian d(u_pred)/d(params) using automatic differentiation.
    
    This implements the key efficiency trick from the paper:
    - Randomly sample a subset of spatial-temporal points
    - Compute Jacobian only at those points
    - This varies between epochs to cover the full solution space
    
    Args:
        model: SCFNO model
        x_base: Base input (batch, ..., d)
        params: Parameters (batch, n_params) - must require grad
        n_sample_points: Number of output points to sample (None = all)
        device: Computation device
    
    Returns:
        jacobian: Tensor of shape (batch, n_sample_points, n_params)
        sample_idx: Indices of sampled points (for matching with true Jacobians)
        u_pred: Full predicted output
    """
    batch_size = params.shape[0]
    n_params = params.shape[1]

    # Forward pass
    u_pred = model.forward_with_params(x_base, params)

    # Flatten spatial-temporal dimensions
    u_flat = u_pred.reshape(batch_size, -1)  # (batch, S*T)
    n_total = u_flat.shape[1]

    # Sample output points
    if n_sample_points is not None and n_sample_points < n_total:
        sample_idx = torch.randperm(n_total, device=device)[:n_sample_points]
    else:
        sample_idx = torch.arange(n_total, device=device)
        n_sample_points = n_total

    u_sampled = u_flat[:, sample_idx]  # (batch, n_sample)

    # Compute Jacobian: d(u_sampled)/d(params)
    # Shape: (batch, n_sample, n_params)
    # 
    # We use the efficient approach: for each parameter p_i,
    # compute d(u_sampled)/d(params[:, p_i]) using a single backward pass
    # with appropriate grad_outputs.
    
    jacobian = torch.zeros(batch_size, len(sample_idx), n_params, device=device)
    
    # Efficient Jacobian computation: for each sampled output point j,
    # compute gradient of u_sampled[:, j] w.r.t. params (all parameters at once)
    # This requires n_sample backward passes (not n_sample * n_params)
    for j in range(len(sample_idx)):
        # Gradient of u_sampled[:, j] (summed over batch) w.r.t. params
        # grad shape: (batch, n_params)
        grad = torch.autograd.grad(
            outputs=u_sampled[:, j].sum(),
            inputs=params,
            create_graph=model.training,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        if grad is not None:
            # grad[:, p_idx] = d(sum_b u_sampled[b, j])/d(params[b, p_idx])
            # This is the Jacobian column for output point j
            jacobian[:, j, :] = grad

    return jacobian, sample_idx, u_pred


def sensitivity_loss(pred_jacobian, true_jacobian):
    """
    Compute sensitivity loss L_s.
    
    L_s = (1/M) * sum ||d(u_hat)/dp - du/dp||^2
    
    Args:
        pred_jacobian: Predicted Jacobian (batch, M, n_params)
        true_jacobian: True Jacobian (batch, M, n_params)
    
    Returns:
        Scalar loss
    """
    return relative_l2_loss(pred_jacobian, true_jacobian)


def train_sc_fno(
    model,
    train_data,
    val_data,
    config,
    device="cpu",
    verbose=True
):
    """
    Train SC-FNO model.
    
    Args:
        model: SCFNO model
        train_data: Dict with keys: x_base, params, targets, jacobians
        val_data: Dict with keys: x_base, params, targets, jacobians
        config: Training configuration dict
        device: Computation device
        verbose: Print training progress
    
    Returns:
        Dict with training history and final metrics
    """
    mode = config.get("mode", "sc_fno")
    n_epochs = config.get("n_epochs", 500)
    batch_size = config.get("batch_size", 16)
    lr = config.get("lr", 1e-3)
    c1 = config.get("c1", 1.0)  # L_u weight
    c2 = config.get("c2", 1.0)  # L_s weight
    c3 = config.get("c3", 1.0)  # L_eq weight
    n_sample_points = config.get("n_sample_points", None)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Create datasets
    x_base_train = torch.FloatTensor(train_data["x_base"])
    params_train = torch.FloatTensor(train_data["params"])
    targets_train = torch.FloatTensor(train_data["targets"])
    jacobians_train = torch.FloatTensor(train_data["jacobians"]) if "jacobians" in train_data else None

    x_base_val = torch.FloatTensor(val_data["x_base"])
    params_val = torch.FloatTensor(val_data["params"])
    targets_val = torch.FloatTensor(val_data["targets"])

    n_train = x_base_train.shape[0]

    train_losses = []
    val_losses = []
    epoch_times = []

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        t_start = time.time()

        # Shuffle training data
        perm = torch.randperm(n_train)
        x_base_train = x_base_train[perm]
        params_train = params_train[perm]
        targets_train = targets_train[perm]
        if jacobians_train is not None:
            jacobians_train = jacobians_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            optimizer.zero_grad()

            x_b = x_base_train[i:i+batch_size].to(device)
            t = targets_train[i:i+batch_size].to(device)

            if mode in ["sc_fno", "sc_fno_pinn"] and jacobians_train is not None:
                j = jacobians_train[i:i+batch_size].to(device)
                # params need to require grad for Jacobian computation
                p = params_train[i:i+batch_size].to(device).requires_grad_(True)
            else:
                p = params_train[i:i+batch_size].to(device)
                j = None

            # Forward pass and primary loss L_u
            if mode in ["sc_fno", "sc_fno_pinn"] and j is not None:
                # Compute Jacobian and u_pred together
                pred_jac, sample_idx, u_pred = compute_jacobian_efficient(
                    model, x_b, p, n_sample_points, device
                )
                
                # Primary loss
                loss_u = relative_l2_loss(u_pred, t)
                total_loss = c1 * loss_u
                
                # Sensitivity loss L_s
                # Get corresponding true Jacobians at sampled points
                j_flat = j.reshape(j.shape[0], -1, j.shape[-1])  # (batch, S*T, n_params)
                true_jac_sampled = j_flat[:, sample_idx, :]
                
                loss_s = sensitivity_loss(pred_jac, true_jac_sampled)
                total_loss = total_loss + c2 * loss_s
            else:
                # Standard forward pass
                u_pred = model.forward_with_params(x_b, p)
                loss_u = relative_l2_loss(u_pred, t)
                total_loss = c1 * loss_u

            # Equation loss L_eq (PINN)
            if mode in ["fno_pinn", "sc_fno_pinn"]:
                equation_fn = config.get("equation_fn", None)
                if equation_fn is not None:
                    loss_eq = equation_fn(u_pred, p, x_b)
                    total_loss = total_loss + c3 * loss_eq

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step()
        epoch_time = time.time() - t_start
        avg_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            x_b_val = x_base_val.to(device)
            p_val = params_val.to(device)
            t_val = targets_val.to(device)

            u_pred_val = model.forward_with_params(x_b_val, p_val)
            val_loss = relative_l2_loss(u_pred_val, t_val).item()
            val_r2 = r2_score_torch(u_pred_val, t_val)

        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train: {avg_loss:.6f} | "
                  f"Val L2: {val_loss:.6f} | "
                  f"Val R2: {val_r2:.4f} | "
                  f"Time: {epoch_time:.2f}s")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epoch_times": epoch_times,
        "best_val_loss": best_val_loss,
        "avg_epoch_time": float(np.mean(epoch_times))
    }


def evaluate_model(model, test_data, device="cpu", n_jac_samples=100, eval_batch_size=32):
    """
    Evaluate model on test data, computing both solution and Jacobian metrics.
    
    Args:
        model: Trained SCFNO model
        test_data: Dict with x_base, params, targets, (optional) jacobians
        device: Computation device
        n_jac_samples: Number of output points to sample for Jacobian evaluation
        eval_batch_size: Batch size for evaluation (to avoid OOM)
    
    Returns dict with R2 and relative L2 for both u and du/dp.
    """
    model.eval()

    x_base_all = torch.FloatTensor(test_data["x_base"])
    params_all = torch.FloatTensor(test_data["params"])
    targets_all = torch.FloatTensor(test_data["targets"])
    n_test = x_base_all.shape[0]

    all_u_pred = []
    all_pred_jac = []
    
    # Process in batches to avoid OOM
    for i in range(0, n_test, eval_batch_size):
        x_b = x_base_all[i:i+eval_batch_size].to(device)
        p = params_all[i:i+eval_batch_size].to(device).requires_grad_(True)
        
        with torch.enable_grad():
            u_pred = model.forward_with_params(x_b, p)
            all_u_pred.append(u_pred.detach().cpu())
            
            if "jacobians" in test_data:
                n_params = p.shape[1]
                batch_size = p.shape[0]
                u_flat = u_pred.reshape(batch_size, -1)
                n_outputs = u_flat.shape[1]
                
                # Sample output points for Jacobian evaluation
                n_sample = min(n_jac_samples, n_outputs)
                sample_idx = torch.randperm(n_outputs, device=device)[:n_sample]
                u_sampled = u_flat[:, sample_idx]
                
                pred_jac = torch.zeros(batch_size, n_sample, n_params, device=device)
                for j in range(n_sample):
                    grad = torch.autograd.grad(
                        u_sampled[:, j].sum(),
                        p,
                        create_graph=False,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    if grad is not None:
                        pred_jac[:, j, :] = grad
                
                all_pred_jac.append((pred_jac.detach().cpu(), sample_idx.cpu()))

    u_pred_all = torch.cat(all_u_pred, dim=0)
    targets = targets_all

    # Solution metrics
    u_r2 = r2_score_torch(u_pred_all, targets)
    u_l2 = relative_l2_loss(u_pred_all, targets).item()

    metrics = {
        "u_r2": u_r2,
        "u_relative_l2": u_l2,
    }

    # Jacobian metrics (if true Jacobians available)
    if "jacobians" in test_data and all_pred_jac:
        true_jac_all = torch.FloatTensor(test_data["jacobians"])
        n_params = params_all.shape[1]
        
        # Collect all predicted Jacobians and corresponding true Jacobians
        all_pred_j_list = []
        all_true_j_list = []
        
        offset = 0
        for pred_jac, sample_idx in all_pred_jac:
            batch_size = pred_jac.shape[0]
            true_jac_batch = true_jac_all[offset:offset+batch_size]
            true_jac_flat = true_jac_batch.reshape(batch_size, -1, n_params)
            true_jac_sampled = true_jac_flat[:, sample_idx, :]
            
            all_pred_j_list.append(pred_jac)
            all_true_j_list.append(true_jac_sampled)
            offset += batch_size
        
        pred_jac_cat = torch.cat(all_pred_j_list, dim=0)  # (N, n_sample, n_params)
        true_jac_cat = torch.cat(all_true_j_list, dim=0)  # (N, n_sample, n_params)
        
        param_names = test_data.get("param_names", [])
        for p_idx in range(n_params):
            pred_j = pred_jac_cat[:, :, p_idx]
            true_j = true_jac_cat[:, :, p_idx]
            j_r2 = r2_score_torch(pred_j, true_j)
            j_l2 = relative_l2_loss(pred_j, true_j).item()
            param_name = param_names[p_idx] if p_idx < len(param_names) else f"p{p_idx}"
            metrics[f"jac_{param_name}_r2"] = j_r2
            metrics[f"jac_{param_name}_relative_l2"] = j_l2

    return metrics


def parameter_inversion(
    model,
    x_base_observed,
    u_observed,
    param_ranges,
    n_iter=1000,
    lr=0.01,
    device="cpu",
    n_restarts=3
):
    """
    Perform parameter inversion using the trained surrogate model.
    
    Uses backpropagation through the surrogate model to optimize parameters
    to match observed solution paths.
    
    Args:
        model: Trained SCFNO model
        x_base_observed: Base input (batch, ..., d) - initial conditions + coords
        u_observed: Observed solution (batch, ..., 1)
        param_ranges: Dict of {param_name: [min, max]}
        n_iter: Number of optimization iterations
        lr: Learning rate for parameter optimization
        device: Computation device
        n_restarts: Number of random restarts
    
    Returns:
        Optimized parameters (batch, n_params)
    """
    model.eval()
    
    x_base = torch.FloatTensor(x_base_observed).to(device)
    u_obs = torch.FloatTensor(u_observed).to(device)
    
    batch_size = x_base.shape[0]
    n_params = len(param_ranges)
    
    best_params = None
    best_loss = float("inf")
    
    for restart in range(n_restarts):
        # Initialize parameters randomly within ranges
        params_init = np.zeros((batch_size, n_params))
        for j, (name, (lo, hi)) in enumerate(param_ranges.items()):
            params_init[:, j] = np.random.uniform(lo, hi, batch_size)
        
        params = torch.FloatTensor(params_init).to(device).requires_grad_(True)
        optimizer = optim.Adam([params], lr=lr)
        
        for i in range(n_iter):
            optimizer.zero_grad()
            
            u_pred = model.forward_with_params(x_base, params)
            loss = relative_l2_loss(u_pred, u_obs)
            
            loss.backward()
            optimizer.step()
            
            # Clamp parameters to valid ranges
            with torch.no_grad():
                for j, (name, (lo, hi)) in enumerate(param_ranges.items()):
                    params[:, j].clamp_(lo, hi)
        
        final_loss = loss.item()
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params.detach().cpu().numpy()
    
    return best_params
