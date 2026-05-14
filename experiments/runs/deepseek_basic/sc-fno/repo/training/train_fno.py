"""
Training loops for all FNO variants.

Implements the four model configurations described in the paper:

Algorithm 1: FNO          - L = L_u
Algorithm 2: SC-FNO       - L = c1*L_u + c2*L_s
Algorithm 3: SC-FNO-PINN  - L = c1*L_u + c2*L_s + c3*L_eq

Plus:        FNO-PINN     - L = L_u + L_eq

All variants use identical FNO architecture; they differ only in loss configuration.
The sensitivity loss L_s uses randomly sampled spatial-temporal points each epoch.

Reference: Section 2.4 and Algorithms 1-3 in Appendix A.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from .data_utils import prepare_dataloaders


def train_fno(model, train_loader, val_loader, epochs=500, lr=0.001,
              device='cpu', verbose=True, patience=50):
    """
    Algorithm 1: FNO Training Loop with L_u loss.
    
    Args:
        model: FNO instance
        train_loader, val_loader: DataLoaders
        epochs: number of training epochs
        lr: learning rate
        device: torch device
        verbose: print progress
        patience: early stopping patience
    
    Returns:
        model, history dict
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    mse_loss = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'epoch_times': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_tensor, u_true, _, _ = batch
            input_tensor = input_tensor.to(device)
            u_true = u_true.to(device)
            
            optimizer.zero_grad()
            u_pred = model(input_tensor)
            loss = mse_loss(u_pred, u_true)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_tensor, u_true, _, _ = batch
                input_tensor = input_tensor.to(device)
                u_true = u_true.to(device)
                
                u_pred = model(input_tensor)
                loss = mse_loss(u_pred, u_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_times'].append(epoch_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"FNO Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, time={epoch_time:.2f}s")
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


def train_sc_fno(model, train_loader, val_loader, epochs=500, lr=0.001,
                 loss_weights=None, sensitivity_subsample=0.3,
                 device='cpu', verbose=True, patience=50):
    """
    Algorithm 2: SC-FNO Training Loop with L_u and L_s losses.
    
    Key differences from Algorithm 1:
    - Computes Jacobian ∂u/∂p using AD on the FNO
    - Randomly samples a subset of spatial-temporal points for sensitivity loss
    - Total loss: L = c1*L_u + c2*L_s
    
    Args:
        model: FNO instance (or SC_FNO wrapper)
        sensitivity_subsample: fraction of points to use for sensitivity loss
        loss_weights: dict with 'c1', 'c2' keys
    
    Returns:
        model, history dict
    """
    if loss_weights is None:
        loss_weights = {'c1': 1.0, 'c2': 1.0}
    
    c1 = loss_weights['c1']
    c2 = loss_weights['c2']
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    mse_loss = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'L_u': [], 'L_s': [], 'epoch_times': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        train_loss = 0.0
        L_u_sum = 0.0
        L_s_sum = 0.0
        
        for batch in train_loader:
            input_tensor, u_true, jac_true, params = batch
            input_tensor = input_tensor.to(device)
            u_true = u_true.to(device)
            jac_true = jac_true.to(device)
            params = params.to(device)
            
            input_tensor.requires_grad_(True)
            
            optimizer.zero_grad()
            u_pred = model(input_tensor)
            
            # L_u: data loss
            L_u = mse_loss(u_pred, u_true)
            
            # L_s: sensitivity loss (on randomly sampled points)
            batch_size = u_pred.shape[0]
            n_pts = u_pred.numel() // batch_size
            n_sample = max(1, int(n_pts * sensitivity_subsample))
            
            # Compute Jacobian at all points (needed for AD)
            jac_pred = compute_jacobian_batch(model, input_tensor, u_pred, params.shape[-1])
            
            # Randomly sample points for sensitivity loss
            if n_sample < n_pts:
                flat_jac_pred = jac_pred.reshape(batch_size, -1)
                flat_jac_true = jac_true.reshape(batch_size, -1)
                indices = torch.randperm(flat_jac_pred.shape[1], device=device)[:n_sample]
                L_s = mse_loss(flat_jac_pred[:, indices], flat_jac_true[:, indices])
            else:
                L_s = mse_loss(jac_pred, jac_true)
            
            total_loss = c1 * L_u + c2 * L_s
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            L_u_sum += L_u.item()
            L_s_sum += L_s.item()
        
        n_batches = len(train_loader)
        train_loss /= n_batches
        L_u_sum /= n_batches
        L_s_sum /= n_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_tensor, u_true, _, _ = batch
                input_tensor = input_tensor.to(device)
                u_true = u_true.to(device)
                u_pred = model(input_tensor)
                loss = mse_loss(u_pred, u_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['L_u'].append(L_u_sum)
        history['L_s'].append(L_s_sum)
        history['epoch_times'].append(epoch_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"SC-FNO Epoch {epoch}: total={train_loss:.6f}, L_u={L_u_sum:.6f}, L_s={L_s_sum:.6f}, val={val_loss:.6f}, time={epoch_time:.2f}s")
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


def compute_jacobian_batch(model, x, u_pred, n_params):
    """
    Compute Jacobian ∂u_pred/∂x for parameter dimensions.
    
    Args:
        model: FNO model
        x: input tensor (requires_grad=True)
        u_pred: model output
        n_params: number of parameter dimensions
    
    Returns:
        jacobian: tensor of same shape as u_pred but with additional param dim
    """
    batch_size = u_pred.shape[0]
    grad_list = []
    
    for i in range(u_pred.shape[-1]):  # For each output dimension
        grad_outputs = torch.ones_like(u_pred[..., i])
        grad = torch.autograd.grad(
            outputs=u_pred[..., i], inputs=x,
            grad_outputs=grad_outputs,
            retain_graph=True, create_graph=True
        )[0]
        grad_list.append(grad)
    
    # Stack along last dim: (batch, *grid, n_params, output_dim)
    jac = torch.stack(grad_list, dim=-1)
    return jac


def train_fno_pinn(model, train_loader, val_loader, pde_residual_fn,
                   epochs=500, lr=0.001, loss_weights=None,
                   device='cpu', verbose=True, patience=50):
    """
    FNO-PINN: Training with L_u + L_eq (PINN equation loss).
    
    Args:
        model: FNO instance
        pde_residual_fn: function(u_pred, t_grid, x_grid, params) -> residual
        loss_weights: dict with 'c1' (L_u), 'c3' (L_eq)
    """
    if loss_weights is None:
        loss_weights = {'c1': 1.0, 'c3': 0.1}
    
    c1 = loss_weights['c1']
    c3 = loss_weights['c3']
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    mse_loss = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'L_u': [], 'L_eq': [], 'epoch_times': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        train_loss = 0.0
        L_u_sum = 0.0
        L_eq_sum = 0.0
        
        for batch in train_loader:
            input_tensor, u_true, _, params = batch
            input_tensor = input_tensor.to(device)
            u_true = u_true.to(device)
            params = params.to(device)
            
            input_tensor.requires_grad_(True)
            optimizer.zero_grad()
            
            u_pred = model(input_tensor)
            
            L_u = mse_loss(u_pred, u_true)
            
            # Compute PDE residual
            residual = pde_residual_fn(u_pred, input_tensor, params)
            L_eq = torch.mean(residual ** 2)
            
            total_loss = c1 * L_u + c3 * L_eq
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            L_u_sum += L_u.item()
            L_eq_sum += L_eq.item()
        
        n_batches = len(train_loader)
        train_loss /= n_batches
        L_u_sum /= n_batches
        L_eq_sum /= n_batches
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_tensor, u_true, _, _ = batch
                input_tensor = input_tensor.to(device)
                u_true = u_true.to(device)
                u_pred = model(input_tensor)
                loss = mse_loss(u_pred, u_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['L_u'].append(L_u_sum)
        history['L_eq'].append(L_eq_sum)
        history['epoch_times'].append(epoch_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"FNO-PINN Epoch {epoch}: total={train_loss:.6f}, L_u={L_u_sum:.6f}, L_eq={L_eq_sum:.6f}, time={epoch_time:.2f}s")
        
        if patience_counter >= patience:
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


def train_sc_fno_pinn(model, train_loader, val_loader, pde_residual_fn,
                      epochs=500, lr=0.001, loss_weights=None,
                      sensitivity_subsample=0.3,
                      device='cpu', verbose=True, patience=50):
    """
    Algorithm 3: SC-FNO-PINN Training with L_u + L_s + L_eq losses.
    
    Combines sensitivity loss and PINN equation loss.
    Total loss: L = c1*L_u + c2*L_s + c3*L_eq
    """
    if loss_weights is None:
        loss_weights = {'c1': 1.0, 'c2': 1.0, 'c3': 0.1}
    
    c1 = loss_weights['c1']
    c2 = loss_weights['c2']
    c3 = loss_weights['c3']
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    mse_loss = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'L_u': [], 'L_s': [], 'L_eq': [], 'epoch_times': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        train_loss = 0.0
        L_u_sum, L_s_sum, L_eq_sum = 0.0, 0.0, 0.0
        
        for batch in train_loader:
            input_tensor, u_true, jac_true, params = batch
            input_tensor = input_tensor.to(device)
            u_true = u_true.to(device)
            jac_true = jac_true.to(device)
            params = params.to(device)
            
            input_tensor.requires_grad_(True)
            optimizer.zero_grad()
            
            u_pred = model(input_tensor)
            
            L_u = mse_loss(u_pred, u_true)
            
            # Sensitivity loss (sampled)
            jac_pred = compute_jacobian_batch(model, input_tensor, u_pred, params.shape[-1])
            
            batch_size = u_pred.shape[0]
            n_pts = jac_pred.reshape(batch_size, -1).shape[1]
            n_sample = max(1, int(n_pts * sensitivity_subsample))
            
            if n_sample < n_pts:
                flat_pred = jac_pred.reshape(batch_size, -1)
                flat_true = jac_true.reshape(batch_size, -1)
                indices = torch.randperm(flat_pred.shape[1], device=device)[:n_sample]
                L_s = mse_loss(flat_pred[:, indices], flat_true[:, indices])
            else:
                L_s = mse_loss(jac_pred, jac_true)
            
            # Equation loss
            residual = pde_residual_fn(u_pred, input_tensor, params)
            L_eq = torch.mean(residual ** 2)
            
            total_loss = c1 * L_u + c2 * L_s + c3 * L_eq
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            L_u_sum += L_u.item()
            L_s_sum += L_s.item()
            L_eq_sum += L_eq.item()
        
        n_batches = len(train_loader)
        train_loss /= n_batches
        L_u_sum /= n_batches
        L_s_sum /= n_batches
        L_eq_sum /= n_batches
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_tensor, u_true, _, _ = batch
                input_tensor = input_tensor.to(device)
                u_true = u_true.to(device)
                u_pred = model(input_tensor)
                loss = mse_loss(u_pred, u_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['L_u'].append(L_u_sum)
        history['L_s'].append(L_s_sum)
        history['L_eq'].append(L_eq_sum)
        history['epoch_times'].append(epoch_time)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"SC-FNO-PINN Epoch {epoch}: total={train_loss:.6f}, L_u={L_u_sum:.6f}, L_s={L_s_sum:.6f}, L_eq={L_eq_sum:.6f}, time={epoch_time:.2f}s")
        
        if patience_counter >= patience:
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history
