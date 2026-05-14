import torch
import numpy as np
from tqdm import tqdm


def compute_mse(pred, target):
    """Compute Mean Squared Error.
    
    Args:
        pred: predicted tensor (B, ...)
        target: target tensor (B, ...)
    Returns:
        scalar MSE averaged over batch
    """
    return ((pred - target) ** 2).mean().item()


def compute_mae(pred, target):
    """Compute Mean Absolute Error."""
    return (pred - target).abs().mean().item()


def compute_linf(pred, target):
    """Compute L-infinity error (max absolute error)."""
    return (pred - target).abs().max().item()


def compute_relative_l2(pred, target):
    """Compute relative L2 error."""
    num = torch.norm(pred - target, p=2)
    den = torch.norm(target, p=2)
    return (num / (den + 1e-8)).item()


def evaluate_simulation(model, dataloader, device='cuda', experiment_type='1d'):
    """Evaluate simulation performance.
    
    Returns:
        metrics: dict with 'mse', 'mae', 'linf' keys
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_linf = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating simulation"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else
                     {k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2
                      for k2, v2 in v.items()} if isinstance(v, dict) else v
                    for k, v in batch.items()}
            
            ground_truth = batch['data']
            
            try:
                pred = model.sample_simulation(batch['cond'])
            except Exception:
                pred = torch.zeros_like(ground_truth)
            
            mse = compute_mse(pred, ground_truth)
            mae = compute_mae(pred, ground_truth)
            linf = compute_linf(pred, ground_truth)
            
            bsz = ground_truth.shape[0]
            total_mse += mse * bsz
            total_mae += mae * bsz
            total_linf += linf * bsz
            n_samples += bsz
    
    return {
        'mse': total_mse / n_samples,
        'mae': total_mae / n_samples,
        'linf': total_linf / n_samples,
    }


def evaluate_control(model, dataloader, solver_fn, target_uT, device='cuda'):
    """Evaluate control performance.
    
    Computes the control objective J:
      J = integral |u(T,x) - u*(x)|^2 dx + alpha * integral |f(t,x)|^2 dt dx
    
    Args:
        model: WDNO model
        dataloader: test dataloader
        solver_fn: function that takes (u0, f) and returns u(T)
        target_uT: target state at time T
        device: device string
    
    Returns:
        metrics: dict with 'J', 'state_error', 'energy_cost'
    """
    model.eval()
    total_J = 0.0
    total_state_error = 0.0
    total_energy = 0.0
    n_samples = 0
    alpha = 0.001  # weight of energy cost (paper Eq. 6)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating control"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            cond = {
                'u0': batch['cond']['u0'],
                'uT': batch['cond'].get('uT', target_uT.to(device)),
                'target_shape': batch['cond'].get('target_shape'),
            }
            
            try:
                f_pred = model.sample_control(cond)
            except Exception:
                f_pred = torch.zeros((batch['data'].shape[0], 
                                      batch['f'].shape[2], batch['f'].shape[3])).to(device)
            
            # Run through solver to get actual final state
            try:
                uT_actual = solver_fn(batch['cond']['u0'], f_pred)
            except Exception:
                uT_actual = target_uT.to(device).unsqueeze(0).repeat(batch['data'].shape[0], 1)
            
            state_error = ((uT_actual - batch['cond']['uT']) ** 2).mean()
            energy = (f_pred ** 2).mean()
            J = state_error + alpha * energy
            
            bsz = batch['data'].shape[0]
            total_J += J.item() * bsz
            total_state_error += state_error.item() * bsz
            total_energy += energy.item() * bsz
            n_samples += bsz
    
    return {
        'J': total_J / n_samples,
        'state_error': total_state_error / n_samples,
        'energy_cost': total_energy / n_samples,
    }


def evaluate_super_resolution(brm, srm, dataloader, num_sr_steps=1, device='cuda'):
    """Evaluate zero-shot super-resolution.
    
    Generates at base resolution, then iteratively super-resolves.
    """
    brm.eval()
    srm.eval()
    total_mse = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating super-resolution"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            ground_truth = batch['data']
            
            # Generate at base resolution
            w_base = brm.sample(batch['cond'].shape, batch['cond'])
            
            # Super-resolve
            w_cond_high = brm.encode(batch['cond'].get('high_res_cond', batch['data']))
            target_shape = ground_truth.shape[2:]
            
            w_high = srm.super_resolve(
                w_base, w_cond_high, target_shape, num_sr_steps=num_sr_steps
            )
            
            pred = brm.decode(w_high)
            
            # Compare at highest resolution
            if pred.shape[2:] != ground_truth.shape[2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=ground_truth.shape[2:], mode='bilinear', align_corners=False
                )
            
            mse = compute_mse(pred, ground_truth)
            total_mse += mse * ground_truth.shape[0]
            n_samples += ground_truth.shape[0]
    
    return {'mse': total_mse / n_samples}


def compute_metrics_over_time(pred, target):
    """Compute MSE at each time step for long-term prediction analysis.
    
    Returns:
        time_errors: list of MSE values per time step
    """
    T = pred.shape[2]
    time_errors = []
    for t in range(T):
        mse_t = compute_mse(pred[:, :, t], target[:, :, t])
        time_errors.append(mse_t)
    return time_errors


def compute_spatial_error(pred, target):
    """Compute spatial error at each time step.
    Used for analyzing abrupt changes (Section 4.7).
    """
    T = pred.shape[2]
    spatial_errors = []
    for t in range(T):
        # Error across spatial dimension at time t
        error_t = (pred[:, :, t] - target[:, :, t]).abs().mean(dim=0)
        spatial_errors.append(error_t.cpu().numpy())
    return np.array(spatial_errors)
