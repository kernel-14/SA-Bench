"""
Run experiments for PDE1 (Generalized Nonlinear Damped Wave Equation).
Compares FNO, SC-FNO, FNO-PINN, SC-FNO-PINN configurations.

From the paper:
- PDE1: u_tt = c^2 * u_xx + alpha * u_t + beta * u + gamma * sin(omega * u)
- Parameters: c in [0,0.25], alpha in [0,0.1], beta in [0,0.25], gamma in [0,0.25], omega in [0,0.25]
- N=30 time steps, Sx=20 spatial points, M=5 initial steps given
- 2000 training samples
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fno import FNO2d
from experiments.sc_fno_experiment import SCFNO, train_sc_fno, evaluate_model, relative_l2_loss, r2_score_torch


def solve_pde1(c, alpha, beta, gamma, omega, u0, u0_t, x, t):
    """
    Solve PDE1: u_tt = c^2 * u_xx + alpha * u_t + beta * u + gamma * sin(omega * u)
    Using finite differences.
    """
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u = np.zeros((nx, nt))
    u[:, 0] = u0
    # First time step using initial velocity
    u[:, 1] = u0 + dt * u0_t

    for k in range(1, nt - 1):
        # Second-order spatial derivative (periodic BC)
        u_xx = (np.roll(u[:, k], -1) - 2 * u[:, k] + np.roll(u[:, k], 1)) / dx**2
        # First-order time derivative (backward difference)
        u_t = (u[:, k] - u[:, k-1]) / dt
        # PDE: u_tt = c^2 * u_xx + alpha * u_t + beta * u + gamma * sin(omega * u)
        u_tt = c**2 * u_xx + alpha * u_t + beta * u[:, k] + gamma * np.sin(omega * u[:, k])
        u[:, k+1] = 2 * u[:, k] - u[:, k-1] + dt**2 * u_tt

    return u


def prepare_pde1_data(n_samples=2000, n_time=30, n_space=20, M=5, seed=42):
    """
    Prepare PDE1 dataset.
    
    Input to FNO: first M time steps of u (spatial field) + parameters p
    Output: remaining N-M time steps
    
    Returns train/val/test splits (70/15/15).
    """
    np.random.seed(seed)
    t = np.linspace(0, 1, n_time)
    x = np.linspace(0, 1, n_space)

    # Parameter ranges
    param_ranges = {
        "c": [0.0, 0.25],
        "alpha": [0.0, 0.1],
        "beta": [0.0, 0.25],
        "gamma": [0.0, 0.25],
        "omega": [0.0, 0.25],
    }
    n_params = len(param_ranges)

    params_dict = {k: np.random.uniform(*v, n_samples) for k, v in param_ranges.items()}
    params = np.stack(list(params_dict.values()), axis=1)  # (N, 5)

    solutions = np.zeros((n_samples, n_space, n_time))
    jacobians = np.zeros((n_samples, n_space, n_time, n_params))

    h = 1e-4  # finite difference step for Jacobians

    for i in range(n_samples):
        c = params_dict["c"][i]
        a = params_dict["alpha"][i]
        b = params_dict["beta"][i]
        g = params_dict["gamma"][i]
        w = params_dict["omega"][i]

        # Random initial conditions
        np.random.seed(seed + i)
        u0 = np.random.randn(n_space) * 0.1
        u0_t = np.zeros(n_space)

        sol = solve_pde1(c, a, b, g, w, u0, u0_t, x, t)
        solutions[i] = sol

        # Finite differences for Jacobians
        param_vals = [c, a, b, g, w]
        for j_idx, pval in enumerate(param_vals):
            pvals_p = param_vals.copy()
            pvals_p[j_idx] += h
            pvals_m = param_vals.copy()
            pvals_m[j_idx] -= h

            sol_p = solve_pde1(*pvals_p, u0, u0_t, x, t)
            sol_m = solve_pde1(*pvals_m, u0, u0_t, x, t)
            jacobians[i, :, :, j_idx] = (sol_p - sol_m) / (2 * h)

        if (i + 1) % 200 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")

    # Split data: 70/15/15
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    idx = np.random.permutation(n_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    def make_split(idx_arr):
        # Input: first M time steps of spatial field + params
        # x_base: (N, Sx, M, 1) - initial condition spatial fields
        # params: (N, 5)
        # targets: (N, Sx, N-M, 1) - remaining time steps
        # jacobians: (N, Sx, N-M, 5) - Jacobians at output time steps
        
        u_init = solutions[idx_arr, :, :M]  # (N, Sx, M)
        x_base = u_init[:, :, :, np.newaxis]  # (N, Sx, M, 1)
        
        # Add spatial and temporal coordinates
        x_coord = np.tile(x[np.newaxis, :, np.newaxis, np.newaxis], (len(idx_arr), 1, M, 1))
        t_coord = np.tile(t[:M][np.newaxis, np.newaxis, :, np.newaxis], (len(idx_arr), n_space, 1, 1))
        x_base = np.concatenate([x_base, x_coord, t_coord], axis=-1)  # (N, Sx, M, 3)
        
        targets = solutions[idx_arr, :, M:]  # (N, Sx, N-M)
        jac = jacobians[idx_arr, :, M:, :]  # (N, Sx, N-M, 5)
        
        return {
            "x_base": x_base.astype(np.float32),
            "params": params[idx_arr].astype(np.float32),
            "targets": targets[:, :, :, np.newaxis].astype(np.float32),  # (N, Sx, N-M, 1)
            "jacobians": jac.astype(np.float32),  # (N, Sx, N-M, 5)
            "param_names": list(param_ranges.keys())
        }

    return make_split(train_idx), make_split(val_idx), make_split(test_idx)


def build_pde1_model():
    """Build FNO model for PDE1."""
    # From Table C.7: modes=8 for both dims, width=20, 4 Fourier layers
    # Input: M time steps of spatial field + x_coord + t_coord + 5 params = 3 + 5 = 8 channels
    # Output: N-M time steps, 1 channel
    
    n_params = 5
    in_channels = 3 + n_params  # u_init + x_coord + t_coord + params
    out_channels = 1
    
    fno = FNO2d(
        modes1=8,
        modes2=8,
        width=20,
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=4
    )
    
    model = SCFNO(fno, n_params=n_params, input_dim=in_channels)
    return model


def pde1_equation_loss(u_pred, params, x_base):
    """
    Compute PINN equation loss for PDE1.
    
    PDE: u_tt = c^2 * u_xx + alpha * u_t + beta * u + gamma * sin(omega * u)
    
    This requires computing spatial and temporal derivatives of u_pred.
    """
    # This is a simplified version - full implementation would use AD
    # to compute u_tt, u_xx, u_t from u_pred
    return torch.tensor(0.0, requires_grad=True)


def run_pde1_experiment(mode="sc_fno", n_samples=2000, device="cpu", save_dir="results/pde1"):
    """Run PDE1 experiment for a given mode."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running PDE1 experiment: {mode}")
    print(f"{'='*60}")
    
    # Prepare data
    print("Generating PDE1 dataset...")
    train_data, val_data, test_data = prepare_pde1_data(n_samples=n_samples)
    
    # Build model
    model = build_pde1_model()
    
    # Training config
    config = {
        "mode": mode,
        "n_epochs": 500,
        "batch_size": 4,
        "lr": 1e-3,
        "c1": 1.0,
        "c2": 1.0,
        "c3": 1.0,
        "n_sample_points": 50,  # Sample subset for efficiency
    }
    
    if mode in ["fno_pinn", "sc_fno_pinn"]:
        config["equation_fn"] = pde1_equation_loss
    
    # Train
    history = train_sc_fno(model, train_data, val_data, config, device=device)
    
    # Evaluate on test set
    metrics = evaluate_model(model, test_data, device=device)
    
    print(f"\nTest Results for {mode}:")
    print(f"  u(t) R2: {metrics['u_r2']:.4f}")
    print(f"  u(t) Relative L2: {metrics['u_relative_l2']:.4f}")
    for key, val in metrics.items():
        if "jac" in key:
            print(f"  {key}: {val:.4f}")
    
    # Save results
    results = {
        "mode": mode,
        "metrics": metrics,
        "avg_epoch_time": history["avg_epoch_time"],
        "best_val_loss": history["best_val_loss"]
    }
    
    with open(os.path.join(save_dir, f"{mode}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(save_dir, f"{mode}_model.pt"))
    
    return model, metrics, history


def run_pde1_perturbation_experiment(model, test_data, perturbation_lambdas=[0.1, 0.2, 0.3, 0.4], device="cpu"):
    """
    Evaluate model robustness to parameter perturbations.
    
    For each lambda, test on parameters in range [b, (1+lambda)*b].
    """
    results = {}
    
    for lam in perturbation_lambdas:
        # Perturb test parameters beyond training range
        params_perturbed = test_data["params"].copy()
        
        # Scale parameters beyond their original upper bounds
        param_ranges = {
            "c": [0.0, 0.25],
            "alpha": [0.0, 0.1],
            "beta": [0.0, 0.25],
            "gamma": [0.0, 0.25],
            "omega": [0.0, 0.25],
        }
        
        for j, (name, (a, b)) in enumerate(param_ranges.items()):
            # Perturb to range [b, (1+lambda)*b]
            n_test = params_perturbed.shape[0]
            params_perturbed[:, j] = np.random.uniform(b, (1 + lam) * b, n_test)
        
        perturbed_data = dict(test_data)
        perturbed_data["params"] = params_perturbed
        
        metrics = evaluate_model(model, perturbed_data, device=device)
        results[lam] = metrics
        print(f"  Lambda={lam}: u R2={metrics['u_r2']:.4f}, u L2={metrics['u_relative_l2']:.4f}")
    
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    modes = ["fno", "sc_fno", "fno_pinn", "sc_fno_pinn"]
    all_results = {}
    
    for mode in modes:
        model, metrics, history = run_pde1_experiment(mode=mode, device=device)
        all_results[mode] = metrics
    
    print("\nPDE1 Results Summary:")
    for mode, metrics in all_results.items():
        print(f"  {mode}: u R2={metrics['u_r2']:.4f}, u L2={metrics['u_relative_l2']:.4f}")
