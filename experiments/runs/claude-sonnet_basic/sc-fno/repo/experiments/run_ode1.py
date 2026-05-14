"""
Run experiments for ODE1 (Composite Harmonic Oscillator).
Compares FNO, SC-FNO, FNO-PINN, SC-FNO-PINN configurations.

From the paper:
- ODE1: du/dt = alpha*sin(alpha*pi*t) + beta*cos(beta*pi*t)
- u(0) = sin(gamma*pi)
- Parameters: alpha in [1,3], beta in [1,3], gamma in [0,1]
- N=100 time steps, M=10 initial steps given
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
from models.fno import FNO1d
from experiments.sc_fno_experiment import SCFNO, train_sc_fno, evaluate_model, relative_l2_loss, r2_score_torch


def prepare_ode1_data(n_samples=2000, n_time=100, M=10, seed=42):
    """
    Prepare ODE1 dataset.
    
    Input to FNO: first M time steps of u + parameters p
    Output: remaining N-M time steps
    
    Returns train/val/test splits (70/15/15).
    """
    np.random.seed(seed)
    t = np.linspace(0, 1, n_time)

    # Parameter ranges
    alphas = np.random.uniform(1.0, 3.0, n_samples)
    betas = np.random.uniform(1.0, 3.0, n_samples)
    gammas = np.random.uniform(0.0, 1.0, n_samples)
    params = np.stack([alphas, betas, gammas], axis=1)  # (N, 3)

    # Analytical solutions
    solutions = np.zeros((n_samples, n_time))
    jacobians = np.zeros((n_samples, n_time, 3))

    for i in range(n_samples):
        a, b, g = alphas[i], betas[i], gammas[i]
        solutions[i] = -1/np.pi * np.cos(a * np.pi * t) + 1/np.pi * np.sin(b * np.pi * t) + np.sin(g * np.pi) + 1/np.pi
        jacobians[i, :, 0] = t * np.sin(a * np.pi * t)  # du/dalpha
        jacobians[i, :, 1] = t * np.cos(b * np.pi * t)  # du/dbeta
        jacobians[i, :, 2] = np.pi * np.cos(g * np.pi)  # du/dgamma

    # Split data: 70/15/15
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    # Ensure val/test have different parameter ranges (not seen during training)
    # Sort by alpha to create a split where test has higher alpha values
    # Actually, the paper uses random split but ensures val/test params not in training
    # We use a simple random split here
    idx = np.random.permutation(n_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    def make_split(idx):
        # Input: first M time steps + params repeated
        # x_base: (N, M, 1) - initial condition values
        # params: (N, 3)
        # targets: (N, N-M) - remaining time steps
        # jacobians: (N, N-M, 3) - Jacobians at output time steps
        
        u_init = solutions[idx, :M]  # (N, M)
        x_base = u_init[:, :, np.newaxis]  # (N, M, 1)
        
        # Add time coordinate to x_base
        t_init = t[:M]
        t_coord = np.tile(t_init[np.newaxis, :, np.newaxis], (len(idx), 1, 1))
        x_base = np.concatenate([x_base, t_coord], axis=-1)  # (N, M, 2)
        
        targets = solutions[idx, M:]  # (N, N-M)
        jac = jacobians[idx, M:, :]  # (N, N-M, 3)
        
        return {
            "x_base": x_base.astype(np.float32),
            "params": params[idx].astype(np.float32),
            "targets": targets[:, :, np.newaxis].astype(np.float32),  # (N, N-M, 1)
            "jacobians": jac[:, :, np.newaxis, :].astype(np.float32),  # (N, N-M, 1, 3)
            "param_names": ["alpha", "beta", "gamma"]
        }

    return make_split(train_idx), make_split(val_idx), make_split(test_idx)


def build_ode1_model(mode="sc_fno"):
    """Build FNO model for ODE1."""
    # From Table C.7: modes=8, width=20, 4 Fourier layers
    # Input: M time steps of u + time coord + 3 params = 2 + 3 = 5 channels
    # Output: N-M time steps = 90 time steps, 1 channel
    
    M = 10
    n_params = 3
    in_channels = 2 + n_params  # u_init + t_coord + params
    out_channels = 1
    
    fno = FNO1d(
        modes=8,
        width=20,
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=4
    )
    
    model = SCFNO(fno, n_params=n_params, input_dim=in_channels)
    return model


def run_ode1_experiment(mode="sc_fno", n_samples=2000, device="cpu", save_dir="results/ode1"):
    """Run ODE1 experiment for a given mode."""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running ODE1 experiment: {mode}")
    print(f"{'='*60}")
    
    # Prepare data
    train_data, val_data, test_data = prepare_ode1_data(n_samples=n_samples)
    
    # Build model
    model = build_ode1_model(mode)
    
    # Training config
    config = {
        "mode": mode,
        "n_epochs": 500,
        "batch_size": 16,
        "lr": 1e-3,
        "c1": 1.0,
        "c2": 1.0,
        "c3": 1.0,
        "n_sample_points": None,  # Use all points for ODE (small)
    }
    
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


def run_all_ode1_experiments(device="cpu"):
    """Run all four model configurations for ODE1."""
    modes = ["fno", "fno_pinn", "sc_fno", "sc_fno_pinn"]
    all_results = {}
    
    for mode in modes:
        model, metrics, history = run_ode1_experiment(mode=mode, device=device)
        all_results[mode] = metrics
    
    # Print comparison table
    print("\n" + "="*80)
    print("ODE1 Results Summary")
    print("="*80)
    print(f"{'Metric':<30} {'FNO':>10} {'SC-FNO':>10} {'SC-FNO-PINN':>12} {'FNO-PINN':>10}")
    print("-"*80)
    
    for metric in ["u_r2", "u_relative_l2"]:
        vals = [all_results.get(m, {}).get(metric, float("nan")) for m in modes]
        print(f"{metric:<30} {vals[0]:>10.4f} {vals[2]:>10.4f} {vals[3]:>12.4f} {vals[1]:>10.4f}")
    
    return all_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_all_ode1_experiments(device=device)
