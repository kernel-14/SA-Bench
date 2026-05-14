"""
Run experiments for PDE4 (Allen-Cahn equation).
Tests SC-FNO on a challenging bifurcation problem.

From the paper:
- PDE4: u_t = epsilon * u_xx + alpha * u - beta * u^3
- u(x,0) = c * tanh(omega * x)
- Parameters: c in [0.1,0.9], alpha in [0.01,1.0], beta in [0.01,1.0],
              omega in [5.0,10.0], epsilon in [0.01,1.0]
- N=30 time steps, Sx=40 spatial points, M=5 initial steps given
- 500 training samples (challenging due to bifurcation)
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
from experiments.sc_fno_experiment import SCFNO, train_sc_fno, evaluate_model


def solve_pde4(c, alpha, beta, omega, epsilon, x, t):
    """
    Solve Allen-Cahn: u_t = epsilon * u_xx + alpha * u - beta * u^3
    u(x,0) = c * tanh(omega * x)
    Using finite differences with periodic BC.
    """
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u0 = c * np.tanh(omega * x)

    u = np.zeros((nx, nt))
    u[:, 0] = u0

    for k in range(nt - 1):
        u_xx = (np.roll(u[:, k], -1) - 2 * u[:, k] + np.roll(u[:, k], 1)) / dx**2
        u_t = epsilon * u_xx + alpha * u[:, k] - beta * u[:, k]**3
        u[:, k+1] = u[:, k] + dt * u_t

    return u


def prepare_pde4_data(n_samples=500, n_time=30, n_space=40, M=5, seed=42):
    """Prepare PDE4 dataset."""
    np.random.seed(seed)
    t = np.linspace(0, 1.0, n_time)
    x = np.linspace(0, 1, n_space, endpoint=False)

    param_ranges = {
        "c": [0.1, 0.9],
        "alpha": [0.01, 1.0],
        "beta": [0.01, 1.0],
        "omega": [5.0, 10.0],
        "epsilon": [0.01, 1.0],
    }
    n_params = len(param_ranges)

    params_dict = {k: np.random.uniform(*v, n_samples) for k, v in param_ranges.items()}
    params = np.stack(list(params_dict.values()), axis=1)

    solutions = np.zeros((n_samples, n_space, n_time))
    jacobians = np.zeros((n_samples, n_space, n_time, n_params))

    h = 1e-4

    for i in range(n_samples):
        c = params_dict["c"][i]
        a = params_dict["alpha"][i]
        b = params_dict["beta"][i]
        w = params_dict["omega"][i]
        eps = params_dict["epsilon"][i]

        sol = solve_pde4(c, a, b, w, eps, x, t)
        solutions[i] = sol

        param_vals = [c, a, b, w, eps]
        for j_idx in range(n_params):
            pvals_p = param_vals.copy()
            pvals_p[j_idx] += h
            pvals_m = param_vals.copy()
            pvals_m[j_idx] -= h

            sol_p = solve_pde4(*pvals_p, x, t)
            sol_m = solve_pde4(*pvals_m, x, t)
            jacobians[i, :, :, j_idx] = (sol_p - sol_m) / (2 * h)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples} samples")

    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    idx = np.random.permutation(n_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    def make_split(idx_arr):
        u_init = solutions[idx_arr, :, :M]
        x_base = u_init[:, :, :, np.newaxis]

        x_coord = np.tile(x[np.newaxis, :, np.newaxis, np.newaxis], (len(idx_arr), 1, M, 1))
        t_coord = np.tile(t[:M][np.newaxis, np.newaxis, :, np.newaxis], (len(idx_arr), n_space, 1, 1))
        x_base = np.concatenate([x_base, x_coord, t_coord], axis=-1)

        targets = solutions[idx_arr, :, M:]
        jac = jacobians[idx_arr, :, M:, :]

        return {
            "x_base": x_base.astype(np.float32),
            "params": params[idx_arr].astype(np.float32),
            "targets": targets[:, :, :, np.newaxis].astype(np.float32),
            "jacobians": jac.astype(np.float32),
            "param_names": list(param_ranges.keys())
        }

    return make_split(train_idx), make_split(val_idx), make_split(test_idx)


def build_pde4_model():
    """Build FNO model for PDE4."""
    n_params = 5
    in_channels = 3 + n_params
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


def run_pde4_experiment(mode="sc_fno", n_samples=500, device="cpu", save_dir="results/pde4"):
    """Run PDE4 experiment."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running PDE4 (Allen-Cahn) experiment: {mode}, N={n_samples}")
    print(f"{'='*60}")

    print("Generating PDE4 dataset...")
    train_data, val_data, test_data = prepare_pde4_data(n_samples=n_samples)

    model = build_pde4_model()

    config = {
        "mode": mode,
        "n_epochs": 500,
        "batch_size": 1,
        "lr": 1e-3,
        "c1": 1.0,
        "c2": 1.0,
        "n_sample_points": 30,
    }

    history = train_sc_fno(model, train_data, val_data, config, device=device)
    metrics = evaluate_model(model, test_data, device=device)

    print(f"\nTest Results for {mode} (N={n_samples}):")
    print(f"  u(t) R2: {metrics['u_r2']:.4f}")
    print(f"  u(t) Relative L2: {metrics['u_relative_l2']:.4f}")
    for key, val in metrics.items():
        if "jac" in key:
            print(f"  {key}: {val:.4f}")

    results = {
        "mode": mode,
        "n_samples": n_samples,
        "metrics": metrics,
        "avg_epoch_time": history["avg_epoch_time"],
    }

    with open(os.path.join(save_dir, f"{mode}_N{n_samples}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(save_dir, f"{mode}_N{n_samples}_model.pt"))

    return model, metrics, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test with different sample sizes as in Table 3
    for n_samples in [500, 100]:
        for mode in ["fno", "sc_fno"]:
            run_pde4_experiment(mode=mode, n_samples=n_samples, device=device)
