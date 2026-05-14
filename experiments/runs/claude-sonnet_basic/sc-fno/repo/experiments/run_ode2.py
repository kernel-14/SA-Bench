"""
Run experiments for ODE2 (Duffing Oscillator).
Compares FNO, SC-FNO, FNO-PINN, SC-FNO-PINN configurations.

From the paper:
- ODE2: x_ddot + delta*x_dot + alpha*x + beta*x^3 = gamma*cos(omega*t)
- x(0) = epsilon, x_dot(0) = zeta
- Parameters: alpha in [0.02,0.06], beta in [0.01,0.03], gamma in [20,60],
              delta in [0.5,1.5], omega in [0.2,0.6], epsilon in [0,0.2], zeta in [0,0.2]
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
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fno import FNO1d
from experiments.sc_fno_experiment import SCFNO, train_sc_fno, evaluate_model


def duffing_rhs(t, state, alpha, beta, gamma, delta, omega):
    """RHS of Duffing oscillator as first-order system."""
    x, v = state
    dxdt = v
    dvdt = gamma * np.cos(omega * t) - delta * v - alpha * x - beta * x**3
    return [dxdt, dvdt]


def prepare_ode2_data(n_samples=2000, n_time=100, M=10, seed=42):
    """
    Prepare ODE2 dataset.
    
    Input to FNO: first M time steps of x + parameters p
    Output: remaining N-M time steps
    
    Returns train/val/test splits (70/15/15).
    """
    np.random.seed(seed)
    t = np.linspace(0, 1, n_time)

    param_ranges = {
        "alpha": [0.02, 0.06],
        "beta": [0.01, 0.03],
        "gamma": [20.0, 60.0],
        "delta": [0.5, 1.5],
        "omega": [0.2, 0.6],
        "epsilon": [0.0, 0.2],
        "zeta": [0.0, 0.2],
    }
    n_params = len(param_ranges)

    params_dict = {k: np.random.uniform(*v, n_samples) for k, v in param_ranges.items()}
    params = np.stack(list(params_dict.values()), axis=1)  # (N, 7)

    solutions = np.zeros((n_samples, n_time))  # x only (not x_dot)
    jacobians = np.zeros((n_samples, n_time, n_params))

    h = 1e-4

    for i in range(n_samples):
        a = params_dict["alpha"][i]
        b = params_dict["beta"][i]
        g = params_dict["gamma"][i]
        d = params_dict["delta"][i]
        w = params_dict["omega"][i]
        eps = params_dict["epsilon"][i]
        zeta = params_dict["zeta"][i]

        x0 = [eps, zeta]
        sol = solve_ivp(duffing_rhs, [0, 1], x0, t_eval=t, args=(a, b, g, d, w))
        solutions[i] = sol.y[0]  # x position only

        # Finite differences for Jacobians
        param_vals = [a, b, g, d, w, eps, zeta]
        for j_idx in range(n_params):
            pvals_p = param_vals.copy()
            pvals_p[j_idx] += h
            pvals_m = param_vals.copy()
            pvals_m[j_idx] -= h

            if j_idx < 5:
                sol_p = solve_ivp(duffing_rhs, [0, 1], x0, t_eval=t, args=tuple(pvals_p[:5]))
                sol_m = solve_ivp(duffing_rhs, [0, 1], x0, t_eval=t, args=tuple(pvals_m[:5]))
            else:
                x0_p = [pvals_p[5], pvals_p[6]]
                x0_m = [pvals_m[5], pvals_m[6]]
                sol_p = solve_ivp(duffing_rhs, [0, 1], x0_p, t_eval=t, args=(a, b, g, d, w))
                sol_m = solve_ivp(duffing_rhs, [0, 1], x0_m, t_eval=t, args=(a, b, g, d, w))

            jacobians[i, :, j_idx] = (sol_p.y[0] - sol_m.y[0]) / (2 * h)

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
        u_init = solutions[idx_arr, :M]
        x_base = u_init[:, :, np.newaxis]

        t_coord = np.tile(t[:M][np.newaxis, :, np.newaxis], (len(idx_arr), 1, 1))
        x_base = np.concatenate([x_base, t_coord], axis=-1)  # (N, M, 2)

        targets = solutions[idx_arr, M:]
        jac = jacobians[idx_arr, M:, :]

        return {
            "x_base": x_base.astype(np.float32),
            "params": params[idx_arr].astype(np.float32),
            "targets": targets[:, :, np.newaxis].astype(np.float32),
            "jacobians": jac[:, :, np.newaxis, :].astype(np.float32),
            "param_names": list(param_ranges.keys())
        }

    return make_split(train_idx), make_split(val_idx), make_split(test_idx)


def build_ode2_model():
    """Build FNO model for ODE2."""
    n_params = 7
    in_channels = 2 + n_params  # x_init + t_coord + params
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


def run_ode2_experiment(mode="sc_fno", n_samples=2000, device="cpu", save_dir="results/ode2"):
    """Run ODE2 experiment."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running ODE2 (Duffing) experiment: {mode}")
    print(f"{'='*60}")

    print("Generating ODE2 dataset...")
    train_data, val_data, test_data = prepare_ode2_data(n_samples=n_samples)

    model = build_ode2_model()

    config = {
        "mode": mode,
        "n_epochs": 500,
        "batch_size": 16,
        "lr": 1e-3,
        "c1": 1.0,
        "c2": 1.0,
        "c3": 1.0,
        "n_sample_points": None,
    }

    history = train_sc_fno(model, train_data, val_data, config, device=device)
    metrics = evaluate_model(model, test_data, device=device)

    print(f"\nTest Results for {mode}:")
    print(f"  u(t) R2: {metrics['u_r2']:.4f}")
    print(f"  u(t) Relative L2: {metrics['u_relative_l2']:.4f}")
    for key, val in metrics.items():
        if "jac" in key:
            print(f"  {key}: {val:.4f}")

    results = {
        "mode": mode,
        "metrics": metrics,
        "avg_epoch_time": history["avg_epoch_time"],
    }

    with open(os.path.join(save_dir, f"{mode}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(save_dir, f"{mode}_model.pt"))

    return model, metrics, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for mode in ["fno", "sc_fno", "fno_pinn", "sc_fno_pinn"]:
        run_ode2_experiment(mode=mode, device=device)
