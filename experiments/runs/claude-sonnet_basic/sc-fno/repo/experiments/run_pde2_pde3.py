"""
Run experiments for PDE2 (Forced Burgers Equation) and PDE3 (Navier-Stokes).
Also includes the high-dimensional parameter space experiment (zoned PDE2).

From the paper:
- PDE2: (1/pi) * u_t + alpha * u * u_x = gamma * u_xx + delta * sin(omega * t)
- Parameters: alpha in [0.1,1.0], gamma in [0.025,0.25], delta in [0.1,0.5], omega in [0.01,0.1]
- N=30 time steps, Sx=40 spatial points, M=5 initial steps given
- 2000 training samples

- PDE3: Navier-Stokes vorticity formulation
- Parameters: alpha in [pi, 5*pi], beta in [pi, 5*pi]
- Sx=Sy=64, t_final=3s
- 1000 training samples
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fno import FNO2d, FNO3d
from experiments.sc_fno_experiment import SCFNO, train_sc_fno, evaluate_model, relative_l2_loss, r2_score_torch


# ============================================================
# PDE2: Forced Burgers Equation
# ============================================================

def solve_pde2(alpha, gamma, delta, omega, x, t):
    """
    Solve Forced Burgers: (1/pi) * u_t + alpha * u * u_x = gamma * u_xx + delta * sin(omega * t)
    Using finite differences with periodic BC.
    """
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Initial condition
    x0 = 0.5
    sigma = 0.3
    u0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) + np.sin(0.5 * np.pi * x)

    u = np.zeros((nx, nt))
    u[:, 0] = u0

    for k in range(nt - 1):
        # Periodic boundary conditions
        u_x = (np.roll(u[:, k], -1) - np.roll(u[:, k], 1)) / (2 * dx)
        u_xx = (np.roll(u[:, k], -1) - 2 * u[:, k] + np.roll(u[:, k], 1)) / dx**2
        # PDE: (1/pi) * u_t + alpha * u * u_x = gamma * u_xx + delta * sin(omega * t)
        u_t = np.pi * (gamma * u_xx - alpha * u[:, k] * u_x + delta * np.sin(omega * t[k]))
        u[:, k+1] = u[:, k] + dt * u_t

    return u


def prepare_pde2_data(n_samples=2000, n_time=30, n_space=40, M=5, seed=42, zoned=False, n_zones=40):
    """
    Prepare PDE2 dataset.
    
    If zoned=True, creates high-dimensional parameter space with 2*n_zones+2 parameters.
    """
    np.random.seed(seed)
    t = np.linspace(0, np.pi, n_time)
    x = np.linspace(0, 1, n_space, endpoint=False)

    if not zoned:
        # Standard PDE2: 4 parameters
        param_ranges = {
            "alpha": [0.1, 1.0],
            "gamma": [0.025, 0.25],
            "delta": [0.1, 0.5],
            "omega": [0.01, 0.1],
        }
        n_params = len(param_ranges)
        params_dict = {k: np.random.uniform(*v, n_samples) for k, v in param_ranges.items()}
        params = np.stack(list(params_dict.values()), axis=1)  # (N, 4)

        solutions = np.zeros((n_samples, n_space, n_time))
        jacobians = np.zeros((n_samples, n_space, n_time, n_params))

        h = 1e-4

        for i in range(n_samples):
            a = params_dict["alpha"][i]
            g = params_dict["gamma"][i]
            d = params_dict["delta"][i]
            w = params_dict["omega"][i]

            sol = solve_pde2(a, g, d, w, x, t)
            solutions[i] = sol

            # Finite differences for Jacobians
            param_vals = [a, g, d, w]
            for j_idx in range(n_params):
                pvals_p = param_vals.copy()
                pvals_p[j_idx] += h
                pvals_m = param_vals.copy()
                pvals_m[j_idx] -= h

                sol_p = solve_pde2(*pvals_p, x, t)
                sol_m = solve_pde2(*pvals_m, x, t)
                jacobians[i, :, :, j_idx] = (sol_p - sol_m) / (2 * h)

            if (i + 1) % 200 == 0:
                print(f"  Generated {i+1}/{n_samples} samples")

        param_names = list(param_ranges.keys())

    else:
        # Zoned PDE2: 2*n_zones + 2 parameters (82 total for n_zones=40)
        # Each zone has its own alpha and delta, plus global gamma and omega
        n_params = 2 * n_zones + 2
        param_names = ([f"alpha_{i}" for i in range(n_zones)] +
                      [f"delta_{i}" for i in range(n_zones)] +
                      ["gamma", "omega"])

        params = np.zeros((n_samples, n_params))
        for i in range(n_samples):
            params[i, :n_zones] = np.random.uniform(0.1, 1.0, n_zones)  # alpha per zone
            params[i, n_zones:2*n_zones] = np.random.uniform(0.1, 0.5, n_zones)  # delta per zone
            params[i, 2*n_zones] = np.random.uniform(0.025, 0.25)  # global gamma
            params[i, 2*n_zones+1] = np.random.uniform(0.01, 0.1)  # global omega

        solutions = np.zeros((n_samples, n_space, n_time))
        jacobians = np.zeros((n_samples, n_space, n_time, n_params))

        h = 1e-4

        def solve_pde2_zoned(params_i, x, t, n_zones):
            """Solve zoned Burgers with spatially varying parameters."""
            nx = len(x)
            nt = len(t)
            dx = x[1] - x[0]
            dt = t[1] - t[0]

            alphas = params_i[:n_zones]
            deltas = params_i[n_zones:2*n_zones]
            gamma = params_i[2*n_zones]
            omega = params_i[2*n_zones+1]

            # Map zones to spatial grid
            zone_size = nx // n_zones
            alpha_field = np.zeros(nx)
            delta_field = np.zeros(nx)
            for z in range(n_zones):
                start = z * zone_size
                end = (z + 1) * zone_size if z < n_zones - 1 else nx
                alpha_field[start:end] = alphas[z]
                delta_field[start:end] = deltas[z]

            x0 = 0.5
            sigma = 0.3
            u0 = np.exp(-(x - x0)**2 / (2 * sigma**2)) + np.sin(0.5 * np.pi * x)

            u = np.zeros((nx, nt))
            u[:, 0] = u0

            for k in range(nt - 1):
                u_x = (np.roll(u[:, k], -1) - np.roll(u[:, k], 1)) / (2 * dx)
                u_xx = (np.roll(u[:, k], -1) - 2 * u[:, k] + np.roll(u[:, k], 1)) / dx**2
                u_t = np.pi * (gamma * u_xx - alpha_field * u[:, k] * u_x +
                               delta_field * np.sin(omega * t[k]))
                u[:, k+1] = u[:, k] + dt * u_t

            return u

        for i in range(n_samples):
            sol = solve_pde2_zoned(params[i], x, t, n_zones)
            solutions[i] = sol

            # Finite differences for Jacobians (only for a subset of params for efficiency)
            for j_idx in range(n_params):
                pvals_p = params[i].copy()
                pvals_p[j_idx] += h
                pvals_m = params[i].copy()
                pvals_m[j_idx] -= h

                sol_p = solve_pde2_zoned(pvals_p, x, t, n_zones)
                sol_m = solve_pde2_zoned(pvals_m, x, t, n_zones)
                jacobians[i, :, :, j_idx] = (sol_p - sol_m) / (2 * h)

            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{n_samples} zoned samples")

    # Split data: 70/15/15
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    idx = np.random.permutation(n_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    def make_split(idx_arr):
        u_init = solutions[idx_arr, :, :M]  # (N, Sx, M)
        x_base = u_init[:, :, :, np.newaxis]  # (N, Sx, M, 1)

        x_coord = np.tile(x[np.newaxis, :, np.newaxis, np.newaxis], (len(idx_arr), 1, M, 1))
        t_coord = np.tile(t[:M][np.newaxis, np.newaxis, :, np.newaxis], (len(idx_arr), n_space, 1, 1))
        x_base = np.concatenate([x_base, x_coord, t_coord], axis=-1)  # (N, Sx, M, 3)

        targets = solutions[idx_arr, :, M:]  # (N, Sx, N-M)
        jac = jacobians[idx_arr, :, M:, :]  # (N, Sx, N-M, n_params)

        return {
            "x_base": x_base.astype(np.float32),
            "params": params[idx_arr].astype(np.float32),
            "targets": targets[:, :, :, np.newaxis].astype(np.float32),
            "jacobians": jac.astype(np.float32),
            "param_names": param_names
        }

    return make_split(train_idx), make_split(val_idx), make_split(test_idx)


def build_pde2_model(n_params=4):
    """Build FNO model for PDE2."""
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


def run_pde2_experiment(mode="sc_fno", n_samples=2000, device="cpu", save_dir="results/pde2"):
    """Run PDE2 experiment."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running PDE2 experiment: {mode}")
    print(f"{'='*60}")

    print("Generating PDE2 dataset...")
    train_data, val_data, test_data = prepare_pde2_data(n_samples=n_samples)

    model = build_pde2_model(n_params=4)

    config = {
        "mode": mode,
        "n_epochs": 500,
        "batch_size": 4,
        "lr": 1e-3,
        "c1": 1.0,
        "c2": 1.0,
        "c3": 1.0,
        "n_sample_points": 50,
    }

    history = train_sc_fno(model, train_data, val_data, config, device=device)
    metrics = evaluate_model(model, test_data, device=device)

    print(f"\nTest Results for {mode}:")
    print(f"  u(t) R2: {metrics['u_r2']:.4f}")
    print(f"  u(t) Relative L2: {metrics['u_relative_l2']:.4f}")

    results = {
        "mode": mode,
        "metrics": metrics,
        "avg_epoch_time": history["avg_epoch_time"],
    }

    with open(os.path.join(save_dir, f"{mode}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), os.path.join(save_dir, f"{mode}_model.pt"))

    return model, metrics, history


def run_pde2_zoned_experiment(mode="sc_fno", n_samples=500, device="cpu", save_dir="results/pde2_zoned"):
    """
    Run high-dimensional parameter space experiment (zoned PDE2).
    82 parameters: 40 alpha + 40 delta + gamma + omega.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running Zoned PDE2 experiment: {mode} (82 params)")
    print(f"{'='*60}")

    print("Generating Zoned PDE2 dataset...")
    train_data, val_data, test_data = prepare_pde2_data(
        n_samples=n_samples, zoned=True, n_zones=40
    )

    model = build_pde2_model(n_params=82)

    config = {
        "mode": mode,
        "n_epochs": 500,
        "batch_size": 1,
        "lr": 1e-3,
        "c1": 1.0,
        "c2": 1.0,
        "c3": 1.0,
        "n_sample_points": 30,
    }

    history = train_sc_fno(model, train_data, val_data, config, device=device)
    metrics = evaluate_model(model, test_data, device=device)

    print(f"\nTest Results for {mode} (Zoned):")
    print(f"  u(t) R2: {metrics['u_r2']:.4f}")
    print(f"  u(t) Relative L2: {metrics['u_relative_l2']:.4f}")

    results = {
        "mode": mode,
        "n_params": 82,
        "metrics": metrics,
        "avg_epoch_time": history["avg_epoch_time"],
    }

    with open(os.path.join(save_dir, f"{mode}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return model, metrics, history


# ============================================================
# PDE3: Navier-Stokes
# ============================================================

def initial_vorticity(X, Y, alpha, beta):
    """Initial vorticity distribution for PDE3."""
    return (np.sin(alpha * X) * np.cos(beta * Y) +
            np.cos(alpha * Y) * np.sin(beta * X) +
            np.sin(alpha * X + beta * Y) * np.cos(alpha * Y - beta * X))


def solve_ns_vorticity(alpha, beta, nx=64, ny=64, Re=1000.0, t_final=3.0, dt=0.01):
    """
    Solve NS vorticity equation using pseudo-spectral method.
    Returns vorticity at t=t_final.
    """
    x = np.linspace(0, 1, nx, endpoint=False)
    y = np.linspace(0, 1, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    omega = initial_vorticity(X, Y, alpha, beta)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0

    n_steps = int(t_final / dt)
    for _ in range(n_steps):
        omega_hat = np.fft.fft2(omega)
        psi_hat = -omega_hat / K2
        psi_hat[0, 0] = 0.0

        u = np.real(np.fft.ifft2(1j * KY * psi_hat))
        v = np.real(np.fft.ifft2(-1j * KX * psi_hat))

        omega_x = np.real(np.fft.ifft2(1j * KX * omega_hat))
        omega_y = np.real(np.fft.ifft2(1j * KY * omega_hat))

        lap_omega = np.real(np.fft.ifft2(-K2 * omega_hat))

        domega_dt = -u * omega_x - v * omega_y + (1.0 / Re) * lap_omega
        omega = omega + dt * domega_dt

    return omega


def prepare_pde3_data(n_samples=1000, n_space=64, seed=42):
    """Prepare PDE3 dataset."""
    np.random.seed(seed)

    alphas = np.random.uniform(np.pi, 5 * np.pi, n_samples)
    betas = np.random.uniform(np.pi, 5 * np.pi, n_samples)
    params = np.stack([alphas, betas], axis=1)

    x = np.linspace(0, 1, n_space, endpoint=False)
    y = np.linspace(0, 1, n_space, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    solutions = np.zeros((n_samples, n_space, n_space))  # vorticity at t=3
    jacobians = np.zeros((n_samples, n_space, n_space, 2))

    h = 1e-3

    for i in range(n_samples):
        a, b = alphas[i], betas[i]

        sol = solve_ns_vorticity(a, b, n_space, n_space)
        solutions[i] = sol

        sol_pa = solve_ns_vorticity(a + h, b, n_space, n_space)
        sol_ma = solve_ns_vorticity(a - h, b, n_space, n_space)
        jacobians[i, :, :, 0] = (sol_pa - sol_ma) / (2 * h)

        sol_pb = solve_ns_vorticity(a, b + h, n_space, n_space)
        sol_mb = solve_ns_vorticity(a, b - h, n_space, n_space)
        jacobians[i, :, :, 1] = (sol_pb - sol_mb) / (2 * h)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_samples} PDE3 samples")

    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    idx = np.random.permutation(n_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    def make_split(idx_arr):
        # Initial vorticity field + coordinates
        omega0 = np.array([initial_vorticity(X, Y, alphas[i], betas[i]) for i in idx_arr])
        x_base = omega0[:, :, :, np.newaxis]  # (N, Sx, Sy, 1)

        # Add spatial coordinates
        x_coord = np.tile(x[np.newaxis, :, np.newaxis, np.newaxis], (len(idx_arr), 1, n_space, 1))
        y_coord = np.tile(y[np.newaxis, np.newaxis, :, np.newaxis], (len(idx_arr), n_space, 1, 1))
        x_base = np.concatenate([x_base, x_coord, y_coord], axis=-1)  # (N, Sx, Sy, 3)

        targets = solutions[idx_arr]  # (N, Sx, Sy)

        return {
            "x_base": x_base.astype(np.float32),
            "params": params[idx_arr].astype(np.float32),
            "targets": targets[:, :, :, np.newaxis].astype(np.float32),  # (N, Sx, Sy, 1)
            "jacobians": jacobians[idx_arr].astype(np.float32),  # (N, Sx, Sy, 2)
            "param_names": ["alpha", "beta"]
        }

    return make_split(train_idx), make_split(val_idx), make_split(test_idx)


def build_pde3_model():
    """Build FNO model for PDE3 (2D spatial)."""
    n_params = 2
    in_channels = 3 + n_params  # omega0 + x_coord + y_coord + params
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


def run_pde3_experiment(mode="sc_fno", n_samples=1000, device="cpu", save_dir="results/pde3"):
    """Run PDE3 experiment."""
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running PDE3 (Navier-Stokes) experiment: {mode}")
    print(f"{'='*60}")

    print("Generating PDE3 dataset...")
    train_data, val_data, test_data = prepare_pde3_data(n_samples=n_samples)

    model = build_pde3_model()

    config = {
        "mode": mode,
        "n_epochs": 500,
        "batch_size": 4,
        "lr": 1e-3,
        "c1": 1.0,
        "c2": 1.0,
        "n_sample_points": 100,
    }

    history = train_sc_fno(model, train_data, val_data, config, device=device)
    metrics = evaluate_model(model, test_data, device=device)

    print(f"\nTest Results for {mode}:")
    print(f"  omega R2: {metrics['u_r2']:.4f}")
    print(f"  omega Relative L2: {metrics['u_relative_l2']:.4f}")

    results = {
        "mode": mode,
        "metrics": metrics,
        "avg_epoch_time": history["avg_epoch_time"],
    }

    with open(os.path.join(save_dir, f"{mode}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return model, metrics, history


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # PDE2 experiments
    for mode in ["fno", "sc_fno"]:
        run_pde2_experiment(mode=mode, device=device)

    # Zoned PDE2 (high-dimensional)
    for mode in ["fno", "sc_fno"]:
        run_pde2_zoned_experiment(mode=mode, n_samples=500, device=device)

    # PDE3 experiments
    for mode in ["fno", "sc_fno"]:
        run_pde3_experiment(mode=mode, device=device)
