"""
Evaluation script for WDNO models.

Computes MSE for simulation tasks and objective I for control tasks,
matching the evaluation protocol described in the paper.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wdno import WDNOSimulation, WDNOControl, SuperResolutionModel
from utils.data_generation import (
    generate_burgers_initial_condition,
    generate_burgers_control,
    solve_burgers_fdm,
    burgers_control_objective,
)


def evaluate_simulation(
    model: WDNOSimulation,
    n_test_samples: int = 50,
    n_time: int = 81,
    n_space: int = 120,
    device: str = 'cuda',
):
    """
    Evaluate simulation MSE following the paper's protocol:
    MSE measured on entire state sequences excluding initial conditions.

    Returns:
        mse: Mean squared error
    """
    model.eval()
    u0_all = generate_burgers_initial_condition(n_space, n_test_samples)
    f_all = generate_burgers_control(n_time - 1, n_space, n_test_samples)
    u_gt = solve_burgers_fdm(u0_all, f_all, n_time=n_time - 1, n_space=n_space)

    # Prepare condition
    condition_u0 = u0_all.unsqueeze(1).unsqueeze(-1).expand(-1, 1, n_time, n_space)
    condition_f = f_all.unsqueeze(1)
    condition_f = torch.cat([condition_f, condition_f[:, :, -1:, :]], dim=2)
    condition = torch.cat([condition_u0, condition_f], dim=1)

    mse_values = []
    for i in range(n_test_samples):
        cond = condition[i:i+1].to(device)
        gt = u_gt[i:i+1].to(device)

        with torch.no_grad():
            pred = model(cond)  # (1, 1, 81, 120)

        # Compute MSE excluding initial condition (t=0)
        pred_traj = pred[0, 0, 1:, :]  # (80, 120)
        gt_traj = gt[0, 1:, :]  # (80, 120)
        mse = torch.mean((pred_traj - gt_traj) ** 2).item()
        mse_values.append(mse)

    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    print(f"Simulation MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    return mean_mse


def evaluate_control(
    model: WDNOControl,
    n_test_samples: int = 50,
    n_time: int = 81,
    n_space: int = 120,
    device: str = 'cuda',
    solver = None,
):
    """
    Evaluate control objective I following the paper's protocol:
    I = ∫|u(T,x) - u*(x)|²dx + α∫|f|²dtdx

    Returns:
        mean_I: Mean control objective value
    """
    model.eval()
    u0_all = generate_burgers_initial_condition(n_space, n_test_samples)

    # Use random targets (in practice, from test set)
    u_target = generate_burgers_initial_condition(n_space, n_test_samples)

    I_values = []
    for i in range(n_test_samples):
        u0 = u0_all[i:i+1]
        target = u_target[i:i+1]

        # Prepare condition: [u0, u_target]
        cond_u0 = u0.unsqueeze(1).unsqueeze(-1).expand(-1, 1, n_time, n_space).to(device)
        cond_ut = target.unsqueeze(1).unsqueeze(-1).expand(-1, 1, n_time, n_space).to(device)
        condition = torch.cat([cond_u0, cond_ut], dim=1)

        with torch.no_grad():
            f_pred = model(condition)  # (1, 1, 81, 120)

        # Decode to original space
        f = f_pred[0, 0, :80, :]  # (80, 120)

        # Solve with ground-truth solver
        u_all = solve_burgers_fdm(u0, f.unsqueeze(0), n_time=80, n_space=120)
        u_T = u_all[0, -1, :]  # (120,)

        I_val = burgers_control_objective(
            u_T.unsqueeze(0), f.unsqueeze(0), target, alpha=0.00002
        ).item()

        I_values.append(I_val)

    mean_I = np.mean(I_values)
    std_I = np.std(I_values)
    print(f"Control Objective I: {mean_I:.6f} ± {std_I:.6f}")
    return mean_I


def evaluate_super_resolution(
    brm: WDNOSimulation,
    srm: SuperResolutionModel,
    n_test_samples: int = 100,
    n_sr_steps: int = 1,
    device: str = 'cuda',
):
    """
    Evaluate zero-shot super-resolution MSE at highest resolution.

    Args:
        brm: Base-Resolution Model
        srm: Super-Resolution Model
        n_sr_steps: Number of super-resolution steps
    """
    brm.eval()
    srm.eval()

    # Generate high-resolution test data
    n_time_base = 81
    n_space_base = 120
    scale = 2 ** n_sr_steps
    n_time_hr = n_time_base * scale
    n_space_hr = n_space_base * scale

    mse_values = []
    for i in tqdm(range(n_test_samples), desc="Evaluating SR"):
        u0 = generate_burgers_initial_condition(n_space_hr, 1)
        f = generate_burgers_control(n_time_hr - 1, n_space_hr, 1)
        u_gt = solve_burgers_fdm(u0, f, n_time=n_time_hr - 1, n_space=n_space_hr)

        # Downsample condition to base resolution
        u0_lo = torch.nn.functional.interpolate(
            u0.unsqueeze(0), size=n_space_base, mode='linear'
        ).squeeze(0)
        f_lo = torch.nn.functional.interpolate(
            f.unsqueeze(0), size=(n_time_base - 1, n_space_base), mode='bilinear'
        ).squeeze(0)

        # Base resolution prediction
        cond_u0 = u0_lo.unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand(-1, 1, n_time_base, n_space_base)
        cond_f = f_lo.unsqueeze(0).unsqueeze(1)
        cond_f = torch.cat([cond_f, cond_f[:, :, -1:, :]], dim=2)
        condition = torch.cat([cond_u0, cond_f], dim=1).to(device)

        with torch.no_grad():
            lo_pred = brm(condition)

        # Super-resolve
        hi_pred = lo_pred
        for step in range(n_sr_steps):
            with torch.no_grad():
                hi_pred = srm.super_resolve(hi_pred, condition)

        # Compute MSE at highest resolution
        mse = torch.mean((hi_pred[:, 0, 1:, :] - u_gt[:, 1:, :].to(device)) ** 2).item()
        mse_values.append(mse)

    mean_mse = np.mean(mse_values)
    print(f"SR MSE ({n_sr_steps}x): {mean_mse:.6f}")
    return mean_mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                       choices=['simulation', 'control', 'super_resolution'])
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--srm_checkpoint', type=str, default=None,
                       help='Path to SRM checkpoint (for super_resolution mode)')
    parser.add_argument('--n_test', type=int, default=50)
    parser.add_argument('--n_sr_steps', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device

    if args.mode == 'simulation':
        model = WDNOSimulation(
            data_shape=(81, 120),
            cond_shape=(2, 81, 120),
            wavelet_type='bior2.4',
            wavelet_mode='periodization',
            n_channels=1,
            n_cond_channels=2,
            is_3d=False,
        ).to(device)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        evaluate_simulation(model, n_test_samples=args.n_test, device=device)

    elif args.mode == 'control':
        model = WDNOControl(
            data_shape=(81, 120),
            cond_shape=(2, 81, 120),
            wavelet_type='bior2.4',
            wavelet_mode='periodization',
            n_channels=1,
            n_cond_channels=2,
            is_3d=False,
        ).to(device)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        evaluate_control(model, n_test_samples=args.n_test, device=device)

    elif args.mode == 'super_resolution':
        brm = WDNOSimulation(
            data_shape=(81, 120),
            cond_shape=(2, 81, 120),
            wavelet_type='bior2.4',
            wavelet_mode='periodization',
            n_channels=1,
            n_cond_channels=2,
            is_3d=False,
        ).to(device)

        brm_checkpoint = torch.load(args.checkpoint, map_location=device)
        brm.load_state_dict(brm_checkpoint['model_state_dict'])

        srm = SuperResolutionModel(
            data_shape=(81 * 2 ** args.n_sr_steps, 120 * 2 ** args.n_sr_steps),
            cond_shape=(2, 81 * 2 ** args.n_sr_steps, 120 * 2 ** args.n_sr_steps),
            wavelet_type='bior2.4',
            wavelet_mode='periodization',
            n_channels=1,
            n_cond_channels=2,
            is_3d=False,
        ).to(device)

        if args.srm_checkpoint:
            srm_checkpoint = torch.load(args.srm_checkpoint, map_location=device)
            srm.load_state_dict(srm_checkpoint['model_state_dict'])

        evaluate_super_resolution(
            brm, srm,
            n_test_samples=args.n_test,
            n_sr_steps=args.n_sr_steps,
            device=device,
        )
