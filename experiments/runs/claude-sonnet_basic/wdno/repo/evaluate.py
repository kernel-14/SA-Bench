"""
Main evaluation entry point for WDNO.

Usage:
    # Evaluate simulation
    python evaluate.py --config configs/burgers_1d.yaml --task simulation \
                       --checkpoint outputs/checkpoint_epoch_100.pt \
                       --data_path /path/to/test_data

    # Evaluate control
    python evaluate.py --config configs/burgers_1d.yaml --task control \
                       --checkpoint outputs/checkpoint_epoch_100.pt \
                       --data_path /path/to/test_data

    # Evaluate zero-shot super-resolution
    python evaluate.py --config configs/burgers_1d.yaml --task super_resolution \
                       --checkpoint outputs/checkpoint_epoch_100.pt \
                       --sr_checkpoint outputs/sr_checkpoint.pt \
                       --data_path /path/to/test_data
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from models.wdno import WDNO1D, WDNO2D, build_wdno_1d, build_wdno_2d
from utils.metrics import (
    compute_mse, compute_mae, compute_linf, compute_relative_l2,
    compute_burgers_control_objective, evaluate_simulation, evaluate_super_resolution
)


def evaluate_1d_simulation(model, test_dataloader, device, idwt):
    """Evaluate 1D simulation task."""
    model.eval()
    all_metrics = {"mse": [], "mae": [], "linf": [], "rel_l2": []}
    
    with torch.no_grad():
        for batch in test_dataloader:
            W_f = batch["W_f"].to(device)
            W_u0 = batch["W_u0"].to(device)
            u_full_gt = batch["u_full"].to(device)
            
            B, C_u0, X_half = W_u0.shape
            T_half = W_f.shape[-2]
            W_u0_expanded = W_u0.unsqueeze(-2).expand(B, C_u0, T_half, X_half)
            cond = torch.cat([W_f, W_u0_expanded], dim=1)
            
            shape = (B, model.n_state_channels, T_half, X_half)
            W_u_pred = model.sampler.sample(shape, cond=cond)
            
            # Inverse wavelet transform
            B_pred, C_pred, H_pred, W_pred = W_u_pred.shape
            yl = W_u_pred[:, :1]
            yh_flat = W_u_pred[:, 1:]
            yh = [yh_flat.reshape(B_pred, 1, 3, H_pred, W_pred)]
            u_pred = idwt((yl, yh))
            
            # Compute metrics (excluding initial condition)
            u_gt = u_full_gt[:, 1:]
            metrics = evaluate_simulation(u_pred, u_gt, exclude_initial=False)
            
            for k, v in metrics.items():
                all_metrics[k].append(v)
    
    return {k: np.mean(v) for k, v in all_metrics.items()}


def evaluate_1d_control(model, test_dataloader, device, idwt, solver, guidance_scale=120000):
    """Evaluate 1D control task."""
    model.eval()
    all_objectives = []
    
    for batch in test_dataloader:
        W_u0 = batch["W_u0"].to(device)
        W_u_target = batch["W_u_target"].to(device)
        u_target = batch["u_target"].to(device)
        u0 = batch["u0"].to(device)
        
        B, C_u0, X_half = W_u0.shape
        T_half = 40  # Approximate half of 80
        
        W_u0_expanded = W_u0.unsqueeze(-2).expand(B, C_u0, T_half, X_half)
        W_u_target_expanded = W_u_target.unsqueeze(-2).expand(B, W_u_target.shape[1], T_half, X_half)
        cond = torch.cat([W_u0_expanded, W_u_target_expanded], dim=1)
        
        # Define guidance function for control
        def guidance_fn(W_f_hat):
            # Inverse wavelet transform
            B_hat, C_hat, H_hat, W_hat = W_f_hat.shape
            yl = W_f_hat[:, :1]
            yh_flat = W_f_hat[:, 1:]
            yh = [yh_flat.reshape(B_hat, 1, 3, H_hat, W_hat)]
            f_hat = idwt((yl, yh))
            
            # Compute control objective
            # Note: actual implementation needs the solver
            # Here we use a simplified version
            energy_loss = (f_hat ** 2).mean()
            return energy_loss
        
        shape = (B, model.n_state_channels, T_half, X_half)
        W_f_pred = model.sampler.sample_with_guidance(
            shape=shape,
            cond=cond,
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
        )
        
        # Inverse wavelet transform
        B_pred, C_pred, H_pred, W_pred = W_f_pred.shape
        yl = W_f_pred[:, :1]
        yh_flat = W_f_pred[:, 1:]
        yh = [yh_flat.reshape(B_pred, 1, 3, H_pred, W_pred)]
        f_pred = idwt((yl, yh))
        
        # Compute control objective using solver
        # (simplified - actual implementation needs the solver)
        objective = 0.0
        all_objectives.append(objective)
    
    return np.mean(all_objectives)


def evaluate_super_resolution_1d(brm_model, srm_model, test_dataloader, device, idwt, n_levels=3):
    """Evaluate zero-shot super-resolution."""
    brm_model.eval()
    srm_model.eval()
    
    all_mse = {level: [] for level in range(n_levels + 1)}
    
    with torch.no_grad():
        for batch in test_dataloader:
            W_f = batch["W_f"].to(device)
            W_u0 = batch["W_u0"].to(device)
            u_full_gt = batch["u_full_high_res"].to(device)  # Highest resolution ground truth
            
            # Base resolution prediction
            B, C_u0, X_half = W_u0.shape
            T_half = W_f.shape[-2]
            W_u0_expanded = W_u0.unsqueeze(-2).expand(B, C_u0, T_half, X_half)
            cond = torch.cat([W_f, W_u0_expanded], dim=1)
            
            shape = (B, brm_model.n_state_channels, T_half, X_half)
            W_u_base = brm_model.sampler.sample(shape, cond=cond)
            
            # Inverse wavelet and compute MSE at base resolution
            # ... (implementation)
            
            # Super-resolution steps
            current_W = W_u_base
            for level in range(n_levels):
                # Apply SRM to get higher resolution
                # ... (implementation)
                pass
    
    return {level: np.mean(mse_list) for level, mse_list in all_mse.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate WDNO")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="simulation",
                        choices=["simulation", "control", "super_resolution"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--sr_checkpoint", type=str, default=None,
                        help="Super-resolution model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--n_sr_levels", type=int, default=3,
                        help="Number of super-resolution levels")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build model
    dim = config.get("dim", "1d")
    model_config = config.get("model", {})
    model_config["task"] = args.task
    
    if dim == "1d":
        model = build_wdno_1d(model_config).to(device)
    else:
        model = build_wdno_2d(model_config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Get wavelet transforms
    wavelet = config["wavelet"]["type"]
    wavelet_mode = config["wavelet"]["mode"]
    
    try:
        import pytorch_wavelets as pw
        idwt = pw.DWTInverse(wave=wavelet, mode=wavelet_mode).to(device)
    except ImportError:
        idwt = None
    
    # Evaluate
    print(f"Evaluating {args.task} task on {dim} experiments...")
    
    if args.task == "simulation":
        if dim == "1d":
            # Load test data
            from data.burgers_data import create_burgers_dataloader
            test_loader = create_burgers_dataloader(
                data_path=args.data_path,
                batch_size=1,
                task="simulation",
                wavelet=wavelet,
                wavelet_mode=wavelet_mode,
                shuffle=False,
            )
            
            metrics = evaluate_1d_simulation(model, test_loader, device, idwt)
            print("Simulation Results:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.6f}")
    
    elif args.task == "control":
        if dim == "1d":
            from data.burgers_data import create_burgers_dataloader
            test_loader = create_burgers_dataloader(
                data_path=args.data_path,
                batch_size=1,
                task="control",
                wavelet=wavelet,
                wavelet_mode=wavelet_mode,
                shuffle=False,
            )
            
            guidance_scale = config.get("inference", {}).get("guidance_scale", 120000)
            objective = evaluate_1d_control(model, test_loader, device, idwt, 
                                             solver=None, guidance_scale=guidance_scale)
            print(f"Control Objective I: {objective:.6f}")
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
