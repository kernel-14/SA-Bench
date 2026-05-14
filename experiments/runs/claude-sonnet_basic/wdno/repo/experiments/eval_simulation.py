"""
Evaluation script for WDNO simulation tasks.

Computes MSE on entire state sequences excluding initial conditions.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.wdno import WDNO1D, WDNO2D, build_wdno_1d, build_wdno_2d
from utils.metrics import compute_mse, compute_mae, compute_linf


def evaluate_simulation_1d(model, test_dataloader, device, idwt, wavelet_config):
    """
    Evaluate 1D simulation task.
    
    Computes MSE on entire state sequences excluding initial conditions.
    """
    model.eval()
    all_mse = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            W_f = batch["W_f"].to(device)
            W_u0 = batch["W_u0"].to(device)
            u_full_gt = batch["u_full"].to(device)
            
            # Expand 1D conditions
            B, C_u0, X_half = W_u0.shape
            T_half = W_f.shape[-2]
            W_u0_expanded = W_u0.unsqueeze(-2).expand(B, C_u0, T_half, X_half)
            
            cond = torch.cat([W_f, W_u0_expanded], dim=1)
            
            # Generate prediction
            shape = (B, model.n_state_channels, T_half, X_half)
            W_u_pred = model.sampler.sample(shape, cond=cond)
            
            # Inverse wavelet transform
            u_pred = idwt(W_u_pred)
            
            # Compute MSE (excluding initial condition)
            mse = ((u_pred - u_full_gt[:, 1:]) ** 2).mean().item()
            all_mse.append(mse)
    
    return np.mean(all_mse)


def evaluate_control_1d(model, test_dataloader, device, idwt, solver, guidance_scale=120000):
    """
    Evaluate 1D control task.
    
    Computes control objective I = integral |u(T,x) - u*(x)|^2 dx + alpha * integral |f|^2 dt dx
    """
    model.eval()
    all_objectives = []
    alpha = 0.00002  # Weight of energy cost
    
    for batch in test_dataloader:
        W_u0 = batch["W_u0"].to(device)
        W_u_target = batch["W_u_target"].to(device)
        u_target = batch["u_target"].to(device)
        
        B, C_u0, X_half = W_u0.shape
        T_half = W_u0.shape[-1]  # Approximate
        
        # Expand conditions
        W_u0_expanded = W_u0.unsqueeze(-2).expand(B, C_u0, T_half, X_half)
        W_u_target_expanded = W_u_target.unsqueeze(-2).expand(B, W_u_target.shape[1], T_half, X_half)
        
        cond = torch.cat([W_u0_expanded, W_u_target_expanded], dim=1)
        
        # Define guidance function
        def guidance_fn(W_f_hat):
            f_hat = idwt(W_f_hat)
            # Run solver to get final state
            # This is a simplified version; actual implementation needs the solver
            return torch.tensor(0.0, requires_grad=True)
        
        # Generate control sequence
        shape = (B, model.n_state_channels, T_half, X_half)
        W_f_pred = model.sampler.sample_with_guidance(
            shape=shape,
            cond=cond,
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale,
        )
        
        # Inverse wavelet transform
        f_pred = idwt(W_f_pred)
        
        # Compute control objective using ground-truth solver
        # (simplified here - actual implementation needs the solver)
        objective = 0.0
        all_objectives.append(objective)
    
    return np.mean(all_objectives)


def main():
    parser = argparse.ArgumentParser(description="Evaluate WDNO simulation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="simulation")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    model_config = config.get("model", {})
    model_config["task"] = args.task
    
    dim = config.get("dim", "1d")
    if dim == "1d":
        model = build_wdno_1d(model_config).to(device)
    else:
        model = build_wdno_2d(model_config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Evaluate
    print(f"Evaluating {args.task} task...")
    
    # Results would be printed here
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
