"""
Main training entry point for WDNO.

Usage:
    # 1D Burgers simulation
    python train.py --config configs/burgers_1d.yaml --task simulation --data_path /path/to/data

    # 1D Burgers control
    python train.py --config configs/burgers_1d.yaml --task control --data_path /path/to/data

    # 2D fluid simulation
    python train.py --config configs/fluid_2d.yaml --task simulation --data_path /path/to/data
"""

import os
import sys
import math
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path

from models.wdno import WDNO1D, WDNO2D, build_wdno_1d, build_wdno_2d


def get_dataloader(config, args):
    """Get appropriate dataloader based on config."""
    dim = config.get("dim", "1d")
    task = args.task
    
    if dim == "1d":
        experiment = config.get("experiment", "burgers_1d")
        
        if experiment == "burgers_1d":
            from data.burgers_data import create_burgers_dataloader
            return create_burgers_dataloader(
                data_path=args.data_path,
                n_samples=config["data"].get("n_train", 40000),
                batch_size=config["training"].get("batch_size", 16),
                task=task,
                wavelet=config["wavelet"]["type"],
                wavelet_mode=config["wavelet"]["mode"],
            )
        elif experiment == "ns_1d":
            from data.navier_stokes import create_ns_dataloader
            return create_ns_dataloader(
                data_path=args.data_path,
                batch_size=config["training"].get("batch_size", 16),
                wavelet=config["wavelet"]["type"],
                wavelet_mode=config["wavelet"]["mode"],
            )
        else:
            raise ValueError(f"Unknown experiment: {experiment}")
    
    elif dim == "2d":
        from data.fluid_2d import create_fluid2d_dataloader
        return create_fluid2d_dataloader(
            data_path=args.data_path,
            batch_size=config["training"].get("batch_size", 16),
            task=task,
            wavelet=config["wavelet"]["type"],
            wavelet_mode=config["wavelet"]["mode"],
        )
    else:
        raise ValueError(f"Unknown dimension: {dim}")


def train_step_1d(model, batch, optimizer, device, task):
    """Single training step for 1D experiments."""
    optimizer.zero_grad()
    
    if task == "simulation":
        W_u = batch["W_u"].to(device)
        W_f = batch["W_f"].to(device)
        W_u0 = batch["W_u0"].to(device)
        
        B, C_u0, X_half = W_u0.shape
        T_half = W_f.shape[-2]
        W_u0_expanded = W_u0.unsqueeze(-2).expand(B, C_u0, T_half, X_half)
        
        cond = torch.cat([W_f, W_u0_expanded], dim=1)
        loss = model.diffusion(W_u, cond=cond)
    
    elif task == "control":
        W_f = batch["W_f"].to(device)
        W_u0 = batch["W_u0"].to(device)
        W_u_target = batch["W_u_target"].to(device)
        
        B, C_u0, X_half = W_u0.shape
        T_half = W_f.shape[-2]
        W_u0_expanded = W_u0.unsqueeze(-2).expand(B, C_u0, T_half, X_half)
        W_u_target_expanded = W_u_target.unsqueeze(-2).expand(B, W_u_target.shape[1], T_half, X_half)
        
        cond = torch.cat([W_u0_expanded, W_u_target_expanded], dim=1)
        loss = model.diffusion(W_f, cond=cond)
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


def train_step_2d(model, batch, optimizer, device, task):
    """Single training step for 2D experiments."""
    optimizer.zero_grad()
    
    if task == "simulation":
        W_state = batch["W_state"].to(device)
        W_control = batch["W_control"].to(device)
        W_density_0 = batch["W_density_0"].to(device)
        
        # Expand 2D initial condition to match 3D dimensions
        B, C_d0, H_half, W_half = W_density_0.shape
        T_half = W_control.shape[-3]
        W_density_0_expanded = W_density_0.unsqueeze(-3).expand(B, C_d0, T_half, H_half, W_half)
        
        cond = torch.cat([W_control, W_density_0_expanded], dim=1)
        loss = model.diffusion(W_state, cond=cond)
    
    elif task == "control":
        W_control = batch["W_control"].to(device)
        W_density_0 = batch["W_density_0"].to(device)
        
        B, C_d0, H_half, W_half = W_density_0.shape
        T_half = W_control.shape[-3]
        W_density_0_expanded = W_density_0.unsqueeze(-3).expand(B, C_d0, T_half, H_half, W_half)
        
        cond = W_density_0_expanded
        loss = model.diffusion(W_control, cond=cond)
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train WDNO")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="simulation",
                        choices=["simulation", "control", "super_resolution"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build model
    dim = config.get("dim", "1d")
    model_config = config.get("model", {})
    model_config["task"] = args.task
    
    if dim == "1d":
        model = build_wdno_1d(model_config).to(device)
    else:
        model = build_wdno_2d(model_config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create dataloader
    dataloader = get_dataloader(config, args)
    print(f"Training samples: {len(dataloader.dataset)}")
    
    # Setup optimizer
    train_config = config.get("training", {})
    lr = train_config.get("lr", 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Setup scheduler
    n_steps = train_config.get("n_steps", 190000)
    n_epochs = math.ceil(n_steps / len(dataloader))
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"Training for {n_epochs} epochs ({n_steps} steps)...")
    
    global_step = start_epoch * len(dataloader)
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            if dim == "1d":
                loss = train_step_1d(model, batch, optimizer, device, args.task)
            else:
                loss = train_step_2d(model, batch, optimizer, device, args.task)
            
            epoch_loss += loss
            global_step += 1
            
            if global_step >= n_steps:
                break
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.6f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}, "
                  f"Steps: {global_step}/{n_steps}")
        
        # Save checkpoint
        if epoch % 100 == 0 or epoch == n_epochs - 1 or global_step >= n_steps:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        if global_step >= n_steps:
            break
    
    print("Training complete!")


if __name__ == "__main__":
    main()
