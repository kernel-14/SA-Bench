"""
Training script for WDNO 2D experiments.

Implements training for 2D incompressible fluid and ERA5 experiments.
Uses 3D U-Net with spatial-temporal 3D convolutions.

Training details:
- Batch size: 16
- Optimizer: Adam
- Learning rate: 1e-4
- Training steps: 190,000
- Learning rate scheduler: cosine annealing
- Hardware: 2 A100 GPUs
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.wdno import WDNO2D, build_wdno_2d
from data.fluid_2d import create_fluid2d_dataloader


def train_epoch_2d(model, dataloader, optimizer, device, task="simulation"):
    """Train for one epoch on 2D data."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
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
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train WDNO 2D model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="simulation",
                        choices=["simulation", "control"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_2d")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build model
    model_config = config.get("model", {})
    model_config["task"] = args.task
    model = build_wdno_2d(model_config)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create dataloader
    train_config = config.get("training", {})
    batch_size = train_config.get("batch_size", 16)
    
    dataloader = create_fluid2d_dataloader(
        data_path=args.data_path,
        batch_size=batch_size,
        task=args.task,
        wavelet=config["wavelet"]["type"],
        wavelet_mode=config["wavelet"]["mode"],
    )
    
    # Setup optimizer
    lr = train_config.get("lr", 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
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
    
    print(f"Training for {n_epochs} epochs ({n_steps} steps)...")
    
    global_step = start_epoch * len(dataloader)
    
    for epoch in range(start_epoch, n_epochs):
        avg_loss = train_epoch_2d(model, dataloader, optimizer, device, task=args.task)
        scheduler.step()
        global_step += len(dataloader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.6f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
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
