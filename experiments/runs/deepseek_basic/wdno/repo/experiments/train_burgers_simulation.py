"""
Training script for 1D Burgers' Equation Simulation using WDNO.

Reproduces the results from Table 1 (1D Burgers' column).
Uses Base-Resolution Model (BRM) only.

Hyperparameters from Table 18:
- UNet: init_dim=128, 4 down/up layers, kernel_size=3, dim_mult=[1,2,4,8]
- Resnet groups: 8, Attention: hidden_dim=32, heads=4
- Batch size: 16, Optimizer: Adam, LR: 1e-4
- Training steps: 190000, LR scheduler: cosine annealing
- DDIM sampling: 50 iterations, eta=1
- Wavelet: bior2.4, mode: periodization
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wdno import WDNOSimulation
from utils.data_generation import generate_burgers_initial_condition, generate_burgers_control, solve_burgers_fdm


def create_burgers_dataset(n_samples: int, n_time: int = 81, n_space: int = 120):
    """Create training dataset for 1D Burgers' simulation."""
    u0_all = generate_burgers_initial_condition(n_space, n_samples)
    f_all = generate_burgers_control(n_time - 1, n_space, n_samples)
    u_all = solve_burgers_fdm(u0_all, f_all, n_time=n_time - 1, n_space=n_space)

    # Data shape for WDNO simulation:
    # - data: u_{[0,T]} shape (N, 1, 81, 120) [time × space as 2D image]
    # - condition: [u0, f] concatenated as (N, 2, 81, 120)
    #   where u0 is repeated across time dim

    data = u_all.unsqueeze(1)  # (N, 1, 81, 120)

    # Create condition: broadcast u0 to match time dimension
    condition_u0 = u0_all.unsqueeze(1).unsqueeze(-1).expand(-1, 1, n_time, n_space)
    condition_f = f_all.unsqueeze(1)  # (N, 1, 80, 120) - pad to 81
    condition_f = torch.cat([
        condition_f,
        condition_f[:, :, -1:, :]
    ], dim=2)  # pad last time step
    condition = torch.cat([condition_u0, condition_f], dim=1)  # (N, 2, 81, 120)

    return data, condition


def train_burgers_simulation(
    n_samples: int = 40000,
    batch_size: int = 16,
    n_epochs: int = 100,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = './checkpoints',
):
    """Train WDNO for 1D Burgers' simulation."""
    os.makedirs(save_dir, exist_ok=True)

    print("Creating dataset...")
    data, condition = create_burgers_dataset(n_samples, n_time=81, n_space=120)

    # Split train/val
    n_train = int(n_samples * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]
    train_cond, val_cond = condition[:n_train], condition[n_train:]

    print(f"Train: {n_train}, Val: {n_samples - n_train}")

    # Initialize WDNO Simulation model
    model = WDNOSimulation(
        data_shape=(81, 120),  # (T, X)
        cond_shape=(2, 81, 120),  # (C_cond, T, X)
        wavelet_type='bior2.4',
        wavelet_mode='periodization',
        n_channels=1,
        n_cond_channels=2,
        timesteps=1000,
        init_dim=128,
        dim_mult=[1, 2, 4, 8],
        resnet_groups=8,
        attn_heads=4,
        attn_dim_head=32,
        ddim_steps=50,
        ddim_eta=1.0,
        is_3d=False,
        learning_rate=lr,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs * (n_train // batch_size))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    steps_per_epoch = n_train // batch_size
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        # Shuffle indices
        perm = torch.randperm(n_train)

        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{n_epochs}"):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            batch_data = train_data[idx].to(device)
            batch_cond = train_cond[idx].to(device)

            loss = model.training_step({
                'data': batch_data,
                'condition': batch_cond
            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}")

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_batch_data = val_data[:batch_size].to(device)
                val_batch_cond = val_cond[:batch_size].to(device)

                val_loss = model.training_step({
                    'data': val_batch_data,
                    'condition': val_batch_cond
                })
                print(f"  Val Loss = {val_loss.item():.6f}")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'wavelet_type': 'bior2.4',
            'wavelet_mode': 'periodization',
            'data_shape': (81, 120),
            'cond_shape': (2, 81, 120),
            'init_dim': 128,
            'dim_mult': [1, 2, 4, 8],
        }
    }, os.path.join(save_dir, 'burgers_simulation_final.pt'))

    print("Training complete!")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    train_burgers_simulation(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )
