"""
Training script for 1D Burgers' Equation Control using WDNO.

Reproduces the results from Table 2a.
Uses energy-guided diffusion for control optimization.

Control objective (Eq. 6):
    I = ∫|u(T,x) - u*(x)|²dx + α∫|f|²dtdx

Hyperparameters from Table 18:
- Same UNet architecture as simulation
- Guidance weight λ = 120000
- Cosine guidance schedule
- DDIM: 50 iterations, eta=1
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

from wdno import WDNOControl
from utils.data_generation import (
    generate_burgers_initial_condition,
    generate_burgers_control,
    solve_burgers_fdm,
    burgers_control_objective,
)


def create_burgers_control_dataset(n_samples: int, n_time: int = 81, n_space: int = 120):
    """Create training dataset for 1D Burgers' control."""
    u0_all = generate_burgers_initial_condition(n_space, n_samples)
    f_all = generate_burgers_control(n_time - 1, n_space, n_samples)
    u_all = solve_burgers_fdm(u0_all, f_all, n_time=n_time - 1, n_space=n_space)

    # Data shape:
    # - data: control sequence f shape (N, 1, 80, 120)
    # - condition: [u0, u_target] — we use last state as target for training

    data = f_all.unsqueeze(1)  # (N, 1, 80, 120) - pad to 81
    data = torch.cat([data, data[:, :, -1:, :]], dim=2)  # (N, 1, 81, 120)

    # Condition: initial condition u0 + target state u_T
    u_target = u_all[:, -1, :]  # (N, 120)
    condition_u0 = u0_all.unsqueeze(1).unsqueeze(-1).expand(-1, 1, n_time, n_space)
    condition_uT = u_target.unsqueeze(1).unsqueeze(-1).expand(-1, 1, n_time, n_space)
    condition = torch.cat([condition_u0, condition_uT], dim=1)  # (N, 2, 81, 120)

    return data, condition, u0_all, u_all


def train_burgers_control(
    n_samples: int = 40000,
    batch_size: int = 16,
    n_epochs: int = 100,
    lr: float = 1e-4,
    guidance_weight: float = 120000.0,
    device: str = 'cuda',
    save_dir: str = './checkpoints',
):
    """Train WDNO for 1D Burgers' control."""
    os.makedirs(save_dir, exist_ok=True)

    print("Creating dataset...")
    data, condition, u0_all, u_all = create_burgers_control_dataset(n_samples)

    n_train = int(n_samples * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]
    train_cond, val_cond = condition[:n_train], condition[n_train:]

    print(f"Train: {n_train}, Val: {n_samples - n_train}")

    # Define objective function for guidance
    def control_objective(f_pred, cond):
        """Compute control objective I for guidance."""
        # f_pred: (B, 1, 81, 120) in original space
        f = f_pred[:, 0, :80, :]  # Remove padded last step
        u_target = cond[:, 1, 0, :]  # Target from condition
        u0 = cond[:, 0, 0, :]  # Initial condition

        # Simulate with solver to get u_T
        u_T = solve_burgers_fdm(u0, f, n_time=80, n_space=120)[:, -1, :]  # (B, 120)

        return burgers_control_objective(u_T, f, u_target).mean()

    # Initialize WDNO Control model
    model = WDNOControl(
        objective_fn=control_objective,
        guidance_weight=guidance_weight,
        data_shape=(81, 120),
        cond_shape=(2, 81, 120),
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

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'wavelet_type': 'bior2.4',
            'wavelet_mode': 'periodization',
            'guidance_weight': guidance_weight,
        }
    }, os.path.join(save_dir, 'burgers_control_final.pt'))

    print("Training complete!")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--guidance_weight', type=float, default=120000.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    train_burgers_control(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        guidance_weight=args.guidance_weight,
        device=args.device,
        save_dir=args.save_dir,
    )
