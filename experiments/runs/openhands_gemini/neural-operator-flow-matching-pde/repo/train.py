
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
import math
import argparse
from tqdm import tqdm
import os
import datetime

from config import P2VAEConfig, FMTConfig, DataConfig, TrainingConfig, parse_args
from model import P2VAE, PDEFoundationModel
from data import get_pde_data_loader

# Helper for learning rate scheduling
def create_lr_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps)) # Linearly increase from 0 to 1
        else:
            # Cosine annealing from 1.0 down to 0.0
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def compute_kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence between a Gaussian distribution
    with parameters mu and log_var and a standard normal distribution.
    KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    Returns the sum over latent dimensions, to be averaged over batch size.
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1, 2, 3])

def train_p2vae(args):
    device = args.device
    
    # 1. Data Loader
    pde_data_loader = get_pde_data_loader(args.p2vae_batch_size, TrainingConfig.NUM_WORKERS)
    p2vae_loader_iterator = iter(pde_data_loader)

    # 2. Model
    p2vae_model = P2VAE(model_size=args.p2vae_model_size).to(device)
    p2vae_model.train()

    # 3. Optimizer
    optimizer = optim.AdamW(p2vae_model.parameters(), 
                            lr=args.p2vae_lr, 
                            betas=P2VAEConfig.BETAS, 
                            weight_decay=P2VAEConfig.WEIGHT_DECAY)

    # 4. Learning Rate Scheduler
    warmup_steps = int(args.p2vae_training_steps * 0.1)
    scheduler = create_lr_scheduler(optimizer, args.p2vae_training_steps, warmup_steps)

    print(f"Starting P2VAE training for {args.p2vae_training_steps} steps...")
    
    scaler = GradScaler() # For mixed precision training

    for step in tqdm(range(args.p2vae_training_steps)):
        optimizer.zero_grad()
        
        # Get a batch (trajectory of 4 states)
        try:
            trajectory_batch = next(p2vae_loader_iterator)
        except StopIteration:
            # Re-initialize the loader iterator if exhausted
            p2vae_loader_iterator = iter(pde_data_loader)
            trajectory_batch = next(p2vae_loader_iterator)
        
        # We only need x_0 for P2VAE training for reconstruction.
        # Pick the first state from the trajectory: [batch, T, C, H, W] -> [batch, C, H, W]
        x_0 = trajectory_batch[:, 0, :, :, :].to(device)

        with autocast():
            reconstruction, mu, log_var = p2vae_model(x_0)
            
            reconstruction_loss = nn.MSELoss(reduction='mean')(reconstruction, x_0)
            kl_loss = torch.mean(compute_kl_divergence(mu, log_var)) # Average KL over batch
            
            total_loss = reconstruction_loss + args.beta_kl_loss * kl_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % TrainingConfig.LOG_INTERVAL == 0:
            print(f"P2VAE Step [{step+1}/{args.p2vae_training_steps}], "
                  f"Loss: {total_loss.item():.4f}, "
                  f"Recon Loss: {reconstruction_loss.item():.4f}, "
                  f"KL Loss: {kl_loss.item():.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.8f}")

        # Evaluation (optional, can be added if a validation set is available in data.py)
        # if (step + 1) % TrainingConfig.EVAL_INTERVAL == 0:
        #     # Implement evaluation logic here
        #     pass

    print("P2VAE training finished.")
    return p2vae_model

def train_fmt(args, p2vae_model: P2VAE):
    device = args.device

    # Ensure P2VAE is in eval mode and frozen
    p2vae_model.eval()
    for param in p2vae_model.parameters():
        param.requires_grad = False

    # 1. Data Loader
    pde_data_loader = get_pde_data_loader(args.fmt_batch_size, TrainingConfig.NUM_WORKERS)
    fmt_loader_iterator = iter(pde_data_loader)

    # 2. Model
    model = PDEFoundationModel(p2vae_model_size=args.p2vae_model_size,
                               fmt_model_size=args.fmt_model_size).to(device)
    
    # Assign the trained P2VAE. Note that P2VAE is already in eval mode and frozen by train_p2vae.
    model.p2vae = p2vae_model 
    model.train() # Set the entire PDEFoundationModel to train mode

    # 3. Optimizer (only for FMT parameters)
    optimizer = optim.AdamW(model.fmt.parameters(), 
                            lr=args.fmt_lr,
                            betas=FMTConfig.BETAS,
                            weight_decay=FMTConfig.WEIGHT_DECAY)

    # 4. Learning Rate Scheduler
    warmup_steps = int(args.fmt_training_steps * 0.1)
    scheduler = create_lr_scheduler(optimizer, args.fmt_training_steps, warmup_steps)

    # 5. Loss Function
    cfm_loss_fn = nn.MSELoss(reduction='mean')

    print(f"Starting FMT training for {args.fmt_training_steps} steps...")

    scaler = GradScaler() # For mixed precision training

    for step in tqdm(range(args.fmt_training_steps)):
        optimizer.zero_grad()

        # Get a batch (trajectory of 4 states: x_0, x_1, x_2, x_3)
        try:
            trajectory_batch = next(fmt_loader_iterator) # [batch, T, C, H, W]
        except StopIteration:
            fmt_loader_iterator = iter(pde_data_loader)
            trajectory_batch = next(fmt_loader_iterator)
        
        trajectory_batch = trajectory_batch.to(device)
        batch_size, T, C, H, W = trajectory_batch.shape

        total_cfm_loss = 0.0
        
        # Initialize h_prev for the first physical step (s=0 to predict x_1 from x_0)
        h_prev = model.fmt.get_initial_h_state(batch_size).to(device) # [batch_size, embed_dim]

        # Loop through physical timesteps within the trajectory
        # Paper uses 4 states (x_0, x_1, x_2, x_3)
        # So we predict (x_1 from x_0), (x_2 from x_1), (x_3 from x_2)
        # This corresponds to s = 0, 1, 2. (T-1 predictions)
        for s in range(T - 1): 
            x_s = trajectory_batch[:, s, :, :, :]       # Current physical state
            x_s_plus_1 = trajectory_batch[:, s+1, :, :, :] # Next physical state

            # Sample t_s and k_s independently for each physical timestep
            t_s = torch.rand(batch_size, device=device) * (FMTConfig.T_UNIFORM_MAX - FMTConfig.T_UNIFORM_MIN) + FMTConfig.T_UNIFORM_MIN
            k_s = torch.rand(batch_size, device=device) * (FMTConfig.K_UNIFORM_MAX - FMTConfig.K_UNIFORM_MIN) + FMTConfig.K_UNIFORM_MIN

            # Forward pass through PDEFoundationModel to get predictions
            with autocast():
                # predicted_velocity is (1-t)g_theta, target_residual is (x_1 - x_t_k)
                predicted_velocity_scaled, target_residual, z_s_t_k = model(
                    x_0=x_s, # Current physical state for interpolation base
                    x_1=x_s_plus_1, # Next physical state for interpolation target
                    t=t_s, # Flow matching t for current physical step
                    k=k_s, # Flow matching k for current physical step
                    h_prev=h_prev # Condition from previous physical step
                )

                # The L_CFM in the paper is:
                # L_CFM = 1/2 E[ || (1 - t_s) g_theta(x_s,t_s^k_s, t_s, h_s-1) - (x_s+1 - x_s,t_s^k_s) ||^2 ]
                # Our `predicted_velocity_scaled` is `(1 - t_s) g_theta`.
                # Our `target_residual` is `(x_s+1 - x_s,t_s^k_s)`.
                step_loss = cfm_loss_fn(predicted_velocity_scaled, target_residual)
                total_cfm_loss += step_loss
            
            # Update h_prev for the next physical step
            # The paper says h_s = p_phi(h_s-1, x_s,t_s^k_s, t_s)
            h_prev = model.fmt.update_diffusion_forcing_gru(h_prev, z_s_t_k, t_s)

        scaler.scale(total_cfm_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if (step + 1) % TrainingConfig.LOG_INTERVAL == 0:
            print(f"FMT Step [{step+1}/{args.fmt_training_steps}], "
                  f"Total CFM Loss: {total_cfm_loss.item():.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.8f}")

    print("FMT training finished.")
    # save_model(model.fmt, "fmt_final.pth") # Placeholder for saving
    return model.fmt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    # Set device
    if torch.cuda.is_available() and args.device == "cuda":
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    print(f"Using device: {args.device}")

    # Stage 1: Train P2VAE
    print("\n--- Stage 1: Training P2VAE ---")
    p2vae_model = train_p2vae(args)
    
    # Stage 2: Train FMT
    print("\n--- Stage 2: Training FMT ---")
    fmt_model = train_fmt(args, p2vae_model)

    print("\nTraining complete!")

if __name__ == "__main__":
    main()

