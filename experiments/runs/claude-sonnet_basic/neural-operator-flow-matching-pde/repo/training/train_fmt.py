"""
Training script for FMT (Flow Marching Transformer).

Training details from the paper:
- AdamW optimizer with beta1=0.9, beta2=0.95
- Cosine learning rate schedule with 10% linear warmup
- Weight decay: 0.01
- Base learning rate: 1e-4 for batch size 256
- Training steps: 100k
- Based on frozen P2VAE-16M latents
- Three variants: FMT-S (6M), FMT-B (42M), FMT-L (138M)
- Input: 4 consecutive latent states (c16p16)
- Temporal pyramid: Down(y0, 8), Down(y1, 4), Down(y2, 2), y3

Conditional flow marching objective:
    L_CFM = 0.5 * E[||(1-t)*g(x_t^k, t, h) - (x_{s+1} - x_{s,t}^k)||^2]

where x_{s,t}^k is the noisy interpolation between x_s and x_{s+1}.
"""

import os
import math
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.p2vae import P2VAE, P2VAE_16M
from models.fmt import FlowMarchingTransformer, FMTSmall, FMTBase, FMTLarge
from models.flow_marching import sample_interpolation, flow_marching_loss
from data.pde_dataset import PDEDataset, create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def scale_lr(base_lr: float, batch_size: int, base_batch_size: int = 256) -> float:
    """Scale learning rate linearly with batch size."""
    return base_lr * batch_size / base_batch_size


def compute_cfm_loss(
    fmt_model: FlowMarchingTransformer,
    latents: list,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute conditional flow marching loss for a batch of 4 consecutive latent states.

    The training objective is:
        L_CFM = 0.5 * E[||(1-t_s)*g(x_{s,t_s}^{k_s}, t_s, h_{s-1}) - (x_{s+1} - x_{s,t_s}^{k_s})||^2]

    where x_{s,t_s}^{k_s} = x_s + t_s*(x_{s+1}-x_s) - (1-t_s)*(1-k_s)*(x_s - z)

    The model takes 4 noisy states as input (temporal pyramid) and predicts the
    velocity for the last frame. The loss is computed for the last frame only.

    Args:
        fmt_model: FlowMarchingTransformer
        latents: list of 4 latent tensors (B, C, H, W) [x0, x1, x2, x3]
        device: computation device

    Returns:
        loss: scalar loss
    """
    B = latents[0].shape[0]

    # Sample independent t and k for each physical timestep
    t_values = torch.rand(B, 4, device=device)
    k_values = torch.rand(B, 4, device=device)

    # Build noisy latent states for each frame using the interpolation kernel:
    # x_{s,t_s}^{k_s} = mu_t + sigma_t * z
    # mu_t = t_s * x_{s+1} + k_s * (1-t_s) * x_s
    # sigma_t = (1-t_s) * (1-k_s)
    noisy_latents = []
    for s in range(4):
        if s < 3:
            # Interpolate between x_s and x_{s+1}
            x0 = latents[s]
            x1 = latents[s + 1]
        else:
            # For the last frame (s=3), interpolate between x2 and x3
            # This represents the noisy version of x3 given context x2
            x0 = latents[s - 1]  # x2
            x1 = latents[s]      # x3

        t_s = t_values[:, s]
        k_s = k_values[:, s]

        x_noisy, _, _ = sample_interpolation(x0, x1, t_s, k_s)
        noisy_latents.append(x_noisy)

    # Forward pass: predict velocity for the target frame (frame 3)
    # The model uses temporal pyramid: all 4 noisy frames as input
    # and predicts velocity for the last frame
    t_target = t_values[:, -1]
    velocity_pred, _ = fmt_model(noisy_latents, t_target, t_all=t_values)

    # Compute loss for the target frame
    # The noisy input is noisy_latents[-1] = x_{3,t3}^{k3}
    # The target is x3 (clean)
    # Loss: ||(1-t3)*g - (x3 - x_{3,t3}^{k3})||^2
    x_noisy_target = noisy_latents[-1]  # Use the same noisy state that was input to the model
    x1_target = latents[-1]             # x3 (clean target)

    loss = flow_marching_loss(velocity_pred, x_noisy_target, x1_target, t_target)

    return loss


def train_fmt(args):
    """Main training function for FMT."""

    # Setup distributed training
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    is_main = local_rank == 0

    # Load frozen P2VAE
    vae = P2VAE_16M(in_channels=3, latent_channels=16)
    if args.vae_checkpoint:
        checkpoint = torch.load(args.vae_checkpoint, map_location="cpu")
        vae.load_state_dict(checkpoint["model_state_dict"])
        if is_main:
            logger.info(f"Loaded P2VAE from {args.vae_checkpoint}")
    vae = vae.to(device)
    vae.eval()
    # Freeze VAE weights
    for param in vae.parameters():
        param.requires_grad = False

    # Create FMT model
    if args.model_size == "S":
        fmt = FMTSmall(latent_channels=16, latent_size=16)
    elif args.model_size == "B":
        fmt = FMTBase(latent_channels=16, latent_size=16)
    elif args.model_size == "L":
        fmt = FMTLarge(latent_channels=16, latent_size=16)
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    fmt = fmt.to(device)

    if is_main:
        num_params = sum(p.numel() for p in fmt.parameters())
        logger.info(f"FMT-{args.model_size} parameters: {num_params:,}")

    if args.distributed:
        fmt = DDP(fmt, device_ids=[local_rank])

    # Scale learning rate
    lr = scale_lr(args.base_lr, args.batch_size)

    # Optimizer (only FMT parameters, VAE is frozen)
    optimizer = AdamW(
        fmt.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # Learning rate scheduler
    num_warmup_steps = int(0.1 * args.num_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.num_steps,
    )

    # Data loading
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=args.distributed,
        trajectory_length=4,  # Need 4 consecutive frames
    )

    # Training loop
    step = 0
    fmt.train()

    if is_main:
        logger.info(f"Starting FMT-{args.model_size} training for {args.num_steps} steps")

    train_iter = iter(train_loader)

    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # batch shape: (B, T, C, H, W) with T=4
        batch = batch.to(device)
        B, T, C, H, W = batch.shape
        assert T == 4, f"Expected 4 frames, got {T}"

        # Encode frames to latent space (frozen VAE)
        with torch.no_grad():
            latents = []
            for t_idx in range(T):
                z = vae.get_latent(batch[:, t_idx], deterministic=True)
                latents.append(z)

        # Compute conditional flow marching loss
        fmt_model = fmt.module if args.distributed else fmt
        loss = compute_cfm_loss(fmt_model, latents, device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fmt.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        if is_main and step % args.log_every == 0:
            logger.info(f"Step {step}/{args.num_steps}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        if is_main and step % args.save_every == 0:
            save_path = Path(args.output_dir) / f"fmt_{args.model_size}_step{step}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": fmt.module.state_dict() if args.distributed else fmt.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
            }, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    if is_main:
        # Save final model
        save_path = Path(args.output_dir) / f"fmt_{args.model_size}_final.pt"
        torch.save({
            "step": step,
            "model_state_dict": fmt.module.state_dict() if args.distributed else fmt.state_dict(),
            "args": vars(args),
        }, save_path)
        logger.info(f"Training complete. Final model saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FMT")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--vae_checkpoint", type=str, default=None, help="Path to P2VAE checkpoint")
    parser.add_argument("--model_size", type=str, default="B", choices=["S", "B", "L"])
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--base_lr", type=float, default=1e-4, help="Base learning rate for batch_size=256")
    parser.add_argument("--num_steps", type=int, default=100_000, help="Number of training steps")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=10_000, help="Save checkpoint every N steps")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fmt(args)
