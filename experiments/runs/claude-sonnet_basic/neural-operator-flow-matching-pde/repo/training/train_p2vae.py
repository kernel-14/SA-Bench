"""
Training script for P2VAE (Pretrained Physics Variational Autoencoder).

Training details from the paper:
- AdamW optimizer with beta1=0.9, beta2=0.995
- Cosine learning rate schedule with 10% linear warmup
- Weight decay: 1e-4
- Base learning rate: 1e-4 for batch size 256
- KL weight: beta = 1e-3
- Training steps: 100k
- Two variants: P2VAE-16M (base_dim=64) and P2VAE-87M (base_dim=128)
- Input: c3p128 (3 channels, 128x128 resolution)
- Output latent: c16p16 (16 channels, 16x16 resolution) -> 12x compression
"""

import os
import math
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.p2vae import P2VAE, P2VAE_16M, P2VAE_87M
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


def train_p2vae(args):
    """Main training function for P2VAE."""

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

    # Create model
    if args.model_size == "16M":
        model = P2VAE_16M(
            in_channels=args.in_channels,
            latent_channels=args.latent_channels,
            kl_weight=args.kl_weight,
        )
    elif args.model_size == "87M":
        model = P2VAE_87M(
            in_channels=args.in_channels,
            latent_channels=args.latent_channels,
            kl_weight=args.kl_weight,
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    model = model.to(device)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"P2VAE-{args.model_size} parameters: {num_params:,}")

    if args.distributed:
        model = DDP(model, device_ids=[local_rank])

    # Scale learning rate
    lr = scale_lr(args.base_lr, args.batch_size)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.995),
        weight_decay=1e-4,
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
    )

    # Training loop
    step = 0
    model.train()

    if is_main:
        logger.info(f"Starting P2VAE training for {args.num_steps} steps")

    train_iter = iter(train_loader)

    while step < args.num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Get a single frame from the trajectory (any frame works for VAE training)
        # batch shape: (B, T, C, H, W) or (B, C, H, W)
        if batch.dim() == 5:
            # Randomly select a frame from the trajectory
            t_idx = torch.randint(0, batch.shape[1], (1,)).item()
            x = batch[:, t_idx].to(device)
        else:
            x = batch.to(device)

        # Forward pass
        x_hat, loss = model(x)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        if is_main and step % args.log_every == 0:
            logger.info(f"Step {step}/{args.num_steps}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        if is_main and step % args.save_every == 0:
            save_path = Path(args.output_dir) / f"p2vae_{args.model_size}_step{step}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": step,
                "model_state_dict": model.module.state_dict() if args.distributed else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
            }, save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    if is_main:
        # Save final model
        save_path = Path(args.output_dir) / f"p2vae_{args.model_size}_final.pt"
        torch.save({
            "step": step,
            "model_state_dict": model.module.state_dict() if args.distributed else model.state_dict(),
            "args": vars(args),
        }, save_path)
        logger.info(f"Training complete. Final model saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train P2VAE")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--model_size", type=str, default="16M", choices=["16M", "87M"])
    parser.add_argument("--in_channels", type=int, default=3, help="Input channels (c3)")
    parser.add_argument("--latent_channels", type=int, default=16, help="Latent channels (c16)")
    parser.add_argument("--kl_weight", type=float, default=1e-3, help="KL divergence weight beta")
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
    train_p2vae(args)
