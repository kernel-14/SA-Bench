"""
Stage 1 training: P2VAE (Pretrained Physics Variational Autoencoder).

Training details from the paper:
  - AdamW optimizer: β1=0.9, β2=0.995, weight_decay=1e-4
  - Cosine LR schedule with 10% linear warmup
  - Base LR=1e-4 for batch_size=256 (scaled by sqrt(batch/256))
  - KL weight β=1e-3
  - 100k steps
  - 4 H-100 GPUs (DDP)
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import DatasetConfig, P2VAEConfig, TrainConfig, get_p2vae_config
from data import PDEDataset, UniformPDEDataset, build_dataset, collate_traj
from model import P2VAE


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, max_steps: int, base_lr: float, warmup_frac: float) -> float:
    warmup_steps = int(max_steps * warmup_frac)
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def scale_lr(base_lr: float, batch_size: int, base_batch: int = 256) -> float:
    """Linear LR scaling with batch size."""
    return base_lr * batch_size / base_batch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    # Config
    cfg: P2VAEConfig = get_p2vae_config(args.model_size)
    data_cfg = DatasetConfig(root_dir=args.data_dir)
    train_cfg = TrainConfig()

    torch.manual_seed(train_cfg.seed + local_rank)

    # Model
    model = P2VAE(cfg).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if is_distributed else model

    # Optimizer
    lr = scale_lr(cfg.lr, args.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # Data
    train_dataset = build_dataset(data_cfg, split="train")
    val_dataset = build_dataset(data_cfg, split="val")

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size // world_size,
            sampler=train_sampler,
            num_workers=data_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_traj,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_traj,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_traj,
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.mixed_precision)

    # Logging
    if is_main and WANDB_AVAILABLE and args.wandb:
        wandb.init(
            project=train_cfg.wandb_project,
            name=f"p2vae_{args.model_size}",
            config={"model_size": args.model_size, "batch_size": args.batch_size},
        )

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    # Training loop
    step = 0
    data_iter = iter(train_loader)

    while step < cfg.max_steps:
        model.train()

        # Fetch batch (cycle through dataloader)
        try:
            frames = next(data_iter)
        except StopIteration:
            if is_distributed:
                train_sampler.set_epoch(step)
            data_iter = iter(train_loader)
            frames = next(data_iter)

        # Use only the first frame for VAE training (single-frame reconstruction)
        x = frames[0].to(device, non_blocking=True)

        # Update LR
        current_lr = get_lr(step, cfg.max_steps, lr, cfg.warmup_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=train_cfg.mixed_precision):
            loss, info = raw_model.loss(x)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        # Logging
        if is_main and step % train_cfg.log_every == 0:
            log_str = (
                f"Step {step}/{cfg.max_steps} | "
                f"loss={loss.item():.4f} | "
                f"recon={info['recon'].item():.4f} | "
                f"kl={info['kl'].item():.4f} | "
                f"lr={current_lr:.2e}"
            )
            print(log_str)
            if WANDB_AVAILABLE and args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/recon": info["recon"].item(),
                    "train/kl": info["kl"].item(),
                    "train/lr": current_lr,
                    "step": step,
                })

        # Validation
        if is_main and step % train_cfg.eval_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_frames in val_loader:
                    x_val = val_frames[0].to(device)
                    with torch.cuda.amp.autocast(enabled=train_cfg.mixed_precision):
                        val_loss, _ = raw_model.loss(x_val)
                    val_losses.append(val_loss.item())
                    if len(val_losses) >= 50:
                        break
            val_loss_mean = sum(val_losses) / len(val_losses)
            print(f"  [Val] step={step} val_loss={val_loss_mean:.4f}")
            if WANDB_AVAILABLE and args.wandb:
                wandb.log({"val/loss": val_loss_mean, "step": step})

        # Checkpoint
        if is_main and step % train_cfg.save_every == 0:
            ckpt_path = os.path.join(
                train_cfg.checkpoint_dir, f"p2vae_{args.model_size}_step{step}.pt"
            )
            torch.save({
                "step": step,
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    if is_main:
        ckpt_path = os.path.join(
            train_cfg.checkpoint_dir, f"p2vae_{args.model_size}_final.pt"
        )
        torch.save({
            "step": step,
            "model_state": raw_model.state_dict(),
            "cfg": cfg,
        }, ckpt_path)
        print(f"Training complete. Final checkpoint: {ckpt_path}")

    if is_distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train P2VAE")
    parser.add_argument("--model_size", type=str, default="16M", choices=["16M", "87M"])
    parser.add_argument("--data_dir", type=str, default="/data/pde")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.max_steps is not None:
        # Allow CLI override
        cfg = get_p2vae_config(args.model_size)
        cfg.max_steps = args.max_steps

    train(args)
