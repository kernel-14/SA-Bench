"""
Stage 2 training: FMT (Flow Marching Transformer).

Training details from the paper:
  - Requires a frozen P2VAE checkpoint (Stage 1 output)
  - AdamW optimizer: β1=0.9, β2=0.95, weight_decay=0.01
  - Cosine LR schedule with 10% linear warmup
  - Base LR=1e-4 for batch_size=256 (scaled linearly)
  - 100k steps
  - 4 H-100 GPUs (DDP)
  - Trains FMT-S (6M), FMT-B (42M), FMT-L (138M)

The FMT is trained on latent representations from the frozen P2VAE.
The conditional flow marching loss (Eq. 11) is computed over 4 consecutive
latent frames with independently sampled t_s, k_s per step.
"""

import argparse
import math
import os
from typing import List

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

from config import DatasetConfig, FMTConfig, TrainConfig, get_fmt_config, get_p2vae_config
from data import build_dataset, collate_traj
from model import FMT, P2VAE


# ---------------------------------------------------------------------------
# LR schedule (same helper as train_vae.py)
# ---------------------------------------------------------------------------

def get_lr(step: int, max_steps: int, base_lr: float, warmup_frac: float) -> float:
    warmup_steps = int(max_steps * warmup_frac)
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def scale_lr(base_lr: float, batch_size: int, base_batch: int = 256) -> float:
    return base_lr * batch_size / base_batch


# ---------------------------------------------------------------------------
# Load frozen P2VAE
# ---------------------------------------------------------------------------

def load_frozen_vae(ckpt_path: str, device: torch.device) -> P2VAE:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    vae = P2VAE(cfg).to(device)
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


# ---------------------------------------------------------------------------
# Encode a batch of frame sequences to latent space
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_frames(
    vae: P2VAE,
    frames: List[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    """Encode a list of raw frames to latent means."""
    return [vae.encode_deterministic(f.to(device)) for f in frames]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    # Config
    fmt_cfg: FMTConfig = get_fmt_config(args.model_size)
    data_cfg = DatasetConfig(root_dir=args.data_dir)
    train_cfg = TrainConfig()

    torch.manual_seed(train_cfg.seed + local_rank)

    # Frozen VAE
    vae = load_frozen_vae(args.vae_ckpt, device)
    if is_main:
        print(f"Loaded frozen P2VAE from {args.vae_ckpt}")

    # FMT model
    model = FMT(fmt_cfg).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if is_distributed else model

    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        print(f"FMT-{args.model_size}: {n_params / 1e6:.1f}M parameters")

    # Optimizer
    lr = scale_lr(fmt_cfg.lr, args.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(fmt_cfg.beta1, fmt_cfg.beta2),
        weight_decay=fmt_cfg.weight_decay,
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

    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.mixed_precision)

    if is_main and WANDB_AVAILABLE and args.wandb:
        wandb.init(
            project=train_cfg.wandb_project,
            name=f"fmt_{args.model_size}",
            config={"model_size": args.model_size, "batch_size": args.batch_size},
        )

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    step = 0
    data_iter = iter(train_loader)

    while step < fmt_cfg.max_steps:
        model.train()

        try:
            frames = next(data_iter)
        except StopIteration:
            if is_distributed:
                train_sampler.set_epoch(step)
            data_iter = iter(train_loader)
            frames = next(data_iter)

        # Encode frames to latent space (frozen VAE, no grad)
        y_seq = encode_frames(vae, frames, device)

        # Update LR
        current_lr = get_lr(step, fmt_cfg.max_steps, lr, fmt_cfg.warmup_frac)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=train_cfg.mixed_precision):
            loss, info = raw_model.compute_cfm_loss(y_seq)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), fmt_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if is_main and step % train_cfg.log_every == 0:
            step_losses = {
                k: v.item() for k, v in info.items() if k.startswith("loss_step")
            }
            log_str = (
                f"Step {step}/{fmt_cfg.max_steps} | "
                f"loss={loss.item():.4f} | "
                f"lr={current_lr:.2e}"
            )
            print(log_str)
            if WANDB_AVAILABLE and args.wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    **{f"train/{k}": v for k, v in step_losses.items()},
                    "step": step,
                })

        if is_main and step % train_cfg.eval_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_frames in val_loader:
                    y_val = encode_frames(vae, val_frames, device)
                    with torch.cuda.amp.autocast(enabled=train_cfg.mixed_precision):
                        val_loss, _ = raw_model.compute_cfm_loss(y_val)
                    val_losses.append(val_loss.item())
                    if len(val_losses) >= 50:
                        break
            val_loss_mean = sum(val_losses) / len(val_losses)
            print(f"  [Val] step={step} val_loss={val_loss_mean:.4f}")
            if WANDB_AVAILABLE and args.wandb:
                wandb.log({"val/loss": val_loss_mean, "step": step})

        if is_main and step % train_cfg.save_every == 0:
            ckpt_path = os.path.join(
                train_cfg.checkpoint_dir,
                f"fmt_{args.model_size}_step{step}.pt",
            )
            torch.save({
                "step": step,
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": fmt_cfg,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    if is_main:
        ckpt_path = os.path.join(
            train_cfg.checkpoint_dir, f"fmt_{args.model_size}_final.pt"
        )
        torch.save({
            "step": step,
            "model_state": raw_model.state_dict(),
            "cfg": fmt_cfg,
        }, ckpt_path)
        print(f"Training complete. Final checkpoint: {ckpt_path}")

    if is_distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FMT")
    parser.add_argument("--model_size", type=str, default="B", choices=["S", "B", "L"])
    parser.add_argument("--vae_ckpt", type=str, required=True,
                        help="Path to frozen P2VAE checkpoint")
    parser.add_argument("--data_dir", type=str, default="/data/pde")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    train(args)
