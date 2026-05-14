"""
Training loop for Hi-MAR.

Supports:
  - Class-conditional generation on ImageNet 256x256
  - Text-to-image generation on MS-COCO 256x256
  - Distributed training (DDP)
  - Mixed precision (bfloat16)
  - EMA (for MS-COCO)
  - Classifier-free guidance training

Usage:
  # Single GPU
  python train.py --dataset imagenet --model Hi-MAR-B --data_path /data/imagenet

  # Multi-GPU (8 GPUs)
  torchrun --nproc_per_node=8 train.py --dataset imagenet --model Hi-MAR-B --data_path /data/imagenet
"""

import argparse
import copy
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from config import (
    Config,
    ModelConfig,
    TrainConfig,
    coco_small_config,
    imagenet_base_config,
    imagenet_huge_config,
    imagenet_large_config,
)
from data import (
    VAETokenizer,
    build_coco_loaders,
    build_imagenet_loaders,
    create_mask,
    sample_mask_ratio_beta,
    sample_mask_ratio_cosine,
    sample_mask_ratio_phase1,
)
from model import build_himar


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model: nn.Module, momentum: float = 0.9999):
        self.momentum = momentum
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.momentum).add_(m_param.data, alpha=1.0 - self.momentum)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Constant LR with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    return base_lr


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model: nn.Module,
    vae: VAETokenizer,
    batch,
    cfg: Config,
    device: torch.device,
    scaler: Optional[GradScaler],
    amp_dtype: torch.dtype,
) -> dict:
    """
    Single training step.

    Returns dict with loss values.
    """
    is_imagenet = cfg.train.dataset == "imagenet"

    if is_imagenet:
        img_large, img_small, labels = batch
        img_large = img_large.to(device, non_blocking=True)
        img_small = img_small.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        text_embeds = None
    else:
        img_large, img_small, text_embeds = batch
        img_large = img_large.to(device, non_blocking=True)
        img_small = img_small.to(device, non_blocking=True)
        text_embeds = text_embeds.to(device, non_blocking=True)
        labels = None

    B = img_large.shape[0]

    # Encode images to tokens using frozen VAE
    with torch.no_grad():
        tokens_large = vae.encode(img_large)   # (B, N_l, token_dim)
        tokens_small = vae.encode(img_small)   # (B, N_s, token_dim)

    N_s = tokens_small.shape[1]
    N_l = tokens_large.shape[1]

    # Sample masking ratios
    ratios_small = sample_mask_ratio_phase1(
        B,
        cfg.train.mask_ratio_min_phase1,
        cfg.train.mask_ratio_max_phase1,
    )

    if cfg.train.mask_strategy_phase2 == "cosine":
        ratios_large = sample_mask_ratio_cosine(B)
    else:
        ratios_large = sample_mask_ratio_beta(B, cfg.train.beta_alpha, cfg.train.beta_beta)

    mask_small = create_mask(B, N_s, ratios_small).to(device)
    mask_large = create_mask(B, N_l, ratios_large).to(device)

    # Forward pass
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(scaler is not None)):
        losses = model(
            tokens_small=tokens_small,
            tokens_large=tokens_large,
            mask_small=mask_small,
            mask_large=mask_large,
            class_labels=labels,
            text_embeds=text_embeds,
        )

    return losses


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Config, args):
    # ---- Setup distributed training ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        dist.init_process_group(backend=cfg.train.dist_backend)
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = local_rank == 0

    # ---- Build model ----
    model_cfg = cfg.model
    model = build_himar(
        model_name=model_cfg.model_name,
        token_dim=model_cfg.token_dim,
        num_tokens_small=model_cfg.num_tokens_small,
        num_tokens_large=model_cfg.num_tokens_large,
        num_classes=model_cfg.num_classes,
        text_embed_dim=model_cfg.text_embed_dim,
        num_diffusion_timesteps=model_cfg.num_diffusion_timesteps,
        beta_schedule=model_cfg.beta_schedule,
        scale_emb_dim=model_cfg.scale_emb_dim,
        mlp_ratio=model_cfg.mlp_ratio,
        dropout=model_cfg.dropout,
        use_cfg=model_cfg.use_cfg,
        cfg_dropout=model_cfg.cfg_dropout,
    ).to(device)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model: {model_cfg.model_name}, Parameters: {num_params:.1f}M")

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if distributed else model

    # ---- EMA ----
    ema = None
    if cfg.train.use_ema:
        ema = EMA(raw_model, cfg.train.ema_momentum)

    # ---- VAE tokenizer (frozen) ----
    vae = VAETokenizer(vae_path=model_cfg.vae_path, device=str(device))

    # ---- Data loaders ----
    if cfg.train.dataset == "imagenet":
        train_loader, val_loader = build_imagenet_loaders(
            data_path=cfg.train.data_path,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            image_size=model_cfg.image_size_large,
            distributed=distributed,
        )
    else:
        train_loader, val_loader = build_coco_loaders(
            image_dir=cfg.train.data_path,
            ann_dir=cfg.train.coco_ann_path,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            image_size=model_cfg.image_size_large,
            clip_model_name=model_cfg.clip_model,
            distributed=distributed,
        )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=(cfg.train.beta1, cfg.train.beta2),
        weight_decay=cfg.train.weight_decay,
    )

    # ---- AMP ----
    amp_dtype = torch.bfloat16 if cfg.train.amp_dtype == "bfloat16" else torch.float16
    scaler = GradScaler() if (cfg.train.use_amp and amp_dtype == torch.float16) else None

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    if cfg.train.resume:
        ckpt = torch.load(cfg.train.resume, map_location="cpu")
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if is_main:
            print(f"Resumed from epoch {start_epoch}")

    # ---- Logging ----
    writer = None
    if is_main:
        output_dir = Path(cfg.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(output_dir / "logs"))

    # ---- Compute warmup steps ----
    steps_per_epoch = len(train_loader)
    if cfg.train.lr_schedule == "constant_with_warmup":
        if cfg.train.dataset == "imagenet":
            warmup_steps = cfg.train.warmup_epochs * steps_per_epoch
        else:
            warmup_steps = cfg.train.warmup_steps
    else:
        warmup_steps = 0

    # ---- Training loop ----
    for epoch in range(start_epoch, cfg.train.epochs):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            # LR schedule
            lr = get_lr(global_step, cfg.train.epochs * steps_per_epoch, cfg.train.lr, warmup_steps)
            set_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)

            losses = train_step(model, vae, batch, cfg, device, scaler, amp_dtype)
            loss = losses["loss"]

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()

            if ema is not None:
                ema.update(raw_model)

            epoch_loss += loss.item()
            epoch_loss1 += losses["loss_phase1"].item()
            epoch_loss2 += losses["loss_phase2"].item()
            global_step += 1

            if is_main and global_step % cfg.train.log_every == 0:
                elapsed = time.time() - t0
                it_per_sec = cfg.train.log_every / elapsed
                print(
                    f"Epoch {epoch:4d} | Step {global_step:7d} | "
                    f"Loss {loss.item():.4f} (L1={losses['loss_phase1'].item():.4f}, "
                    f"L2={losses['loss_phase2'].item():.4f}) | "
                    f"LR {lr:.2e} | {it_per_sec:.1f} it/s"
                )
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/loss_phase1", losses["loss_phase1"].item(), global_step)
                    writer.add_scalar("train/loss_phase2", losses["loss_phase2"].item(), global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                t0 = time.time()

        # ---- Save checkpoint ----
        if is_main and (epoch + 1) % cfg.train.save_every == 0:
            ckpt = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": cfg,
            }
            if ema:
                ckpt["ema"] = ema.state_dict()
            ckpt_path = Path(cfg.train.output_dir) / f"checkpoint_epoch{epoch:04d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    if writer:
        writer.close()

    if distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hi-MAR")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "coco"])
    parser.add_argument("--model", type=str, default="Hi-MAR-B",
                        choices=["Hi-MAR-S", "Hi-MAR-B", "Hi-MAR-L", "Hi-MAR-H"])
    parser.add_argument("--data_path", type=str, default="/data/imagenet")
    parser.add_argument("--coco_ann_path", type=str, default="/data/coco/annotations")
    parser.add_argument("--vae_path", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "imagenet":
        if args.model == "Hi-MAR-B":
            cfg = imagenet_base_config()
        elif args.model == "Hi-MAR-L":
            cfg = imagenet_large_config()
        else:
            cfg = imagenet_huge_config()
    else:
        cfg = coco_small_config()

    # Override from CLI
    cfg.model.model_name = args.model
    cfg.model.vae_path = args.vae_path
    cfg.train.data_path = args.data_path
    cfg.train.coco_ann_path = args.coco_ann_path
    cfg.train.output_dir = args.output_dir
    cfg.train.num_workers = args.num_workers
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.resume:
        cfg.train.resume = args.resume
    if args.no_amp:
        cfg.train.use_amp = False

    train(cfg, args)


if __name__ == "__main__":
    main()
