"""
Training loop for GPT and nGPT on OpenWebText.

Key nGPT training differences (Section 2.6, step 7):
  - Adam (no weight decay) instead of AdamW
  - No learning rate warmup
  - normalize_weights() called after every optimizer step

Supports single-GPU and multi-GPU (DDP) training.

Usage:
    # Single GPU
    python train.py --model_type ngpt --model_size 0.5B --seq_len 4096

    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node=8 train.py --model_type ngpt --model_size 0.5B
"""

import os
import sys
import copy
import math
import time
import argparse
import logging
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import ModelConfig, TrainConfig, MODEL_CONFIGS
from model import build_model, count_parameters, NGPT
from data import build_dataloader, infinite_loader

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning rate schedule — cosine annealing (Table 3)
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine annealing with optional linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    if step >= cfg.max_steps:
        return cfg.lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


# ---------------------------------------------------------------------------
# Optimizer construction
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Build Adam (nGPT) or AdamW (GPT) optimizer.

    For nGPT, weight decay is 0 and all parameters use the same optimizer
    (no parameter group splitting needed).  For GPT, weight decay is applied
    to 2-D parameters only (standard practice).
    """
    if cfg.model_type == "ngpt":
        # nGPT: Adam with no weight decay (Section 2.6, step 7)
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )
    else:
        # GPT: AdamW with weight decay on 2-D params
        decay_params = [p for p in model.parameters() if p.dim() >= 2]
        nodecay_params = [p for p in model.parameters() if p.dim() < 2]
        param_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, val_loader, cfg: TrainConfig, device: torch.device) -> float:
    model.eval()
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    ctx = torch.autocast(device_type=device.type, dtype=dtype)

    total_loss = 0.0
    n_batches = 0
    for x, y in val_loader:
        if n_batches >= cfg.eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, optimizer, step: int, val_loss: float, cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    ckpt = {
        "step": step,
        "val_loss": val_loss,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(cfg),
    }
    path = os.path.join(cfg.out_dir, f"ckpt_{step:07d}.pt")
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("step", 0), ckpt.get("val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig):
    # Distributed setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend=cfg.backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_master = rank == 0
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True

    if is_master:
        logger.info(f"Training {cfg.model_type.upper()} {cfg.model_size} | "
                    f"seq_len={cfg.seq_len} | steps={cfg.max_steps}")

    # Model
    model_cfg = copy.deepcopy(MODEL_CONFIGS[cfg.model_size])
    model_cfg.model_type = cfg.model_type
    model_cfg.max_seq_len = cfg.seq_len

    model = build_model(model_cfg).to(device)

    if cfg.compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    if is_master:
        n_params = count_parameters(model)
        logger.info(f"Parameters: {n_params / 1e6:.1f}M")

    # Optimizer
    optimizer = build_optimizer(model, cfg)

    # Data
    per_device_batch = cfg.global_batch_size // world_size
    train_path = os.path.join(cfg.dataset_path, "train.bin")
    val_path = os.path.join(cfg.dataset_path, "val.bin")

    train_loader = build_dataloader(
        train_path, cfg.seq_len, per_device_batch,
        num_workers=cfg.num_workers, streaming=True,
        rank=rank, world_size=world_size,
    )
    val_loader = build_dataloader(
        val_path, cfg.seq_len, per_device_batch,
        num_workers=cfg.num_workers, streaming=False,
        rank=rank, world_size=world_size,
    )

    train_iter = infinite_loader(train_loader)

    # Mixed precision context
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    ctx = torch.autocast(device_type=device.type, dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    # Optional: WandB logging
    if is_master and cfg.wandb_project:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))

    model.train()
    step = 0
    t0 = time.time()

    for step in range(1, cfg.max_steps + 1):
        # Update learning rate
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        with ctx:
            _, loss = model(x, y)

        scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ----------------------------------------------------------------
        # nGPT: normalize all weight matrices after every optimizer step
        # (Section 2.6, step 2).  Must be applied to the raw (non-DDP)
        # model so that the actual parameters are updated.
        # ----------------------------------------------------------------
        if cfg.model_type == "ngpt":
            raw_model = model.module if isinstance(model, DDP) else model
            raw_model.normalize_weights()

        # Logging
        if is_master and step % cfg.log_interval == 0:
            dt = time.time() - t0
            tokens_per_sec = (cfg.log_interval * cfg.global_batch_size * cfg.seq_len) / dt
            logger.info(
                f"step {step:7d} | loss {loss.item():.4f} | "
                f"lr {lr:.2e} | {tokens_per_sec / 1e6:.2f}M tok/s"
            )
            if cfg.wandb_project:
                import wandb
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": step})
            t0 = time.time()

        # Validation
        if is_master and step % cfg.eval_interval == 0:
            val_loss = evaluate(model, val_loader, cfg, device)
            logger.info(f"step {step:7d} | val_loss {val_loss:.4f}")
            if cfg.wandb_project:
                import wandb
                wandb.log({"val/loss": val_loss, "step": step})

        # Checkpoint
        if is_master and step % cfg.save_interval == 0:
            val_loss = evaluate(model, val_loader, cfg, device)
            save_checkpoint(model, optimizer, step, val_loss, cfg)

    if is_master:
        logger.info("Training complete.")
        val_loss = evaluate(model, val_loader, cfg, device)
        save_checkpoint(model, optimizer, step, val_loss, cfg)

    if ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train GPT or nGPT on OpenWebText")

    parser.add_argument("--model_type", type=str, default="ngpt", choices=["gpt", "ngpt"])
    parser.add_argument("--model_size", type=str, default="0.5B", choices=["0.5B", "1B"])
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--global_batch_size", type=int, default=512)
    parser.add_argument("--dataset_path", type=str, default="data/openwebtext")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)

    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--out_dir", type=str, default="checkpoints")

    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--backend", type=str, default="nccl")

    args = parser.parse_args()

    cfg = TrainConfig(
        model_type=args.model_type,
        model_size=args.model_size,
        seq_len=args.seq_len,
        global_batch_size=args.global_batch_size,
        dataset_path=args.dataset_path,
        num_workers=args.num_workers,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        grad_clip=args.grad_clip,
        max_steps=args.max_steps,
        min_lr_ratio=args.min_lr_ratio,
        dtype=args.dtype,
        compile=args.compile,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        save_interval=args.save_interval,
        out_dir=args.out_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        backend=args.backend,
    )

    # Apply model-type defaults
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    elif args.model_type == "gpt":
        cfg.weight_decay = 0.1

    if args.warmup_steps is not None:
        cfg.warmup_steps = args.warmup_steps
    elif args.model_type == "gpt":
        cfg.warmup_steps = 2000

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
