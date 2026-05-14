"""Pretraining loop for OLMoE.

Implements the full pretraining recipe from the paper (§2, Appendix B):
  - AdamW optimizer with beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1
  - Cosine LR schedule with 2500 warmup steps, peak LR=4e-4, min LR=4e-5
  - Annealing phase: final 100B tokens with linear LR decay to 0
  - Gradient clipping at 1.0
  - Mixed precision BF16 training
  - FP32 gradient reduction and optimizer states
  - Total: 5.133T tokens (1.3 epochs of OLMoE-Mix)
  - Checkpoints every 5000 steps
  - All parameters weight decayed (including RMSNorm and embeddings, §4.2.3-4)
"""

import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import OLMoEConfig, TrainConfig
from data import build_pretraining_dataloader
from model import OLMoE, build_olmoe_1b_7b


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
    annealing_start_step: int,
    annealing_min_lr: float = 0.0,
) -> float:
    """Compute learning rate for a given step.

    Schedule (§2, Appendix B):
    1. Linear warmup from 0 to peak_lr over warmup_steps
    2. Cosine decay from peak_lr to min_lr over (annealing_start_step - warmup_steps)
    3. Linear decay from min_lr to 0 over annealing phase (final 100B tokens)
    """
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)

    if step < annealing_start_step:
        # Cosine decay
        progress = (step - warmup_steps) / max(1, annealing_start_step - warmup_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * cosine_factor

    # Linear annealing to 0
    annealing_steps = total_steps - annealing_start_step
    progress = (step - annealing_start_step) / max(1, annealing_steps)
    return annealing_min_lr + (min_lr - annealing_min_lr) * (1.0 - progress)


def set_lr(optimizer: AdamW, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: OLMoE, config: TrainConfig) -> AdamW:
    """Build AdamW optimizer.

    All parameters are weight decayed, including RMSNorm and embeddings (§4.2.3-4).
    This differs from common practice of excluding norm/embedding params.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    return optimizer


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    step: int,
    tokens_seen: int,
    save_dir: str,
    config: OLMoEConfig,
):
    save_path = Path(save_dir) / f"step_{step:08d}"
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "tokens_seen": tokens_seen,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path / "checkpoint.pt",
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    checkpoint_path: str,
) -> Dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {"step": ckpt["step"], "tokens_seen": ckpt["tokens_seen"]}


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(
    model: OLMoE,
    batch: Dict[str, torch.Tensor],
    optimizer: AdamW,
    scaler: Optional[GradScaler],
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Single training step with mixed precision and gradient clipping."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

    with torch.autocast(device_type=device.type, dtype=dtype):
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]

    optimizer.zero_grad()

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    return {
        "loss": loss.item(),
        "ce_loss": out["ce_loss"].item(),
        "load_balance_loss": out["load_balance_loss"].item(),
        "router_z_loss": out["router_z_loss"].item(),
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: OLMoEConfig):
    """Full pretraining loop for OLMoE-1B-7B."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.train.seed)

    # Build model
    model = build_olmoe_1b_7b()
    model = model.to(device)

    param_counts = model.get_num_params()
    print(f"Model parameters: total={param_counts['total']/1e9:.2f}B, "
          f"active={param_counts['active']/1e9:.2f}B")

    # Build optimizer
    optimizer = build_optimizer(model, config.train)

    # Mixed precision scaler (only for fp16; bf16 doesn't need it)
    scaler = None
    if config.train.dtype == "float16":
        scaler = GradScaler()

    # Compute step counts
    tokens_per_step = (
        config.train.global_batch_size_tokens
        * config.train.gradient_accumulation_steps
    )
    total_steps = config.train.total_tokens // tokens_per_step
    annealing_steps = config.train.annealing_tokens // tokens_per_step
    annealing_start_step = total_steps - annealing_steps

    print(f"Total steps: {total_steps}, annealing starts at step {annealing_start_step}")

    # Resume from checkpoint
    start_step = 0
    tokens_seen = 0
    if config.train.resume_from:
        state = load_checkpoint(model, optimizer, config.train.resume_from)
        start_step = state["step"]
        tokens_seen = state["tokens_seen"]
        print(f"Resumed from step {start_step}, tokens seen: {tokens_seen}")

    # Build dataloader
    dataloader = build_pretraining_dataloader(
        data_dir=config.train.data_path,
        seq_len=config.train.seq_len,
        batch_size=config.train.batch_size_per_device,
        seed=config.train.seed,
        num_workers=config.train.num_workers,
        max_tokens=config.train.total_tokens,
    )

    # Training loop
    model.train()
    step = start_step
    t0 = time.time()

    for batch in dataloader:
        if step >= total_steps:
            break

        # Update learning rate
        lr = get_lr(
            step=step,
            warmup_steps=config.train.warmup_steps,
            total_steps=total_steps,
            peak_lr=config.train.learning_rate,
            min_lr=config.train.min_lr,
            annealing_start_step=annealing_start_step,
            annealing_min_lr=config.train.annealing_min_lr,
        )
        set_lr(optimizer, lr)

        # Training step
        metrics = training_step(model, batch, optimizer, scaler, config.train, device)
        tokens_seen += tokens_per_step
        step += 1

        # Logging
        if step % config.train.log_interval == 0:
            elapsed = time.time() - t0
            tokens_per_sec = tokens_per_step * config.train.log_interval / elapsed
            print(
                f"step={step:8d} | tokens={tokens_seen/1e9:.2f}B | "
                f"loss={metrics['loss']:.4f} | ce={metrics['ce_loss']:.4f} | "
                f"lb={metrics['load_balance_loss']:.4f} | rz={metrics['router_z_loss']:.6f} | "
                f"lr={lr:.2e} | grad_norm={metrics['grad_norm']:.3f} | "
                f"tok/s={tokens_per_sec:.0f}"
            )
            t0 = time.time()

        # Checkpointing every 5000 steps (paper releases intermediate checkpoints)
        if step % config.train.save_interval_steps == 0:
            save_checkpoint(model, optimizer, step, tokens_seen, config.train.save_dir, config)
            print(f"Saved checkpoint at step {step}")

    # Final checkpoint
    save_checkpoint(model, optimizer, step, tokens_seen, config.train.save_dir, config)
    print(f"Training complete. Final step: {step}, tokens seen: {tokens_seen/1e12:.3f}T")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain OLMoE")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--total_tokens", type=int, default=5_133_000_000_000)
    parser.add_argument("--batch_size_per_device", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OLMoEConfig()
    cfg.train.data_path = args.data_path
    cfg.train.save_dir = args.save_dir
    cfg.train.resume_from = args.resume_from
    cfg.train.total_tokens = args.total_tokens
    cfg.train.batch_size_per_device = args.batch_size_per_device
    cfg.train.seq_len = args.seq_len
    cfg.train.seed = args.seed

    train(cfg)
