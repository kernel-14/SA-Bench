"""
Training loop for gated-attention language models.

Implements the training setup described in Sec. 3.1:
  - AdamW optimizer with cosine LR schedule and linear warmup
  - BF16 mixed precision
  - Gradient clipping
  - MoE auxiliary losses (Z-loss + load-balancing loss)
  - Optional Weights & Biases logging

Usage:
    python train.py --variant G1_elementwise --model moe_15a2b
    python train.py --variant baseline --model dense_1_7b_28l
    python train.py --variant G1_elementwise --model dense_1_7b_48l --max_lr 8e-3
"""

import argparse
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from config import (
    GatingConfig,
    ModelConfig,
    TrainingConfig,
    GATING_VARIANTS,
    get_dense_1_7b_config,
    get_dense_1_7b_deep_config,
    get_moe_15a2b_config,
    get_moe_training_config,
    get_dense_400b_training_config,
    get_dense_3_5t_training_config,
    get_dense_1t_training_config,
)
from model import GatedTransformerLM, build_model, count_parameters
from data import BinaryShardDataset, make_train_dataloader, make_eval_dataloader, EvalDataset


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainingConfig) -> float:
    """Cosine LR schedule with linear warmup (Sec. 3.1)."""
    if step < cfg.warmup_steps:
        return cfg.max_lr * step / cfg.warmup_steps
    if step >= cfg.total_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.max_lr - cfg.min_lr) * cosine_decay


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.AdamW:
    """AdamW with weight decay applied only to non-bias/norm parameters."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model: GatedTransformerLM,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.AdamW,
    scaler: Optional[GradScaler],
    cfg: TrainingConfig,
    step: int,
    device: torch.device,
) -> dict[str, float]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    use_amp = cfg.dtype == "bfloat16" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]
        aux_loss = out["aux_loss"]

    if cfg.gradient_accumulation_steps > 1:
        loss = loss / cfg.gradient_accumulation_steps

    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    metrics = {
        "loss": loss.item() * cfg.gradient_accumulation_steps,
        "aux_loss": aux_loss.item(),
    }

    if (step + 1) % cfg.gradient_accumulation_steps == 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        metrics["grad_norm"] = grad_norm.item()

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return metrics


# ---------------------------------------------------------------------------
# Evaluation: perplexity on held-out sets
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: GatedTransformerLM,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, labels=labels)
        # CE loss is already mean over non-ignored tokens
        seq_len = (labels != -100).sum().item()
        total_loss += out["loss"].item() * seq_len
        total_tokens += seq_len
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    train_data_dir: str,
    eval_data_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    use_wandb: bool = False,
    run_name: str = "gated_attn",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = build_model(model_cfg).to(device)
    param_counts = count_parameters(model)
    print(
        f"Model: {param_counts['total'] / 1e9:.2f}B total params, "
        f"{param_counts['activated'] / 1e9:.2f}B activated"
    )

    # Optimizer
    optimizer = build_optimizer(model, train_cfg)

    # AMP scaler (not needed for BF16 on Ampere+, but kept for compatibility)
    scaler = GradScaler() if train_cfg.dtype == "float16" else None

    # Data
    train_dataset = BinaryShardDataset(
        train_data_dir, seq_len=train_cfg.seq_len, split="train"
    )
    train_loader = make_train_dataloader(
        train_dataset,
        batch_size=train_cfg.batch_size // max(1, torch.cuda.device_count()),
    )

    eval_loader = None
    if eval_data_dir is not None:
        eval_dataset = EvalDataset.from_file(
            Path(eval_data_dir) / "val.bin", seq_len=train_cfg.seq_len
        )
        eval_loader = make_eval_dataloader(eval_dataset)

    # Resume
    start_step = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    # W&B
    if use_wandb:
        import wandb
        wandb.init(project="gated-attention-llm", name=run_name, config={
            "model": model_cfg.__dict__,
            "training": train_cfg.__dict__,
        })

    output_dir = Path(train_cfg.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = start_step
    t0 = time.time()
    data_iter = iter(train_loader)

    while step < train_cfg.total_steps:
        # Update LR
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Fetch batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        metrics = train_step(model, batch, optimizer, scaler, train_cfg, step, device)
        step += 1

        # Logging
        if step % train_cfg.log_interval == 0:
            dt = time.time() - t0
            tokens_per_sec = (
                train_cfg.batch_size * train_cfg.seq_len * train_cfg.log_interval / dt
            )
            print(
                f"step {step:7d} | loss {metrics['loss']:.4f} | "
                f"aux {metrics.get('aux_loss', 0):.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {metrics.get('grad_norm', 0):.3f} | "
                f"{tokens_per_sec / 1e6:.2f}M tok/s"
            )
            if use_wandb:
                import wandb
                wandb.log({"train/loss": metrics["loss"], "train/lr": lr, "step": step})
            t0 = time.time()

        # Evaluation
        if step % train_cfg.eval_interval == 0 and eval_loader is not None:
            ppl = evaluate_perplexity(model, eval_loader, device)
            print(f"  eval PPL: {ppl:.4f}")
            if use_wandb:
                import wandb
                wandb.log({"eval/ppl": ppl, "step": step})

        # Checkpoint
        if step % train_cfg.save_interval == 0:
            ckpt_path = output_dir / f"step_{step:07d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "model_cfg": model_cfg,
                    "train_cfg": train_cfg,
                },
                ckpt_path,
            )
            print(f"  saved checkpoint: {ckpt_path}")

    print("Training complete.")
    if use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gated-attention LM")
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=list(GATING_VARIANTS.keys()),
        help="Gating variant (see config.GATING_VARIANTS)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dense_1_7b_28l",
        choices=["dense_1_7b_28l", "dense_1_7b_48l", "moe_15a2b"],
    )
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--eval_data_dir", type=str, default=None)
    parser.add_argument("--max_lr", type=float, default=None, help="Override max LR")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--sandwich_norm", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    gating = GATING_VARIANTS[args.variant]

    if args.model == "dense_1_7b_28l":
        model_cfg = get_dense_1_7b_config(
            num_layers=28, gating=gating, sandwich_norm=args.sandwich_norm
        )
        train_cfg = get_dense_400b_training_config()
    elif args.model == "dense_1_7b_48l":
        model_cfg = get_dense_1_7b_deep_config(
            gating=gating, sandwich_norm=args.sandwich_norm
        )
        train_cfg = get_dense_400b_training_config()
    elif args.model == "moe_15a2b":
        model_cfg = get_moe_15a2b_config(gating=gating)
        train_cfg = get_moe_training_config()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # CLI overrides
    if args.max_lr is not None:
        train_cfg.max_lr = args.max_lr
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.total_steps is not None:
        train_cfg.total_steps = args.total_steps
    train_cfg.output_dir = args.output_dir

    run_name = args.run_name or f"{args.model}_{args.variant}"

    train(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_data_dir=args.train_data_dir,
        eval_data_dir=args.eval_data_dir,
        resume_from=args.resume_from,
        use_wandb=args.wandb,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
