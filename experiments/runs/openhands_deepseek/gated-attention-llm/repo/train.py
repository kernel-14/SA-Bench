"""
Training loop for Gated Attention LLMs.

Implements paper training settings:
- AdamW optimizer (beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)
- Cosine LR schedule with linear warmup
- BF16 mixed precision
- Gradient clipping at 1.0
- Distributed training support (DDP)
- Both dense and MoE model training
"""

import os
import math
import time
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, Tuple

from config import Config, ModelConfig, TrainingConfig
from model import Transformer, create_model, get_parameter_count
from data import create_dataloader, get_eval_dataloaders


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 3e-5,
):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_distributed() -> Tuple[int, int, bool]:
    """Setup distributed training environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    is_main = rank == 0
    return local_rank, world_size, is_main


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 100))


def train_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    max_grad_norm: float,
    is_moe: bool,
    use_amp: bool,
) -> float:
    """Single training step."""
    input_ids, labels = batch
    input_ids = input_ids.cuda()
    labels = labels.cuda()

    optimizer.zero_grad()

    if use_amp and scaler is not None:
        with autocast(dtype=torch.bfloat16):
            logits, loss = model(input_ids, labels=labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, loss = model(input_ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

    return loss.item()


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    is_moe: bool,
) -> float:
    """Single evaluation step."""
    input_ids, labels = batch
    input_ids = input_ids.cuda()
    labels = labels.cuda()

    _, loss = model(input_ids, labels=labels)
    return loss.item()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    is_moe: bool,
    max_eval_batches: int = 100,
) -> Dict[str, float]:
    """Evaluate model on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if i >= max_eval_batches:
            break
        loss = eval_step(model, batch, is_moe)
        total_loss += loss * batch[0].numel()
        total_tokens += batch[0].numel()

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = compute_perplexity(avg_loss)

    return {"loss": avg_loss, "ppl": ppl}


def train(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    exp_name: str = "gated_attention",
    output_dir: str = "outputs",
    resume_from: Optional[str] = None,
):
    """Main training function."""
    local_rank, world_size, is_main = setup_distributed()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main:
        print(f"Starting training: {exp_name}")
        print(f"Model config: {model_config}")
        print(f"Training config: {training_config}")
        os.makedirs(output_dir, exist_ok=True)

    # Create model
    model = create_model(model_config)
    model = model.to(device)

    total_params, trainable_params = get_parameter_count(model)
    if is_main:
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Resume from checkpoint if specified
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        if is_main:
            print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint.get("step", 0)

    # Wrap with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Create dataloader
    train_dataloader = create_dataloader(
        data_path=training_config.data_path,
        seq_len=training_config.seq_len,
        batch_size=training_config.batch_size // world_size,
        split="train",
        num_workers=4,
        streaming=True,
    )

    # Calculate total steps
    total_tokens = training_config.total_tokens
    tokens_per_step = training_config.batch_size * training_config.seq_len
    total_steps = total_tokens // tokens_per_step

    if is_main:
        print(f"Total training steps: {total_steps}")
        print(f"Tokens per step: {tokens_per_step:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.max_lr,
        betas=(training_config.beta1, training_config.beta2),
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
    )

    # LR scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=training_config.warmup_steps,
        total_steps=total_steps,
        min_lr=training_config.min_lr,
    )

    # Mixed precision
    use_amp = training_config.use_amp and training_config.dtype == "bfloat16"
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    # Training loop
    model.train()
    global_step = start_step
    total_loss_accum = 0.0
    start_time = time.time()

    # For loss spike detection (smoothing)
    smoothed_loss = None

    for step, batch in enumerate(train_dataloader, start=start_step):
        loss = train_step(
            model, batch, optimizer, scaler,
            training_config.max_grad_norm,
            model_config.model_type == "moe",
            use_amp,
        )

        scheduler.step()
        global_step += 1

        # Accumulate loss for logging
        total_loss_accum += loss

        # Update smoothed loss
        if smoothed_loss is None:
            smoothed_loss = loss
        else:
            smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss

        # Logging
        if is_main and global_step % training_config.log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss_accum / training_config.log_interval
            ppl = compute_perplexity(avg_loss)
            lr = scheduler.get_last_lr()[0]

            print(
                f"Step {global_step}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"PPL: {ppl:.2f} | "
                f"Smoothed Loss: {smoothed_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )
            total_loss_accum = 0.0

        # Save checkpoint
        if is_main and global_step % training_config.save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
            save_dict = {
                "step": global_step,
                "model_state_dict": model.module.state_dict()
                if isinstance(model, DDP)
                else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
            }
            torch.save(save_dict, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # Evaluation
        if is_main and global_step % training_config.eval_interval == 0:
            eval_dir = os.path.join(training_config.data_path, "eval")
            if os.path.exists(eval_dir):
                eval_loaders = get_eval_dataloaders(
                    eval_dir,
                    training_config.seq_len,
                    training_config.batch_size // world_size,
                )
                eval_results = {}
                for split_name, loader in eval_loaders.items():
                    result = evaluate(
                        model.module if isinstance(model, DDP) else model,
                        loader,
                        is_moe=model_config.model_type == "moe",
                    )
                    eval_results[split_name] = result["ppl"]

                avg_ppl = sum(eval_results.values()) / len(eval_results) if eval_results else float("inf")
                print(f"Eval PPL: {eval_results} | Avg: {avg_ppl:.4f}")

                # Save eval metrics
                with open(os.path.join(output_dir, "eval_metrics.jsonl"), "a") as f:
                    json.dump({
                        "step": global_step,
                        "eval_ppl": eval_results,
                        "avg_ppl": avg_ppl,
                    }, f)
                    f.write("\n")

    # Final save
    if is_main:
        final_path = os.path.join(output_dir, f"{exp_name}_final.pt")
        save_dict = {
            "step": global_step,
            "model_state_dict": model.module.state_dict()
            if isinstance(model, DDP)
            else model.state_dict(),
            "config": model_config,
        }
        torch.save(save_dict, final_path)
        print(f"Training complete. Final model saved to {final_path}")

    cleanup_distributed()


def main():
    """Entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Gated Attention LLM")
    parser.add_argument("--model_type", type=str, default="dense",
                        choices=["dense", "moe"])
    parser.add_argument("--gating_position", type=str, default=None,
                        choices=["G1", "G2", "G3", "G4", "G5", None])
    parser.add_argument("--gating_granularity", type=str, default=None,
                        choices=["elementwise", "headwise", None])
    parser.add_argument("--gating_head_specific", action="store_true", default=True)
    parser.add_argument("--gating_mode", type=str, default="multiplicative",
                        choices=["multiplicative", "additive"])
    parser.add_argument("--gating_activation", type=str, default="sigmoid",
                        choices=["sigmoid", "silu", "identity"])
    parser.add_argument("--use_sandwich_norm", action="store_true", default=False)
    parser.add_argument("--n_layers", type=int, default=28)
    parser.add_argument("--d_model", type=int, default=2048)
    parser.add_argument("--max_lr", type=float, default=4e-3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default="data/tokens")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--exp_name", type=str, default="gated_attention")
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    # Build configs
    model_config = ModelConfig(
        model_type=args.model_type,
        n_layers=args.n_layers,
        d_model=args.d_model,
        gating_position=args.gating_position,
        gating_granularity=args.gating_granularity,
        gating_head_specific=args.gating_head_specific,
        gating_mode=args.gating_mode,
        gating_activation=args.gating_activation,
        use_sandwich_norm=args.use_sandwich_norm,
    )

    training_config = TrainingConfig(
        max_lr=args.max_lr,
        batch_size=args.batch_size,
        data_path=args.data_path,
    )

    train(
        model_config=model_config,
        training_config=training_config,
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
