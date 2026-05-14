"""
MDM training script.

Trains a Masked Diffusion Model on one of: sudoku, zebra, nae_sat, text.

Usage:
  python train_mdm.py --task sudoku --model_size 6M --epochs 300 --lr 0.001 --batch_size 128
  python train_mdm.py --task nae_sat --N 25 --P 275 --model_size 19M
  python train_mdm.py --task text --model_size 170M --data_path data/slimpajama
"""

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import (
    ExperimentConfig, MODEL_CONFIGS,
    get_sudoku_mdm_config, get_zebra_mdm_config, get_nae_sat_config,
)
from mdm import MDM
from data import (
    get_sudoku_loaders, get_zebra_loaders, get_nae_sat_loaders, get_text_loaders,
)
from utils import (
    set_seed, get_logger, build_optimizer, build_cosine_lr_scheduler,
    save_checkpoint, load_checkpoint, log_metrics,
)

logger = get_logger("train_mdm")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: MDM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    grad_clip: float = 1.0,
    step: int = 0,
    log_every: int = 100,
    use_wandb: bool = False,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x0 = batch["x0"].to(device)

        loss = model.compute_loss(x0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        step += 1

        if step % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            log_metrics(
                {"loss": loss.item(), "lr": lr},
                step=step,
                use_wandb=use_wandb,
                prefix="train",
            )

    return total_loss / max(n_batches, 1), step


@torch.no_grad()
def evaluate(
    model: MDM,
    loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x0 = batch["x0"].to(device)
        loss = model.compute_loss(x0)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    task: str,
    model_size: str = "6M",
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 128,
    max_iters: Optional[int] = None,
    data_path: Optional[str] = None,
    output_dir: str = "outputs",
    device: str = "cuda",
    seed: int = 42,
    use_wandb: bool = False,
    resume_from: Optional[str] = None,
    # NAE-SAT specific
    N: int = 25,
    P: int = 275,
    # Noise schedule
    noise_schedule_type: str = "linear",
    # Logging
    log_every: int = 100,
    eval_every: int = 500,
    save_every: int = 1000,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = device if torch.cuda.is_available() else "cpu"

    # Build data loaders
    if task == "sudoku":
        data_path = data_path or "data/sudoku"
        train_loader, test_loader, _ = get_sudoku_loaders(
            data_path, batch_size=batch_size
        )
        vocab_size = 10
        seq_len = 81
    elif task == "zebra":
        data_path = data_path or "data/zebra"
        train_loader, test_loader = get_zebra_loaders(
            data_path, batch_size=batch_size
        )
        vocab_size = 6
        seq_len = 25
    elif task == "nae_sat":
        train_loader, test_loader = get_nae_sat_loaders(
            N=N, P=P, batch_size=batch_size
        )
        vocab_size = 5  # 0=mask, 1-3=latent values, 4=padding
        seq_len = 512
    elif task == "text":
        data_path = data_path or "data/slimpajama"
        train_loader, test_loader = get_text_loaders(
            data_path, seq_len=2048, batch_size=batch_size
        )
        vocab_size = 32_000
        seq_len = 2048
    else:
        raise ValueError(f"Unknown task: {task}")

    # Build model
    model_config = MODEL_CONFIGS[model_size]
    model = MDM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        model_config=model_config,
        noise_schedule_type=noise_schedule_type,
    ).to(device)

    n_params = model.count_parameters()
    logger.info(f"Model: {model_size} | Parameters: {n_params:,}")

    # Build optimizer and scheduler
    if max_iters is None:
        max_iters = epochs * len(train_loader)

    warmup_steps = min(2000, max_iters // 10)
    optimizer = build_optimizer(model, lr=lr)
    scheduler = build_cosine_lr_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=max_iters,
        lr_min_ratio=0.1,
    )

    start_step = 0
    if resume_from:
        info = load_checkpoint(model, optimizer, scheduler, resume_from, device)
        start_step = info["step"]
        logger.info(f"Resumed from step {start_step}")

    if use_wandb:
        import wandb
        wandb.init(
            project="masked-diffusion-token-ordering",
            name=f"mdm_{task}_{model_size}",
            config={
                "task": task, "model_size": model_size, "lr": lr,
                "batch_size": batch_size, "n_params": n_params,
            },
        )

    # Training loop
    step = start_step
    best_val_loss = float("inf")

    for epoch in range(epochs):
        if step >= max_iters:
            break

        avg_loss, step = train_epoch(
            model, train_loader, optimizer, scheduler,
            device=device, step=step,
            log_every=log_every, use_wandb=use_wandb,
        )

        if epoch % max(1, epochs // 20) == 0:
            val_loss = evaluate(model, test_loader, device)
            log_metrics(
                {"loss": val_loss, "epoch": epoch},
                step=step, use_wandb=use_wandb, prefix="val",
            )
            logger.info(
                f"Epoch {epoch}/{epochs} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, step, val_loss,
                    output_dir, "best_model.pt",
                )

        if step % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, step, avg_loss,
                output_dir, f"checkpoint_step{step}.pt",
            )

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, step, avg_loss,
        output_dir, "final_model.pt",
    )
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train MDM")
    parser.add_argument("--task", type=str, default="sudoku",
                        choices=["sudoku", "zebra", "nae_sat", "text"])
    parser.add_argument("--model_size", type=str, default="6M",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/mdm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--noise_schedule", type=str, default="linear",
                        choices=["linear", "cosine"])
    # NAE-SAT
    parser.add_argument("--N", type=int, default=25)
    parser.add_argument("--P", type=int, default=275)
    # Logging
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        task=args.task,
        model_size=args.model_size,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        use_wandb=args.use_wandb,
        resume_from=args.resume_from,
        N=args.N,
        P=args.P,
        noise_schedule_type=args.noise_schedule,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
