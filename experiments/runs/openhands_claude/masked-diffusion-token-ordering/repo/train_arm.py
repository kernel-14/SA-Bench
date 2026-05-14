"""
ARM training script.

Trains an Autoregressive Model on one of: sudoku, zebra, text.

Supports two training modes:
  1. Left-to-right (standard ARM): --no_ordering
  2. Order-aware (ARM with ground-truth ordering): --use_ordering

For π-learner scaling law experiments (Section 3.2), use:
  python train_arm.py --task text --pi_learner --permutation_type random

Usage:
  python train_arm.py --task sudoku --model_size 42M --epochs 300 --use_ordering
  python train_arm.py --task sudoku --model_size 42M --epochs 300
  python train_arm.py --task text --model_size 170M --pi_learner --permutation_type closer
"""

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import MODEL_CONFIGS
from arm import ARM, PiLearner
from data import (
    get_sudoku_loaders, get_zebra_loaders, get_text_loaders,
    sample_permutation,
)
from utils import (
    set_seed, get_logger, build_optimizer, build_cosine_lr_scheduler,
    save_checkpoint, load_checkpoint, log_metrics,
)

logger = get_logger("train_arm")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch_arm(
    model: ARM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    use_ordering: bool = False,
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
        ordering = batch.get("ordering")
        if ordering is not None and use_ordering:
            ordering = ordering.to(device)
        else:
            ordering = None

        loss = model.compute_loss(x0, ordering)

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
                step=step, use_wandb=use_wandb, prefix="train",
            )

    return total_loss / max(n_batches, 1), step


def train_epoch_pi_learner(
    model: PiLearner,
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
                step=step, use_wandb=use_wandb, prefix="train",
            )

    return total_loss / max(n_batches, 1), step


@torch.no_grad()
def evaluate_arm(
    model: ARM,
    loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x0 = batch["x0"].to(device)
        loss = model.compute_loss(x0, ordering=None)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_pi_learner(
    model: PiLearner,
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
    model_size: str = "42M",
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
    use_ordering: bool = False,
    # π-learner specific
    pi_learner: bool = False,
    permutation_type: str = "random",
    permutation_seed: Optional[int] = None,
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
            data_path, batch_size=batch_size, use_ordering=use_ordering
        )
        vocab_size = 10
        seq_len = 81
    elif task == "zebra":
        data_path = data_path or "data/zebra"
        train_loader, test_loader = get_zebra_loaders(
            data_path, batch_size=batch_size, use_ordering=use_ordering
        )
        vocab_size = 6
        seq_len = 25
    elif task == "text":
        data_path = data_path or "data/slimpajama"
        train_loader, test_loader = get_text_loaders(
            data_path, seq_len=2048, batch_size=batch_size
        )
        vocab_size = 32_000
        seq_len = 2048
    else:
        raise ValueError(f"Unknown task: {task}")

    model_config = MODEL_CONFIGS[model_size]

    if pi_learner:
        # π-learner: ARM with fixed permutation and learnable pos embeddings
        pi = sample_permutation(
            seq_len, permutation_type, seed=permutation_seed or seed
        ).to(device)
        model = PiLearner(
            vocab_size=vocab_size,
            seq_len=seq_len,
            model_config=model_config,
            pi=pi,
        ).to(device)
        logger.info(
            f"π-learner | permutation_type={permutation_type} | "
            f"Parameters: {model.count_parameters():,}"
        )
    else:
        model = ARM(
            vocab_size=vocab_size,
            seq_len=seq_len,
            model_config=model_config,
        ).to(device)
        logger.info(
            f"ARM | use_ordering={use_ordering} | "
            f"Parameters: {model.count_parameters():,}"
        )

    if max_iters is None:
        max_iters = epochs * len(train_loader)

    warmup_steps = min(2000, max_iters // 10)
    optimizer = build_optimizer(model, lr=lr)
    scheduler = build_cosine_lr_scheduler(
        optimizer, warmup_steps=warmup_steps, total_steps=max_iters, lr_min_ratio=0.1
    )

    start_step = 0
    if resume_from:
        info = load_checkpoint(model, optimizer, scheduler, resume_from, device)
        start_step = info["step"]
        logger.info(f"Resumed from step {start_step}")

    if use_wandb:
        import wandb
        run_name = f"pi_learner_{permutation_type}" if pi_learner else f"arm_{task}_{model_size}"
        if use_ordering:
            run_name += "_ordered"
        wandb.init(
            project="masked-diffusion-token-ordering",
            name=run_name,
            config={
                "task": task, "model_size": model_size, "lr": lr,
                "batch_size": batch_size, "use_ordering": use_ordering,
                "pi_learner": pi_learner, "permutation_type": permutation_type,
            },
        )

    step = start_step
    best_val_loss = float("inf")

    for epoch in range(epochs):
        if step >= max_iters:
            break

        if pi_learner:
            avg_loss, step = train_epoch_pi_learner(
                model, train_loader, optimizer, scheduler,
                device=device, step=step,
                log_every=log_every, use_wandb=use_wandb,
            )
        else:
            avg_loss, step = train_epoch_arm(
                model, train_loader, optimizer, scheduler,
                device=device, use_ordering=use_ordering, step=step,
                log_every=log_every, use_wandb=use_wandb,
            )

        if epoch % max(1, epochs // 20) == 0:
            if pi_learner:
                val_loss = evaluate_pi_learner(model, test_loader, device)
            else:
                val_loss = evaluate_arm(model, test_loader, device)

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

    save_checkpoint(
        model, optimizer, scheduler, step, avg_loss,
        output_dir, "final_model.pt",
    )
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train ARM")
    parser.add_argument("--task", type=str, default="sudoku",
                        choices=["sudoku", "zebra", "text"])
    parser.add_argument("--model_size", type=str, default="42M",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/arm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--use_ordering", action="store_true",
                        help="Use ground-truth token ordering for order-aware ARM training")
    # π-learner
    parser.add_argument("--pi_learner", action="store_true",
                        help="Train a π-learner with fixed permutation")
    parser.add_argument("--permutation_type", type=str, default="random",
                        choices=["identity", "random", "closer", "much_closer"])
    parser.add_argument("--permutation_seed", type=int, default=None)
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
        use_ordering=args.use_ordering,
        pi_learner=args.pi_learner,
        permutation_type=args.permutation_type,
        permutation_seed=args.permutation_seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
