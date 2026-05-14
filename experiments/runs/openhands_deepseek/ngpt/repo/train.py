
"""Training loop for GPT and nGPT models.

Implements:
- Adam/AdamW optimizer with cosine annealing schedule
- Gradient accumulation for large effective batch sizes
- Weight normalization step for nGPT (after each batch)
- Mixed precision (bfloat16) training
- Evaluation on validation set and downstream tasks
"""

import math
import os
import time
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast

from config import TrainConfig, OptimConfig, nGPTConfig
from model import create_model
from data import create_dataloader, OpenWebTextDataset


def get_cosine_schedule(step: int, total_steps: int, initial_lr: float, final_lr: float = 0.0) -> float:
    """Cosine annealing learning rate schedule."""
    if step >= total_steps:
        return final_lr
    return final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + math.cos(math.pi * step / total_steps))


def get_warmup_cosine_schedule(
    step: int, total_steps: int, warmup_steps: int, initial_lr: float, final_lr: float = 0.0
) -> float:
    """Cosine annealing with linear warmup (for GPT baseline)."""
    if step < warmup_steps:
        return initial_lr * (step + 1) / warmup_steps
    return get_cosine_schedule(step - warmup_steps, total_steps - warmup_steps, initial_lr, final_lr)


class Trainer:
    """Trainer for GPT and nGPT models."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ngpt = config.use_ngpt

        # Create model
        self.model = create_model(config, use_ngpt=self.use_ngpt)
        self.model = self.model.to(self.device)
        self.model.train()

        # Setup optimizer
        optim_config = config.optim
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=optim_config.initial_lr,
            betas=(optim_config.beta1, optim_config.beta2),
            eps=optim_config.epsilon,
            weight_decay=optim_config.weight_decay if optim_config.optimizer == "adamw" else 0.0,
        )

        # Training state
        self.global_step = 0
        self.total_tokens_processed = 0
        self.best_val_loss = float('inf')

        # For nGPT: normalize weights after model initialization
        if self.use_ngpt:
            self._normalize_ngpt_weights()

    def _normalize_ngpt_weights(self):
        """Normalize all weight matrices and embeddings for nGPT (Section 2.6, step 2)."""
        self.model.normalize_all_weights()

    def get_lr(self) -> float:
        if self.config.use_ngpt:
            # nGPT: no warmup (Table 3, Section 2.6 step 7)
            return get_cosine_schedule(
                self.global_step,
                self.config.total_iters,
                self.config.optim.initial_lr,
                self.config.optim.final_lr,
            )
        else:
            # GPT: with warmup (Table 3)
            return get_warmup_cosine_schedule(
                self.global_step,
                self.config.total_iters,
                self.config.optim.warmup_steps,
                self.config.optim.initial_lr,
                self.config.optim.final_lr,
            )

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss for next-token prediction."""
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with optional gradient accumulation."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            logits = self.model(input_ids)
            loss = self.compute_loss(logits, labels)

        # Scale loss for gradient accumulation
        loss = loss / self.config.grad_acc_steps
        loss.backward()

        # Gradient clipping
        if self.config.optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)

        return {"loss": loss.item() * self.config.grad_acc_steps}

    def optimizer_step(self):
        """Perform optimizer step with LR scheduling."""
        # Update learning rate
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()
        self.optimizer.zero_grad()

        # For nGPT: normalize all weights after each step (Section 2.6, step 2)
        if self.use_ngpt:
            self._normalize_ngpt_weights()

    @torch.no_grad()
    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='sum',
                )

            total_loss += loss.item()
            total_tokens += labels.numel()

        self.model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(avg_loss)
        return {"val_loss": avg_loss, "val_perplexity": perplexity}

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch (or specified number of steps)."""
        metrics = {"loss": 0.0, "tokens": 0}
        start_time = time.time()

        for batch in train_loader:
            step_metrics = self.train_step(batch)
            metrics["loss"] += step_metrics["loss"]
            metrics["tokens"] += batch["input_ids"].numel()

            self.global_step += 1

            if self.global_step % self.config.grad_acc_steps == 0:
                self.optimizer_step()

            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = metrics["loss"] / self.config.log_interval if metrics["tokens"] > 0 else 0.0
                ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
                tokens_per_sec = metrics["tokens"] / (time.time() - start_time + 1e-8)
                print(
                    f"Step {self.global_step}/{self.config.total_iters} | "
                    f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                    f"LR: {self.get_lr():.2e} | Tokens/s: {tokens_per_sec:.0f}"
                )
                # Reset accumulators
                metrics["loss"] = 0.0
                metrics["tokens"] = 0
                start_time = time.time()

            # Check for end of training
            if self.global_step >= self.config.total_iters:
                break

        return metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]


def main():
    """Main training entry point.

    Example usage:

    # Train 0.5B nGPT with 4k context:
    python train.py --preset 0.5B --use-ngpt --seq-len 4096

    # Train 1B GPT baseline with 1k context:
    python train.py --preset 1.0B --no-ngpt --seq-len 1024
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train GPT or nGPT")
    parser.add_argument("--preset", type=str, default="0.5B", choices=["0.5B", "1.0B"])
    parser.add_argument("--use-ngpt", action="store_true", default=True, help="Use nGPT (default: True)")
    parser.add_argument("--no-ngpt", action="store_false", dest="use_ngpt", help="Use baseline GPT")
    parser.add_argument("--seq-len", type=int, default=4096, choices=[1024, 4096, 8192])
    parser.add_argument("--total-iters", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=None, help="Override default learning rate")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--grad-acc-steps", type=int, default=1)
    args = parser.parse_args()

    # Build config from preset
    from config import ModelConfig, OptimConfig, DataConfig, TrainConfig

    model_cfg = ModelConfig.presets()[args.preset]
    model_cfg.max_seq_len = args.seq_len

    optim_cfg = OptimConfig()
    if args.use_ngpt:
        optim_cfg = optim_cfg.to_ngpt()
    else:
        optim_cfg = optim_cfg.to_gpt()
    optim_cfg.global_batch_size = args.batch_size

    if args.lr is not None:
        optim_cfg.initial_lr = args.lr
    else:
        # Default learning rates based on context length and model size
        # (from Figure 7 and paper experiments)
        if args.seq_len == 1024:
            optim_cfg.initial_lr = 2.0e-3
        elif args.seq_len == 4096:
            optim_cfg.initial_lr = 1.0e-3
        else:
            optim_cfg.initial_lr = 5.0e-4

    data_cfg = DataConfig(seq_len=args.seq_len)

    config = TrainConfig(
        model=model_cfg,
        optim=optim_cfg,
        data=data_cfg,
        total_iters=args.total_iters,
        grad_acc_steps=args.grad_acc_steps,
        use_ngpt=args.use_ngpt,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create datasets
    train_dataset = OpenWebTextDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len + 1,  # +1 for next-token prediction target
        split="train",
    )
    val_dataset = OpenWebTextDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len + 1,
        split="val",
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.optim.global_batch_size // config.grad_acc_steps,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.optim.global_batch_size // config.grad_acc_steps,
        shuffle=False,
    )

    # Create trainer
    trainer = Trainer(config)

    print(f"Training {'nGPT' if args.use_ngpt else 'GPT'} model:")
    print(f"  Preset: {args.preset}")
    print(f"  Layers: {model_cfg.n_layers}")
    print(f"  d_model: {model_cfg.d_model}")
    print(f"  n_heads: {model_cfg.n_heads}")
    print(f"  Context: {args.seq_len}")
    print(f"  LR: {optim_cfg.initial_lr}")
    print(f"  Total steps: {args.total_iters}")
    print(f"  Parameters: {sum(p.numel() for p in trainer.model.parameters()) / 1e6:.1f}M")

    # Training loop
    while trainer.global_step < config.total_iters:
        trainer.train_epoch(train_loader)

        # Periodic evaluation
        if trainer.global_step % config.eval_interval == 0:
            val_metrics = trainer.evaluate(val_loader)
            print(
                f"Eval Step {trainer.global_step} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val PPL: {val_metrics['val_perplexity']:.2f}"
            )

            # Save best checkpoint
            if val_metrics["val_loss"] < trainer.best_val_loss:
                trainer.best_val_loss = val_metrics["val_loss"]
                trainer.save_checkpoint(os.path.join(args.output_dir, "best_model.pt"))

        # Periodic checkpointing
        if trainer.global_step % config.save_interval == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_step_{trainer.global_step}.pt")
            )

    # Final save
    trainer.save_checkpoint(os.path.join(args.output_dir, "final_model.pt"))
    print("Training complete!")


if __name__ == "__main__":
    main()
