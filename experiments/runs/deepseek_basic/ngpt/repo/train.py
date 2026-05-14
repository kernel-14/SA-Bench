"""
Training script for nGPT (Normalized Transformer) and baseline GPT.

Implements the training procedure described in the paper:
- Adam optimizer without weight decay for nGPT
- AdamW with weight decay for baseline GPT
- Cosine annealing learning rate schedule
- No warmup for nGPT
- Weight normalization after each optimizer step for nGPT
"""

import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import nGPT, BaselineGPT, create_ngpt_model


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, data, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len - 1

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
):
    """
    Cosine annealing learning rate scheduler with optional warmup.
    For nGPT: warmup_steps = 0 (no warmup).
    For GPT: warmup_steps = 2000.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step_ngpt(model, batch, optimizer, scheduler, device):
    """Single training step for nGPT."""
    model.train()
    x, y = batch
    x, y = x.to(device), y.to(device)

    # Forward pass
    logits, loss = model(x, targets=y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping (optional, not explicitly mentioned for nGPT)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Optimizer step
    optimizer.step()

    # CRITICAL: Normalize all matrices after optimizer step (Section 2.6, step 2)
    model.normalize_weights()

    # Update learning rate
    scheduler.step()

    return loss.item()


def train_step_gpt(model, batch, optimizer, scheduler, device):
    """Single training step for baseline GPT."""
    model.train()
    x, y = batch
    x, y = x.to(device), y.to(device)

    logits, loss = model(x, targets=y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, eval_loader, device, max_batches=None):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in eval_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, targets=y)
        n_tokens = (y != -1).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        n_batches += 1

        if max_batches and n_batches >= max_batches:
            break

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def estimate_perplexity(model, data_loader, device, max_batches=10):
    """Estimate perplexity on a dataset."""
    loss = evaluate(model, data_loader, device, max_batches)
    return math.exp(loss)


def train(
    model,
    train_loader,
    eval_loader,
    optimizer,
    scheduler,
    device,
    total_steps: int,
    eval_every: int = 1000,
    log_every: int = 100,
    model_type: str = 'ngpt',
    save_checkpoint_path: str = None,
):
    """Main training loop."""
    train_step_fn = train_step_ngpt if model_type == 'ngpt' else train_step_gpt

    best_eval_loss = float('inf')
    train_losses = []
    eval_losses = []

    print(f"Starting training for {total_steps} steps (model: {model_type})")
    start_time = time.time()

    step = 0
    train_iter = iter(train_loader)

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = train_step_fn(model, batch, optimizer, scheduler, device)
        train_losses.append(loss)

        if step % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            print(
                f"Step {step:6d}/{total_steps} | "
                f"Loss: {loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

        if step % eval_every == 0 and step > 0:
            eval_loss = evaluate(model, eval_loader, device)
            eval_losses.append((step, eval_loss))
            perplexity = math.exp(eval_loss)
            print(
                f"--- Eval @ step {step}: "
                f"Loss: {eval_loss:.4f}, "
                f"Perplexity: {perplexity:.2f} ---"
            )

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                if save_checkpoint_path:
                    torch.save(
                        {
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'eval_loss': eval_loss,
                            'model_type': model_type,
                        },
                        save_checkpoint_path,
                    )
                    print(f"Checkpoint saved to {save_checkpoint_path}")

        step += 1

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    return {
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'best_eval_loss': best_eval_loss,
        'total_time': total_time,
    }


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description='Train nGPT or GPT models')
    parser.add_argument('--model_type', type=str, default='ngpt',
                        choices=['ngpt', 'gpt'],
                        help='Type of model to train')
    parser.add_argument('--model_size', type=str, default='0.5B',
                        choices=['0.5B', '1B'],
                        help='Model size configuration')
    parser.add_argument('--vocab_size', type=int, default=32000,
                        help='Vocabulary size')
    parser.add_argument('--max_seq_len', type=int, default=4096,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay (only for GPT)')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps (0 for nGPT, 2000 for GPT)')
    parser.add_argument('--total_steps', type=int, default=100000,
                        help='Total training steps')
    parser.add_argument('--eval_every', type=int, default=1000,
                        help='Evaluate every N steps')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    if args.model_type == 'ngpt':
        model = create_ngpt_model(
            config=args.model_size,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
        )
    else:
        configs = {
            '0.5B': {'d_model': 1024, 'n_heads': 16, 'n_layers': 24, 'd_mlp': 4096},
            '1B': {'d_model': 1280, 'n_heads': 20, 'n_layers': 36, 'd_mlp': 5120},
        }
        cfg = configs[args.model_size]
        model = BaselineGPT(
            vocab_size=args.vocab_size,
            d_model=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],
            d_mlp=cfg['d_mlp'],
            max_seq_len=args.max_seq_len,
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_type} ({args.model_size}), "
          f"Parameters: {n_params / 1e6:.1f}M")

    # Configure optimizer
    if args.model_type == 'ngpt':
        optimizer = model.configure_optimizers(args.learning_rate)
    else:
        optimizer = model.configure_optimizers(
            args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
    )

    print("Training setup complete.")
    print("Note: This is a static reproduction. For actual training, "
          "provide data loaders and run the training loop.")
    print("The model architecture and training logic match the paper description.")


if __name__ == '__main__':
    main()
