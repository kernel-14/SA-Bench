"""Main pretraining script for OLMoE-1B-7B.

Usage:
    python scripts/train.py --data_path /path/to/data --output_dir /path/to/output

This script implements the pretraining procedure from Section 2 and Appendix B.
"""

import argparse
import os
import sys
import math
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Add the repository root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from olmoe.models.configuration import OLMoEConfig
from olmoe.models.transformer import OLMoEModel, create_olmoe_model
from olmoe.training.trainer import OLMoETrainer, get_cosine_schedule_with_warmup
from olmoe.data.pretraining import OLMoEPretrainingDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train OLMoE-1B-7B")
    parser.add_argument(
        "--data_path",
        type=str,
        nargs="+",
        required=True,
        help="Paths to pretraining data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size in samples (per device)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1220000,
        help="Maximum number of training steps (1.2M for 5T tokens)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2500,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Configuration
    config = OLMoEConfig(
        batch_size_samples=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    # Create model
    print(f"Creating OLMoE-1B-7B model...")
    model = create_olmoe_model()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (~{total_params / 1e9:.1f}B)")

    # Create dataset
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    except ImportError:
        print("Warning: transformers not available, using dummy tokenizer")
        # Dummy tokenizer for demonstration
        class DummyTokenizer:
            pad_token_id = 0
            def encode(self, text):
                return [hash(c) % config.vocab_size for c in text[:1000]]
        tokenizer = DummyTokenizer()

    dataset = OLMoEPretrainingDataset(
        data_paths=args.data_path,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Create trainer
    trainer = OLMoETrainer(model=model, config=config)

    # Set up learning rate scheduler
    trainer.scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
        min_lr_ratio=config.min_lr / config.peak_lr,
    )

    # Training loop
    print(f"Starting training for {args.max_steps} steps...")
    os.makedirs(args.output_dir, exist_ok=True)

    step = 0
    total_tokens = 0
    start_time = time.time()

    for batch in dataloader:
        if step >= args.max_steps:
            break

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        metrics = trainer.train_step(input_ids, labels)

        total_tokens += input_ids.numel()

        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            print(
                f"Step {step}/{args.max_steps} | "
                f"CE Loss: {metrics['ce_loss']:.4f} | "
                f"LB Loss: {metrics.get('lb_loss', 0):.4f} | "
                f"RZ Loss: {metrics.get('rz_loss', 0):.4f} | "
                f"Total Loss: {metrics['total_loss']:.4f} | "
                f"Grad Norm: {metrics['grad_norm']:.4f} | "
                f"LR: {metrics['lr']:.2e} | "
                f"Tokens/s: {tokens_per_sec:.0f}"
            )

        if step % args.save_every == 0 and step > 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{step}.pt")
            trainer.save_checkpoint(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        step += 1

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, "checkpoint-final.pt")
    trainer.save_checkpoint(final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")
    print(f"Total tokens processed: {total_tokens:,}")


if __name__ == "__main__":
    main()
