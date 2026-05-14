#!/usr/bin/env python3
"""
Fine-tuning script for MoE-POT.

Usage:
    python finetune.py --checkpoint checkpoints/pretrain/pretrain_final.pt \
                       --dataset NS_1e-5 \
                       --data_path data/fno_ns_1e-5.npy \
                       --model_size tiny
"""

import argparse
import os
import sys

import torch
from torch.nn.parallel import DataParallel

from moe_pot.model import create_moe_pot_tiny, create_moe_pot_small, create_moe_pot_medium
from moe_pot.datasets import create_dataloaders
from moe_pot.trainer import MoEPOTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="MoE-POT Fine-tuning")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--model_size", type=str, default="tiny",
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name for fine-tuning")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to fine-tuning dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/finetune",
                        help="Directory to save fine-tuned checkpoints")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--warmup_epochs", type=int, default=40,
                        help="Number of warmup epochs")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max_channels", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--no_freeze_router", action="store_true",
                        help="Do not freeze router during fine-tuning")
    parser.add_argument("--downstream", action="store_true",
                        help="Fine-tune for downstream task (500 epochs, 100 warmup)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Adjust epochs for downstream tasks
    if args.downstream:
        args.num_epochs = 500
        args.warmup_epochs = 100

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Create model
    model_kwargs = {
        "in_channels": args.max_channels,
        "patch_size": args.patch_size,
        "num_timesteps": args.num_timesteps,
    }

    if args.model_size == "tiny":
        model = create_moe_pot_tiny(**model_kwargs)
    elif args.model_size == "small":
        model = create_moe_pot_small(**model_kwargs)
    elif args.model_size == "medium":
        model = create_moe_pot_medium(**model_kwargs)

    # Create trainer and load pre-trained weights
    trainer = MoEPOTTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
    )
    trainer.load_checkpoint(args.checkpoint)

    # Create data loaders
    dataset_config = [{
        "name": args.dataset,
        "path": args.data_path,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "weight": 1.0,
    }]

    train_loader = create_dataloaders(
        dataset_configs=dataset_config,
        num_input_timesteps=args.num_timesteps,
        target_resolution=args.resolution,
        max_channels=args.max_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="train",
    )

    val_loader = create_dataloaders(
        dataset_configs=dataset_config,
        num_input_timesteps=args.num_timesteps,
        target_resolution=args.resolution,
        max_channels=args.max_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="test",
    )

    # Fine-tune
    save_dir = os.path.join(args.output_dir, args.dataset)
    trainer.finetune(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        save_dir=save_dir,
        freeze_router=not args.no_freeze_router,
    )


if __name__ == "__main__":
    main()
