#!/usr/bin/env python3
"""
Pre-training script for MoE-POT.

Usage:
    python pretrain.py --config configs/pretrain_tiny.yaml
    python pretrain.py --model_size tiny --data_dir /path/to/data
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel

from moe_pot.model import create_moe_pot_tiny, create_moe_pot_small, create_moe_pot_medium
from moe_pot.datasets import create_dataloaders, DEFAULT_PRETRAIN_DATASETS
from moe_pot.trainer import MoEPOTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="MoE-POT Pre-training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--model_size", type=str, default="tiny",
                        choices=["tiny", "small", "medium"],
                        help="Model size variant")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing PDE datasets")
    parser.add_argument("--output_dir", type=str, default="checkpoints/pretrain",
                        help="Directory to save checkpoints")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of pre-training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=200,
                        help="Number of warmup epochs")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Total batch size (across all GPUs)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_timesteps", type=int, default=10,
                        help="Number of input timesteps T")
    parser.add_argument("--patch_size", type=int, default=8,
                        help="Spatial patch size P")
    parser.add_argument("--resolution", type=int, default=128,
                        help="Target spatial resolution")
    parser.add_argument("--max_channels", type=int, default=4,
                        help="Maximum number of channels (for padding)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log metrics every N epochs")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset_configs(data_dir: str, max_channels: int = 4) -> list:
    """Build dataset configurations from data directory."""
    configs = []
    for default_config in DEFAULT_PRETRAIN_DATASETS:
        config = default_config.copy()
        # Update path to use data_dir
        filename = os.path.basename(config["path"])
        config["path"] = os.path.join(data_dir, filename)
        if os.path.exists(config["path"]):
            configs.append(config)
        else:
            print(f"Warning: Dataset not found at {config['path']}, skipping.")
    return configs


def main():
    args = parse_args()

    # Load config if provided
    if args.config is not None:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU(s)")
    else:
        device = torch.device("cpu")
        num_gpus = 0
        print("Using CPU")

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
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: MoE-POT-{args.model_size.capitalize()}")
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")

    # Multi-GPU support
    if num_gpus > 1:
        model = DataParallel(model)

    # Build dataset configurations
    dataset_configs = build_dataset_configs(args.data_dir, args.max_channels)

    if len(dataset_configs) == 0:
        print("Error: No datasets found. Please check --data_dir.")
        sys.exit(1)

    print(f"Found {len(dataset_configs)} datasets for pre-training:")
    for config in dataset_configs:
        print(f"  - {config['name']}: {config['path']}")

    # Create data loaders
    train_loader = create_dataloaders(
        dataset_configs=dataset_configs,
        num_input_timesteps=args.num_timesteps,
        target_resolution=args.resolution,
        max_channels=args.max_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split="train",
    )

    # Create per-dataset validation loaders
    val_loaders = {}
    for config in dataset_configs:
        try:
            val_loader = create_dataloaders(
                dataset_configs=[config],
                num_input_timesteps=args.num_timesteps,
                target_resolution=args.resolution,
                max_channels=args.max_channels,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                split="test",
            )
            val_loaders[config["name"]] = val_loader
        except Exception as e:
            print(f"Warning: Could not create validation loader for {config['name']}: {e}")

    # Create trainer
    trainer = MoEPOTTrainer(
        model=model.module if isinstance(model, DataParallel) else model,
        device=device,
        learning_rate=args.lr,
        use_amp=not args.no_amp,
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Pre-train
    trainer.pretrain(
        train_loader=train_loader,
        val_loaders=val_loaders if val_loaders else None,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        save_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
