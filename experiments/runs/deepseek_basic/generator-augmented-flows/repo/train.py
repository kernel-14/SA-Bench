"""
Main training script for consistency models with Generator-Augmented Flows.

Supports three coupling strategies:
- IC (independent coupling): baseline consistency training
- OT (minibatch optimal transport): batch-OT consistency training  
- GC (generator-augmented coupling): our method with joint learning

Usage:
    python train.py --config configs/cifar10_ict.py --coupling gc --mu 0.5
    python train.py --config configs/cifar10_ict.py --coupling ic
    python train.py --config configs/cifar10_ict.py --coupling ot

Based on the paper "Improving Consistency Models with Generator-Augmented Flows".
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from importlib.machinery import SourceFileLoader

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consistency_models.model import ConsistencyModel, SongUNet
from consistency_models.training import (
    train_consistency_model,
    ConsistencyTrainingConfig,
)
from consistency_models.metrics import evaluate_model, InceptionV3FeatureExtractor
from consistency_models.scheduling import noise_schedule_karras


def load_config(config_path: str) -> dict:
    """Load a configuration from a Python file."""
    loader = SourceFileLoader("config", config_path)
    mod = loader.load_module()
    return mod.config


def get_dataset(name: str, resolution: int, train: bool = True) -> datasets.VisionDataset:
    """
    Get a dataset by name.

    Args:
        name: "cifar10", "imagenet32", "celeba64", "lsun_church"
        resolution: Image resolution
        train: Whether to return train or test set

    Returns:
        Dataset object
    """
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution) if name != "cifar10" else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1]
    ])

    if name == "cifar10":
        return datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )
    elif name == "imagenet32":
        # ImageNet 32x32
        return datasets.ImageFolder(
            root="./data/imagenet32/train" if train else "./data/imagenet32/val",
            transform=transform,
        )
    elif name == "celeba64":
        return datasets.CelebA(
            root="./data", split="train" if train else "valid",
            download=True, transform=transform,
        )
    elif name == "lsun_church":
        return datasets.LSUN(
            root="./data", classes=["church_outdoor_train" if train else "church_outdoor_val"],
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


def build_model(config: dict, device: torch.device) -> ConsistencyModel:
    """
    Build a consistency model from configuration.

    Args:
        config: Configuration dictionary
        device: Torch device

    Returns:
        ConsistencyModel instance
    """
    img_resolution = config["img_resolution"]
    img_channels = config.get("img_channels", 3)

    # Build SongUNet backbone
    # Handle variable num_blocks per resolution
    num_blocks = config["num_blocks"]
    channel_mult = config["channel_mult"]

    network = SongUNet(
        img_resolution=img_resolution,
        in_channels=img_channels,
        out_channels=img_channels,
        model_channels=config["model_channels"],
        channel_mult=channel_mult,
        num_blocks=num_blocks if isinstance(num_blocks, int) else num_blocks[0],
        attn_resolutions=config.get("attn_resolutions", []),
        dropout=config.get("dropout", 0.0),
        embedding_type=config.get("embedding_type", "positional"),
    )

    # Wrap in consistency model
    model = ConsistencyModel(
        network=network,
        sigma_data=config.get("sigma_data", 0.5),
        sigma_min=config.get("sigma_min", 0.002),
    )

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Train consistency models with Generator-Augmented Flows"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration file")
    parser.add_argument("--coupling", type=str, default=None,
                        choices=["ic", "ot", "gc"],
                        help="Coupling type (overrides config)")
    parser.add_argument("--mu", type=float, default=None,
                        help="Joint learning factor μ for GC (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Total training steps (overrides config)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate a trained model")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load config
    base_config = load_config(args.config)

    # Override config with CLI args
    if args.coupling is not None:
        base_config["coupling"] = args.coupling
    if args.mu is not None:
        base_config["gc_mu"] = args.mu
    if args.batch_size is not None:
        base_config["batch_size"] = args.batch_size
    if args.lr is not None:
        base_config["learning_rate"] = args.lr
    if args.total_steps is not None:
        base_config["total_steps"] = args.total_steps

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(base_config, f, indent=2)

    # Build model
    model = build_model(base_config, device)
    print(f"Model built with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint.get("step", 0)
    else:
        start_step = 0

    if args.eval_only:
        # Evaluation only
        dataset = get_dataset(base_config["dataset"], base_config["img_resolution"], train=False)
        dataloader = DataLoader(
            dataset, batch_size=base_config["batch_size"],
            shuffle=False, num_workers=4, pin_memory=True,
        )
        feature_extractor = InceptionV3FeatureExtractor().to(device)
        results = evaluate_model(
            model, dataloader, feature_extractor,
            num_samples=50000, batch_size=base_config["batch_size"],
            device=device,
        )
        print("Evaluation Results:")
        print(f"  FID: {results['fid']:.2f}")
        print(f"  KID: {results['kid']:.4f}")
        print(f"  IS:  {results['is_mean']:.2f} ± {results['is_std']:.2f}")
        return

    # Get dataset
    train_dataset = get_dataset(base_config["dataset"], base_config["img_resolution"], train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=base_config["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )

    # Create training config
    train_config = ConsistencyTrainingConfig(
        sigma_min=base_config["sigma_min"],
        sigma_max=base_config["sigma_max"],
        rho=base_config["rho"],
        s0=base_config["s0"],
        s1=base_config["s1"],
        P_mean=base_config["P_mean"],
        P_std=base_config["P_std"],
        batch_size=base_config["batch_size"],
        total_steps=base_config["total_steps"],
        learning_rate=base_config["learning_rate"],
        optimizer=base_config["optimizer"],
        ema_decay=base_config.get("ema_decay", 0.999),
        coupling=base_config["coupling"],
        gc_mu=base_config.get("gc_mu", 0.5),
        ot_solver=base_config.get("ot_solver", "sinkhorn"),
        sigma_data=base_config.get("sigma_data", 0.5),
        distance_fn_name=base_config.get("distance_fn", "pseudo_huber"),
        distance_fn_kwargs=base_config.get("distance_kwargs", {}),
        log_every=base_config.get("log_every", 100),
        eval_every=base_config.get("eval_every", 5000),
    )

    coupling_name = {"ic": "Independent Coupling", "ot": "Mini-batch OT", "gc": f"Generator-Augmented (μ={train_config.gc_mu})"}
    print(f"\n{'='*60}")
    print(f"Training with {coupling_name[base_config['coupling']]}")
    print(f"Dataset: {base_config['dataset']} ({base_config['img_resolution']}x{base_config['img_resolution']})")
    print(f"Batch size: {base_config['batch_size']}")
    print(f"Total steps: {base_config['total_steps']}")
    print(f"Learning rate: {base_config['learning_rate']}")
    print(f"Optimizer: {base_config['optimizer']}")
    print(f"{'='*60}\n")

    # Training callback for checkpointing
    def save_callback(step, loss, model):
        if step > 0 and step % train_config.eval_every == 0:
            checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"model_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "loss": loss,
                "config": base_config,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Train
    stats = train_consistency_model(
        model, train_loader, train_config, device,
        callback=save_callback,
    )

    # Save final model
    final_path = os.path.join(args.output_dir, "checkpoints", "model_final.pt")
    torch.save({
        "step": train_config.total_steps,
        "model_state_dict": model.state_dict(),
        "config": base_config,
    }, final_path)
    print(f"Final model saved to {final_path}")

    # Save stats
    with open(os.path.join(args.output_dir, "training_stats.json"), "w") as f:
        json.dump({k: v for k, v in stats.items() if k not in ["step_times"]}, f, indent=2)

    print("\nTraining complete!")
    print(f"Average loss: {np.mean(stats['losses'][-100:]):.6f}")


if __name__ == "__main__":
    main()
