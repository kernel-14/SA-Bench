#!/usr/bin/env python3
"""
Evaluation script for MoE-POT.

Evaluates a trained model on test datasets and reports L2RE metrics.
Also runs interpretability analysis if requested.

Usage:
    python evaluate.py --checkpoint checkpoints/pretrain/pretrain_final.pt \
                       --model_size tiny \
                       --data_dir data \
                       --zero_shot
"""

import argparse
import os
import json
from typing import Dict

import torch

from moe_pot.model import create_moe_pot_tiny, create_moe_pot_small, create_moe_pot_medium
from moe_pot.datasets import create_dataloaders, DEFAULT_PRETRAIN_DATASETS
from moe_pot.trainer import MoEPOTTrainer, l2_relative_error
from moe_pot.interpretability import run_interpretability_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="MoE-POT Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="tiny",
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_timesteps", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max_channels", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--zero_shot", action="store_true",
                        help="Evaluate in zero-shot mode (no fine-tuning)")
    parser.add_argument("--interpretability", action="store_true",
                        help="Run interpretability analysis")
    parser.add_argument("--rollout_steps", type=int, default=1,
                        help="Number of auto-regressive rollout steps")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Load checkpoint
    trainer = MoEPOTTrainer(model=model, device=device)
    trainer.load_checkpoint(args.checkpoint)

    # Build dataset configs
    dataset_configs = []
    for default_config in DEFAULT_PRETRAIN_DATASETS:
        config = default_config.copy()
        filename = os.path.basename(config["path"])
        config["path"] = os.path.join(args.data_dir, filename)
        if os.path.exists(config["path"]):
            dataset_configs.append(config)

    if len(dataset_configs) == 0:
        print("No datasets found. Please check --data_dir.")
        return

    # Evaluate on each dataset
    results = {}
    dataset_loaders = {}

    for config in dataset_configs:
        name = config["name"]
        print(f"\nEvaluating on {name}...")

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
            dataset_loaders[name] = val_loader

            metrics = trainer.evaluate(val_loader, rollout_steps=args.rollout_steps)
            results[name] = metrics
            print(f"  L2RE: {metrics['l2re']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {"error": str(e)}

    # Print summary table
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<20} {'L2RE':<10}")
    print("-" * 30)
    for name, metrics in results.items():
        if "l2re" in metrics:
            print(f"{name:<20} {metrics['l2re']:.4f}")
        else:
            print(f"{name:<20} ERROR")

    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Run interpretability analysis
    if args.interpretability and len(dataset_loaders) > 0:
        print("\nRunning interpretability analysis...")
        interp_results = run_interpretability_analysis(
            model=model,
            dataset_loaders=dataset_loaders,
            device=device,
            output_dir=os.path.join(args.output_dir, "interpretability"),
        )


if __name__ == "__main__":
    main()
