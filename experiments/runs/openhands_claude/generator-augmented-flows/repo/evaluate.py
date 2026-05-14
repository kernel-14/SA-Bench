"""
Evaluation script for trained consistency models.

Computes FID, KID, and IS on 50,000 generated samples vs training images.
"""

import argparse
import os

import torch
import yaml

from src.model import build_consistency_model
from src.data import get_dataset, get_dataloader
from src.metrics import EvaluationMetrics


def evaluate(cfg: dict, args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = build_consistency_model(cfg).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict"))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Dataset for real images
    dataset = get_dataset(
        name=cfg["dataset"],
        root=cfg["data_root"],
        resolution=cfg["image_resolution"],
        train=True,
    )
    dataloader = get_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Evaluator
    evaluator = EvaluationMetrics(
        device=device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )

    print(f"Evaluating with {args.num_samples} samples, {args.num_steps} sampling step(s)...")
    metrics = evaluator.evaluate(
        model=model,
        real_dataloader=dataloader,
        sigma_max=cfg["sigma_max"],
        num_steps=args.num_steps,
    )

    print("\n=== Evaluation Results ===")
    print(f"FID:  {metrics['fid']:.4f}")
    print(f"KID:  {metrics['kid_mean'] * 100:.4f} ± {metrics['kid_std'] * 100:.4f} (×10²)")
    print(f"IS:   {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f}")

    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained consistency model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of samples for evaluation (default: 50000)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for generation")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="Number of sampling steps (1=one-step generation)")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args)


if __name__ == "__main__":
    main()
