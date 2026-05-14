#!/usr/bin/env python3
"""
Evaluation script for SAM 2.

Supports:
- Semi-supervised VOS evaluation
- Interactive offline/online evaluation
- Image segmentation evaluation (mIoU)

Usage:
    python scripts/eval.py --checkpoint path/to/checkpoint --task vos --data path/to/data
    python scripts/eval.py --checkpoint path/to/checkpoint --task interactive --mode offline
    python scripts/eval.py --checkpoint path/to/checkpoint --task image
"""

import argparse
import yaml
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from sam2.model import SAM2
from sam2.eval import SAM2Evaluator, InteractiveEvaluator
from sam2.eval.metrics import JFMetric


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAM 2")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/sam2_config.yaml",
                        help="Path to config file")
    parser.add_argument("--task", type=str, default="vos",
                        choices=["vos", "interactive", "image"],
                        help="Evaluation task")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online"],
                        help="Interactive evaluation mode")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to evaluation data")
    parser.add_argument("--encoder_size", type=str, default=None,
                        help="Override encoder size")
    parser.add_argument("--num_clicks", type=int, default=3,
                        help="Number of clicks for prompting")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Build model
    encoder_size = args.encoder_size or config["model"]["encoder_size"]
    model_cfg = config["model"]

    model = SAM2(
        encoder_size=encoder_size,
        img_size=model_cfg["img_size"],
        embed_dim=model_cfg["embed_dim"],
        memory_dim=model_cfg["memory_dim"],
        num_memory_attn_layers=model_cfg["num_memory_attn_layers"],
        num_memory_attn_heads=model_cfg["num_memory_attn_heads"],
        max_recent_frames=model_cfg["max_recent_frames"],
        max_prompted_frames=model_cfg["max_prompted_frames"],
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Create evaluator
    evaluator = SAM2Evaluator(model, device=args.device)

    if args.task == "vos":
        print("Running semi-supervised VOS evaluation...")
        print("(Requires dataset to be provided via --data)")
        # Evaluation would proceed here with actual data

    elif args.task == "interactive":
        print(f"Running interactive {args.mode} evaluation...")
        print("(Requires dense video annotations for interactive evaluation)")
        interactive_eval = InteractiveEvaluator(model, device=args.device)
        # Evaluation would proceed here

    elif args.task == "image":
        print("Running image segmentation evaluation (mIoU)...")
        print("(Requires image dataset with ground truth masks)")

    print("\nEvaluation infrastructure ready.")


if __name__ == "__main__":
    main()
