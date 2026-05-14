#!/usr/bin/env python3
"""
Training script for SAM 2.

Implements the two-stage training process:
1. Pre-training on SA-1B (static images)
2. Full training on mixed video + image data
3. Optional fine-tuning with 16-frame sequences

Usage:
    python scripts/train.py --config configs/sam2_config.yaml --stage pretrain
    python scripts/train.py --config configs/sam2_config.yaml --stage full
    python scripts/train.py --config configs/sam2_config.yaml --stage finetune
"""

import argparse
import yaml
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from sam2.model import SAM2
from sam2.training import SAM2Trainer, SAM2Loss, InteractivePromptSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM 2")
    parser.add_argument("--config", type=str, default="configs/sam2_config.yaml",
                        help="Path to config file")
    parser.add_argument("--stage", type=str, default="pretrain",
                        choices=["pretrain", "full", "finetune"],
                        help="Training stage")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--encoder_size", type=str, default=None,
                        help="Override encoder size")
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
        num_multimask_outputs=model_cfg["num_multimask_outputs"],
        num_mask_decoder_blocks=model_cfg["num_mask_decoder_blocks"],
    )

    print(f"Built SAM 2 with {encoder_size} encoder")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Determine stage config
    if args.stage == "pretrain":
        stage_cfg = config["pretraining"]
        layer_decay_map = {
            "tiny": config["pretraining"]["layer_decay_T"],
            "small": config["pretraining"]["layer_decay_S"],
            "base_plus": config["pretraining"]["layer_decay_Bplus"],
            "large": config["pretraining"]["layer_decay_L"],
        }
        layer_decay = layer_decay_map.get(encoder_size, 0.9)
    elif args.stage == "full":
        stage_cfg = config["full_training"]
        layer_decay_map = {
            "tiny": config["pretraining"]["layer_decay_T"],
            "small": config["pretraining"]["layer_decay_S"],
            "base_plus": config["pretraining"]["layer_decay_Bplus"],
            "large": config["pretraining"]["layer_decay_L"],
        }
        layer_decay = layer_decay_map.get(encoder_size, 0.9)
    else:
        stage_cfg = config["finetuning"]
        layer_decay = 0.9

    # Create trainer
    trainer = SAM2Trainer(
        model=model,
        loss_fn=SAM2Loss(
            focal_weight=config["pretraining"]["mask_loss_focal_weight"],
            dice_weight=config["pretraining"]["mask_loss_dice_weight"],
            iou_weight=config["pretraining"]["iou_loss_weight"],
        ),
        base_lr=stage_cfg.get("learning_rate", 4e-4),
        weight_decay=config["pretraining"]["weight_decay"],
        max_grad_norm=config["pretraining"]["gradient_clip_max"],
        warmup_steps=config["pretraining"]["warmup_steps"],
        cooldown_steps=config["pretraining"]["cooldown_steps"],
        total_steps=stage_cfg.get("steps", 90000),
        timescale=config["pretraining"]["lr_timescale"],
        layer_decay=layer_decay,
        device=args.device,
    )

    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_step = checkpoint["current_step"]
        print(f"Resumed from step {trainer.current_step}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nStarting {args.stage} training...")
    print(f"Stage config:")
    for k, v in stage_cfg.items():
        print(f"  {k}: {v}")

    # Note: Actual training requires data loading setup
    # This is a skeleton that would be connected to a data pipeline
    print("\nTraining infrastructure ready.")
    print("Connect to data pipeline to begin training.")


if __name__ == "__main__":
    main()
