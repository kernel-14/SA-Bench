"""
Training script for gated attention models.

Supports both dense (1.7B) and MoE (15A2B) model configurations
as described in the paper.

Training settings from the paper:
  - MoE models: max LR 2e-3, cosine decay to 3e-5, 1k warmup, bsz 1024, 100k steps
  - Dense 1.7B (400B tokens): max LR 4e-3, bsz 1024
  - Dense 1.7B (3.5T tokens): max LR 4.5e-3, bsz 2048
  - Dense 1.7B 48-layer (stability): max LR 4e-3 or 8e-3, bsz 1024 or 4096
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
):
    """
    Cosine learning rate schedule with linear warmup.
    
    From the paper: warmup to max LR in 1k steps, then cosine decay to 3e-5.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


@dataclass
class TrainingConfig:
    """Training configuration matching the paper's experimental settings."""
    
    # Model type
    model_type: str = "dense"  # "dense" or "moe"
    
    # Training data
    data_path: str = "data/train"
    eval_data_path: str = "data/eval"
    
    # Training hyperparameters
    max_lr: float = 4e-3
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    total_steps: int = 100000
    batch_size: int = 1024  # Global batch size
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # AdamW optimizer
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Sequence length
    seq_len: int = 4096
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "checkpoints"
    
    # Mixed precision
    use_bf16: bool = True
    
    # Distributed training
    local_rank: int = 0
    world_size: int = 1


def build_model(model_type: str, gating_config: dict):
    """Build model based on type and gating configuration."""
    if model_type == "dense":
        from models.transformer import GatedTransformerModel, TransformerConfig
        config = TransformerConfig(**gating_config)
        return GatedTransformerModel(config)
    elif model_type == "moe":
        from models.moe_transformer import MoEConfig, MoETransformerModel
        config = MoEConfig(**gating_config)
        return MoETransformerModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return math.exp(loss)


def train(args):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    gating_config = {
        "gating_position": args.gating_position,
        "gating_granularity": args.gating_granularity,
        "head_specific": args.head_specific,
        "gating_type": args.gating_type,
        "gating_activation": args.gating_activation,
        "use_sandwich_norm": args.use_sandwich_norm,
    }
    
    model = build_model(args.model_type, gating_config)
    model = model.to(device)
    
    if args.use_bf16:
        model = model.to(torch.bfloat16)
    
    # Count parameters
    param_counts = model.get_num_params()
    print(f"Model parameters: {param_counts}")
    
    # Optimizer
    # Separate weight decay for different parameter groups
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name for nd in ["bias", "norm", "embedding"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_steps,
        min_lr_ratio=args.min_lr / args.max_lr,
    )
    
    # Training loop
    model.train()
    step = 0
    total_loss = 0.0
    
    print(f"Starting training with {args.model_type} model")
    print(f"Gating: position={args.gating_position}, granularity={args.gating_granularity}")
    print(f"Max LR: {args.max_lr}, Batch size: {args.batch_size}")
    
    # Note: In practice, you would load actual training data here
    # The paper uses a 3.5T token dataset with multilingual, math, and general content
    # For this reproduction, we provide the training infrastructure
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Training config saved to {config_path}")
    print("Training loop ready. Provide data loader to begin training.")
    
    return model, optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Train gated attention model")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="dense",
                        choices=["dense", "moe"])
    
    # Gating configuration
    parser.add_argument("--gating_position", type=str, default="sdpa_output",
                        choices=["sdpa_output", "value", "key", "query", "dense_output", "none"],
                        help="Position to apply gating (G1-G5 from paper)")
    parser.add_argument("--gating_granularity", type=str, default="elementwise",
                        choices=["elementwise", "headwise"])
    parser.add_argument("--head_specific", action="store_true", default=True,
                        help="Use head-specific gating (vs head-shared)")
    parser.add_argument("--gating_type", type=str, default="multiplicative",
                        choices=["multiplicative", "additive"])
    parser.add_argument("--gating_activation", type=str, default="sigmoid",
                        choices=["sigmoid", "silu", "identity", "rmsnorm", "ns_sigmoid"])
    parser.add_argument("--use_sandwich_norm", action="store_true", default=False)
    
    # Training hyperparameters
    parser.add_argument("--max_lr", type=float, default=4e-3)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Data
    parser.add_argument("--data_path", type=str, default="data/train")
    parser.add_argument("--eval_data_path", type=str, default="data/eval")
    parser.add_argument("--seq_len", type=int, default=4096)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)
    
    # Mixed precision
    parser.add_argument("--use_bf16", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Handle "none" gating position
    if args.gating_position == "none":
        args.gating_position = None
    
    train(args)


if __name__ == "__main__":
    main()
