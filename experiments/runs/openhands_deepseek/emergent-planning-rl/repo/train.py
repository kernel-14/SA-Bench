"""
Main training script for DRC agent on Sokoban.

Reproduces the training setup from Guez et al. (2019):
- DRC(3,3) agent with 32 channels
- IMPALA training for 250M transitions
- Boxoban unfiltered training set (900k levels)
"""

import os
import sys
import argparse
import numpy as np
import torch

from configs.config import Config
from models.drc import DRCNet
from training.trainer import IMPALATrainer
from environment.sokoban import parse_boxoban_level


def load_boxoban_levels(data_dir: str, split: str = "unfiltered") -> list:
    """
    Load Boxoban levels from text files.

    The Boxoban dataset is expected to be in:
        data_dir/boxoban-levels-master/{split}/{train,valid,test}/

    Returns:
        levels: list of level strings
    """
    levels = []
    level_dir = os.path.join(data_dir, "boxoban-levels-master", split, "train")
    if not os.path.exists(level_dir):
        # Try alternative path
        level_dir = os.path.join(data_dir, split, "train")
    if not os.path.exists(level_dir):
        print(f"Warning: Level directory {level_dir} not found. Using synthetic levels.")
        return _generate_synthetic_levels(1000)

    for filename in sorted(os.listdir(level_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(level_dir, filename)
            with open(filepath, "r") as f:
                content = f.read()
            # Split content into individual levels (separated by blank lines or ;)
            level_strs = content.strip().split("\n\n")
            if len(level_strs) <= 1:
                level_strs = content.strip().split(";")
            levels.extend([s.strip() for s in level_strs if s.strip()])

    return levels


def _generate_synthetic_levels(num_levels: int) -> list:
    """Generate simple synthetic Sokoban levels for testing."""
    import random
    levels = []
    for _ in range(num_levels):
        grid = []
        for r in range(8):
            row = ""
            for c in range(8):
                if r == 0 or r == 7 or c == 0 or c == 7:
                    row += "#"
                else:
                    row += " "
            grid.append(row)
        # Place targets and boxes
        grid[1] = grid[1][:1] + "." + grid[1][2:4] + "." + grid[1][5:]
        grid[2] = grid[2][:1] + "$" + grid[2][2:4] + "$" + grid[2][5:]
        # Place agent
        grid[6] = grid[6][:2] + "@" + grid[6][3:]
        levels.append("\n".join(grid))
    return levels


def main():
    parser = argparse.ArgumentParser(description="Train DRC agent on Sokoban")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing Boxoban levels")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on (cuda or cpu)")
    parser.add_argument("--D", type=int, default=3,
                        help="Number of ConvLSTM layers")
    parser.add_argument("--N", type=int, default=3,
                        help="Number of internal ticks per step")
    parser.add_argument("--hidden_channels", type=int, default=32,
                        help="Number of hidden channels")
    parser.add_argument("--total_transitions", type=int, default=250_000_000,
                        help="Total training transitions")
    parser.add_argument("--checkpoint_interval", type=int, default=1_000_000,
                        help="Transitions between checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    config = Config()
    config.drc.D = args.D
    config.drc.N = args.N
    config.drc.hidden_channels = args.hidden_channels
    config.training.total_transitions = args.total_transitions
    config.checkpoint_interval = args.checkpoint_interval
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    config.data_dir = args.data_dir

    # Create model
    model = DRCNet(
        input_channels=config.sokoban.num_channels,
        hidden_channels=config.drc.hidden_channels,
        num_layers=config.drc.D,
        num_ticks=config.drc.N,
        num_actions=5,
        kernel_size=config.drc.kernel_size,
        padding=config.drc.padding,
        grid_size=config.sokoban.grid_size,
        bottom_up_skip=config.drc.bottom_up_skip,
        top_down_skip=config.drc.top_down_skip,
        pool_and_inject=config.drc.pool_and_inject,
    )

    print(f"Model: DRC({config.drc.D},{config.drc.N})")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Load levels
    train_levels = load_boxoban_levels(config.data_dir, "unfiltered")
    print(f"Loaded {len(train_levels)} training levels")

    # Also load some validation levels for evaluation
    valid_levels = load_boxoban_levels(config.data_dir, "valid")
    if not valid_levels:
        valid_levels = train_levels[:1000]  # Use subset for validation
    print(f"Using {len(valid_levels)} validation levels")

    # Create trainer
    trainer = IMPALATrainer(
        model=model,
        device=device,
        gamma=config.training.gamma,
        vtrace_lambda=config.training.vtrace_lambda,
        baseline_cost=config.training.baseline_cost,
        entropy_cost=config.training.entropy_cost,
        action_l2_penalty=config.training.action_l2_penalty,
        head_l2_regularization=config.training.head_l2_regularization,
        learning_rate=config.training.learning_rate,
        final_learning_rate=config.training.final_learning_rate,
        total_transitions=config.training.total_transitions,
        unroll_length=config.training.unroll_length,
        batch_size=config.training.batch_size,
        checkpoint_dir=config.checkpoint_dir,
        log_dir=config.log_dir,
        checkpoint_interval=config.checkpoint_interval,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("Starting training...")
    trainer.train(
        levels=train_levels,
        eval_levels=valid_levels,
        save_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
