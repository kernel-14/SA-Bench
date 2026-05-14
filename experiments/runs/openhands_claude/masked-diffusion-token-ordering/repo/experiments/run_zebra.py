"""
Zebra (Einstein) puzzle experiment (Section 4.2, 4.3, Table 3).

Trains MDM (19M) and ARM (42M, with/without ordering) on Zebra puzzles.
Evaluates all inference strategies and reports accuracy.

Usage:
  python experiments/run_zebra.py
  python experiments/run_zebra.py --eval_only --mdm_checkpoint outputs/zebra/mdm/best_model.pt
"""

import sys
import os
import argparse
from typing import Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIGS
from mdm import MDM
from arm import ARM
from data import get_zebra_loaders, ZebraPuzzleGenerator
from inference import mdm_solve_puzzle
from train_mdm import train as train_mdm
from train_arm import train as train_arm
from evaluate import evaluate_mdm_puzzle, evaluate_arm_puzzle
from utils import get_logger, load_checkpoint, set_seed

logger = get_logger("run_zebra")


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_zebra_dataset(data_path: str, seed: int = 42):
    """Generate Zebra puzzle dataset if it doesn't exist."""
    train_path = os.path.join(data_path, "zebra_train.json")
    test_path = os.path.join(data_path, "zebra_test.json")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.info(f"Generating Zebra puzzle dataset at {data_path}...")
        generator = ZebraPuzzleGenerator(seed=seed)
        generator.save_dataset(data_path, num_train=100_000, num_test=10_000)
        logger.info("Dataset generated.")
    else:
        logger.info(f"Zebra dataset already exists at {data_path}.")


# ---------------------------------------------------------------------------
# Full Zebra experiment
# ---------------------------------------------------------------------------

def run_zebra_experiment(
    data_path: str = "data/zebra",
    output_dir: str = "outputs/zebra",
    device: str = "cuda",
    seed: int = 42,
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_steps: int = 50,
    gumbel_noise_coeff: float = 0.5,
    eval_only: bool = False,
    mdm_checkpoint: Optional[str] = None,
    arm_checkpoint: Optional[str] = None,
    arm_ordered_checkpoint: Optional[str] = None,
    use_wandb: bool = False,
):
    set_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataset
    prepare_zebra_dataset(data_path, seed=seed)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    if not eval_only:
        # Train MDM (19M parameters, Section D.2)
        logger.info("Training MDM (19M) on Zebra puzzles...")
        train_mdm(
            task="zebra",
            model_size="19M",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            data_path=data_path,
            output_dir=os.path.join(output_dir, "mdm"),
            device=device,
            seed=seed,
            use_wandb=use_wandb,
        )
        mdm_checkpoint = os.path.join(output_dir, "mdm", "best_model.pt")

        # Train ARM without ordering (42M parameters)
        logger.info("Training ARM (42M) without ordering on Zebra puzzles...")
        train_arm(
            task="zebra",
            model_size="42M",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            data_path=data_path,
            output_dir=os.path.join(output_dir, "arm_no_order"),
            device=device,
            seed=seed,
            use_ordering=False,
            use_wandb=use_wandb,
        )
        arm_checkpoint = os.path.join(output_dir, "arm_no_order", "best_model.pt")

        # Train ARM with ordering (42M parameters)
        logger.info("Training ARM (42M) with ordering on Zebra puzzles...")
        train_arm(
            task="zebra",
            model_size="42M",
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            data_path=data_path,
            output_dir=os.path.join(output_dir, "arm_ordered"),
            device=device,
            seed=seed,
            use_ordering=True,
            use_wandb=use_wandb,
        )
        arm_ordered_checkpoint = os.path.join(output_dir, "arm_ordered", "best_model.pt")

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    _, test_loader = get_zebra_loaders(
        data_path, batch_size=batch_size, use_ordering=True
    )

    results = {}

    # Evaluate MDM with all strategies
    if mdm_checkpoint and os.path.exists(mdm_checkpoint):
        model_config = MODEL_CONFIGS["19M"]
        mdm_model = MDM(vocab_size=6, seq_len=25, model_config=model_config).to(device)
        load_checkpoint(mdm_model, None, None, mdm_checkpoint, device)

        for strategy in ["vanilla", "top_prob", "top_prob_margin"]:
            logger.info(f"Evaluating MDM ({strategy})...")
            metrics = evaluate_mdm_puzzle(
                mdm_model, test_loader, strategy, num_steps,
                gumbel_noise_coeff, device, task="zebra"
            )
            key = f"MDM ({strategy})"
            results[key] = metrics["puzzle_accuracy"]
            logger.info(f"  {key}: {metrics['puzzle_accuracy']*100:.2f}%")

    # Evaluate ARM without ordering
    if arm_checkpoint and os.path.exists(arm_checkpoint):
        model_config = MODEL_CONFIGS["42M"]
        arm_model = ARM(vocab_size=6, seq_len=25, model_config=model_config).to(device)
        load_checkpoint(arm_model, None, None, arm_checkpoint, device)

        logger.info("Evaluating ARM (w/o ordering)...")
        metrics = evaluate_arm_puzzle(
            arm_model, test_loader, device, use_ordering=False, task="zebra"
        )
        results["ARM (w/o ordering)"] = metrics["puzzle_accuracy"]
        logger.info(f"  ARM (w/o ordering): {metrics['puzzle_accuracy']*100:.2f}%")

    # Evaluate ARM with ordering
    if arm_ordered_checkpoint and os.path.exists(arm_ordered_checkpoint):
        model_config = MODEL_CONFIGS["42M"]
        arm_ordered = ARM(vocab_size=6, seq_len=25, model_config=model_config).to(device)
        load_checkpoint(arm_ordered, None, None, arm_ordered_checkpoint, device)

        logger.info("Evaluating ARM (with ordering)...")
        metrics = evaluate_arm_puzzle(
            arm_ordered, test_loader, device, use_ordering=True, task="zebra"
        )
        results["ARM (with ordering)"] = metrics["puzzle_accuracy"]
        logger.info(f"  ARM (with ordering): {metrics['puzzle_accuracy']*100:.2f}%")

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("Table 3: Zebra Puzzle Results")
    logger.info(f"{'Method':<30} {'Accuracy':<15}")
    logger.info("-" * 45)
    for method, acc in results.items():
        logger.info(f"{method:<30} {acc*100:.2f}%")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Zebra puzzle experiment")
    parser.add_argument("--data_path", type=str, default="data/zebra")
    parser.add_argument("--output_dir", type=str, default="outputs/zebra")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--gumbel_noise_coeff", type=float, default=0.5)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--mdm_checkpoint", type=str, default=None)
    parser.add_argument("--arm_checkpoint", type=str, default=None)
    parser.add_argument("--arm_ordered_checkpoint", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_zebra_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_steps=args.num_steps,
        gumbel_noise_coeff=args.gumbel_noise_coeff,
        eval_only=args.eval_only,
        mdm_checkpoint=args.mdm_checkpoint,
        arm_checkpoint=args.arm_checkpoint,
        arm_ordered_checkpoint=args.arm_ordered_checkpoint,
        use_wandb=args.use_wandb,
    )
