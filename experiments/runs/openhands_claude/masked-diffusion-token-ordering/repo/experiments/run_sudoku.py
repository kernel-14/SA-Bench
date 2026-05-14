"""
Sudoku experiment (Section 4.2, 4.3, 4.5, Tables 2 and 5).

Trains MDM (6M) and ARM (42M, with/without ordering) on Sudoku puzzles.
Evaluates all inference strategies and reports accuracy.

Usage:
  # Train all models
  python experiments/run_sudoku.py --train_all

  # Evaluate a trained MDM
  python experiments/run_sudoku.py --eval_only --mdm_checkpoint outputs/sudoku/mdm/best_model.pt

  # Full experiment (train + eval)
  python experiments/run_sudoku.py
"""

import sys
import os
import argparse
from typing import Dict, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIGS
from mdm import MDM
from arm import ARM
from data import get_sudoku_loaders
from inference import mdm_solve_puzzle
from train_mdm import train as train_mdm
from train_arm import train as train_arm
from evaluate import evaluate_mdm_puzzle, evaluate_arm_puzzle
from utils import get_logger, load_checkpoint, set_seed, compute_sudoku_accuracy

logger = get_logger("run_sudoku")


# ---------------------------------------------------------------------------
# Full Sudoku experiment
# ---------------------------------------------------------------------------

def run_sudoku_experiment(
    data_path: str = "data/sudoku",
    output_dir: str = "outputs/sudoku",
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
    hard_test: bool = False,
    use_wandb: bool = False,
):
    set_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    if not eval_only:
        # Train MDM (6M parameters, Section D.2)
        logger.info("Training MDM (6M) on Sudoku...")
        train_mdm(
            task="sudoku",
            model_size="6M",
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
        logger.info("Training ARM (42M) without ordering on Sudoku...")
        train_arm(
            task="sudoku",
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
        logger.info("Training ARM (42M) with ordering on Sudoku...")
        train_arm(
            task="sudoku",
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
    _, test_loader, hard_test_loader = get_sudoku_loaders(
        data_path, batch_size=batch_size, use_ordering=True
    )
    eval_loader = hard_test_loader if hard_test else test_loader
    test_name = "hard test" if hard_test else "test"

    results = {}

    # Evaluate MDM with all strategies
    if mdm_checkpoint and os.path.exists(mdm_checkpoint):
        model_config = MODEL_CONFIGS["6M"]
        mdm_model = MDM(vocab_size=10, seq_len=81, model_config=model_config).to(device)
        load_checkpoint(mdm_model, None, None, mdm_checkpoint, device)

        for strategy in ["vanilla", "top_prob", "top_prob_margin"]:
            logger.info(f"Evaluating MDM ({strategy}) on {test_name} set...")
            metrics = evaluate_mdm_puzzle(
                mdm_model, eval_loader, strategy, num_steps,
                gumbel_noise_coeff, device, task="sudoku"
            )
            key = f"MDM ({strategy})"
            results[key] = metrics["puzzle_accuracy"]
            logger.info(f"  {key}: {metrics['puzzle_accuracy']*100:.2f}%")

    # Evaluate ARM without ordering
    if arm_checkpoint and os.path.exists(arm_checkpoint):
        model_config = MODEL_CONFIGS["42M"]
        arm_model = ARM(vocab_size=10, seq_len=81, model_config=model_config).to(device)
        load_checkpoint(arm_model, None, None, arm_checkpoint, device)

        logger.info(f"Evaluating ARM (w/o ordering) on {test_name} set...")
        metrics = evaluate_arm_puzzle(
            arm_model, eval_loader, device, use_ordering=False, task="sudoku"
        )
        results["ARM (w/o ordering)"] = metrics["puzzle_accuracy"]
        logger.info(f"  ARM (w/o ordering): {metrics['puzzle_accuracy']*100:.2f}%")

    # Evaluate ARM with ordering
    if arm_ordered_checkpoint and os.path.exists(arm_ordered_checkpoint):
        model_config = MODEL_CONFIGS["42M"]
        arm_ordered = ARM(vocab_size=10, seq_len=81, model_config=model_config).to(device)
        load_checkpoint(arm_ordered, None, None, arm_ordered_checkpoint, device)

        logger.info(f"Evaluating ARM (with ordering) on {test_name} set...")
        metrics = evaluate_arm_puzzle(
            arm_ordered, eval_loader, device, use_ordering=True, task="sudoku"
        )
        results["ARM (with ordering)"] = metrics["puzzle_accuracy"]
        logger.info(f"  ARM (with ordering): {metrics['puzzle_accuracy']*100:.2f}%")

    # Print summary table
    table_name = "Table 5 (Hard Sudoku)" if hard_test else "Table 2 (Sudoku)"
    logger.info(f"\n{'='*60}")
    logger.info(f"{table_name}")
    logger.info(f"{'Method':<30} {'Accuracy':<15}")
    logger.info("-" * 45)
    for method, acc in results.items():
        logger.info(f"{method:<30} {acc*100:.2f}%")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Sudoku experiment")
    parser.add_argument("--data_path", type=str, default="data/sudoku")
    parser.add_argument("--output_dir", type=str, default="outputs/sudoku")
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
    parser.add_argument("--hard_test", action="store_true",
                        help="Evaluate on hard test set (Table 5)")
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sudoku_experiment(
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
        hard_test=args.hard_test,
        use_wandb=args.use_wandb,
    )
