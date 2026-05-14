"""
Data preparation scripts for all experiments.

Usage:
  python experiments/prepare_data.py --task sudoku --data_path data/sudoku
  python experiments/prepare_data.py --task zebra --data_path data/zebra
  python experiments/prepare_data.py --task slimpajama --data_path data/slimpajama
"""

import sys
import os
import argparse
import csv
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import ZebraPuzzleGenerator
from utils import get_logger

logger = get_logger("prepare_data")


# ---------------------------------------------------------------------------
# Sudoku data preparation
# ---------------------------------------------------------------------------

def prepare_sudoku(
    data_path: str,
    kaggle_csv: Optional[str] = None,
    seed: int = 42,
):
    """
    Prepare Sudoku dataset from Radcliffe (2020) Kaggle dataset.

    The dataset is available at: https://www.kaggle.com/dsv/1495975
    Download 'sudoku.csv' and pass it as kaggle_csv.

    Shah et al. (2024) filtered puzzles solvable with 7 fixed strategies.
    For the hard test set, we use the remaining puzzles.

    If kaggle_csv is not provided, generates a small synthetic dataset for testing.
    """
    os.makedirs(data_path, exist_ok=True)

    if kaggle_csv and os.path.exists(kaggle_csv):
        logger.info(f"Processing Kaggle Sudoku dataset from {kaggle_csv}...")
        _process_kaggle_sudoku(kaggle_csv, data_path, seed)
    else:
        logger.info("Kaggle CSV not provided. Generating synthetic Sudoku dataset...")
        _generate_synthetic_sudoku(data_path, seed)


def _process_kaggle_sudoku(kaggle_csv: str, data_path: str, seed: int):
    """Process the Kaggle Sudoku CSV into train/test/hard_test splits."""
    import random
    rng = random.Random(seed)

    all_puzzles = []
    with open(kaggle_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle = row.get("puzzle", row.get("quizzes", ""))
            solution = row.get("solution", row.get("solutions", ""))
            difficulty = row.get("difficulty", "0")
            if len(puzzle) == 81 and len(solution) == 81:
                all_puzzles.append({
                    "puzzle": puzzle,
                    "solution": solution,
                    "difficulty": float(difficulty) if difficulty else 0.0,
                })

    logger.info(f"Loaded {len(all_puzzles)} puzzles from Kaggle dataset.")

    # Split: easy puzzles (solvable with 7 strategies) → train/test
    # Hard puzzles (require backtracking) → hard_test
    # Approximate: use difficulty score as proxy
    easy = [p for p in all_puzzles if p["difficulty"] < 0.5]
    hard = [p for p in all_puzzles if p["difficulty"] >= 0.5]

    rng.shuffle(easy)
    n_train = int(0.9 * len(easy))
    train_data = easy[:n_train]
    test_data = easy[n_train:]

    _write_sudoku_csv(os.path.join(data_path, "sudoku_train.csv"), train_data)
    _write_sudoku_csv(os.path.join(data_path, "sudoku_test.csv"), test_data)
    _write_sudoku_csv(os.path.join(data_path, "sudoku_hard_test.csv"), hard)

    logger.info(
        f"Split: {len(train_data)} train, {len(test_data)} test, "
        f"{len(hard)} hard_test"
    )


def _generate_synthetic_sudoku(data_path: str, seed: int):
    """Generate a small synthetic Sudoku dataset for testing."""
    import random
    rng = random.Random(seed)

    def generate_sudoku():
        """Generate a valid Sudoku puzzle using backtracking."""
        grid = [[0] * 9 for _ in range(9)]
        _fill_sudoku(grid, rng)
        solution = "".join(str(grid[r][c]) for r in range(9) for c in range(9))

        # Create puzzle by removing some cells
        puzzle_grid = [row[:] for row in grid]
        cells = [(r, c) for r in range(9) for c in range(9)]
        rng.shuffle(cells)
        for r, c in cells[:50]:  # remove 50 cells
            puzzle_grid[r][c] = 0
        puzzle = "".join(
            str(puzzle_grid[r][c]) if puzzle_grid[r][c] != 0 else "."
            for r in range(9) for c in range(9)
        )
        return puzzle, solution

    def _fill_sudoku(grid, rng):
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    nums = list(range(1, 10))
                    rng.shuffle(nums)
                    for n in nums:
                        if _is_valid(grid, r, c, n):
                            grid[r][c] = n
                            if _fill_sudoku(grid, rng):
                                return True
                            grid[r][c] = 0
                    return False
        return True

    def _is_valid(grid, r, c, n):
        if n in grid[r]:
            return False
        if n in [grid[i][c] for i in range(9)]:
            return False
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if grid[i][j] == n:
                    return False
        return True

    logger.info("Generating synthetic Sudoku puzzles (this may take a while)...")
    n_train, n_test = 1000, 200

    train_data = [dict(zip(["puzzle", "solution"], generate_sudoku()))
                  for _ in range(n_train)]
    test_data = [dict(zip(["puzzle", "solution"], generate_sudoku()))
                 for _ in range(n_test)]

    _write_sudoku_csv(os.path.join(data_path, "sudoku_train.csv"), train_data)
    _write_sudoku_csv(os.path.join(data_path, "sudoku_test.csv"), test_data)
    _write_sudoku_csv(os.path.join(data_path, "sudoku_hard_test.csv"), test_data)

    logger.info(f"Generated {n_train} train and {n_test} test puzzles.")


def _write_sudoku_csv(path: str, data: list):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["puzzle", "solution"])
        writer.writeheader()
        writer.writerows(data)
    logger.info(f"Wrote {len(data)} puzzles to {path}")


# ---------------------------------------------------------------------------
# Zebra data preparation
# ---------------------------------------------------------------------------

def prepare_zebra(data_path: str, seed: int = 42):
    """Generate Zebra puzzle dataset."""
    generator = ZebraPuzzleGenerator(seed=seed)
    generator.save_dataset(data_path, num_train=100_000, num_test=10_000)
    logger.info(f"Zebra dataset saved to {data_path}")


# ---------------------------------------------------------------------------
# Slimpajama data preparation
# ---------------------------------------------------------------------------

def prepare_slimpajama(
    data_path: str,
    seq_len: int = 2048,
    max_tokens: int = 10_000_000_000,  # 10B tokens
):
    """
    Download and tokenize the Slimpajama dataset.

    This requires the 'datasets' library and a LLaMA tokenizer.
    The tokenized data is saved as binary files for fast loading.
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("Please install 'datasets' and 'transformers' packages.")
        return

    os.makedirs(data_path, exist_ok=True)

    logger.info("Loading Slimpajama dataset (this may take a while)...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    for split in ["train", "validation"]:
        output_path = os.path.join(data_path, f"{split}.bin")
        if os.path.exists(output_path):
            logger.info(f"{split} split already exists at {output_path}")
            continue

        logger.info(f"Processing {split} split...")
        dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split=split,
            streaming=True,
        )

        import numpy as np
        tokens = []
        n_tokens = 0

        for example in dataset:
            text = example["text"]
            ids = tokenizer.encode(text, add_special_tokens=False)
            tokens.extend(ids)
            n_tokens += len(ids)

            if n_tokens >= max_tokens:
                break

        # Save as binary
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(output_path)
        logger.info(f"Saved {len(arr):,} tokens to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--task", type=str, required=True,
                        choices=["sudoku", "zebra", "slimpajama", "all"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kaggle_csv", type=str, default=None,
                        help="Path to Kaggle Sudoku CSV file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.task == "sudoku" or args.task == "all":
        data_path = args.data_path or "data/sudoku"
        prepare_sudoku(data_path, args.kaggle_csv, args.seed)

    if args.task == "zebra" or args.task == "all":
        data_path = args.data_path or "data/zebra"
        prepare_zebra(data_path, args.seed)

    if args.task == "slimpajama" or args.task == "all":
        data_path = args.data_path or "data/slimpajama"
        prepare_slimpajama(data_path)
