"""
Sudoku Puzzle Dataset and Evaluation
=====================================
Implements Sudoku puzzle handling for the MDM experiments.

The Sudoku puzzle is represented as a sequence of 81 tokens (9x9 grid),
where each token is a digit 1-9 (or 0 for empty cells).

For MDM training/inference:
- Token 0 = mask token (also used for empty cells in the puzzle)
- Tokens 1-9 = digits

The paper uses the dataset from Shah et al. (2024), which filters puzzles
solvable with 7 fixed strategies (no backtracking needed).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import os


# Sudoku constants
GRID_SIZE = 9
SEQ_LEN = 81  # 9x9 = 81 cells
VOCAB_SIZE = 11  # 0=mask, 1-9=digits, 10=separator (optional)
MASK_TOKEN = 0


def is_valid_sudoku(grid: np.ndarray) -> bool:
    """
    Check if a completed 9x9 Sudoku grid is valid.
    
    Args:
        grid: 9x9 numpy array with values 1-9
    
    Returns:
        True if valid, False otherwise
    """
    if grid.shape != (9, 9):
        return False
    
    # Check all values are 1-9
    if not np.all((grid >= 1) & (grid <= 9)):
        return False
    
    # Check rows
    for i in range(9):
        if len(set(grid[i])) != 9:
            return False
    
    # Check columns
    for j in range(9):
        if len(set(grid[:, j])) != 9:
            return False
    
    # Check 3x3 boxes
    for bi in range(3):
        for bj in range(3):
            box = grid[bi*3:(bi+1)*3, bj*3:(bj+1)*3].flatten()
            if len(set(box)) != 9:
                return False
    
    return True


def sequence_to_grid(seq: np.ndarray) -> np.ndarray:
    """Convert a flat sequence of 81 tokens to a 9x9 grid."""
    return seq.reshape(9, 9)


def grid_to_sequence(grid: np.ndarray) -> np.ndarray:
    """Convert a 9x9 grid to a flat sequence of 81 tokens."""
    return grid.flatten()


class SudokuDataset(Dataset):
    """
    Dataset for Sudoku puzzles.
    
    Each sample is a pair (puzzle, solution) where:
    - puzzle: sequence of 81 tokens with some cells masked (0)
    - solution: complete sequence of 81 tokens (1-9)
    
    The dataset format follows Shah et al. (2024).
    """
    
    def __init__(self, data_path: str = None, 
                 puzzles: np.ndarray = None,
                 solutions: np.ndarray = None,
                 max_samples: int = None):
        """
        Args:
            data_path: path to CSV file with puzzles and solutions
            puzzles: pre-loaded puzzle array (n_samples, 81)
            solutions: pre-loaded solution array (n_samples, 81)
            max_samples: maximum number of samples to load
        """
        if data_path is not None:
            self.puzzles, self.solutions = self._load_from_file(data_path, max_samples)
        elif puzzles is not None and solutions is not None:
            self.puzzles = puzzles
            self.solutions = solutions
        else:
            raise ValueError("Either data_path or (puzzles, solutions) must be provided")
        
        if max_samples is not None:
            self.puzzles = self.puzzles[:max_samples]
            self.solutions = self.solutions[:max_samples]
    
    def _load_from_file(self, path: str, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load puzzles from CSV file."""
        import csv
        
        puzzles = []
        solutions = []
        
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header if present
            
            for i, row in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break
                
                if len(row) >= 2:
                    puzzle_str = row[0].strip()
                    solution_str = row[1].strip()
                    
                    # Parse puzzle: '.' or '0' for empty cells
                    puzzle = np.array([int(c) if c.isdigit() and c != '0' else 0 
                                      for c in puzzle_str if c in '0123456789.'], dtype=np.int64)
                    solution = np.array([int(c) for c in solution_str if c.isdigit()], dtype=np.int64)
                    
                    if len(puzzle) == 81 and len(solution) == 81:
                        puzzles.append(puzzle)
                        solutions.append(solution)
        
        return np.array(puzzles), np.array(solutions)
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        puzzle = torch.tensor(self.puzzles[idx], dtype=torch.long)
        solution = torch.tensor(self.solutions[idx], dtype=torch.long)
        return puzzle, solution
    
    def get_masked_puzzle(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a puzzle with masked cells (0) and its solution."""
        puzzle = torch.tensor(self.puzzles[idx], dtype=torch.long)
        solution = torch.tensor(self.solutions[idx], dtype=torch.long)
        return puzzle, solution


def evaluate_sudoku_accuracy(model: torch.nn.Module,
                              dataset: SudokuDataset,
                              strategy: str = 'top_prob_margin',
                              n_steps: int = 50,
                              gumbel_noise: float = 0.5,
                              batch_size: int = 64,
                              device: torch.device = None,
                              max_samples: int = None) -> float:
    """
    Evaluate MDM accuracy on Sudoku puzzles.
    
    Args:
        model: trained MDM model
        dataset: Sudoku dataset
        strategy: inference strategy ('vanilla', 'top_prob', 'top_prob_margin')
        n_steps: number of inference steps
        gumbel_noise: Gumbel noise coefficient
        batch_size: batch size for inference
        device: computation device
        max_samples: maximum number of samples to evaluate
    
    Returns:
        accuracy: fraction of correctly solved puzzles
    """
    from adaptive_inference import mdm_sample_greedy, MASK_TOKEN
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    n_correct = 0
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        # Get batch
        puzzles = []
        solutions = []
        for i in range(start, end):
            p, s = dataset[i]
            puzzles.append(p)
            solutions.append(s)
        
        puzzles = torch.stack(puzzles).to(device)
        solutions = torch.stack(solutions).to(device)
        
        # Run inference
        generated = mdm_sample_greedy(
            model, puzzles, n_steps=n_steps,
            strategy=strategy, gumbel_noise=gumbel_noise
        )
        
        # Check correctness: all 81 cells must match the solution
        correct = (generated == solutions).all(dim=-1)
        n_correct += correct.sum().item()
    
    return n_correct / n_samples


def generate_synthetic_sudoku_data(n_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Sudoku puzzles for testing.
    
    This is a simple generator that creates valid Sudoku puzzles by:
    1. Generating a valid solution
    2. Removing some cells to create the puzzle
    
    Note: For the actual paper experiments, use the dataset from Shah et al. (2024).
    """
    rng = np.random.RandomState(seed)
    
    puzzles = []
    solutions = []
    
    # Base valid Sudoku solution
    base = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7, 8, 9, 1],
        [5, 6, 7, 8, 9, 1, 2, 3, 4],
        [8, 9, 1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8, 9, 1, 2],
        [6, 7, 8, 9, 1, 2, 3, 4, 5],
        [9, 1, 2, 3, 4, 5, 6, 7, 8],
    ])
    
    for _ in range(n_samples):
        # Shuffle rows within bands and columns within stacks
        solution = base.copy()
        
        # Shuffle rows within each band
        for band in range(3):
            rows = list(range(band * 3, (band + 1) * 3))
            rng.shuffle(rows)
            solution[band*3:(band+1)*3] = solution[rows]
        
        # Shuffle columns within each stack
        for stack in range(3):
            cols = list(range(stack * 3, (stack + 1) * 3))
            rng.shuffle(cols)
            solution[:, stack*3:(stack+1)*3] = solution[:, cols]
        
        # Shuffle bands
        bands = [0, 1, 2]
        rng.shuffle(bands)
        solution = np.vstack([solution[b*3:(b+1)*3] for b in bands])
        
        # Shuffle stacks
        stacks = [0, 1, 2]
        rng.shuffle(stacks)
        solution = np.hstack([solution[:, s*3:(s+1)*3] for s in stacks])
        
        # Relabel digits
        perm = rng.permutation(9) + 1
        relabeled = np.zeros_like(solution)
        for old, new in enumerate(perm, 1):
            relabeled[solution == old] = new
        solution = relabeled
        
        # Create puzzle by masking some cells
        puzzle = solution.copy()
        n_remove = rng.randint(40, 60)  # Remove 40-60 cells
        remove_idx = rng.choice(81, size=n_remove, replace=False)
        puzzle.flat[remove_idx] = 0
        
        puzzles.append(puzzle.flatten())
        solutions.append(solution.flatten())
    
    return np.array(puzzles), np.array(solutions)
