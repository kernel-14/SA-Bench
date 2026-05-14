"""
Zebra (Einstein) Puzzle Dataset and Evaluation
================================================
Implements Zebra puzzle handling for the MDM experiments.

The Zebra puzzle (also known as Einstein's puzzle) is a logic puzzle
where you must determine the attributes of 5 houses based on clues.

For MDM training/inference:
- The puzzle is represented as a sequence of tokens
- The model must fill in the missing attributes

The paper uses the dataset from Shah et al. (2024).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
import os


class ZebraDataset(Dataset):
    """
    Dataset for Zebra (Einstein) puzzles.
    
    Each sample is a pair (puzzle, solution) where:
    - puzzle: sequence with some attributes masked (0)
    - solution: complete sequence with all attributes
    
    The dataset format follows Shah et al. (2024).
    """
    
    def __init__(self, data_path: str = None,
                 puzzles: np.ndarray = None,
                 solutions: np.ndarray = None,
                 max_samples: int = None):
        """
        Args:
            data_path: path to data file
            puzzles: pre-loaded puzzle array
            solutions: pre-loaded solution array
            max_samples: maximum number of samples
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
    
    def _load_from_file(self, path: str, max_samples: int = None):
        """Load puzzles from file."""
        import json
        
        puzzles = []
        solutions = []
        
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                
                data = json.loads(line.strip())
                if 'puzzle' in data and 'solution' in data:
                    puzzles.append(data['puzzle'])
                    solutions.append(data['solution'])
        
        return np.array(puzzles), np.array(solutions)
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        puzzle = torch.tensor(self.puzzles[idx], dtype=torch.long)
        solution = torch.tensor(self.solutions[idx], dtype=torch.long)
        return puzzle, solution


def evaluate_zebra_accuracy(model: torch.nn.Module,
                              dataset: ZebraDataset,
                              strategy: str = 'top_prob_margin',
                              n_steps: int = 50,
                              gumbel_noise: float = 0.5,
                              batch_size: int = 64,
                              device: torch.device = None,
                              max_samples: int = None) -> float:
    """
    Evaluate MDM accuracy on Zebra puzzles.
    
    Args:
        model: trained MDM model
        dataset: Zebra dataset
        strategy: inference strategy
        n_steps: number of inference steps
        gumbel_noise: Gumbel noise coefficient
        batch_size: batch size
        device: computation device
        max_samples: maximum number of samples
    
    Returns:
        accuracy: fraction of correctly solved puzzles
    """
    from adaptive_inference import mdm_sample_greedy
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    n_correct = 0
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        puzzles = []
        solutions = []
        for i in range(start, end):
            p, s = dataset[i]
            puzzles.append(p)
            solutions.append(s)
        
        puzzles = torch.stack(puzzles).to(device)
        solutions = torch.stack(solutions).to(device)
        
        generated = mdm_sample_greedy(
            model, puzzles, n_steps=n_steps,
            strategy=strategy, gumbel_noise=gumbel_noise
        )
        
        correct = (generated == solutions).all(dim=-1)
        n_correct += correct.sum().item()
    
    return n_correct / n_samples


def generate_synthetic_zebra_data(n_samples: int, seq_len: int = 50,
                                    vocab_size: int = 30,
                                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Zebra-like puzzle data for testing.
    
    Note: For actual paper experiments, use the dataset from Shah et al. (2024).
    """
    rng = np.random.RandomState(seed)
    
    solutions = rng.randint(1, vocab_size, size=(n_samples, seq_len))
    
    # Create puzzles by masking some positions
    puzzles = solutions.copy()
    for i in range(n_samples):
        n_mask = rng.randint(seq_len // 4, seq_len // 2)
        mask_idx = rng.choice(seq_len, size=n_mask, replace=False)
        puzzles[i, mask_idx] = 0
    
    return puzzles.astype(np.int64), solutions.astype(np.int64)
