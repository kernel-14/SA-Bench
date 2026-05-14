## datasets.py

"""
Dataset implementations for the masked diffusion reproduction project.

Defines an abstract base class :class:`BaseDataset` and four concrete subclasses:

- :class:`TextDataset`  – tokenised text with optional fixed permutation (π‑learner).
- :class:`NAESATDataset` – synthetic L&O‑NAE‑SAT distribution.
- :class:`SudokuDataset`  – Sudoku puzzles with clue mask.
- :class:`ZebraDataset`  – Einstein / Zebra puzzles with clue mask.

All datasets are configurable through the global :class:`ExperimentConfig` object
and return tensors suitable for training or evaluation.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Local imports  – only configs is needed at module level; utils may be used
# inside methods if required, but we avoid unnecessary horizontal coupling.
from configs import ExperimentConfig


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseDataset(Dataset):
    """
    Abstract base for all project datasets.

    Args:
        config: Full experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self._seed = config.seed
        # We maintain a separate generator so that global torch state is untouched
        self._rng = torch.Generator()
        if self._seed is not None:
            self._rng.manual_seed(self._seed)
        # Common shortcuts
        self.max_len = config.model.max_seq_length
        self.vocab_size = config.model.vocab_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a data sample. Must be overridden."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return size of dataset. Must be overridden."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TextDataset – for π‑learner scaling‑law experiments
# ---------------------------------------------------------------------------

class TextDataset(BaseDataset):
    """
    Tokenised text dataset with optional fixed positional permutation.

    The permutation is applied statically to every sequence, enabling
    experiments with order‑agnostic (MDM) and order‑aware (ARM) training.

    Args:
        config: Experiment configuration.
        split: One of ``'train'``, ``'val'``, ``'test'``. Currently only used
            to select which part of the corpus is loaded (implemented via
            a simple train/val split).
    """

    def __init__(self, config: ExperimentConfig, split: str = "train") -> None:
        super().__init__(config)
        self.split = split

        # Tokenizer
        tokenizer_name = config.data.tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Ensure the tokenizer has a pad token; if not, set eos_token as pad.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load raw text lines
        text_path = Path(config.data.text_data_path)
        if not text_path.exists():
            raise FileNotFoundError(f"Text data not found: {text_path}")

        with open(text_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        # Simple train/val split (90/10) – can be refined as needed.
        n = len(lines)
        n_train = int(0.9 * n)
        if split == "train":
            self._lines = lines[:n_train]
        else:
            self._lines = lines[n_train:]

        # Generate permutation once
        perm_type = config.data.perm_type or "identity"
        self._perm = self._generate_permutation(perm_type)

    def _generate_permutation(self, perm_type: str) -> torch.Tensor:
        """Build permutation tensor of length ``max_len``."""
        L = self.max_len
        if perm_type == "identity":
            return torch.arange(L)
        elif perm_type == "uniform":
            return torch.randperm(L, generator=self._rng)
        elif perm_type == "closer":
            # Apply L/10 random swaps
            perm = torch.arange(L)
            num_swaps = max(1, L // 10)
            for _ in range(num_swaps):
                i = torch.randint(0, L, (1,), generator=self._rng).item()
                j = torch.randint(0, L, (1,), generator=self._rng).item()
                perm[i], perm[j] = perm[j].clone(), perm[i].clone()
            return perm
        elif perm_type == "much_closer":
            # Apply sqrt(L) random swaps
            perm = torch.arange(L)
            num_swaps = max(1, int(L ** 0.5))
            for _ in range(num_swaps):
                i = torch.randint(0, L, (1,), generator=self._rng).item()
                j = torch.randint(0, L, (1,), generator=self._rng).item()
                perm[i], perm[j] = perm[j].clone(), perm[i].clone()
            return perm
        else:
            raise ValueError(f"Unknown perm_type: {perm_type}")

    def __len__(self) -> int:
        return len(self._lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        line = self._lines[idx]
        # Tokenize with padding / truncation to max_len
        enc = self.tokenizer(
            line,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)  # (L,)

        # For causal LM loss, labels are the same. The model will shift internally.
        labels = input_ids.clone()

        # Apply fixed positional permutation to both
        input_ids = input_ids[self._perm]
        labels = labels[self._perm]

        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# NAESATDataset – L&O‑NAE‑SAT synthetic distribution
# ---------------------------------------------------------------------------

class NAESATDataset(BaseDataset):
    """
    Synthetic dataset for the L&O‑NAE‑SAT distribution.

    Each sample is a clean sequence of length ``max_seq_length`` consisting of:
    - ``N`` latent tokens (values in ``{0, 1, ..., m-1}``)
    - ``P`` observation tokens (computed via NAE predicate on triples)
    - padding with ``PAD_TOKEN_ID`` up to ``max_seq_length``.

    Args:
        config: Experiment configuration.
    """

    # As specified in Section 3.3 / Appendix C.2.1
    PAD_TOKEN_ID: int = 2

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)

        # Retrieve NAE‑SAT specific parameters
        nae_config = config.data.nae_sat
        self.N = nae_config.N
        self.P = nae_config.P
        self.m = nae_config.vocab_size  # number of actual token values (0 … m-1)

        if self.P < 0:
            raise ValueError(f"P must be non‑negative, got {self.P}")

        # Pre‑select P distinct unordered triples from {0,…,N-1}
        all_triples = list(itertools.combinations(range(self.N), 3))
        if self.P > len(all_triples):
            raise ValueError(
                f"P={self.P} exceeds number of distinct triples "
                f"({len(all_triples)}) for N={self.N}."
            )
        self._triples = random.Random(self._seed).sample(all_triples, self.P)

        # The maximum length is set by the model config; we pad to that.
        # The total useful tokens are N + P; must not exceed max_len.
        if self.N + self.P > self.max_len:
            raise ValueError(
                f"Total tokens N+P ({self.N}+{self.P}) exceeds "
                f"max_seq_length ({self.max_len})."
            )

    def __len__(self) -> int:
        # Synthetic dataset – large fixed size to support many iterations.
        return 1_000_000

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Use a deterministic internal rng based on idx for reproducibility
        rng = torch.Generator()
        rng.manual_seed(self._seed + idx)

        # Sample latent tokens uniformly
        latents = torch.randint(0, self.m, (self.N,), generator=rng)

        # Compute observation tokens
        obs = torch.zeros(self.P, dtype=torch.long)
        for t_idx, (i, j, k) in enumerate(self._triples):
            nae = 1 - (latents[i] == latents[j] == latents[k]).int()
            obs[t_idx] = nae

        # Build full sequence, padding with PAD_TOKEN_ID
        seq = torch.full((self.max_len,), self.PAD_TOKEN_ID, dtype=torch.long)
        seq[: self.N] = latents
        seq[self.N : self.N + self.P] = obs

        return {"input_ids": seq}


# ---------------------------------------------------------------------------
# SudokuDataset – 9×9 Sudoku puzzles
# ---------------------------------------------------------------------------

class SudokuDataset(BaseDataset):
    """
    Dataset for Sudoku puzzles.

    Each item provides:
    - ``input_ids``: full solution grid (81 integers, 1–9).
    - ``clue_mask``: binary mask (1‑hot) indicating clue positions.

    The puzzle file is expected to have one line per puzzle, containing
    the puzzle and solution strings separated by a delimiter (default ``\t``).
    Digits are 0–9; 0 represents an empty cell.

    Args:
        config: Experiment configuration.
        split: One of ``'train'``, ``'test'``, ``'hard_test'``.
    """

    def __init__(self, config: ExperimentConfig, split: str = "train") -> None:
        super().__init__(config)

        # Determine which file to load
        sudoku_config = config.data.sudoku
        file_map = {
            "train": sudoku_config.train_file,
            "test": sudoku_config.test_file,
            "hard_test": sudoku_config.hard_test_file,
        }
        file_path = file_map.get(split)
        if file_path is None:
            raise ValueError(f"Unknown split '{split}' for SudokuDataset.")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Sudoku file not found: {file_path}")

        self._puzzles: List[Tuple[torch.Tensor, torch.Tensor]] = []
        delimiter = "\t"        # typical for Shah et al. dataset
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(delimiter)
                if len(parts) != 2:
                    # Try comma as fallback
                    parts = line.split(",")
                    if len(parts) != 2:
                        raise ValueError(
                            f"Expected puzzle and solution separated by tab or comma, "
                            f"got: '{line}'"
                        )
                puzzle_str, sol_str = parts
                puzzle = torch.tensor([int(ch) for ch in puzzle_str], dtype=torch.long)
                solution = torch.tensor([int(ch) for ch in sol_str], dtype=torch.long)
                if puzzle.shape[0] != 81 or solution.shape[0] != 81:
                    raise ValueError("Sudoku puzzle must have exactly 81 cells.")
                self._puzzles.append((solution, puzzle))

    def __len__(self) -> int:
        return len(self._puzzles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        solution, puzzle = self._puzzles[idx]
        # Clue mask: 1 where puzzle digit is non‑zero, 0 otherwise
        clue_mask = (puzzle != 0).long()
        return {
            "input_ids": solution,
            "clue_mask": clue_mask,
        }


# ---------------------------------------------------------------------------
# ZebraDataset – Einstein/Zebra puzzles
# ---------------------------------------------------------------------------

class ZebraDataset(BaseDataset):
    """
    Dataset for Zebra (Einstein) logic puzzles.

    Each line contains a puzzle representation and a solution representation,
    separated by a delimiter. The format is assumed to be space‑separated
    integer tokens, with ``0`` indicating an empty/masked position.

    The returned dictionary contains:
    - ``input_ids``: the solution tokens.
    - ``clue_mask``: binary mask marking non‑zero tokens in the puzzle.

    Args:
        config: Experiment configuration.
        split: One of ``'train'`` or ``'test'``.
    """

    def __init__(self, config: ExperimentConfig, split: str = "train") -> None:
        super().__init__(config)

        zebra_config = config.data.zebra
        file_map = {
            "train": zebra_config.train_file,
            "test": zebra_config.test_file,
        }
        file_path = file_map.get(split)
        if file_path is None:
            raise ValueError(f"Unknown split '{split}' for ZebraDataset.")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Zebra file not found: {file_path}")

        self._puzzles: List[Tuple[torch.Tensor, torch.Tensor]] = []
        delimiter = "\t"
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(delimiter)
                if len(parts) != 2:
                    # alternative: comma
                    parts = line.split(",")
                    if len(parts) != 2:
                        raise ValueError(
                            f"Expected puzzle and solution separated by tab or comma, "
                            f"got: '{line}'"
                        )
                puzzle_str, sol_str = parts
                puzzle_tokens = [int(t) for t in puzzle_str.split()]
                solution_tokens = [int(t) for t in sol_str.split()]
                puzzle = torch.tensor(puzzle_tokens, dtype=torch.long)
                solution = torch.tensor(solution_tokens, dtype=torch.long)
                # The vocabulary size may be provided by config; we trust it.
                self._puzzles.append((solution, puzzle))

    def __len__(self) -> int:
        return len(self._puzzles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        solution, puzzle = self._puzzles[idx]
        clue_mask = (puzzle != 0).long()
        return {
            "input_ids": solution,
            "clue_mask": clue_mask,
        }


# ---------------------------------------------------------------------------
# Optional: convenience factory function (not required by design, but useful)
# ---------------------------------------------------------------------------

def create_dataset(config: ExperimentConfig, split: str = "train") -> Dataset:
    """
    Instantiate the appropriate dataset based on ``config.task``.

    Args:
        config: Experiment configuration.
        split: Data split name.

    Returns:
        A :class:`Dataset` instance.

    Raises:
        ValueError: If the task is not recognised.
    """
    task = config.task
    if task in ("scaling",):
        return TextDataset(config, split=split)
    elif task == "nae_sat":
        return NAESATDataset(config)
    elif task == "sudoku":
        return SudokuDataset(config, split=split)
    elif task == "zebra":
        return ZebraDataset(config, split=split)
    else:
        raise ValueError(f"Unknown task '{task}' – cannot create dataset.")
