"""
Dataset implementations for all experiments in the paper.

Datasets:
  1. L&O-NAE-SAT  — synthetic latents-and-observations distribution (Section 3.3, D.1.1)
  2. Sudoku        — 9×9 logic puzzle (Section 4.2, D.2)
  3. Zebra         — Einstein/Zebra puzzle (Section 4.2, D.2)
  4. Text          — Slimpajama for scaling law experiments (Section 3.2, C.1)
"""

import os
import json
import math
import random
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# L&O-NAE-SAT Dataset (Section 3.3 / D.1.1)
# ---------------------------------------------------------------------------

class NAESATDataset(Dataset):
    """
    Latents-and-Observations distribution with NAE-SAT observations.

    Sequence structure (Section C.2.1):
      - Positions 0..N-1:     latent tokens (values in {1, ..., m})
      - Positions N..N+P-1:   observation tokens (NAE values: 1 or 2)
      - Positions N+P..511:   padding tokens (value = pad_value)

    NAE(x_{i1}, x_{i2}, x_{i3}) = 1 - 1[x_{i1} = x_{i2} = x_{i3}]
    Returns 1 if not all equal, 0 if all equal.
    In our encoding: observation value = 1 (NAE satisfied) or 2 (NAE violated).

    The paper uses (N, P) pairs: (25,275), (30,270), (40,260), (50,250), (100,200).
    For Section C.2.1: (N, P) = (20, 280), seq_len = 512, pad_len = 212.
    """

    MASK_TOKEN = 0

    def __init__(
        self,
        N: int,
        P: int,
        m: int = 3,                  # vocabulary size for latents (values 1..m)
        seq_len: int = 512,
        num_samples: int = 100_000,
        seed: int = 42,
        pad_value: int = 2,          # padding token value (distinct from mask=0)
    ):
        super().__init__()
        self.N = N
        self.P = P
        self.m = m
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.pad_value = pad_value

        rng = np.random.default_rng(seed)

        # Pre-generate fixed random triples for observations (fixed across all samples)
        # Each observation j corresponds to a random triple (i1, i2, i3) from [N]
        self.triples = np.array([
            rng.choice(N, size=3, replace=False) for _ in range(P)
        ])  # (P, 3)

        # Generate all sequences
        self.sequences = self._generate(rng)

    def _nae(self, a: int, b: int, c: int) -> int:
        """NAE(a, b, c) = 1 if not all equal, 0 if all equal."""
        return 0 if (a == b == c) else 1

    def _generate(self, rng: np.random.Generator) -> np.ndarray:
        """Generate num_samples sequences."""
        sequences = np.zeros((self.num_samples, self.seq_len), dtype=np.int64)

        for idx in range(self.num_samples):
            # Sample latent tokens from {1, ..., m}
            latents = rng.integers(1, self.m + 1, size=self.N)

            # Compute observation tokens
            obs = np.array([
                self._nae(latents[i1], latents[i2], latents[i3]) + 1
                for i1, i2, i3 in self.triples
            ])  # values in {1, 2}

            # Assemble sequence: [latents | observations | padding]
            seq = np.concatenate([
                latents,
                obs,
                np.full(self.seq_len - self.N - self.P, self.pad_value),
            ])
            sequences[idx] = seq

        return sequences

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = torch.from_numpy(self.sequences[idx]).long()
        return {"x0": seq, "latent_end": self.N, "obs_end": self.N + self.P}

    def get_vocab_size(self) -> int:
        # 0=mask, 1..m=latent values, m+1=NAE-satisfied, m+2=NAE-violated, pad_value
        return self.m + 3  # conservative upper bound


# ---------------------------------------------------------------------------
# Sudoku Dataset (Section D.2)
# ---------------------------------------------------------------------------

class SudokuDataset(Dataset):
    """
    Sudoku puzzle dataset.

    Each puzzle is represented as a sequence of 81 tokens (9×9 grid, row-major).
    Token values: 0 = mask/empty, 1-9 = digits.

    The dataset from Shah et al. (2024) / Radcliffe (2020) is expected as a CSV
    with columns 'puzzle' and 'solution' (strings of 81 digits, '.' or '0' for empty).

    For order-aware ARM training, the ordering is derived from the solution:
    cells are ordered by the number of constraints they satisfy (most constrained first),
    which corresponds to the strategy used in Shah et al. (2024).
    """

    MASK_TOKEN = 0
    VOCAB_SIZE = 10  # 0=mask, 1-9=digits

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        use_ordering: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.use_ordering = use_ordering
        self.data = self._load(data_path, split, max_samples)

    def _load(self, data_path: str, split: str,
              max_samples: Optional[int]) -> List[Dict]:
        filepath = os.path.join(data_path, f"sudoku_{split}.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Sudoku dataset not found at {filepath}. "
                "Download from https://www.kaggle.com/dsv/1495975 "
                "and preprocess with experiments/prepare_sudoku.py"
            )

        import csv
        data = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                puzzle_str = row.get("puzzle", row.get("quizzes", ""))
                solution_str = row.get("solution", row.get("solutions", ""))
                if len(puzzle_str) == 81 and len(solution_str) == 81:
                    data.append({
                        "puzzle": puzzle_str,
                        "solution": solution_str,
                    })
                if max_samples and len(data) >= max_samples:
                    break
        return data

    def _parse_grid(self, s: str) -> torch.Tensor:
        """Convert 81-char string to tensor of ints (0 for empty/mask)."""
        tokens = []
        for c in s:
            if c in "123456789":
                tokens.append(int(c))
            else:
                tokens.append(0)
        return torch.tensor(tokens, dtype=torch.long)

    def _compute_ordering(self, puzzle: torch.Tensor,
                          solution: torch.Tensor) -> torch.Tensor:
        """
        Compute the generation order for order-aware ARM training.

        Strategy: order empty cells by most-constrained-first (cells with fewest
        remaining candidates are filled first), then append given cells.
        This mirrors the approach in Shah et al. (2024).
        """
        given_positions = (puzzle != 0).nonzero(as_tuple=True)[0]
        empty_positions = (puzzle == 0).nonzero(as_tuple=True)[0]

        # For simplicity, order empty cells by row then column (left-to-right within rows)
        # A more faithful implementation would use constraint propagation ordering
        ordering = torch.cat([given_positions, empty_positions])
        return ordering

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        puzzle = self._parse_grid(item["puzzle"])
        solution = self._parse_grid(item["solution"])

        result = {"x0": solution, "puzzle": puzzle}

        if self.use_ordering:
            ordering = self._compute_ordering(puzzle, solution)
            result["ordering"] = ordering

        return result


# ---------------------------------------------------------------------------
# Zebra (Einstein) Puzzle Dataset (Section D.2)
# ---------------------------------------------------------------------------

class ZebraDataset(Dataset):
    """
    Zebra / Einstein puzzle dataset.

    Each puzzle has 5 houses × 5 attributes = 25 tokens.
    Each attribute has 5 possible values (encoded as 1-5, 0=mask).

    Attributes: nationality, color, drink, pet, cigarette (standard Einstein puzzle).

    The dataset from Shah et al. (2024) is expected as a JSON file with entries:
      {"puzzle": [...], "solution": [...], "ordering": [...]}
    where puzzle/solution are lists of 25 ints and ordering is the generation order.
    """

    MASK_TOKEN = 0
    NUM_HOUSES = 5
    NUM_ATTRIBUTES = 5
    NUM_VALUES = 5
    SEQ_LEN = 25  # 5 × 5
    VOCAB_SIZE = 6  # 0=mask, 1-5=values

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        use_ordering: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.use_ordering = use_ordering
        self.data = self._load(data_path, split, max_samples)

    def _load(self, data_path: str, split: str,
              max_samples: Optional[int]) -> List[Dict]:
        filepath = os.path.join(data_path, f"zebra_{split}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Zebra dataset not found at {filepath}. "
                "Generate using experiments/generate_zebra.py"
            )

        with open(filepath, "r") as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        solution = torch.tensor(item["solution"], dtype=torch.long)
        puzzle = torch.tensor(item.get("puzzle", [0] * self.SEQ_LEN), dtype=torch.long)

        result = {"x0": solution, "puzzle": puzzle}

        if self.use_ordering and "ordering" in item:
            result["ordering"] = torch.tensor(item["ordering"], dtype=torch.long)

        return result


# ---------------------------------------------------------------------------
# Text Dataset (Slimpajama) for scaling law experiments (Section C.1)
# ---------------------------------------------------------------------------

class SlimpajamaDataset(Dataset):
    """
    Slimpajama dataset for π-learner scaling law experiments (Section 3.2).

    Loads pre-tokenized sequences of length seq_len from the Slimpajama dataset.
    Expects pre-processed token files (e.g., from the MDLM codebase).
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        seq_len: int = 2048,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.data = self._load(data_path, split, max_samples)

    def _load(self, data_path: str, split: str,
              max_samples: Optional[int]) -> torch.Tensor:
        filepath = os.path.join(data_path, f"{split}.bin")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Slimpajama data not found at {filepath}. "
                "Pre-process using experiments/prepare_slimpajama.py"
            )

        data = np.memmap(filepath, dtype=np.uint16, mode="r")
        n_sequences = len(data) // self.seq_len
        if max_samples:
            n_sequences = min(n_sequences, max_samples)
        data = data[:n_sequences * self.seq_len].reshape(n_sequences, self.seq_len)
        return torch.from_numpy(data.astype(np.int64))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x0": self.data[idx]}


# ---------------------------------------------------------------------------
# Permutation utilities for π-learner experiments (Section C.1)
# ---------------------------------------------------------------------------

def sample_permutation(
    seq_len: int,
    permutation_type: str = "random",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample a permutation π of {0, ..., seq_len-1}.

    Types (Section C.1):
      "identity":    π = [0, 1, ..., L-1]  (ARM training)
      "random":      π ~ Unif(S_L)          (MDM training)
      "closer":      L/10 random swaps from identity
      "much_closer": sqrt(L) random swaps from identity
    """
    rng = np.random.default_rng(seed)
    pi = np.arange(seq_len)

    if permutation_type == "identity":
        pass
    elif permutation_type == "random":
        rng.shuffle(pi)
    elif permutation_type == "closer":
        n_swaps = seq_len // 10
        for _ in range(n_swaps):
            i, j = rng.choice(seq_len, size=2, replace=False)
            pi[i], pi[j] = pi[j], pi[i]
    elif permutation_type == "much_closer":
        n_swaps = int(math.sqrt(seq_len))
        for _ in range(n_swaps):
            i, j = rng.choice(seq_len, size=2, replace=False)
            pi[i], pi[j] = pi[j], pi[i]
    else:
        raise ValueError(f"Unknown permutation type: {permutation_type}")

    return torch.from_numpy(pi).long()


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def get_nae_sat_loaders(
    N: int,
    P: int,
    m: int = 3,
    seq_len: int = 512,
    num_train: int = 100_000,
    num_test: int = 10_000,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = NAESATDataset(N, P, m, seq_len, num_train, seed=seed)
    test_dataset = NAESATDataset(N, P, m, seq_len, num_test, seed=seed + 1)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def get_sudoku_loaders(
    data_path: str,
    batch_size: int = 128,
    use_ordering: bool = False,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = SudokuDataset(data_path, "train", use_ordering)
    test_dataset = SudokuDataset(data_path, "test", use_ordering=False)
    hard_test_dataset = SudokuDataset(data_path, "hard_test", use_ordering=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    hard_test_loader = DataLoader(
        hard_test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader, hard_test_loader


def get_zebra_loaders(
    data_path: str,
    batch_size: int = 128,
    use_ordering: bool = False,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ZebraDataset(data_path, "train", use_ordering)
    test_dataset = ZebraDataset(data_path, "test", use_ordering=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def get_text_loaders(
    data_path: str,
    seq_len: int = 2048,
    batch_size: int = 512,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = SlimpajamaDataset(data_path, "train", seq_len, max_train_samples)
    val_dataset = SlimpajamaDataset(data_path, "val", seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Zebra puzzle generator (for creating the dataset from scratch)
# ---------------------------------------------------------------------------

class ZebraPuzzleGenerator:
    """
    Generates Zebra / Einstein puzzles.

    Standard 5-house, 5-attribute puzzle with known constraints.
    Generates random valid puzzles and their solutions.
    """

    ATTRIBUTES = ["nationality", "color", "drink", "pet", "cigarette"]
    NUM_ATTRIBUTES = 5
    NUM_HOUSES = 5
    NUM_VALUES = 5

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_solution(self) -> List[List[int]]:
        """Generate a random valid Zebra puzzle solution."""
        # solution[attr][house] = value (1-indexed)
        solution = []
        for _ in range(self.NUM_ATTRIBUTES):
            perm = list(range(1, self.NUM_VALUES + 1))
            self.rng.shuffle(perm)
            solution.append(perm)
        return solution

    def solution_to_sequence(self, solution: List[List[int]]) -> List[int]:
        """
        Convert solution to flat sequence of length 25.
        Layout: [attr0_house0, attr0_house1, ..., attr4_house4]
        """
        seq = []
        for attr in range(self.NUM_ATTRIBUTES):
            for house in range(self.NUM_HOUSES):
                seq.append(solution[attr][house])
        return seq

    def generate_puzzle(
        self, solution: List[List[int]], num_given: int = 5
    ) -> List[int]:
        """Create a partial puzzle by masking some cells."""
        seq = self.solution_to_sequence(solution)
        puzzle = [0] * len(seq)
        given_indices = self.rng.sample(range(len(seq)), num_given)
        for i in given_indices:
            puzzle[i] = seq[i]
        return puzzle

    def generate_dataset(
        self, num_samples: int, num_given: int = 5
    ) -> List[Dict]:
        data = []
        for _ in range(num_samples):
            solution = self.generate_solution()
            seq = self.solution_to_sequence(solution)
            puzzle = self.generate_puzzle(solution, num_given)
            # Ordering: given cells first, then empty cells left-to-right
            given = [i for i, v in enumerate(puzzle) if v != 0]
            empty = [i for i, v in enumerate(puzzle) if v == 0]
            ordering = given + empty
            data.append({
                "solution": seq,
                "puzzle": puzzle,
                "ordering": ordering,
            })
        return data

    def save_dataset(self, data_path: str, num_train: int = 100_000,
                     num_test: int = 10_000):
        os.makedirs(data_path, exist_ok=True)
        train_data = self.generate_dataset(num_train)
        test_data = self.generate_dataset(num_test)

        with open(os.path.join(data_path, "zebra_train.json"), "w") as f:
            json.dump(train_data, f)
        with open(os.path.join(data_path, "zebra_test.json"), "w") as f:
            json.dump(test_data, f)
