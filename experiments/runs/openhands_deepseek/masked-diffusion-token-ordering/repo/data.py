import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import os
from typing import Tuple, List, Optional, Dict
from torch.utils.data import Dataset, DataLoader, IterableDataset
from config import DataConfig, ModelConfig
from itertools import combinations


class LONAESATDataset(Dataset):
    """
    L&O-NAE-SAT distribution from Section 3.3.
    - N latent tokens, sampled uniformly from {0, 1}
    - P observation tokens, each determined by NAE on a random triple of latents
    - NAE(a, b, c) = 1 - 1[a=b=c]
    - Sequence: [latent_0, ..., latent_{N-1}, obs_0, ..., obs_{P-1}]
    - Additional padding with value 2 to reach max_seq_len
    """

    def __init__(
        self,
        N: int,
        P: int,
        max_seq_len: int,
        size: int = 10000,
        seed: int = 42,
    ):
        self.N = N
        self.P = P
        self.L = N + P
        self.max_seq_len = max_seq_len
        self.size = size
        self.rng = np.random.RandomState(seed)

        # Fix random triples for observations
        self.triples = []
        # We need P triples; just sample with replacement
        for _ in range(P):
            triple = tuple(sorted(self.rng.choice(N, size=3, replace=False)))
            self.triples.append(triple)

        # pre-generate data
        self.data = []
        for _ in range(size):
            latents = self.rng.randint(0, 2, size=N).astype(np.int64)
            observations = np.zeros(P, dtype=np.int64)
            for j, (i1, i2, i3) in enumerate(self.triples):
                val = 1 - int(latents[i1] == latents[i2] == latents[i3])
                observations[j] = val
            seq = np.concatenate([latents, observations])
            # Pad with value 2 to max_seq_len
            padded = np.full(max_seq_len, 2, dtype=np.int64)
            padded[:self.L] = seq
            self.data.append(padded)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.L


class SudokuDataset(Dataset):
    """
    Sudoku puzzle dataset.
    Input: 81-cell puzzle with some cells filled (1-9) and empty (0).
    Target: full solution.

    Based on Shah et al. (2024) dataset, created from Radcliffe (2020).
    Training: puzzles solvable with 7 fixed strategies.
    Hard test: remaining puzzles requiring backtracking/new strategies.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        size: int = 10000,
        max_seq_len: int = 512,
        seed: int = 42,
        hard: bool = False,
    ):
        self.size = size
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)

        if data_path is not None and os.path.exists(data_path):
            self.data = self._load_from_file(data_path, hard)
        else:
            self.data = self._generate_synthetic(size)

    def _load_from_file(self, path: str, hard: bool) -> List[Dict]:
        with open(path, 'r') as f:
            all_data = json.load(f)
        if hard:
            return all_data.get('hard', all_data.get('test', []))
        return all_data.get('train', all_data)

    def _generate_synthetic(self, size: int) -> List[Dict]:
        """Generate simple synthetic Sudoku-like data as fallback."""
        data = []
        for _ in range(size):
            # Generate a random valid-ish grid (simplified)
            solution = self.rng.randint(1, 10, size=81).astype(np.int64).tolist()
            # Mask some cells
            puzzle = solution.copy()
            mask_positions = self.rng.choice(81, size=40, replace=False)
            for pos in mask_positions:
                puzzle[pos] = 0
            data.append({'puzzle': puzzle, 'solution': solution})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        puzzle = torch.tensor(item['puzzle'], dtype=torch.long)
        solution = torch.tensor(item['solution'], dtype=torch.long)
        # Pad to max_seq_len
        if puzzle.shape[0] < self.max_seq_len:
            pad_len = self.max_seq_len - puzzle.shape[0]
            puzzle = F.pad(puzzle, (0, pad_len), value=10)  # padding token
            solution = F.pad(solution, (0, pad_len), value=10)
        return puzzle, solution


class ZebraPuzzleDataset(Dataset):
    """
    Zebra (Einstein) puzzle dataset from Shah et al. (2024).

    The puzzles involve assigning attributes to houses based on logical constraints.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        size: int = 10000,
        max_seq_len: int = 1024,
        seed: int = 42,
    ):
        self.size = size
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)

        if data_path is not None and os.path.exists(data_path):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = self._generate_synthetic(size)

    def _generate_synthetic(self, size: int) -> List[Dict]:
        data = []
        for _ in range(size):
            num_houses = 5
            num_attrs = 5
            seq_len = num_houses * num_attrs * 2
            puzzle = self.rng.randint(1, num_houses + 2, size=seq_len).astype(np.int64).tolist()
            solution = puzzle.copy()
            mask_count = seq_len // 3
            mask_positions = self.rng.choice(seq_len, size=mask_count, replace=False)
            for pos in mask_positions:
                puzzle[pos] = 0
            data.append({'puzzle': puzzle, 'solution': solution})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        puzzle = torch.tensor(item['puzzle'], dtype=torch.long)
        solution = torch.tensor(item['solution'], dtype=torch.long)
        if puzzle.shape[0] < self.max_seq_len:
            pad_len = self.max_seq_len - puzzle.shape[0]
            puzzle = F.pad(puzzle, (0, pad_len), value=50)
            solution = F.pad(solution, (0, pad_len), value=50)
        return puzzle, solution


class TextDataset(IterableDataset):
    """
    Streaming text dataset for tokenized text data (e.g., SlimPajama).
    Yields sequences of token IDs of length max_seq_len.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        tokenizer=None,
        max_seq_len: int = 2048,
        seed: int = 42,
    ):
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.data_path = data_path
        self.tokenizer = tokenizer

    def __iter__(self):
        rng = random.Random(self.seed)
        if self.data_path is not None and os.path.exists(self.data_path):
            yield from self._iter_from_file(rng)
        else:
            yield from self._iter_random(rng)

    def _iter_from_file(self, rng):
        buffer = []
        with open(self.data_path, 'r') as f:
            for line in f:
                if self.tokenizer:
                    tokens = self.tokenizer.encode(line.strip())
                else:
                    tokens = [ord(c) % 256 for c in line.strip()]
                buffer.extend(tokens)
                while len(buffer) >= self.max_seq_len:
                    chunk = torch.tensor(buffer[:self.max_seq_len], dtype=torch.long)
                    buffer = buffer[self.max_seq_len:]
                    yield chunk

    def _iter_random(self, rng):
        vocab_size = 50257
        while True:
            tokens = torch.randint(0, vocab_size, (self.max_seq_len,), dtype=torch.long)
            yield tokens


class PermutationDataloader:
    """
    Wraps a dataloader and applies a permutation π to the sequences.
    Used for π-learner experiments (Section 3.2).
    """

    def __init__(
        self,
        dataloader: DataLoader,
        permutation: torch.Tensor,
        mask_token: int = 0,
    ):
        self.dataloader = dataloader
        self.permutation = permutation
        self.mask_token = mask_token

    def __iter__(self):
        for batch in self.dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x_permuted = x[:, self.permutation]
            yield x_permuted

    def __len__(self):
        return len(self.dataloader)


def sample_permutation(
    L: int,
    distribution: str,
    rng: np.random.RandomState,
) -> torch.Tensor:
    """
    Sample a permutation π over [0, ..., L-1].

    - "identity": identity permutation
    - "uniform": uniformly random permutation
    - "closer": L/10 random swaps from identity
    - "much_closer": sqrt(L) random swaps from identity
    """
    perm = list(range(L))
    if distribution == "identity":
        pass
    elif distribution == "uniform":
        rng.shuffle(perm)
    elif distribution == "closer":
        n_swaps = max(1, L // 10)
        for _ in range(n_swaps):
            i, j = rng.choice(L, size=2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]
    elif distribution == "much_closer":
        n_swaps = max(1, int(np.sqrt(L)))
        for _ in range(n_swaps):
            i, j = rng.choice(L, size=2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]
    else:
        raise ValueError(f"Unknown permutation distribution: {distribution}")
    return torch.tensor(perm, dtype=torch.long)


def get_dataloader(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    batch_size: int,
    split: str = "train",
    num_workers: int = 4,
    data_path: Optional[str] = None,
) -> DataLoader:
    """Create the appropriate dataloader based on config."""
    if data_cfg.dataset == "lonaesat":
        size = 10000 if split == "train" else 1000
        dataset = LONAESATDataset(
            N=data_cfg.N_latent,
            P=data_cfg.P_obs,
            max_seq_len=model_cfg.max_seq_len,
            size=size,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=num_workers, pin_memory=True)

    elif data_cfg.dataset == "sudoku":
        dataset = SudokuDataset(
            data_path=data_path,
            size=10000 if split == "train" else 1000,
            max_seq_len=model_cfg.max_seq_len,
            hard=(split == "hard_test"),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=num_workers, pin_memory=True)

    elif data_cfg.dataset == "zebra":
        dataset = ZebraPuzzleDataset(
            data_path=data_path,
            size=10000 if split == "train" else 1000,
            max_seq_len=model_cfg.max_seq_len,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=num_workers, pin_memory=True)

    elif data_cfg.dataset == "text":
        dataset = TextDataset(
            data_path=data_path,
            max_seq_len=model_cfg.max_seq_len,
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    else:
        raise ValueError(f"Unknown dataset: {data_cfg.dataset}")


class HumanEvalInfillDataset(Dataset):
    """
    HumanEval Infill dataset (Bavarian et al., 2022).
    Three variants:
    - Single-line: fill one line
    - Multi-line: fill multiple contiguous lines
    - Split: fill split regions
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        variant: str = "single",
        max_seq_len: int = 2048,
        seed: int = 42,
    ):
        self.variant = variant
        self.max_seq_len = max_seq_len
        self.seed = seed

        if data_path is not None and os.path.exists(data_path):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = self._generate_dummy()

    def _generate_dummy(self) -> List[Dict]:
        return [{
            'prefix': 'def f(x):',
            'suffix': '    return x',
            'middle': '    y = x + 1\n    z = y * 2',
            'task_id': f'dummy_{i}'
        } for i in range(164)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full = item['prefix'] + '\n' + item['middle'] + '\n' + item['suffix']
        tokens = [ord(c) % 256 for c in full]
        padded = np.full(self.max_seq_len, 0, dtype=np.int64)
        padded[:len(tokens)] = tokens[:self.max_seq_len]
        return torch.from_numpy(padded)
