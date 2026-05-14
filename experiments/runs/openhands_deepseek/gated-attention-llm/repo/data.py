"""
Data loading and preprocessing for language model training.

The paper trains on subsets of a 3.5T high-quality token dataset encompassing
multilingual, math, and general knowledge content with context length 4096.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Tuple, Iterator
import os
import json


class TextDataset(Dataset):
    """
    Map-style dataset for pre-tokenized text data stored as numpy memmap.
    Assumes data is a flat array of token IDs.
    """
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        split: str = "train",
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.split = split

        # Load tokenized data
        if os.path.isdir(data_path):
            files = sorted([
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith(".bin") or f.endswith(".npy") or f.endswith(".idx")
            ])
            if not files:
                raise FileNotFoundError(f"No .bin/.npy/.idx files found in {data_path}")

            # Load all files
            arrays = []
            for f in files:
                if f.endswith(".bin"):
                    arr = np.memmap(f, dtype=np.uint32, mode="r")
                elif f.endswith(".npy"):
                    arr = np.load(f, mmap_mode="r")
                else:
                    arr = np.fromfile(f, dtype=np.uint32)
                arrays.append(arr)
            self.data = np.concatenate(arrays)
        else:
            if data_path.endswith(".bin"):
                self.data = np.memmap(data_path, dtype=np.uint32, mode="r")
            elif data_path.endswith(".npy"):
                self.data = np.load(data_path, mmap_mode="r")
            else:
                self.data = np.fromfile(data_path, dtype=np.uint32)

        # Calculate number of sequences
        self.n_tokens = len(self.data)
        self.n_sequences = (self.n_tokens - 1) // self.seq_len

        # Shuffle sequence indices deterministically
        rng = np.random.RandomState(seed)
        self.indices = np.arange(self.n_sequences)
        if split == "train":
            rng.shuffle(self.indices)

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_idx = self.indices[idx]
        start = seq_idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for target

        tokens = self.data[start:end].astype(np.int64)
        input_ids = torch.from_numpy(tokens[:-1])
        labels = torch.from_numpy(tokens[1:])

        return input_ids, labels


class StreamingTextDataset(IterableDataset):
    """
    Iterable dataset for streaming large text corpora.

    Supports data sharding across distributed workers via worker_info.
    """
    def __init__(
        self,
        data_path: str,
        seq_len: int,
        split: str = "train",
        seed: int = 42,
    ):
        self.data_path = data_path
        self.seq_len = seq_len
        self.split = split
        self.seed = seed

        # Count total tokens
        if os.path.isdir(data_path):
            self.files = sorted([
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith(".bin") or f.endswith(".npy") or f.endswith(".idx")
            ])
        else:
            self.files = [data_path]

        self.total_tokens = 0
        for f in self.files:
            if f.endswith(".bin"):
                arr = np.memmap(f, dtype=np.uint32, mode="r")
            elif f.endswith(".npy"):
                arr = np.load(f, mmap_mode="r")
            else:
                arr = np.fromfile(f, dtype=np.uint32)
            self.total_tokens += len(arr)

    def _token_generator(self, worker_id: int, num_workers: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate (input, label) pairs."""
        rng = np.random.RandomState(self.seed + worker_id)

        for file_idx, file_path in enumerate(rng.permutation(len(self.files))):
            f = self.files[file_idx]
            if f.endswith(".bin"):
                arr = np.memmap(f, dtype=np.uint32, mode="r")
            elif f.endswith(".npy"):
                arr = np.load(f, mmap_mode="r")
            else:
                arr = np.fromfile(f, dtype=np.uint32)

            n_tokens = len(arr)
            n_seqs = (n_tokens - 1) // self.seq_len

            # Shard across workers
            worker_seqs = n_seqs // num_workers
            worker_start = worker_id * worker_seqs
            worker_end = worker_start + worker_seqs if worker_id < num_workers - 1 else n_seqs

            indices = np.arange(worker_start, worker_end)
            if self.split == "train":
                rng.shuffle(indices)

            for idx in indices:
                start = idx * self.seq_len
                end = start + self.seq_len + 1
                tokens = arr[start:end].astype(np.int64)
                yield (
                    torch.from_numpy(tokens[:-1].copy()),
                    torch.from_numpy(tokens[1:].copy()),
                )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        return self._token_generator(worker_id, num_workers)


def create_dataloader(
    data_path: str,
    seq_len: int,
    batch_size: int,
    split: str = "train",
    num_workers: int = 4,
    streaming: bool = False,
    seed: int = 42,
) -> DataLoader:
    """
    Create a dataloader for language model training.

    Args:
        data_path: Path to tokenized data file or directory
        seq_len: Sequence length for training
        batch_size: Batch size per device
        split: 'train' or 'val'
        num_workers: Number of data loading workers
        streaming: Whether to use streaming dataset for large corpora
        seed: Random seed for shuffling

    Returns:
        DataLoader instance
    """
    if streaming:
        dataset = StreamingTextDataset(data_path, seq_len, split, seed)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataset = TextDataset(data_path, seq_len, split, seed)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    return dataloader


def get_eval_dataloaders(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    eval_splits: Optional[List[str]] = None,
):
    """
    Create dataloaders for evaluation on held-out test sets.
    Paper evaluates on diverse domains: English, Chinese, Code, Math, Law, Literature.
    """
    if eval_splits is None:
        eval_splits = ["english", "chinese", "code", "math", "law", "literature"]

    eval_loaders = {}
    for split in eval_splits:
        path = os.path.join(data_dir, split)
        if os.path.exists(path):
            eval_loaders[split] = create_dataloader(
                path, seq_len, batch_size, split="val", num_workers=2, streaming=False
            )

    return eval_loaders


class TokenizerWrapper:
    """Simple tokenizer interface for text processing during evaluation."""
    def __init__(self, vocab_size: int = 151936):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        raise NotImplementedError("Replace with actual tokenizer")


def process_eval_sample(
    text: str,
    tokenizer: TokenizerWrapper,
    max_length: int = 4096,
) -> torch.Tensor:
    """Tokenize a single text for evaluation."""
    tokens = tokenizer.encode(text, max_length=max_length)
    return torch.tensor([tokens], dtype=torch.long)
