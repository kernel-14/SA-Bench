"""
Dataset loading and preprocessing for pre-training.

The paper trains on a 3.5T high-quality token dataset encompassing:
  - Multilingual text (English, Chinese)
  - Math
  - Code
  - General knowledge

This module provides:
  1. A streaming dataset wrapper for pre-tokenised binary shards (the format
     used by most large-scale LLM pre-training pipelines).
  2. A HuggingFace datasets adapter for use with publicly available datasets.
  3. A DataLoader factory that handles packing sequences to the target length.
"""

import os
import math
import random
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


# ---------------------------------------------------------------------------
# Binary shard dataset (pre-tokenised uint16 / uint32 token arrays)
# ---------------------------------------------------------------------------

class BinaryShardDataset(IterableDataset):
    """Streams packed sequences from pre-tokenised binary shard files.

    Each shard is a flat array of token IDs stored as uint16 (vocab ≤ 65535)
    or uint32.  Sequences are packed end-to-end; the dataset slices them into
    fixed-length chunks of `seq_len` tokens.

    Args:
        data_dir:   Directory containing *.bin shard files.
        seq_len:    Context length (default 4096, as used in the paper).
        split:      'train' or 'val'.
        dtype:      numpy dtype of the token arrays ('uint16' or 'uint32').
        seed:       Random seed for shard shuffling.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        seq_len: int = 4096,
        split: str = "train",
        dtype: str = "uint16",
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.split = split
        self.dtype = np.dtype(dtype)
        self.seed = seed

        pattern = f"{split}_*.bin" if split != "all" else "*.bin"
        self.shards = sorted(self.data_dir.glob(pattern))
        if not self.shards:
            # Fall back to any .bin files
            self.shards = sorted(self.data_dir.glob("*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"No .bin shards found in {data_dir}")

    def _iter_shard(self, path: Path) -> Iterator[np.ndarray]:
        data = np.fromfile(path, dtype=self.dtype)
        num_chunks = len(data) // (self.seq_len + 1)
        for i in range(num_chunks):
            chunk = data[i * (self.seq_len + 1) : (i + 1) * (self.seq_len + 1)]
            yield chunk.astype(np.int64)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        shards = list(self.shards)

        rng = random.Random(self.seed)
        rng.shuffle(shards)

        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        for shard_path in shards:
            for chunk in self._iter_shard(shard_path):
                input_ids = torch.from_numpy(chunk[:-1])
                labels = torch.from_numpy(chunk[1:])
                yield {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# HuggingFace datasets adapter
# ---------------------------------------------------------------------------

class HFTextDataset(IterableDataset):
    """Wraps a HuggingFace dataset for language model pre-training.

    Tokenises text on-the-fly and packs tokens into fixed-length sequences.

    Args:
        dataset_name:   HF dataset identifier (e.g. 'allenai/c4').
        tokenizer:      HF tokenizer instance.
        seq_len:        Context length.
        split:          Dataset split.
        text_column:    Column name containing text.
        streaming:      Use streaming mode (recommended for large datasets).
        seed:           Shuffle seed.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        seq_len: int = 4096,
        split: str = "train",
        text_column: str = "text",
        streaming: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_column = text_column

        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            trust_remote_code=True,
        )
        if streaming:
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10_000)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        for example in self.dataset:
            text = example[self.text_column]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens.append(self.tokenizer.eos_token_id)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Evaluation dataset (perplexity on held-out test sets)
# ---------------------------------------------------------------------------

class EvalDataset(Dataset):
    """Fixed-length evaluation dataset for perplexity measurement.

    Supports the diverse held-out test sets used in the paper:
    English, Chinese, Code, Math, Law, Literature.
    """

    def __init__(
        self,
        tokens: Union[np.ndarray, list[int]],
        seq_len: int = 4096,
    ):
        self.seq_len = seq_len
        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=np.int64)
        self.tokens = tokens
        self.num_chunks = len(tokens) // (seq_len + 1)

    @classmethod
    def from_file(cls, path: Union[str, Path], seq_len: int = 4096, dtype: str = "uint16"):
        tokens = np.fromfile(path, dtype=np.dtype(dtype)).astype(np.int64)
        return cls(tokens, seq_len)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {
            "input_ids": torch.from_numpy(chunk[:-1]),
            "labels": torch.from_numpy(chunk[1:]),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_train_dataloader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def make_eval_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def load_tokenizer(name_or_path: str = "Qwen/Qwen2-7B"):
    """Load the Qwen2 tokenizer (vocab size 152064, as used in the paper)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
    return tokenizer
