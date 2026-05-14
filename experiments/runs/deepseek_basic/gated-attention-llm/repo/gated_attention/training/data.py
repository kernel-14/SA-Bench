"""
Data loading utilities for gated LLM training.

Implements data preprocessing matching the paper's description:
  - 3.5T high-quality tokens encompassing multilingual, math, and general knowledge
  - Sequence length 4096
  - Global batch size 1024 (or 2048, 4096 depending on experiment)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


@dataclass
class DataConfig:
    """Data configuration matching paper settings."""
    seq_len: int = 4096
    tokenizer_name: str = "qwen"  # Qwen team, Alibaba Group
    data_mixture: str = "multilingual_math_general"
    num_workers: int = 4
    prefetch_factor: int = 2


class StreamingTextDataset(IterableDataset):
    """Streaming text dataset for large-scale pretraining.

    Handles tokenization and sequence packing for 3.5T token training.
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer,
        seq_len: int = 4096,
        buffer_size: int = 10000,
    ):
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer_size = buffer_size

    def _tokenize_stream(self):
        """Tokenize data stream into fixed-length sequences."""
        buffer = []
        for path in self.data_paths:
            with open(path, "r") as f:
                for line in f:
                    tokens = self.tokenizer.encode(line.strip())
                    buffer.extend(tokens)

                    # Yield full sequences
                    while len(buffer) >= self.seq_len + 1:
                        seq = buffer[:self.seq_len + 1]
                        buffer = buffer[self.seq_len:]
                        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
                        labels = torch.tensor(seq[1:], dtype=torch.long)
                        yield {"input_ids": input_ids, "labels": labels}

    def __iter__(self):
        return self._tokenize_stream()


class PackedTextDataset(Dataset):
    """Memory-mapped packed text dataset for efficient training.

    Pre-tokenized and packed into fixed-length sequences.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 4096,
    ):
        self.seq_len = seq_len
        self.data = torch.load(data_path)  # (num_sequences, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with standard LLM training settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
