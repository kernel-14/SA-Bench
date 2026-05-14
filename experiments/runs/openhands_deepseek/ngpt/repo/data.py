
"""Data loading for OpenWebText dataset (Gokaslan & Cohen, 2019).

Used for training both GPT and nGPT as described in Section 3.
Uses the LLaMA-2 tokenizer with 32k vocabulary.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Iterator, Optional
import numpy as np


class OpenWebTextDataset(Dataset):
    """OpenWebText dataset loaded from pre-tokenized files.

    The dataset is tokenized using LLaMA-2 tokenizer (32k vocab) and stored as
    flat sequence of token IDs, chunked into seq_len windows.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 4096,
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.split = split

        # Load tokenized data
        data_path = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(data_path):
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        else:
            self.data = np.array([], dtype=np.uint16)

        self.num_tokens = len(self.data)
        self.num_chunks = max(0, self.num_tokens // seq_len)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = torch.from_numpy(self.data[start:end].astype(np.int64))
        # For autoregressive training: input is tokens[:-1], target is tokens[1:]
        x = input_ids[:-1]
        y = input_ids[1:]

        return {"input_ids": x, "labels": y}


class StreamingDataset(IterableDataset):
    """Streaming dataset for large-scale training.

    Reads from multiple pre-tokenized shard files, shuffling across shards.
    This avoids loading the entire dataset into memory.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 4096,
        split: str = "train",
        shuffle_buffer_size: int = 1000,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size

        # Find all shard files
        pattern = os.path.join(data_dir, f"{split}_shard_*.bin")
        import glob
        self.shard_files = sorted(glob.glob(pattern))

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Each worker processes a subset of shards
            per_worker = len(self.shard_files) // worker_info.num_workers
            worker_id = worker_info.id
            shard_files = self.shard_files[
                worker_id * per_worker : (worker_id + 1) * per_worker
            ]
        else:
            shard_files = self.shard_files

        # Shuffle shard order
        np.random.shuffle(shard_files)

        for shard_path in shard_files:
            data = np.memmap(shard_path, dtype=np.uint16, mode='r')
            num_tokens = len(data)
            num_chunks = max(0, num_tokens // self.seq_len)

            chunk_indices = np.random.permutation(num_chunks) if num_chunks > 0 else []

            for idx in chunk_indices:
                start = idx * self.seq_len
                end = start + self.seq_len

                input_ids = torch.from_numpy(data[start:end].astype(np.int64))
                x = input_ids[:-1]
                y = input_ids[1:]

                yield {"input_ids": x, "labels": y}


def create_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create a DataLoader with appropriate settings for training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not isinstance(dataset, IterableDataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
