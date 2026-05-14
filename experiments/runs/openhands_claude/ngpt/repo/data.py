"""
Data loading for the OpenWebText dataset (Gokaslan & Cohen, 2019).

The paper trains on OpenWebText with the LLaMA-2 tokenizer (32k vocabulary).
We support two modes:
  1. Pre-tokenized binary files (fast, recommended for large-scale training).
  2. On-the-fly tokenization from raw text files (for smaller experiments).

Pre-tokenization script: python data.py --tokenize --data_dir <path>
"""

import os
import math
import struct
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Optional, Iterator


# ---------------------------------------------------------------------------
# Pre-tokenized binary dataset (memory-mapped for efficiency)
# ---------------------------------------------------------------------------

class TokenizedDataset(Dataset):
    """Memory-mapped dataset of pre-tokenized token IDs stored as uint16.

    Expected file format: flat binary file of uint16 token IDs.
    Compatible with the format produced by nanoGPT's data preparation scripts.
    """

    def __init__(self, data_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        # Number of complete (input, target) pairs
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


class StreamingTokenDataset(Dataset):
    """Streaming dataset that samples random windows from a token array.

    Suitable for training where the dataset is much larger than memory.
    """

    def __init__(self, data_path: str, seq_len: int, seed: int = 42):
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_tokens // self.seq_len

    def __getitem__(self, idx: int):
        # Random offset for data augmentation
        start = self.rng.integers(0, self.n_tokens - self.seq_len - 1)
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    data_path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 4,
    streaming: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader for the tokenized dataset.

    Args:
        data_path:   Path to the binary token file (uint16).
        seq_len:     Sequence length (context window).
        batch_size:  Per-device batch size.
        num_workers: DataLoader worker processes.
        streaming:   Use random-window streaming (True) or sequential (False).
        rank:        Process rank for distributed training.
        world_size:  Total number of processes.
        seed:        Random seed.
    """
    if streaming:
        dataset = StreamingTokenDataset(data_path, seq_len, seed=seed + rank)
    else:
        dataset = TokenizedDataset(data_path, seq_len)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and streaming is False),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def infinite_loader(loader: DataLoader) -> Iterator:
    """Wrap a DataLoader to yield batches indefinitely."""
    while True:
        for batch in loader:
            yield batch


# ---------------------------------------------------------------------------
# Tokenization helper (requires sentencepiece / transformers)
# ---------------------------------------------------------------------------

def tokenize_openwebtext(
    data_dir: str,
    output_dir: str,
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    val_fraction: float = 0.0005,
    num_proc: int = 8,
):
    """Tokenize the OpenWebText dataset and save as binary uint16 files.

    Requires: datasets, transformers, sentencepiece

    Args:
        data_dir:       Directory containing raw OpenWebText text files, or
                        "openwebtext" to download via HuggingFace datasets.
        output_dir:     Where to write train.bin and val.bin.
        tokenizer_name: HuggingFace tokenizer identifier.
        val_fraction:   Fraction of data to use for validation.
        num_proc:       Number of parallel tokenization workers.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # LLaMA-2 tokenizer does not add BOS by default in encode(); add manually
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    if os.path.isdir(data_dir):
        dataset = load_dataset("text", data_files={"train": os.path.join(data_dir, "*.txt")})
    else:
        dataset = load_dataset("openwebtext", num_proc=num_proc)

    split = dataset["train"].train_test_split(
        test_size=val_fraction, seed=2357, shuffle=True
    )
    split["val"] = split.pop("test")

    def tokenize(example):
        ids = tokenizer.encode(example["text"])
        ids.append(eos_id)
        return {"ids": ids, "len": len(ids)}

    tokenized = split.map(
        tokenize,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=num_proc,
    )

    for split_name, dset in tokenized.items():
        total = sum(dset["len"])
        arr = np.memmap(
            os.path.join(output_dir, f"{split_name}.bin"),
            dtype=np.uint16,
            mode="w+",
            shape=(total,),
        )
        idx = 0
        for sample in dset:
            ids = sample["ids"]
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)
        arr.flush()
        print(f"Wrote {total:,} tokens to {split_name}.bin")


# ---------------------------------------------------------------------------
# CLI for pre-tokenization
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize OpenWebText for nGPT training")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--data_dir", type=str, default="openwebtext")
    parser.add_argument("--output_dir", type=str, default="data/openwebtext")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--val_fraction", type=float, default=0.0005)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    if args.tokenize:
        tokenize_openwebtext(
            args.data_dir,
            args.output_dir,
            args.tokenizer,
            args.val_fraction,
            args.num_proc,
        )
