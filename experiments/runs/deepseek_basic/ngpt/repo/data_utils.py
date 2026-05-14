"""
Data loading and preprocessing utilities for training nGPT.

Supports loading the OpenWebText dataset and preparing it for training.
Uses the LLaMA-2 tokenizer (32k tokens) as described in the paper.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class OpenWebTextDataset(Dataset):
    """
    Dataset for OpenWebText corpus.

    The paper uses OpenWebText (Gokaslan & Cohen, 2019) which represents
    a good approximation of OpenAI's internal dataset used to train GPT-2 models.
    """

    def __init__(self, data_path, seq_len, split='train', split_ratio=0.99):
        """
        Args:
            data_path: Path to tokenized data (numpy memmap or file)
            seq_len: Sequence length for training
            split: 'train' or 'val'
            split_ratio: Fraction of data to use for training
        """
        self.seq_len = seq_len

        # Load tokenized data
        if data_path.endswith('.npy'):
            data = np.load(data_path)
        elif data_path.endswith('.bin'):
            data = np.memmap(data_path, dtype=np.uint16, mode='r')
        else:
            # Assume it's a text file with integer tokens
            data = np.loadtxt(data_path, dtype=np.int64)

        # Split into train/val
        n_train = int(len(data) * split_ratio)
        if split == 'train':
            self.data = data[:n_train]
        else:
            self.data = data[n_train:]

        # Ensure we have enough data
        n_sequences = max(0, len(self.data) - seq_len)
        self.data = self.data[:n_sequences + seq_len]

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_openwebtext_loaders(
    data_path,
    seq_len,
    batch_size,
    num_workers=4,
    split_ratio=0.99,
):
    """
    Create train and validation data loaders for OpenWebText.

    Args:
        data_path: Path to tokenized data
        seq_len: Sequence length
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        split_ratio: Train/val split ratio

    Returns:
        train_loader, val_loader
    """
    train_dataset = OpenWebTextDataset(data_path, seq_len, 'train', split_ratio)
    val_dataset = OpenWebTextDataset(data_path, seq_len, 'val', split_ratio)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, val_loader


def tokenize_with_llama2(text_path, output_path, tokenizer_name='gpt2'):
    """
    Tokenize text using GPT-2 tokenizer (as an approximation for LLaMA-2 tokenizer).
    The paper uses LLaMA-2 tokenizer with 32k tokens.

    For a full reproduction, the actual LLaMA-2 tokenizer should be used.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    tokens = np.array(tokens, dtype=np.uint16)
    np.save(output_path, tokens)

    print(f"Tokenized {len(tokens)} tokens, saved to {output_path}")
    return tokens
