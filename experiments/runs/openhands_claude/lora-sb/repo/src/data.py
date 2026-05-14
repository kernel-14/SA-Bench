"""Dataset loading and preprocessing for all three benchmarks.

Benchmarks:
1. Math Reasoning: MetaMathQA (train) → GSM8K + MATH (eval)
2. Commonsense Reasoning: COMMONSENSE170K (train) → 8 datasets (eval)
3. Natural Language Understanding: GLUE (train + eval)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

METAMATH_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: {output}"
)

METAMATH_INFERENCE_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

COMMONSENSE_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: {output}"
)

COMMONSENSE_INFERENCE_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


# ---------------------------------------------------------------------------
# Math Reasoning: MetaMathQA
# ---------------------------------------------------------------------------

def load_metamath(
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 512,
    n_train: int = 50000,
    seed: int = 42,
) -> Dataset:
    """Load and tokenize MetaMathQA training dataset (50K samples)."""
    dataset = load_dataset("meta-math/MetaMathQA", split="train")

    if len(dataset) > n_train:
        dataset = dataset.shuffle(seed=seed).select(range(n_train))

    def format_and_tokenize(example):
        text = METAMATH_PROMPT.format(
            instruction=example["query"],
            output=example["response"],
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
    return dataset


def load_gsm8k(split: str = "test") -> Dataset:
    """Load GSM8K evaluation dataset."""
    return load_dataset("gsm8k", "main", split=split)


def load_math_dataset(split: str = "test") -> Dataset:
    """Load MATH evaluation dataset."""
    return load_dataset("hendrycks/competition_math", split=split, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Commonsense Reasoning: COMMONSENSE170K
# ---------------------------------------------------------------------------

COMMONSENSE_DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Easy",
    "ARC-Challenge",
    "openbookqa",
]

COMMONSENSE_HF_NAMES = {
    "boolq": ("google/boolq", None),
    "piqa": ("piqa", None),
    "social_i_qa": ("social_i_qa", None),
    "hellaswag": ("hellaswag", None),
    "winogrande": ("winogrande", "winogrande_xl"),
    "ARC-Easy": ("ai2_arc", "ARC-Easy"),
    "ARC-Challenge": ("ai2_arc", "ARC-Challenge"),
    "openbookqa": ("openbookqa", "main"),
}


def load_commonsense170k(
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 256,
    seed: int = 42,
) -> Dataset:
    """Load COMMONSENSE170K training dataset.

    This dataset is from LLM-Adapters (Hu et al., 2023) and combines 8 commonsense
    reasoning datasets. We load it from the HuggingFace hub.
    """
    dataset = load_dataset("MuskumPillerum/commonsense_170k", split="train")

    def format_and_tokenize(example):
        text = COMMONSENSE_PROMPT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)
    return dataset


def load_commonsense_eval(dataset_name: str) -> Dataset:
    """Load a single commonsense reasoning evaluation dataset."""
    hf_name, config = COMMONSENSE_HF_NAMES[dataset_name]
    if config:
        return load_dataset(hf_name, config, split="validation", trust_remote_code=True)
    return load_dataset(hf_name, split="validation", trust_remote_code=True)


# ---------------------------------------------------------------------------
# GLUE
# ---------------------------------------------------------------------------

GLUE_TASKS = ["cola", "rte", "mrpc", "stsb", "qnli", "sst2"]

GLUE_NUM_LABELS = {
    "cola": 2,
    "rte": 2,
    "mrpc": 2,
    "stsb": 1,  # regression
    "qnli": 2,
    "sst2": 2,
}

GLUE_MAX_SEQ_LEN = {
    "cola": 128,
    "rte": 256,
    "mrpc": 128,
    "stsb": 128,
    "qnli": 256,
    "sst2": 128,
}


def load_glue_task(
    task_name: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: Optional[int] = None,
) -> DatasetDict:
    """Load and tokenize a GLUE task."""
    dataset = load_dataset("glue", task_name)
    seq_len = max_seq_len or GLUE_MAX_SEQ_LEN.get(task_name, 128)

    def tokenize_fn(examples):
        if task_name == "stsb":
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                max_length=seq_len,
                padding="max_length",
            )
        elif task_name in ("rte", "mrpc", "qnli"):
            key1 = "sentence1" if task_name in ("rte", "mrpc") else "question"
            key2 = "sentence2" if task_name in ("rte", "mrpc") else "sentence"
            return tokenizer(
                examples[key1],
                examples[key2],
                truncation=True,
                max_length=seq_len,
                padding="max_length",
            )
        elif task_name in ("cola", "sst2"):
            return tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=seq_len,
                padding="max_length",
            )
        else:
            raise ValueError(f"Unknown GLUE task: {task_name}")

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


# ---------------------------------------------------------------------------
# DataLoader utilities
# ---------------------------------------------------------------------------

def get_init_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    n_samples: int,
    batch_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for LoRA-SB initialization (0.1% of dataset).

    Randomly selects n_samples from the dataset.
    """
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    selected = indices[:n_samples]
    subset = Subset(dataset, selected)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_causal_lm_collate_fn(tokenizer),
    )


def _causal_lm_collate_fn(tokenizer: PreTrainedTokenizer):
    """Collate function for causal LM datasets."""
    def collate(batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }
    return collate


def get_causal_lm_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for causal LM training."""
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_causal_lm_collate_fn(tokenizer),
    )


def get_glue_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for GLUE tasks."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
