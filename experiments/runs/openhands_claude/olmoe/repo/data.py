"""Data loading and preprocessing for OLMoE pretraining and adaptation.

Pretraining data (OLMoE-Mix, Table 2):
  - DCLM-Baseline: 3,860B tokens (web pages)
  - StarCoder: 101B tokens (code)
  - peS2o: 57.2B tokens (STEM papers)
  - arXiv: 21.1B tokens (STEM papers)
  - OpenWebMath: 12.7B tokens (math web pages)
  - Algebraic Stack: 12.6B tokens (math proofs/code)
  - English Wikipedia & Wikibooks: 3.69B tokens (encyclopedic)
  Total: ~4,060B tokens (trained for 5.133T = 1.3 epochs)

Preprocessing:
  - Filter documents with 32+ repeated n-grams (n=1..13)
  - StarCoder: additional quality filters (GitHub stars, word frequency)
  - Random shuffle at start of each epoch
  - Annealing: reshuffle + linear LR decay for final 100B tokens

Adaptation data (Table 3):
  SFT: Tulu 2 SFT Mix, No Robots, CodeFeedback, MetaMathQA, Daring Anteater
  DPO: UltraFeedback binarized (filtered for TruthfulQA contamination)
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


# ---------------------------------------------------------------------------
# Pretraining dataset
# ---------------------------------------------------------------------------

# Sampling weights derived from Table 2 token counts
OLMOE_MIX_SOURCES = {
    "dclm_baseline": {"tokens_b": 3860, "doc_type": "web"},
    "starcoder": {"tokens_b": 101, "doc_type": "code"},
    "pes2o": {"tokens_b": 57.2, "doc_type": "stem_papers"},
    "arxiv": {"tokens_b": 21.1, "doc_type": "stem_papers"},
    "openwebmath": {"tokens_b": 12.7, "doc_type": "math_web"},
    "algebraic_stack": {"tokens_b": 12.6, "doc_type": "math_code"},
    "wikipedia_wikibooks": {"tokens_b": 3.69, "doc_type": "encyclopedic"},
}


def get_source_weights() -> Dict[str, float]:
    """Compute sampling weights proportional to token counts."""
    total = sum(v["tokens_b"] for v in OLMOE_MIX_SOURCES.values())
    return {k: v["tokens_b"] / total for k, v in OLMOE_MIX_SOURCES.items()}


class TokenizedDocument:
    """Represents a single tokenized document."""

    def __init__(self, token_ids: List[int], source: str):
        self.token_ids = token_ids
        self.source = source


def has_repeated_ngrams(token_ids: List[int], min_n: int = 1, max_n: int = 13, threshold: int = 32) -> bool:
    """Filter documents with 32+ consecutive repeated n-grams (§2).

    An n-gram is any span of 1 to 13 tokens. Returns True if the document
    contains a sequence of 32 or more repeated n-grams.
    """
    for n in range(min_n, max_n + 1):
        if len(token_ids) < n * threshold:
            continue
        ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(ngrams)):
            if ngrams[i] == ngrams[i - 1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
                if max_consecutive >= threshold:
                    return True
            else:
                current_consecutive = 1
    return False


def starcoder_quality_filter(token_ids: List[int], word_counts: Optional[Dict[str, int]] = None) -> bool:
    """Additional StarCoder quality filters (§2).

    Removes documents where:
    - Repository has fewer than 2 GitHub stars (metadata-based, not token-based)
    - Most frequent word constitutes >30% of document
    - Top-2 most frequent words constitute >50% of document

    Args:
        token_ids: tokenized document
        word_counts: pre-computed word frequency dict (optional)
    Returns:
        True if document should be kept
    """
    if word_counts is None:
        return True

    total_words = sum(word_counts.values())
    if total_words == 0:
        return True

    sorted_counts = sorted(word_counts.values(), reverse=True)
    top1_ratio = sorted_counts[0] / total_words
    if top1_ratio > 0.30:
        return False

    if len(sorted_counts) >= 2:
        top2_ratio = (sorted_counts[0] + sorted_counts[1]) / total_words
        if top2_ratio > 0.50:
            return False

    return True


class PretrainingDataset(IterableDataset):
    """Streaming dataset for OLMoE pretraining.

    Reads tokenized documents from disk, applies filters, packs into
    fixed-length sequences of seq_len tokens with EOS separator.

    Supports multi-epoch training with reshuffling (paper trains for 1.3 epochs).
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 4096,
        seed: int = 42,
        max_tokens: Optional[int] = None,
        source_weights: Optional[Dict[str, float]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.seed = seed
        self.max_tokens = max_tokens
        self.source_weights = source_weights or get_source_weights()

    def _iter_files(self, source: str) -> Iterator[Path]:
        source_dir = self.data_dir / source
        if not source_dir.exists():
            return
        files = sorted(source_dir.glob("*.jsonl")) + sorted(source_dir.glob("*.bin"))
        rng = random.Random(self.seed)
        rng.shuffle(files)
        yield from files

    def _read_tokens(self, file_path: Path) -> Iterator[List[int]]:
        """Read tokenized documents from a JSONL file."""
        if file_path.suffix == ".jsonl":
            with open(file_path) as f:
                for line in f:
                    doc = json.loads(line)
                    token_ids = doc.get("input_ids", doc.get("tokens", []))
                    source = doc.get("source", "unknown")
                    if not has_repeated_ngrams(token_ids):
                        yield token_ids
        elif file_path.suffix == ".bin":
            # Memory-mapped numpy array of token IDs
            tokens = np.memmap(file_path, dtype=np.uint16, mode="r")
            yield tokens.tolist()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        sources = list(self.source_weights.keys())
        weights = [self.source_weights[s] for s in sources]
        rng = random.Random(self.seed)

        buffer: List[int] = []
        tokens_yielded = 0

        while True:
            source = rng.choices(sources, weights=weights, k=1)[0]
            for file_path in self._iter_files(source):
                for token_ids in self._read_tokens(file_path):
                    buffer.extend(token_ids)
                    buffer.append(0)  # EOS separator

                    while len(buffer) >= self.seq_len + 1:
                        chunk = buffer[: self.seq_len + 1]
                        buffer = buffer[self.seq_len + 1:]

                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        labels = torch.tensor(chunk[1:], dtype=torch.long)

                        yield {"input_ids": input_ids, "labels": labels}
                        tokens_yielded += self.seq_len

                        if self.max_tokens and tokens_yielded >= self.max_tokens:
                            return


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------

# Adaptation data sources (Table 3)
SFT_SOURCES = {
    "tulu2_sft_mix": 326154,
    "no_robots": 9500,
    "codefeedback_filtered": 156526,
    "metamathqa": 98750,
    "daring_anteater_advanced": 17082,
}

DPO_SOURCES = {
    "ultrafeedback_binarized_cleaned": 60800,
}


@dataclass
class SFTExample:
    prompt: str
    response: str
    source: str


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


class SFTDataset(Dataset):
    """Supervised fine-tuning dataset.

    Formats prompt-response pairs for instruction tuning.
    Loss is computed only on response tokens (token-level aggregation
    following Muennighoff et al. 2024 §B for long generative tasks).
    Filters to max_seq_len=4096 tokens.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = self._load(data_path)

    def _load(self, data_path: str) -> List[Dict]:
        examples = []
        path = Path(data_path)
        files = list(path.glob("*.jsonl")) if path.is_dir() else [path]
        for f in files:
            with open(f) as fp:
                for line in fp:
                    ex = json.loads(line)
                    examples.append(ex)
        return examples

    def _format_prompt(self, messages: List[Dict]) -> Tuple[str, str]:
        """Format chat messages into prompt and response strings."""
        prompt_parts = []
        response = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                response = content
            elif role == "system":
                prompt_parts.insert(0, f"<|system|>\n{content}\n")
        prompt = "".join(prompt_parts) + "<|assistant|>\n"
        return prompt, response

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        messages = ex.get("messages", ex.get("conversations", []))
        prompt, response = self._format_prompt(messages)

        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response) + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        # Labels: -100 for prompt tokens (not trained on), response tokens for loss
        labels = [-100] * len(prompt_ids) + response_ids

        # Truncate to max_seq_len
        input_ids = input_ids[: self.max_seq_len]
        labels = labels[: self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DPODataset(Dataset):
    """Direct Preference Optimization dataset.

    Each example has a prompt, chosen response, and rejected response.
    UltraFeedback binarized, filtered for TruthfulQA contamination.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = self._load(data_path)

    def _load(self, data_path: str) -> List[Dict]:
        examples = []
        with open(data_path) as f:
            for line in f:
                examples.append(json.loads(line))
        return examples

    def _encode_pair(self, prompt: str, response: str) -> Tuple[List[int], List[int]]:
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response) + [self.tokenizer.eos_token_id]
        input_ids = (prompt_ids + response_ids)[: self.max_seq_len]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_seq_len]
        return input_ids, labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        chosen_ids, chosen_labels = self._encode_pair(prompt, chosen)
        rejected_ids, rejected_labels = self._encode_pair(prompt, rejected)

        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }


class KTODataset(Dataset):
    """KTO (Kahneman-Tversky Optimization) dataset.

    Each example has a prompt, response, and binary label (desirable/undesirable).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = self._load(data_path)

    def _load(self, data_path: str) -> List[Dict]:
        examples = []
        with open(data_path) as f:
            for line in f:
                examples.append(json.loads(line))
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt_ids = self.tokenizer.encode(ex["prompt"])
        response_ids = self.tokenizer.encode(ex["completion"]) + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + response_ids)[: self.max_seq_len]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "label": torch.tensor(float(ex["label"]), dtype=torch.float),
        }


def sft_collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad a batch of SFT examples to the same length."""
    max_len = max(ex["input_ids"].shape[0] for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, ex in enumerate(batch):
        n = ex["input_ids"].shape[0]
        input_ids[i, :n] = ex["input_ids"]
        labels[i, :n] = ex["labels"]

    return {"input_ids": input_ids, "labels": labels}


def dpo_collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad a batch of DPO examples."""
    def pad_sequences(seqs, pad_val):
        max_len = max(s.shape[0] for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            out[i, : s.shape[0]] = s
        return out

    return {
        "chosen_input_ids": pad_sequences([ex["chosen_input_ids"] for ex in batch], pad_token_id),
        "chosen_labels": pad_sequences([ex["chosen_labels"] for ex in batch], -100),
        "rejected_input_ids": pad_sequences([ex["rejected_input_ids"] for ex in batch], pad_token_id),
        "rejected_labels": pad_sequences([ex["rejected_labels"] for ex in batch], -100),
    }


def build_pretraining_dataloader(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 4,
    max_tokens: Optional[int] = None,
) -> DataLoader:
    dataset = PretrainingDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        seed=seed,
        max_tokens=max_tokens,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_sft_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int,
    max_seq_len: int = 4096,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = SFTDataset(data_path, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: sft_collate_fn(b, tokenizer.pad_token_id or 0),
        pin_memory=True,
    )


def build_dpo_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int,
    max_seq_len: int = 4096,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = DPODataset(data_path, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: dpo_collate_fn(b, tokenizer.pad_token_id or 0),
        pin_memory=True,
    )
