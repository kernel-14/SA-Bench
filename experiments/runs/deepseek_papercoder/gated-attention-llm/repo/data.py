## data.py
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
import datasets
from typing import List, Dict, Optional, Union, Iterator
import random

class DataModule:
    """
    Data loading module for multi‑domain language model training and evaluation.

    Handles streaming loading, tokenization, interleaving, and chunking of large
    text corpora, producing fixed‑length sequences suitable for causal language modelling.
    The module is designed to faithfully reproduce the data pipeline described in the
    "Gated Attention for Large Language Models" paper.
    """

    def __init__(
        self,
        config: Dict,
        tokenizer: AutoTokenizer,
        seq_length: int = 4096,
        batch_size: int = 1,
    ):
        """
        Initialise the DataModule.

        Args:
            config: The full configuration dictionary (as loaded from config.yaml).
                    The 'data' section is used for dataset descriptions.
            tokenizer: HuggingFace tokenizer (should have eos_token).
            seq_length: Length of sequences to produce for training/evaluation.
            batch_size: Per‑device batch size (for training; evaluation uses batch_size=1).
        """
        self.config = config
        self.data_cfg = config["data"]
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size

        # Ensure the tokenizer has an eos token
        if self.tokenizer.eos_token is None:
            # Add a default eos token if missing (e.g., "</s>")
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})
        self.eos_token_id = self.tokenizer.eos_token_id

        # Mapping from evaluation domain name to (dataset_name, config, split)
        self._eval_domain_map = {
            "c4_en_test": ("c4", "en", "validation"),
            "c4_zh_test": ("c4", "zh", "validation"),
            "code_test": ("the_stack", "python", "validation"),
            "math_test": ("mathpile", None, "validation"),
        }

    def get_train_dataloader(self) -> DataLoader:
        """
        Build a training DataLoader that yields batches of token sequences.

        The training data is a weighted interleaving of multiple streaming datasets,
        globally shuffled and chunked into fixed‑length windows.

        Returns:
            DataLoader producing dicts of {"input_ids": tensor, "labels": tensor}.
        """
        # Load each training dataset as a streaming tokenised iterable
        streams = []
        weights = []
        for entry in self.data_cfg["train_datasets"]:
            ds_stream = self._load_streaming_dataset(entry)
            streams.append(ds_stream)
            weights.append(entry["weight"])

        # Normalise weights
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("Training dataset weights must sum to a positive number.")
        weights = [w / total_weight for w in weights]

        # Interleave them according to weights and shuffle
        interleaved = self._interleave_streams(streams, weights)

        # In distributed training, split the data across processes
        if torch.distributed.is_initialized():
            interleaved = datasets.distributed.split_dataset_by_node(
                interleaved,
                world_size=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
            )

        # Wrap in an IterableDataset that yields fixed‑length chunks
        chunked_dataset = _ChunkedTokenDataset(
            token_stream=interleaved,
            seq_length=self.seq_length,
            eos_token_id=self.eos_token_id,
            drop_last=True,       # discard incomplete sequences during training
        )

        dataloader = DataLoader(
            chunked_dataset,
            batch_size=self.batch_size,
            num_workers=0,         # streaming iterable datasets are incompatible with multiprocessing
            pin_memory=True,
        )
        return dataloader

    def get_eval_dataloader(self, domain_name: str) -> DataLoader:
        """
        Build an evaluation DataLoader for a single held‑out domain.

        Args:
            domain_name: One of the keys defined in the `eval_datasets` list
                         of the configuration (e.g., "c4_en_test").

        Returns:
            DataLoader yielding batches (batch_size=1) of token sequences,
            including the final (potentially padded) chunk for exhaustive perplexity evaluation.
        """
        if domain_name not in self._eval_domain_map:
            raise ValueError(f"Unknown evaluation domain: {domain_name}")

        dataset_name, config, split = self._eval_domain_map[domain_name]
        entry = {"name": dataset_name, "split": split}
        if config is not None:
            entry["config"] = config

        ds_stream = self._load_streaming_dataset(entry)
        # For evaluation, we want to cover all tokens; drop_last=False yields a final chunk
        # that may be padded.
        chunked_dataset = _ChunkedTokenDataset(
            token_stream=ds_stream,
            seq_length=self.seq_length,
            eos_token_id=self.eos_token_id,
            drop_last=False,
        )
        dataloader = DataLoader(
            chunked_dataset,
            batch_size=1,          # single sequence per batch for exact loss tracking
            num_workers=0,
            pin_memory=True,
        )
        return dataloader

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize_fn(self, example: Dict) -> Dict:
        """
        Tokenize a single text example and append an EOS token.

        The function tries to extract text from the 'text' field first; if absent,
        it uses the first string‑typed column of the example.

        Args:
            example: A dictionary containing the raw text.

        Returns:
            A dictionary with a single key "input_ids", containing the token list.
        """
        text = None
        # Try common text field names
        for key in ("text", "content"):
            if key in example:
                text = example[key]
                break
        if text is None:
            # fallback: use the first column that is a string
            for key, value in example.items():
                if isinstance(value, str):
                    text = value
                    break
            if text is None:
                raise ValueError("Could not locate text field in example.")

        # Tokenize without adding special tokens (the EOS will be appended manually)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids.append(self.eos_token_id)
        return {"input_ids": token_ids}

    def _load_streaming_dataset(self, dataset_info: Dict) -> IterableDataset:
        """
        Load a single dataset in streaming mode, apply tokenization, and return
        an iterable that yields token lists.

        Args:
            dataset_info: Dictionary with keys: 'name', 'split', and optionally 'config'.

        Returns:
            An IterableDataset yielding dictionaries with key 'input_ids' (list of ints).
        """
        name = dataset_info["name"]
        split = dataset_info["split"]
        config = dataset_info.get("config", None)

        ds = datasets.load_dataset(
            name,
            config,
            split=split,
            streaming=True,
            cache_dir=self.data_cfg.get("cache_dir", None),
        )
        # Remove all columns except the tokenised ones; map lazily
        ds = ds.map(
            self._tokenize_fn,
            remove_columns=ds.column_names,
            batched=False,
        )
        return ds

    def _interleave_streams(
        self,
        streams: List[IterableDataset],
        weights: List[float],
        seed: int = 42,
    ) -> IterableDataset:
        """
        Interleave multiple streaming datasets according to the given weights and apply
        a global shuffle.

        Args:
            streams: List of tokenised IterableDataset objects.
            weights: List of probabilities (summing to 1) for each stream.
            seed: Random seed for reproducibility.

        Returns:
            A single interleaved IterableDataset.
        """
        if len(streams) == 1:
            interleaved = streams[0]
        else:
            interleaved = datasets.interleave_datasets(
                [stream for stream in streams],
                probabilities=weights,
                seed=seed,
            )
        # Shuffle with a buffer to break local order; 1M tokens is a typical choice.
        interleaved = interleaved.shuffle(buffer_size=1_000_000, seed=seed)
        return interleaved


class _ChunkedTokenDataset(IterableDataset):
    """
    Internal IterableDataset that consumes a stream of variable‑length token lists and
    yields fixed‑length sequences suitable for language modelling.

    The dataset drops incomplete chunks when `drop_last=True`, or pads the final chunk
    (with label masking) when `drop_last=False`.
    """

    def __init__(
        self,
        token_stream: IterableDataset,
        seq_length: int,
        eos_token_id: int,
        drop_last: bool = True,
    ):
        """
        Args:
            token_stream: An iterable that yields dicts with key 'input_ids' (list of ints).
            seq_length: Desired sequence length (number of tokens in input_ids).
            eos_token_id: Token ID used for padding (if drop_last=False).
            drop_last: Whether to discard the final incomplete sequence (training) or pad it.
        """
        self.token_stream = token_stream
        self.seq_length = seq_length
        self.eos_token_id = eos_token_id
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []
        for example in self.token_stream:
            token_ids = example["input_ids"]
            buffer.extend(token_ids)

            while len(buffer) >= self.seq_length + 1:
                # Extract a full sequence of length seq_length + 1
                chunk = buffer[: self.seq_length + 1]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}
                # Remove the first seq_length tokens from the buffer (non‑overlapping windows)
                buffer = buffer[self.seq_length :]

        # Handle the final incomplete chunk if drop_last is False
        if not self.drop_last and len(buffer) > 1:
            # Pad to seq_length + 1; the labels for padding positions are set to -100
            padded = buffer + [self.eos_token_id] * (self.seq_length + 1 - len(buffer))
            input_ids = torch.tensor(padded[:-1], dtype=torch.long)
            labels = torch.tensor(padded[1:], dtype=torch.long)
            # Set labels for padding positions to -100 (ignore index)
            labels[len(buffer) - 1 :] = -100
            yield {"input_ids": input_ids, "labels": labels}
