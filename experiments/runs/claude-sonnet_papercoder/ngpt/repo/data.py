```python
## data.py
"""Data pipeline for nGPT and GPT experiment reproduction.

This module handles downloading, tokenizing, caching, and serving batches of
OpenWebText data for training, and PG19 data for length extrapolation
evaluation (Appendix A.8 of the nGPT paper).

The module exposes two public classes:
    - OpenWebTextDataset: Primary training/validation data source.
    - PG19Dataset: Length extrapolation evaluation dataset.

All configuration values are sourced from the Config dataclass (config.py),
which in turn is populated from config.yaml. No values are hardcoded here.

Typical usage:
    from config import Config
    from data import OpenWebTextDataset, PG19Dataset

    config = Config.ngpt_500m(context_length=4096)
    dataset = OpenWebTextDataset(config)

    # Training batch
    x, y = dataset.get_batch('train', device='cuda')

    # Validation loop
    for x_val, y_val in dataset.get_val_loader(steps=100):
        ...

    # Length extrapolation
    pg19 = PG19Dataset(config, dataset.tokenizer)
    x_long, y_long = pg19.get_batch_at_length(length=8192, device='cuda')
"""

import logging
import os
import pathlib
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch

from config import Config
from utils import setup_logger


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = setup_logger("data")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create a directory and all parent directories if they do not exist.

    Args:
        path: The directory path to create.
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def _load_tokenizer(
    tokenizer_name: str,
    tokenizer_fallback: str,
) -> Tuple[object, int, bool]:
    """Load a HuggingFace tokenizer with fallback support.

    Attempts to load the primary tokenizer (LLaMA-2). If unavailable due to
    license restrictions or network issues, falls back to GPT-2. The actual
    vocab size is returned so callers can update the config accordingly.

    Args:
        tokenizer_name: Primary tokenizer identifier (e.g.,
            "meta-llama/Llama-2-7b-hf").
        tokenizer_fallback: Fallback tokenizer identifier (e.g., "gpt2").

    Returns:
        A tuple of (tokenizer, vocab_size, used_fallback) where:
            - tokenizer: The loaded HuggingFace tokenizer object.
            - vocab_size: The actual vocabulary size of the loaded tokenizer.
            - used_fallback: True if the fallback tokenizer was used.
    """
    from transformers import AutoTokenizer  # type: ignore

    # Attempt to load the primary tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )
        vocab_size = tokenizer.vocab_size
        logger.info(
            "Loaded primary tokenizer '%s' with vocab_size=%d.",
            tokenizer_name,
            vocab_size,
        )
        return tokenizer, vocab_size, False

    except (OSError, EnvironmentError, Exception) as exc:
        logger.warning(
            "Failed to load primary tokenizer '%s': %s. "
            "Falling back to '%s'. "
            "NOTE: Model vocab_size will be %s (GPT-2), not 32000 (LLaMA-2). "
            "This changes model parameter count and results will not exactly "
            "match the paper.",
            tokenizer_name,
            exc,
            tokenizer_fallback,
            "50257",
        )

    # Load fallback tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_fallback,
        use_fast=True,
    )
    vocab_size = tokenizer.vocab_size
    logger.info(
        "Loaded fallback tokenizer '%s' with vocab_size=%d.",
        tokenizer_fallback,
        vocab_size,
    )
    return tokenizer, vocab_size, True


def _get_eos_token_id(tokenizer: object) -> int:
    """Safely retrieve the EOS token ID from a tokenizer.

    Falls back to 0 if the tokenizer does not define an EOS token, which
    should not happen for LLaMA-2 or GPT-2 but is handled defensively.

    Args:
        tokenizer: A HuggingFace tokenizer object.

    Returns:
        The integer EOS token ID.
    """
    eos_id: Optional[int] = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        logger.warning(
            "Tokenizer has no eos_token_id. Using 0 as EOS token. "
            "This may cause incorrect document boundary handling."
        )
        return 0
    return int(eos_id)


def _tokenize_batch(
    examples: dict,
    tokenizer: object,
    eos_token_id: int,
) -> dict:
    """Tokenize a batch of text documents for use with datasets.map().

    Encodes each document and appends an EOS token to mark document
    boundaries. Empty documents are skipped.

    Args:
        examples: A batch dict from HuggingFace datasets with key "text".
        tokenizer: The HuggingFace tokenizer to use for encoding.
        eos_token_id: The EOS token ID to append after each document.

    Returns:
        A dict with key "input_ids" containing a list of token ID lists.
    """
    all_ids: List[List[int]] = []
    for text in examples["text"]:
        if not text or not text.strip():
            continue
        ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            continue
        ids.append(eos_token_id)
        all_ids.append(ids)
    return {"input_ids": all_ids}


# ---------------------------------------------------------------------------
# OpenWebTextDataset
# ---------------------------------------------------------------------------

class OpenWebTextDataset:
    """Dataset class for OpenWebText training and validation data.

    Downloads, tokenizes, and caches the OpenWebText corpus as flat binary
    arrays. Provides efficient random-access batch sampling via numpy memmaps.

    The dataset is split 90/10 into train and validation sets following the
    nanoGPT convention (config.yaml data.train_val_split: 0.9).

    Attributes:
        config: The experiment configuration.
        tokenizer: The loaded HuggingFace tokenizer.
        vocab_size: The actual vocabulary size of the loaded tokenizer.
            May differ from config.vocab_size if the fallback tokenizer
            was used.
        train_data: Memory-mapped numpy array of training token IDs (uint16).
        val_data: Memory-mapped numpy array of validation token IDs (uint16).
    """

    def __init__(self, config: Config) -> None:
        """Initialize the dataset, loading from cache or downloading.

        Attempts to load the LLaMA-2 tokenizer (config.tokenizer_name).
        Falls back to GPT-2 (config.tokenizer_fallback) if unavailable.
        Checks for cached binary files; downloads and tokenizes if missing.

        Args:
            config: The experiment configuration. Key fields used:
                - config.tokenizer_name: Primary tokenizer identifier.
                - config.tokenizer_fallback: Fallback tokenizer identifier.
                - config.vocab_size: Expected vocabulary size (may be updated
                  if fallback tokenizer is used).
                - config.cache_dir: Directory for cached binary files.
                - config.train_val_split: Fraction of data for training.
                - config.context_length: Sequence length for batch sampling.
                - config.batch_size: Global batch size for batch sampling.
        """
        self.config: Config = config

        # Load tokenizer with fallback support
        self.tokenizer, self.vocab_size, used_fallback = _load_tokenizer(
            tokenizer_name=config.tokenizer_name,
            tokenizer_fallback=config.tokenizer_fallback,
        )

        # Warn if vocab size differs from config expectation
        if self.vocab_size != config.vocab_size:
            logger.warning(
                "Actual tokenizer vocab_size=%d differs from config.vocab_size=%d. "
                "The model will be built with vocab_size=%d. "
                "Update config.vocab_size to suppress this warning.",
                self.vocab_size,
                config.vocab_size,
                self.vocab_size,
            )

        # Determine cache file paths
        self._train_bin: str = os.path.join(config.cache_dir, "train.bin")
        self._val_bin: str = os.path.join(config.cache_dir, "val.bin")

        # Check for partial cache (one file exists but not the other)
        train_exists: bool = os.path.isfile(self._train_bin)
        val_exists: bool = os.path.isfile(self._val_bin)

        if train_exists and not val_exists:
            logger.warning(
                "Partial cache detected: '%s' exists but '%s' does not. "
                "Deleting partial cache and re-tokenizing.",
                self._train_bin,
                self._val_bin,
            )
            os.remove(self._train_bin)
            train_exists = False

        if val_exists and not train_exists:
            logger.warning(
                "Partial cache detected: '%s' exists but '%s' does not. "
                "Deleting partial cache and re-tokenizing.",
                self._val_bin,
                self._train_bin,
            )
            os.remove(self._val_bin)
            val_exists = False

        # Download and tokenize if cache is missing
        if not train_exists or not val_exists:
            logger.info(
                "Cache not found at '%s'. Downloading and tokenizing OpenWebText. "
                "This may take 30-60 minutes on first run.",
                config.cache_dir,
            )
            self._download_and_tokenize()
        else:
            logger.info(
                "Found cached data at '%s'. Loading from cache.",
                config.cache_dir,
            )

        # Load memmaps
        self._load_from_cache()

        logger.info(
            "OpenWebTextDataset ready: %d train tokens, %d val tokens.",
            len(self.train_data),
            len(self.val_data),
        )

    def _download_and_tokenize(self) -> None:
        """Download OpenWebText, tokenize, and save as binary memmaps.

        Downloads the dataset from HuggingFace, tokenizes all documents using
        the loaded tokenizer (with EOS token appended per document), packs
        tokens into a flat array, splits 90/10 train/val, and saves as uint16
        numpy memmaps.

        The tokenization uses datasets.map() with batched=True for efficiency.
        Empty documents are skipped. The resulting binary files are saved to
        config.cache_dir/train.bin and config.cache_dir/val.bin.

        Raises:
            RuntimeError: If the dataset download fails or produces zero tokens.
        """
        import datasets as hf_datasets  # type: ignore

        _ensure_dir(self.config.cache_dir)

        eos_token_id: int = _get_eos_token_id(self.tokenizer)
        logger.info(
            "Using EOS token ID=%d for document boundary marking.",
            eos_token_id,
        )

        # Load OpenWebText — only has a 'train' split
        logger.info(
            "Loading dataset '%s' from HuggingFace...",
            self.config.dataset_name,
        )
        raw_dataset = hf_datasets.load_dataset(
            self.config.dataset_name,
            split="train",
            trust_remote_code=True,
        )
        logger.info(
            "Dataset loaded: %d documents.",
            len(raw_dataset),
        )

        # Tokenize using datasets.map() for efficiency
        # Use a closure to capture tokenizer and eos_token_id
        tokenizer_ref = self.tokenizer
        eos_id_ref: int = eos_token_id

        def _tokenize_fn(examples: dict) -> dict:
            return _tokenize_batch(examples, tokenizer_ref, eos_id_ref)

        logger.info("Tokenizing documents (this may take a while)...")
        tokenized = raw_dataset.map(
            _tokenize_fn,
            batched=True,
            batch_size=1000,
            num_proc=min(4, os.cpu_count() or 1),
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing OpenWebText",
        )

        # Concatenate all token ID lists into a single flat array
        logger.info("Concatenating all token sequences...")
        all_ids: List[int] = []
        for example in tokenized:
            ids = example.get("input_ids", [])
            if ids:
                all_ids.extend(ids)

        n_total: int = len(all_ids)
        if n_total == 0:
            raise RuntimeError(
                "Tokenization produced zero tokens. "
                "Check the dataset and tokenizer configuration."
            )

        logger.info(
            "Total tokens after tokenization: %d (%.2fB).",
            n_total,
            n_total / 1e9,
        )

        # Split train/val
        split_idx: int = int(n_total * self.config.train_val_split)
        train_ids: List[int] = all_ids[:split_idx]
        val_ids: List[int] = all_ids[split_idx:]

        logger.info(
            "Train/val split: %d train tokens (%.2fB), %d val tokens (%.2fB).",
            len(train_ids),
            len(train_ids) / 1e9,
            len(val_ids),
            len(val_ids) / 1e9,
        )

        # Validate uint16 range
        # Both LLaMA-2 (32000) and GPT-2 (50257) fit in uint16 (max 65535)
        max_token_id: int = max(all_ids) if all_ids else 0
        if max_token_id > 65535:
            raise RuntimeError(
                f"Maximum token ID {max_token_id} exceeds uint16 range (65535). "
                "Cannot save as uint16. Check tokenizer configuration."
            )

        # Save train tokens as uint16 memmap
        logger.info("Saving train tokens to '%s'...", self._train_bin)
        train_arr = np.memmap(
            self._train_bin,
            dtype=np.uint16,
            mode="w+",
            shape=(len(train_ids),),
        )
        train_arr[:] = np.array(train_ids, dtype=np.uint16)
        train_arr.flush()
        del train_arr  # Close file handle before re-opening in read mode

        # Save val tokens as uint16 memmap
        logger.info("Saving val tokens to '%s'...", self._val_bin)
        val_arr = np.memmap(
            self._val_bin,
            dtype=np.uint16,
            mode="w+",
            shape=(len(val_ids),),
        )
        val_arr[:] = np.array(val_ids, dtype=np.uint16)
        val_arr.flush()
        del val_arr  # Close file handle before re-opening in read mode

        logger.info(
            "Tokenization complete. Saved %d train tokens and %d val tokens.",
            len(train_ids),
            len(val_ids),
        )

    def _load_from_cache(self) -> None:
        """Memory-map the cached binary files for efficient random access.

        Opens train.bin and val.bin as read-only numpy memmaps. The OS will
        page in only the needed chunks during batch sampling, avoiding the
        need to load the full ~18GB dataset into RAM.

        Sets:
            self.train_data: Read-only memmap of training tokens (uint16).
            self.val_data: Read-only memmap of validation tokens (uint16).

        Raises:
            FileNotFoundError: If either binary file does not exist.
        """
        if not os.path.isfile(self._train_bin):
            raise FileNotFoundError(
                f"Training data cache not found at '{self._train_bin}'. "
                "Call _download_and_tokenize() first."
            )
        if not os.path.isfile(self._val_bin):
            raise FileNotFoundError(
                f"Validation data cache not found at '{self._val_bin}'. "
                "Call _download_and_tokenize() first."
            )

        self.train_data: np.ndarray = np.memmap(
            self._train_bin,
            dtype=np.uint16,
            mode="r",
        )
        self.val_data: np.ndarray = np.memmap(
            self._val_bin,
            dtype=np.uint16,
            mode="r",
        )

        logger.debug(
            "Loaded memmaps: train=%d tokens, val=%d tokens.",
            len(self.train_data),
            len(self.val_data),
        )

    def get_batch(
        self,
        split: str,
        device: str = "cpu",
        batch_size: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (input, target) token sequences.

        Randomly samples starting positions from the data array and extracts
        context_length-token sequences. The target sequence is the input
        shifted by one position (next-token prediction).

        Args:
            split: Either "train" or "val" to select the data source.
            device: The torch device to move tensors to (e.g., "cuda", "cpu").
            batch_size: Number of sequences per batch. If None, uses
                config.batch_size. In DDP setups, pass the per-process
                batch size (global_batch_size // world_size).
            context_length: Sequence length. If None, uses
                config.context_length.

        Returns:
            A tuple (x, y) where:
                - x: Input token tensor of shape (batch_size, context_length),
                  dtype=torch.int64.
                - y: Target token tensor of shape (batch_size, context_length),
                  dtype=torch.int64. y[i, t] = x[i, t+1] (next-token target).

        Raises:
            ValueError: If split is not "train" or "val".
            ValueError: If the data array is too short for the requested
                context_length.
        """
        if split not in ("train", "val"):
            raise ValueError(
                f"split must be 'train' or 'val', got '{split}'."
            )

        # Resolve batch size and context length
        effective_batch_size: int = (
            batch_size if batch_size is not None else self.config.batch_size
        )
        effective_context_length: int = (
            context_length
            if context_length is not None
            else self.config.context_length
        )

        # Select data source
        data: np.ndarray = (
            self.train_data if split == "train" else self.val_data
        )

        # Validate data length
        min_required: int = effective_context_length + 1
        if len(data) < min_required:
            raise ValueError(
                f"Data array for split='{split}' has {len(data)} tokens, "
                f"but context_length={effective_context_length} requires at "
                f"least {min_required} tokens."
            )

        # Sample random starting positions
        # Valid range: [0, len(data) - context_length - 1]
        max_start: int = len(data) - effective_context_length - 1
        ix: torch.Tensor = torch.randint(
            low=0,
            high=max_start,
            size=(effective_batch_size,),
        )

        # Extract sequences and convert uint16 → int64 for embedding lookup
        # PyTorch does not natively support uint16 tensors, so we cast via numpy
        x_list: List[torch.Tensor] = [
            torch.from_numpy(
                data[i : i + effective_context_length].astype(np.int64)
            )
            for i in ix.tolist()
        ]
        y_list: List[torch.Tensor] = [
            torch.from_numpy(
                data[i + 1 : i + 1 + effective_context_length].astype(np.int64)
            )
            for i in ix.tolist()
        ]

        x: torch.Tensor = torch.stack(x_list)  # (batch_size, context_length)
        y: torch.Tensor = torch.stack(y_list)  # (batch_size, context_length)

        # Move to device with non_blocking=True for GPU efficiency
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        return x, y

    def get_val_loader(
        self,
        steps: int,
        device: str = "cpu",
        batch_size: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield a fixed sequence of validation batches for reproducible evaluation.

        Uses a fixed random seed (seed=0) to ensure the same validation batches
        are used every time this method is called, regardless of the current
        training RNG state. The training RNG state is saved and restored after
        the generator is exhausted.

        Args:
            steps: Number of validation batches to yield. Corresponds to
                config.eval_steps (default 100 from config.yaml).
            device: The torch device to move tensors to.
            batch_size: Per-batch size. If None, uses config.batch_size.
            context_length: Sequence length. If None, uses config.context_length.

        Yields:
            Tuples of (x, y) validation batches, each of shape
            (batch_size, context_length) with dtype=torch.int64.
        """
        # Save current RNG state to restore after validation
        rng_state: torch.Tensor = torch.get_rng_state()
        numpy_rng_state = np.random.get_state()

        # Fix seed for reproducible validation batches
        torch.manual_seed(0)
        np.random.seed(0)

        try:
            for _ in range(steps):
                yield self.get_batch(
                    split="val",
                    device=device,
                    batch_size=batch_size,
                    context_length=context_length,
                )
        finally:
            # Always restore training RNG state, even if an exception occurs
            torch.set_rng_state(rng_state)
            np.random.set_state(numpy_rng_state)


# ---------------------------------------------------------------------------
# PG19Dataset
# ---------------------------------------------------------------------------

class PG19Dataset:
    """Dataset class for PG19 length extrapolation evaluation.

    Loads the PG19 (Project Gutenberg) dataset for evaluating perplexity at
    context lengths from 1K to 32K tokens, as described in Appendix A.8 of
    the nGPT paper (Figure 14).

    PG19 consists of full-length books, making it suitable for testing
    long-context extrapolation beyond the training context length.

    Attributes:
        config: The experiment configuration.
        tokenizer: The HuggingFace tokenizer (shared with OpenWebTextDataset).
        data: Memory-mapped numpy array of all PG19 test tokens (uint16).
        eval_lengths: List of context lengths to evaluate (from config).
    """

    # Default PG19 cache filename
    _PG19_BIN_FILENAME: str = "pg19_test.bin"

    def __init__(self, config: Config, tokenizer: object) -> None:
        """Initialize the PG19 dataset, loading from cache or downloading.

        Accepts the tokenizer instance from OpenWebTextDataset to avoid
        re-loading it. Uses the PG19 test split for evaluation.

        Args:
            config: The experiment configuration. Key fields used:
                - config.cache_dir: Directory for cached binary files.
                - config.dataset_name: Not used directly (PG19 is hardcoded).
            tokenizer: A HuggingFace tokenizer object, typically the same
                instance used by OpenWebTextDataset.
        """
        self.config: Config = config
        self.tokenizer: object = tokenizer

        # Evaluation lengths from config.yaml evaluation.length_extrapolation
        self.eval_lengths: List[int] = [1024, 2048, 4096, 8192, 16384, 32768]

        # Cache file path
        self._pg19_bin: str = os.path.join(config.cache_dir, self._PG19_BIN_FILENAME)

        # Download and tokenize if cache is missing
        if not os.path.isfile(self._pg19_bin):
            logger.info(
                "PG19 cache not found at '%s'. Downloading and tokenizing...",
                self._pg19_bin,
            )
            self._download_and_tokenize()
        else:
            logger.info(
                "Found PG19 cache at '%s'. Loading from cache.",
                self._pg19_bin,
            )

        # Load memmap
        self.data: np.ndarray = np.memmap(
            self._pg19_bin,
            dtype=np.uint16,
            mode="r",
        )

        logger.info(
            "PG19Dataset ready: %d tokens (%.2fB).",
            len(self.data),
            len(self.data) / 1e9,
        )

    def _download_and_tokenize(self) -> None:
        """Download PG19 test split, tokenize, and save as a binary memmap.

        Downloads PG19 from HuggingFace datasets, tokenizes all documents
        using the stored tokenizer (with EOS token appended per document),
        concatenates into a flat array, and saves as a uint16 numpy memmap.

        Raises:
            RuntimeError: If the dataset download fails or produces zero tokens.
        """
        import datasets as hf_datasets  # type: ignore

        _ensure_dir(self.config.cache_dir)

        eos_token_id: int = _get_eos_token_id(self.tokenizer)

        logger.info("Loading PG19 test split from HuggingFace...")
        try:
            raw_dataset = hf_datasets.load_dataset(
                "pg19",
                split="test",
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load PG19 dataset: {exc}. "
                "Ensure you have internet access and the 'datasets' library "
                "is installed. PG19 may require additional setup."
            ) from exc

        logger.info("PG19 test split loaded: %d documents.", len(raw_dataset))

        # Tokenize all documents
        tokenizer_ref = self.tokenizer
        eos_id_ref: int = eos_token_id

        def _tokenize_fn(examples: dict) -> dict:
            return _tokenize_batch(examples, tokenizer_ref, eos_id_ref)

        # PG19 uses "text" column
        text_column: str = "text"
        if text_column not in raw_dataset.column_names:
            # Some versions use different column names
            available_cols = raw_dataset.column_names
            logger.warning(
                "PG19 dataset does not have a 'text' column. "
                "Available columns: %s. Attempting to use first column.",
                available_cols,
            )
            text_column = available_cols[0]

        # Rename column to 'text' if needed for _tokenize_batch compatibility
        if text_column != "text":
            raw_dataset = raw_dataset.rename_column(text_column, "text")

        logger.info("Tokenizing PG19 documents...")
        tokenized = raw_dataset.map(
            _tokenize_fn,
            batched=True,
            batch_size=10,  # PG19 documents are very long; use small batches
            num_proc=min(2, os.cpu_count() or 1),
            remove_columns=raw_dataset.column_names,
            desc="Tokenizing PG19",
        )

        # Concatenate all token ID lists
        logger.info("Concatenating PG19 token sequences...")
        all_ids: List[int] = []
        for example in tokenized:
            ids = example.get("input_ids", [])
            if ids:
                all_ids.extend(ids)

        n_total: int = len(all_ids)
        if n_total == 0:
            raise RuntimeError(
                "PG19 tokenization produced zero tokens. "
                "Check the dataset and tokenizer configuration."
            )

        logger.info(
            "PG19 total tokens: %d (%.2fB).",
            n_total,
            n_total / 1e9,
        )

        # Validate uint16 range
        max_token_id: int = max(all_ids) if all_ids else 0
        if max_token_id > 65535:
            raise RuntimeError(
                f"Maximum PG19 token ID {max_token_id} exceeds uint16 range. "
                "Cannot save as uint16."
            )

        # Save as uint16 memmap
        logger.info("Saving PG19 tokens to '%s'...", self._pg19_bin)
        arr = np.memmap(
            self._pg19_bin,
            dtype=np.uint16,
            mode="w+",
            shape=(n_total,),
        )
        arr[:] = np.array(all_ids, dtype=np.uint16)
        arr.flush()
        del arr  # Close file handle

        logger.info("PG19 tokenization complete. Saved %d tokens.", n_total)

    def get_batch_at_length(
        self,
        length: int,
        device: str = "cpu",
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batches of a specific context length for perplexity evaluation.

        Used by Evaluator.evaluate_length_extrapolation() to test the model's
        ability to handle sequences longer than its training context length
        (Appendix A.8, Figure 14).

        Args:
            length: The context length to evaluate. Should be one of the
                values in self.eval_lengths (1024, 2048, 4096, 8192, 16384,
                32768). Values outside this range are accepted but may not
                correspond to paper-reported results.
            device: The torch device to move tensors to.
            n_samples: Number of sequences to sample. Defaults to 10 for
                a reasonable perplexity estimate without excessive compute.

        Returns:
            A tuple (x, y) where:
                - x: Input token tensor of shape (n_samples, length),
                  dtype=torch.int64.
                - y: Target token tensor of shape (n_samples, length),
                  dtype=torch.int64.

        Raises:
            ValueError: If the PG19 data array is too short for the requested
                length (should not happen for lengths up to 32K).
        """
        # Validate data length
        min_required: int = length + 1
        if len(self.data) < min_required:
            raise