"""
dataset_loader.py

Data loading and preprocessing for the NGPT reproduction project.

Provides the ``DatasetLoader`` class that:
- Downloads and tokenizes the OpenWebText corpus with the LLaMA‑2 tokenizer.
- Chunks the data into fixed‑length blocks appropriate for autoregressive language modelling.
- Splits the blocks into training (99%) and validation (1%) sets.
- Caches the processed tensors to disk for fast reloading.
- Creates a PyTorch ``DataLoader`` compatible with single‑GPU or distributed (DDP) training.
- Exposes a ``get_batch()`` method that continuously yields training/validation batches.

All settings are taken from the ``Config`` object (``config.yaml``).
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset
from config import Config
from typing import Tuple, Optional


class DatasetLoader:
    """
    Data loader for OpenWebText with LLaMA‑2 tokenizer.

    Instantiate one loader per split (``'train'`` or ``'val'``).  If the required
    pre‑processed tensor cache does not exist, the first instantiation will download,
    tokenize, chunk, and split the entire dataset.  Subsequent instantiations (for
    the other split or for later runs) will simply load the existing cache.

    Parameters
    ----------
    config : Config
        Global configuration object containing data paths, model dimensions, and
        training batch size.
    split : str
        One of ``'train'`` or ``'val'``.
    """

    def __init__(self, config: Config, split: str):
        """
        Initialise the DatasetLoader.

        Args:
            config: The global configuration.
            split: ``'train'`` or ``'val'``.
        """
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        self.config = config
        self.split = split
        self.max_seq_len = config.model.max_seq_len
        self.batch_size = config.training.batch_size
        self.dataset_path = config.data.dataset_path
        self.tokenizer_name = config.data.tokenizer_name
        self.val_ratio = config.data.val_ratio

        # Load the tokenizer
        # The LLaMA‑2 tokenizer typically requires authentication; ensure that
        # the user has run `huggingface-cli login` or set use_auth_token appropriately.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=False,
        )
        # Set the padding token to the EOS token (useful only if any padding occurs,
        # which it shouldn't in our chunked‑data pipeline).
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Prepare the data (cache if necessary) and create the DataLoader.
        self._prepare_data()
        self.dataset, self.dataloader = self._create_dataloader()

        # Iterator for the get_batch() convenience method.
        self._data_iter = iter(self.dataloader)

    # ------------------------------------------------------------------
    # Public methods matching the design
    # ------------------------------------------------------------------

    def load_data(self) -> Tuple[TensorDataset, DataLoader]:
        """
        Return the underlying PyTorch dataset and DataLoader.

        This method is provided for compatibility with the design interface.
        It returns the dataset and the loader that were created during
        initialisation.

        Returns
        -------
        dataset : TensorDataset
            The dataset containing (input, target) token blocks.
        dataloader : DataLoader
            A DataLoader that iterates over the dataset with the appropriate
            batch size, shuffling, and distributed settings.
        """
        return self.dataset, self.dataloader

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch the next batch of (inputs, targets) from the DataLoader.

        This method uses an internal iterator that is automatically reset
        when the DataLoader is exhausted.  It is designed for training loops
        that prefer explicit batch retrieval over Python's native ``for``
        loop.

        Returns
        -------
        inputs : torch.Tensor
            Input token indices of shape ``(batch_size, max_seq_len)``.
        targets : torch.Tensor
            Target token indices (shifted by one position) of the same shape.
        """
        try:
            x, y = next(self._data_iter)
        except StopIteration:
            # Re‑create the iterator to loop over the DataLoader again.
            self._data_iter = iter(self.dataloader)
            x, y = next(self._data_iter)
        return x, y

    # ------------------------------------------------------------------
    # Private methods for data processing and caching
    # ------------------------------------------------------------------

    def _cache_path(self, split: str) -> str:
        """
        Return the file path for cached tensors of the given split.

        The cache file is named ``openwebtext_<split>_<max_seq_len>.pt`` and
        stored under ``self.dataset_path``.

        Args:
            split: ``'train'`` or ``'val'``.

        Returns:
            Absolute file path.
        """
        os.makedirs(self.dataset_path, exist_ok=True)
        return os.path.join(
            self.dataset_path, f"openwebtext_{split}_{self.max_seq_len}.pt"
        )

    def _prepare_data(self) -> None:
        """
        Ensure that the cached tensors for the requested split exist.

        If the tensor file for the requested split is not found, this method
        triggers the full data processing pipeline (download + tokenize +
        chunk + split) and saves the results for both splits.  This design
        guarantees that both train and validation caches are created at once,
        avoiding redundant processing when both loaders are instantiated.
        """
        cache_file = self._cache_path(self.split)
        if not os.path.isfile(cache_file):
            self._process_full_dataset()

    def _process_full_dataset(self) -> None:
        """
        Download OpenWebText, tokenize, chunk, split, and cache.

        This is the most expensive operation; it is only executed once when
        no cached data is available.  The steps are:

        1. Download OpenWebText from the Hugging Face hub.
        2. Tokenize every document, appending an EOS token after each.
        3. Concatenate all token sequences into one flat array.
        4. Chunk into blocks of ``max_seq_len + 1`` tokens and form
           ``(input, target)`` pairs.
        5. Shuffle the blocks with a fixed seed and split into training and
           validation sets according to ``val_ratio``.
        6. Save the resulting tensors (as ``torch.LongTensor``) to disk.
        """
        print("Downloading and preprocessing OpenWebText (this may take a while)...")
        os.makedirs(self.dataset_path, exist_ok=True)

        # ------------------------------------------------------------------
        # 1. Load dataset
        # ------------------------------------------------------------------
        dataset = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)

        # ------------------------------------------------------------------
        # 2. Tokenization function (applied per document)
        # ------------------------------------------------------------------
        def tokenize_function(example: dict) -> dict:
            # `example["text"]` is a string.  We tokenize without truncation
            # and append the EOS token to separate documents.
            token_ids = self.tokenizer(
                example["text"],
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            token_ids.append(self.tokenizer.eos_token_id)
            return {"input_ids": token_ids}

        # Apply tokenization.  Using batched=True can speed things up, but
        # we must be careful not to exceed memory.  Here we use batched=False
        # for simplicity; a production implementation could add batching.
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=["text"],
            batched=False,
            desc="Tokenizing documents",
        )

        # ------------------------------------------------------------------
        # 3. Concatenate all token sequences into one flat array
        # ------------------------------------------------------------------
        all_token_lists = tokenized_dataset["input_ids"]
        # Convert each list to a numpy array and concatenate.
        all_ids_np = np.concatenate([np.asarray(seq, dtype=np.int64) for seq in all_token_lists])

        # ------------------------------------------------------------------
        # 4. Chunk into blocks of (max_seq_len + 1)
        # ------------------------------------------------------------------
        total_length = all_ids_np.shape[0]
        block_size = self.max_seq_len + 1
        n_blocks = total_length // block_size

        # Discard any trailing tokens that do not form a full block.
        usable_length = n_blocks * block_size
        all_ids_np = all_ids_np[:usable_length]

        # Reshape and extract inputs and targets.
        chunks = all_ids_np.reshape(n_blocks, block_size)
        x = chunks[:, :-1].copy()          # inputs
        y = chunks[:, 1:].copy()           # targets

        # ------------------------------------------------------------------
        # 5. Shuffle and split
        # ------------------------------------------------------------------
        # Use a fixed generator for reproducibility of the train/val split.
        generator = torch.Generator()
        generator.manual_seed(42)
        perm = torch.randperm(n_blocks, generator=generator).numpy()

        val_count = max(1, int(n_blocks * self.val_ratio))
        train_count = n_blocks - val_count

        train_indices = perm[:train_count]
        val_indices = perm[train_count:train_count + val_count]

        # Convert slices to tensors.
        train_x = torch.from_numpy(x[train_indices]).long()
        train_y = torch.from_numpy(y[train_indices]).long()
        val_x = torch.from_numpy(x[val_indices]).long()
        val_y = torch.from_numpy(y[val_indices]).long()

        # ------------------------------------------------------------------
        # 6. Save cached tensors
        # ------------------------------------------------------------------
        def save_split(prefix: str, inputs: torch.Tensor, targets: torch.Tensor):
            path = self._cache_path(prefix)
            torch.save(
                {"inputs": inputs, "targets": targets},
                path,
            )

        save_split("train", train_x, train_y)
        save_split("val", val_x, val_y)

        print(
            f"Preprocessing complete: saved {train_count} training blocks "
            f"and {val_count} validation blocks (block size = {self.max_seq_len})."
        )

    def _load_split(self, split: str) -> TensorDataset:
        """
        Load the cached tensors for a given split and return a ``TensorDataset``.

        Args:
            split: ``'train'`` or ``'val'``.

        Returns:
            A ``TensorDataset`` containing ``(inputs, targets)``.
        """
        cache_file = self._cache_path(split)
        if not os.path.isfile(cache_file):
            raise FileNotFoundError(
                f"Cached split '{split}' not found at {cache_file}. "
                "Please run the preprocessing step first."
            )

        data = torch.load(cache_file)
        return TensorDataset(data["inputs"], data["targets"])

    def _create_dataloader(self) -> Tuple[TensorDataset, DataLoader]:
        """
        Load the dataset for the current split and build a DataLoader.

        The DataLoader is configured to respect PyTorch's distributed
        data parallel (DDP) context: if DDP is initialised, a
        ``DistributedSampler`` is used to ensure each GPU processes
        a unique subset of data.  Otherwise, simple shuffling is used.

        Returns
        -------
        dataset : TensorDataset
            The loaded dataset.
        dataloader : DataLoader
            The configured DataLoader.
        """
        dataset = self._load_split(self.split)

        # Determine batch size per process.
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            batch_size_per_gpu = self.batch_size // world_size
        else:
            batch_size_per_gpu = self.batch_size

        # Create a distributed sampler if necessary.
        sampler: Optional[DistributedSampler] = None
        shuffle: bool = (self.split == "train")

        if torch.distributed.is_initialized():
            sampler = DistributedSampler(
                dataset,
                shuffle=shuffle,
                seed=42,
                drop_last=(self.split == "train"),
            )
            shuffle = False  # sampler handles shuffling

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_gpu,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=(self.split == "train"),
        )

        return dataset, dataloader

