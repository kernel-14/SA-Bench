import os
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from typing import Dict, List, Tuple

# Assuming Config is in a sibling file config.py
from config import Config


class DataModule:
    """
    Manages loading, processing, and batching of the OpenWebText dataset for
    causal language modeling. It uses the HuggingFace `transformers` library
    for tokenization and `datasets` for efficient data handling.
    """

    def __init__(self, config: Config):
        """
        Initializes the DataModule with the given configuration.

        Args:
            config: An instance of the Config dataclass containing all
                    hyperparameters and dataset settings.
        """
        self.config = config
        self.tokenizer = None
        self.dataset: DatasetDict = None
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None

        # LLaMA-style tokenizers often lack a dedicated padding token.
        # Setting eos_token as pad_token is a common practice for causal LM.
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.data_config.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if tokenizer's block_size (max_length) matches config's block_size
        # The tokenizer's model_max_length is often very large by default (e.g., 1e30)
        # We enforce block_size in group_texts
        if self.config.model_config.block_size > self.tokenizer.model_max_length and self.tokenizer.model_max_length < 1e10: # Check for reasonable model_max_length
            print(f"Warning: Config block_size ({self.config.model_config.block_size}) is larger than tokenizer's model_max_length ({self.tokenizer.model_max_length}). This might lead to unexpected behavior if not handled by truncation.")


    def prepare_data(self) -> None:
        """
        Downloads and loads the OpenWebText dataset.
        This method typically runs once, potentially only on the main process
        in a distributed setup to avoid redundant downloads.
        """
        if self.dataset is None:
            print(f"Loading dataset: {self.config.data_config.dataset_name}")
            self.dataset = load_dataset(self.config.data_config.dataset_name)
            print(f"Dataset loaded: {self.dataset}")

    def setup(self, stage: str = None) -> None:
        """
        Preprocesses the raw dataset by tokenizing and then concatenating
        and chunking sequences into fixed-size blocks suitable for causal
        language modeling. This method populates `self.train_dataset` and
        `self.val_dataset`.

        Args:
            stage: Optional; 'fit' or 'test'. Used to separate setup logic
                   for different stages. (Not explicitly used here, but common in frameworks).
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not prepared. Call prepare_data() first.")

        num_proc = min(os.cpu_count(), 8) # Use up to 8 CPU cores for multiprocessing

        def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
            """Tokenizes a batch of text examples."""
            return self.tokenizer(examples['text'], truncation=False, return_attention_mask=False)

        print("Tokenizing dataset...")
        tokenized_datasets = self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=['text'],
            load_from_cache_file=True, # Use cached files if available
            desc="Running tokenizer on dataset",
        )
        print("Dataset tokenized.")

        block_size = self.config.model_config.block_size

        def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
            """
            Concatenates all texts in a batch and then chunks them into
            fixed-size blocks.
            """
            # Concatenate all texts
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])

            # Drop the small remainder, if any, to ensure all samples are block_size
            total_length = (total_length // block_size) * block_size

            # Split by block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            # For causal language modeling, labels are usually the input shifted by one.
            # The model's loss function will typically handle this internally.
            result["labels"] = result["input_ids"].copy()
            return result

        print(f"Grouping texts into blocks of size {block_size}...")
        processed_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True, # Use cached files if available
            desc=f"Grouping texts in chunks of {block_size}",
        )
        print("Texts grouped.")

        self.train_dataset = processed_datasets['train']
        self.val_dataset = processed_datasets['validation'] # Assuming 'validation' split exists in OpenWebText

        print(f"Train dataset size: {len(self.train_dataset)} samples")
        print(f"Validation dataset size: {len(self.val_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training set, configured for distributed
        training if multiple GPUs are used.
        """
        # Calculate per-GPU batch size
        per_gpu_batch_size = self.config.training_config.global_batch_size // self.config.system_config.num_gpus

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=True,
                seed=self.config.system_config.seed, # Ensure consistent shuffling across processes
            )
        else:
            sampler = RandomSampler(self.train_dataset)

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=per_gpu_batch_size,
            sampler=sampler,
            drop_last=True,  # Crucial for DistributedSampler to ensure consistent batch sizes
            num_workers=min(os.cpu_count(), 4), # Reduce workers to avoid too many processes
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the validation set, configured similarly to
        the training loader but typically without shuffling.
        """
        # Calculate per-GPU batch size
        per_gpu_batch_size = self.config.training_config.global_batch_size // self.config.system_config.num_gpus

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=False,  # No shuffling for validation
            )
        else:
            sampler = SequentialSampler(self.val_dataset)

        dataloader = DataLoader(
            self.val_dataset,
            batch_size=per_gpu_batch_size,
            sampler=sampler,
            drop_last=True,  # Maintain consistent batch sizes for distributed evaluation
            num_workers=min(os.cpu_count(), 4), # Reduce workers to avoid too many processes
            pin_memory=True,
        )
        return dataloader

