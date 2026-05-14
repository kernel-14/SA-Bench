## dataset_loader.py
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple, Dict
import logging

class DatasetLoader:
    """
    Handles loading, preprocessing, and batching of the OpenWebText dataset.
    Includes tokenization, context length enforcement, and PyTorch tensor batching.
    """

    def __init__(self, config: Dict):
        """
        Initializes the DatasetLoader with configuration parameters.

        Args:
            config (Dict): Configuration loaded from config.yaml.
        """
        self.config = config
        dataset_config = config["dataset"]
        self.tokenizer_name = dataset_config["tokenizer"]
        self.vocab_size = dataset_config["vocabulary_size"]
        self.context_lengths = dataset_config.get("max_tokens", [1000, 4000, 8000])
        self.batch_size = config["training"]["batch_size"]
        self.dtype = torch.bfloat16 if config["hardware"]["dtype"] == "bfloat16" else torch.float

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            model_max_length=max(self.context_lengths),
            use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("DatasetLoader initialized.")

    def load_data(self) -> Dataset:
        """
        Load the OpenWebText dataset using the Hugging Face datasets library.

        Returns:
            Dataset: Hugging Face dataset object with train, validation, and test splits.
        """
        try:
            dataset = load_dataset("openwebtext", cache_dir="./cache")
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenWebText dataset: {e}")
        
        if not {"train", "validation", "test"}.issubset(dataset):
            raise ValueError("OpenWebText dataset must contain 'train', 'validation', and 'test' splits.")

        self.logger.info("Successfully loaded OpenWebText dataset.")
        return dataset

    def process_data(self, data: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes, processes, and batches the data into PyTorch tensors.

        Args:
            data (Dataset): The Hugging Face dataset split (train/validation/test).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tokenized input and target sequences as tensors.
        """
        self.logger.info("Starting data preprocessing.")

        def tokenize_function(example):
            """
            Tokenize input text and enforce context length constraints.

            Args:
                example (dict): Single example from the OpenWebText dataset.

            Returns:
                dict: Tokenized input and target sequences.
            """
            tokenized_output = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=max(self.context_lengths),
                padding="max_length",
                return_tensors="pt"
            )
            # Input sequence: full tokenized text.
            # Target sequence: shifted by one token for autoregressive modeling.
            input_ids = tokenized_output["input_ids"].squeeze(0)
            target_ids = torch.clone(input_ids)
            target_ids[:-1] = input_ids[1:]  # Shift sequence left
            target_ids[-1] = self.tokenizer.pad_token_id  # Pad last token

            return {"input_ids": input_ids, "target_ids": target_ids}

        # Process dataset: tokenize and create input-target pairs
        processed = data.map(
            tokenize_function,
            batched=False,  # Process each entry individually
            remove_columns=["text"],  # Remove raw text after tokenization
            num_proc=4,  # Use multiprocessing to speed up tokenization
        )

        self.logger.info("Dataset tokenization complete.")

        # Convert dataset into PyTorch format
        dataset_tensors = self._convert_to_tensors(processed)

        return dataset_tensors

    def _convert_to_tensors(self, dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a tokenized Dataset object to PyTorch tensors.

        Args:
            dataset (Dataset): Pre-tokenized dataset with 'input_ids' and 'target_ids'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Batches of input and target tensors.
        """
        # Extract input-target pairs
        inputs = torch.stack([torch.tensor(example["input_ids"], dtype=self.dtype)
                              for example in dataset])
        targets = torch.stack([torch.tensor(example["target_ids"], dtype=self.dtype)
                               for example in dataset])

        self.logger.info(f"Converted dataset into tensors with shapes: inputs={inputs.shape}, targets={targets.shape}")
        return inputs, targets

    def create_dataloader(self, inputs: torch.Tensor, targets: torch.Tensor) -> DataLoader:
        """
        Creates a PyTorch DataLoader from tokenized inputs and targets.

        Args:
            inputs (torch.Tensor): Input sequences of shape (num_samples, context_length).
            targets (torch.Tensor): Target sequences of shape (num_samples, context_length).

        Returns:
            DataLoader: A PyTorch DataLoader for iterating over batches.
        """
        # Create a PyTorch Dataset from tensors
        class TokenDataset(Dataset):
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                return self.inputs[idx], self.targets[idx]

        dataset = TokenDataset(inputs, targets)

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle dataset for training
            num_workers=4,  # Use multiprocessing for data loading
            pin_memory=True
        )

        self.logger.info(f"Dataloader created with batch size {self.batch_size}.")
        return dataloader
