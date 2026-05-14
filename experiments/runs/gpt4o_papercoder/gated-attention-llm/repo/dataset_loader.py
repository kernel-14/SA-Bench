# dataset_loader.py

import os
from typing import Dict, Any
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

class DatasetLoader:
    """
    Handles loading datasets for training and evaluation. 
    Provides methods for tokenizing and batching data based on the configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DatasetLoader with configuration parameters.

        Args:
            config (dict): Configuration dictionary from config.yaml.
        """
        self.config = config
        self.training_path = self.config["dataset"]["path_training"]
        self.evaluation_path = self.config["dataset"]["path_evaluation"]
        self.context_length = self.config["dataset"].get("context_length", 4096)
        self.extended_context_length = self.config["dataset"].get("extended_context_length", 32000)
        self.vocab_size = self.config["dataset"].get("vocabulary_size", 50000)
        self.batch_size = self.config["training"].get("batch_size", 1024)
        self.tokenizer = self._initialize_tokenizer(self.config["model"]["type"])
    
    def _initialize_tokenizer(self, model_type: str) -> AutoTokenizer:
        """
        Initializes the tokenizer based on the specified model type.

        Args:
            model_type (str): Model type defined in the config.

        Returns:
            AutoTokenizer: A tokenizer instance.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_type)
            tokenizer.model_max_length = self.context_length
            tokenizer.vocab_size = self.vocab_size
            return tokenizer
        except Exception as e:
            raise ValueError(f"Failed to initialize tokenizer for model type '{model_type}': {str(e)}")

    def load_training_data(self) -> DataLoader:
        """
        Loads and tokenizes the training data.

        Returns:
            DataLoader: PyTorch DataLoader with tokenized training data.
        """
        if not os.path.exists(self.training_path):
            raise FileNotFoundError(f"Training dataset path does not exist: {self.training_path}")
        
        # Load raw dataset
        print(f"Loading training data from {self.training_path}...")
        raw_dataset = load_dataset(self._guess_dataset_format(self.training_path), data_files=self.training_path)
        
        # Tokenization
        print("Tokenizing training data...")
        tokenized_dataset = raw_dataset.map(
            lambda examples: self.tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=self.context_length
            ),
            batched=True,
            remove_columns=raw_dataset["train"].column_names
        )

        # Convert to PyTorch DataLoader
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        train_dataloader = DataLoader(
            tokenized_dataset["train"], 
            batch_size=self.batch_size, 
            shuffle=True
        )
        return train_dataloader

    def load_evaluation_data(self) -> Dict[str, DataLoader]:
        """
        Loads and tokenizes evaluation data for multiple benchmarks.

        Returns:
            dict: A dictionary of benchmark name to DataLoader.
        """
        if not os.path.exists(self.evaluation_path):
            raise FileNotFoundError(f"Evaluation dataset path does not exist: {self.evaluation_path}")

        # Load all benchmarks
        print(f"Loading evaluation datasets from {self.evaluation_path}...")
        benchmark_files = {
            benchmark["name"]: os.path.join(self.evaluation_path, benchmark["name"])
            for benchmark in self.config["evaluation"]["benchmarks"]
        }

        benchmark_loaders = {}
        for benchmark_name, benchmark_path in benchmark_files.items():
            if not os.path.exists(benchmark_path):
                raise FileNotFoundError(f"Benchmark dataset path does not exist: {benchmark_path}")

            print(f"Processing benchmark: {benchmark_name}")
            raw_dataset = load_dataset(self._guess_dataset_format(benchmark_path), data_files=benchmark_path)
            
            # Tokenization
            print(f"Tokenizing evaluation dataset: {benchmark_name}")
            tokenized_dataset = raw_dataset.map(
                lambda examples: self.tokenizer(
                    examples["text"], 
                    truncation=True, 
                    padding="max_length", 
                    max_length=self.extended_context_length
                ),
                batched=True,
                remove_columns=raw_dataset["validation"].column_names
            )

            # Convert to PyTorch DataLoader
            tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
            eval_dataloader = DataLoader(
                tokenized_dataset["validation"], 
                batch_size=self.batch_size, 
                shuffle=False
            )
            benchmark_loaders[benchmark_name] = eval_dataloader

        return benchmark_loaders

    def _guess_dataset_format(self, path: str) -> str:
        """
        Guesses the data format based on file extension.

        Args:
            path (str): Path to the dataset file.

        Returns:
            str: Dataset format compatible with `load_dataset`.
        """
        extension = os.path.splitext(path)[-1].lower()
        if extension == ".json":
            return "json"
        elif extension == ".csv":
            return "csv"
        elif extension == ".txt":
            return "text"
        else:
            raise ValueError(f"Unsupported dataset format: {extension}")
