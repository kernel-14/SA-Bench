"""
dataset_loader.py
Handles dataset loading, preprocessing, and preparation for PyTorch training, validation, and testing.
This implementation adheres to the design and logic specifications, integrating with the config.yaml file.
"""

import os
import json
from typing import List, Tuple, Dict
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class DatasetLoader:
    """
    DatasetLoader: Handles dataset preparation for MetaMathQA, COMMONSENSE170K, and GLUE benchmarks.
    Provides PyTorch-compatible DataLoader objects for train, validation, and test splits.
    """
    def __init__(self, dataset_name: str, config: dict) -> None:
        """
        Initialize DatasetLoader with dataset-specific parameters and configuration.

        Args:
            dataset_name (str): Name of the dataset to be loaded (e.g., "MetaMathQA", "COMMONSENSE170K", "GLUE").
            config (dict): Configuration dictionary loaded from `config.yaml`.
        """
        self.dataset_name = dataset_name
        self.config = config

        # Extract dataset file paths and settings from config
        self.train_path = config.get('dataset', {}).get('train_path', '')
        self.test_path = config.get('dataset', {}).get('test_path', '')
        self.fraction_for_init = config.get('dataset', {}).get('fraction_for_init', 0.001)
        self.num_init_samples = config.get('dataset', {}).get('num_init_samples', 50)
        self.batch_size = config.get('training', {}).get('batch_size', 1)
        self.max_seq_len = config.get('model', {}).get('max_seq_len', 512)
        self.tokenizer_name = config.get('model', {}).get('base_name', 'roberta-large')
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Ensure paths are valid
        if not os.path.exists(self.train_path):
            raise ValueError(f"Train path does not exist: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise ValueError(f"Test path does not exist: {self.test_path}")

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and preprocess data, returning DataLoaders for train, validation, and test sets.
        
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
        """
        # Step 1: Load raw data
        train_data = self._load_json(self.train_path)
        test_data = self._load_json(self.test_path)

        # Step 2: Split training data into train/validation/init subsets
        train_data, val_data = self._split_data(train_data, (0.8, 0.2))
        init_data = train_data[:self.num_init_samples]  # Extract initialization subset

        # Step 3: Tokenize datasets
        train_tokenized = [self._tokenize(sample) for sample in train_data]
        val_tokenized = [self._tokenize(sample) for sample in val_data]
        test_tokenized = [self._tokenize(sample) for sample in test_data]

        # Step 4: Prepare DataLoader objects
        train_loader = self._prepare_dataloader(train_tokenized, batch_size=self.batch_size, shuffle=True)
        val_loader = self._prepare_dataloader(val_tokenized, batch_size=self.batch_size, shuffle=False)
        test_loader = self._prepare_dataloader(test_tokenized, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _prepare_dataloader(self, data: List[Dict], batch_size: int, shuffle: bool = False) -> DataLoader:
        """
        Convert tokenized data into a PyTorch DataLoader.

        Args:
            data (List[Dict]): Tokenized data samples.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: PyTorch DataLoader object.
        """
        dataset = KeyValueDataset(data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _tokenize(self, sample: Dict) -> Dict:
        """
        Tokenize a sample using the HuggingFace tokenizer.

        Args:
            sample (Dict): Raw data sample from the dataset.

        Returns:
            Dict: Tokenized sample with input_ids, attention_mask, and labels (if applicable).
        """
        if self.dataset_name == "MetaMathQA":
            # Tokenize question and answer
            inputs = self.tokenizer(
                text=sample['question'],
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels = float(sample['answer']) if 'answer' in sample else None

        elif self.dataset_name == "COMMONSENSE170K":
            # Tokenize prompt and options for multiple-choice format
            prompt = sample['prompt']
            options = sample['options']
            tokenized_options = [
                self.tokenizer(
                    text=f"{prompt} {option}",
                    max_length=self.max_seq_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ) for option in options
            ]
            inputs = {
                'input_ids': torch.cat([t['input_ids'] for t in tokenized_options], dim=0),
                'attention_mask': torch.cat([t['attention_mask'] for t in tokenized_options], dim=0)
            }
            labels = sample.get('label', -1)

        elif self.dataset_name == "GLUE":
            # Tokenize sentences for GLUE tasks
            sentence1 = sample.get('sentence1', '')
            sentence2 = sample.get('sentence2', None)
            inputs = self.tokenizer(
                text=sentence1,
                text_pair=sentence2,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels = sample.get('label', -1)

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        tokenized = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels) if labels is not None else None
        }
        return tokenized

    def _load_json(self, dataset_path: str) -> List[Dict]:
        """
        Load a JSON file and return the parsed data.

        Args:
            dataset_path (str): Path to the JSON file.

        Returns:
            List[Dict]: List of data samples loaded from the JSON file.
        """
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        return data

    def _split_data(self, data: List[Dict], split_ratios: Tuple[float, float]) -> Tuple[List[Dict], List[Dict]]:
        """
        Split data into subsets based on given ratios.

        Args:
            data (List[Dict]): List of data samples.
            split_ratios (Tuple[float, float]): Split ratios for train and validation data.

        Returns:
            Tuple[List[Dict], List[Dict]]: Train and validation data subsets.
        """
        train_ratio, val_ratio = split_ratios
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        return train_data, val_data


class KeyValueDataset(Dataset):
    """
    Simple PyTorch Dataset wrapper for tokenized data.
    """
    def __init__(self, data: List[Dict]) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
