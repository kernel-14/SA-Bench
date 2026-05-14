# dataset_loader.py

import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
from typing import Dict, List, Tuple
import pandas as pd

class DatasetLoader:
    def __init__(self, config: dict):
        """Initialize the DatasetLoader with configuration settings from config.yaml.
        
        Args:
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.dataset_sources = config['dataset']['sources']  # List of dataset sources
        self.resolution = config['dataset']['resolution']  # Spatial resolution, e.g., 128x128
        self.channels = config['dataset']['channels']  # Number of physical fields (channels)
        self.split_ratios = config['dataset']['split']  # Data split ratios (train/val/test)
        self.precision = config['dataset']['precision']  # Data precision (e.g., float16)
        self.seed = config['logging']['seed']  # Random seed for reproducibility

        # Ensure deterministic behavior
        self._set_random_seed(self.seed)

    def load_dataset(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load, preprocess, and split the dataset into train, validation, and test sets.
        
        Returns:
            dict: A dictionary containing the splits (train, val, test) with `inputs` and `targets`.
        """
        datasets = []
        for source in self.dataset_sources:
            raw_data = self._load_raw_data(source)
            processed_data = self.preprocess(raw_data)
            datasets.append(processed_data)
        
        combined_data = self._combine_datasets(datasets)
        splits = self._split_data(combined_data)
        return splits

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        """Preprocess raw data to meet the required resolution, precision, and channel format.
        
        Args:
            data (np.ndarray): Raw data array.
        
        Returns:
            torch.Tensor: Preprocessed data as a tensor.
        """
        # Rescale spatial dimensions
        resized_data = resize(torch.tensor(data), [self.resolution, self.resolution])
        
        # Normalize data to the range [0, 1]
        normalized_data = resized_data.float() / resized_data.max()
        
        # Adjust channel dimensions
        if normalized_data.shape[0] < self.channels:
            # Pad with zeros to match the required number of channels
            padding = torch.zeros(
                (self.channels - normalized_data.shape[0], self.resolution, self.resolution),
                dtype=torch.float32
            )
            normalized_data = torch.cat((normalized_data, padding), dim=0)
        elif normalized_data.shape[0] > self.channels:
            # Truncate excess channels
            normalized_data = normalized_data[:self.channels]
        
        # Convert to target precision (float16)
        return normalized_data.half() if self.precision == 'float16' else normalized_data.float()

    def _split_data(self, data: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Partition the dataset into train, validation, and test splits.
        
        Args:
            data (torch.Tensor): Combined, preprocessed dataset tensor.
        
        Returns:
            dict: Dictionary containing train, val, and test splits.
        """
        # Shuffle data for randomness
        num_samples = data.shape[0]
        indices = list(range(num_samples))
        random.shuffle(indices)

        # Calculate split indices
        train_idx = int(self.split_ratios['train'] * num_samples)
        val_idx = train_idx + int(self.split_ratios['valid'] * num_samples)

        # Create splits for data
        train_data = data[indices[:train_idx]]
        val_data = data[indices[train_idx:val_idx]]
        test_data = data[indices[val_idx:]]

        return {
            'train': {'inputs': train_data, 'targets': train_data},
            'val': {'inputs': val_data, 'targets': val_data},
            'test': {'inputs': test_data, 'targets': test_data}
        }

    def _combine_datasets(self, datasets: List[torch.Tensor]) -> torch.Tensor:
        """Combine multiple dataset tensors into a single tensor for processing.
        
        Args:
            datasets (List[torch.Tensor]): List of individual dataset tensors.
        
        Returns:
            torch.Tensor: Unified dataset tensor.
        """
        return torch.cat(datasets, dim=0)

    def _load_raw_data(self, source: str) -> np.ndarray:
        """Load raw data from a given dataset source.
        
        Args:
            source (str): Dataset source name (e.g., FNO-v or PDEBench).
        
        Returns:
            np.ndarray: Raw dataset as a NumPy array.
        """
        # Example path logic to load data (adjust according to actual format)
        data_path = f"./datasets/{source}.npz"  # Assuming data is stored in .npz format
        try:
            data = np.load(data_path)['data']
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        return data

    def _set_random_seed(self, seed: int) -> None:
        """Set the random seed for consistent behavior across splits.
        
        Args:
            seed (int): Random seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

