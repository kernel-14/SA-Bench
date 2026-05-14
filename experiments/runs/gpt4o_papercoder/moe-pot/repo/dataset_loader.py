# dataset_loader.py

import os
from typing import Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from utilities import load_yaml_config, inject_gaussian_noise, compute_balanced_sampling_weights
import numpy as np

class CustomDataset(Dataset):
    """
    Helper dataset to wrap tensor data for PyTorch compatibility.
    """
    def __init__(self, data: torch.Tensor, targets: torch.Tensor = None):
        """
        Initialize the dataset.

        Args:
            data (torch.Tensor): Input data tensor.
            targets (torch.Tensor, optional): Target tensor corresponding to the data.
        """
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        if self.targets is not None:
            return self.data[index], self.targets[index]
        return self.data[index]


class DatasetLoader:
    def __init__(self, config: Dict):
        """
        Initialize the dataset loader with a configuration dictionary.

        Args:
            config (Dict): Configuration dictionary derived from config.yaml.
        """
        self.config = config
        self.spatial_resolution = config['preprocessing']['resolution']
        self.noise_amplitude = config['preprocessing']['noise_amplitude']
        self.normalization = config['preprocessing']['normalization']
        self.masking = config['preprocessing']['masking']
        self.balanced_sampling = config['preprocessing']['balanced_sampling']
        self.interpolate = config['preprocessing']['interpolation']
        self.datasets_path = config.get('datasets_path', './datasets')  # Default dataset path is './datasets'

    def load_pretraining_data(self) -> Tuple[Dataset, Dataset]:
        """
        Load and preprocess datasets for pretraining. Includes Gaussian noise injection.

        Returns:
            Tuple[Dataset, Dataset]: Training and validation datasets.
        """
        # Step 1: Load the pretraining dataset names from config
        dataset_names = self.config['datasets']['pretraining']
        all_data, all_targets = [], []

        for dataset_name in dataset_names:
            dataset_path = os.path.join(self.datasets_path, dataset_name)
            data, targets = self.__load_dataset_from_path(dataset_path)
            
            # Preprocess dataset
            data = self.__interpolate_resolution(data, self.spatial_resolution)
            data = self.__apply_masking(data) if self.masking else data
            data = self.__normalize_data(data) if self.normalization else data

            # Inject noise for pretraining only
            data = inject_gaussian_noise(data, amplitude=self.noise_amplitude)

            all_data.append(data)
            all_targets.append(targets)

        # Concatenate all pretraining datasets
        concatenated_data = torch.cat(all_data, dim=0)
        concatenated_targets = torch.cat(all_targets, dim=0) if all_targets[0] is not None else None

        # Balanced sampling weights
        if self.balanced_sampling:
            dataset_sizes = {name: len(data) for name, data in zip(dataset_names, all_data)}
            print("Dataset sizes", dataset_sizes)
            sampling_weights = compute_balanced_sampling_weights(dataset_sizes)
            concatenated_data = self.__apply_balanced_sampling(concatenated_data, sampling_weights)
            
        # Train-validation split
        train_size = int(0.8 * len(concatenated_data))
        val_size = len(concatenated_data) - train_size

        train_data, val_data = torch.utils.data.random_split(concatenated_data, [train_size, val_size])
        
        return CustomDataset(train_data, concatenated_targets), CustomDataset(val_data, concatenated_targets)

    def load_finetune_data(self) -> Tuple[Dataset, Dataset]:
        """
        Load and preprocess datasets for fine-tuning.

        Returns:
            Tuple[Dataset, Dataset]: Training and validation datasets for fine-tuning.
        """
        # Step 1: Load the fine-tuning dataset names from config
        subsets = self.config['datasets']['finetuning']['subsets']
        all_data, all_targets = [], []

        for subset_name in subsets:
            dataset_path = os.path.join(self.datasets_path, subset_name)
            data, targets = self.__load_dataset_from_path(dataset_path)

            # Preprocess dataset
            data = self.__interpolate_resolution(data, self.spatial_resolution)
            data = self.__apply_masking(data) if self.masking else data
            data = self.__normalize_data(data) if self.normalization else data

            all_data.append(data)
            all_targets.append(targets)

        # Concatenate all fine-tuning datasets
        concatenated_data = torch.cat(all_data, dim=0)
        concatenated_targets = torch.cat(all_targets, dim=0) if all_targets[0] is not None else None

        # Train-validation split
        train_size = int(0.8 * len(concatenated_data))
        val_size = len(concatenated_data) - train_size

        train_data, val_data = torch.utils.data.random_split(concatenated_data, [train_size, val_size])
        
        return CustomDataset(train_data, concatenated_targets), CustomDataset(val_data, concatenated_targets)

    def load_downstream_data(self) -> Tuple[Dataset, Dataset]:
        """
        Load and preprocess datasets for downstream tasks.

        Returns:
            Tuple[Dataset, Dataset]: Training and evaluation datasets for downstream tasks.
        """
        # Step 1: Load the downstream dataset names from config
        downstream_names = self.config['datasets']['downstream']
        all_data, all_targets = [], []

        for dataset_name in downstream_names:
            dataset_path = os.path.join(self.datasets_path, dataset_name)
            data, targets = self.__load_dataset_from_path(dataset_path)

            # Preprocess dataset
            data = self.__interpolate_resolution(data, self.spatial_resolution)
            data = self.__apply_masking(data) if self.masking else data
            data = self.__normalize_data(data) if self.normalization else data

            all_data.append(data)
            all_targets.append(targets)

        # Concatenate all downstream datasets
        concatenated_data = torch.cat(all_data, dim=0)
        concatenated_targets = torch.cat(all_targets, dim=0) if all_targets[0] is not None else None

        # Train-eval split
        train_size = int(0.8 * len(concatenated_data))
        eval_size = len(concatenated_data) - train_size

        train_data, eval_data = torch.utils.data.random_split(concatenated_data, [train_size, eval_size])
        
        return CustomDataset(train_data, concatenated_targets), CustomDataset(eval_data, concatenated_targets)

    def __load_dataset_from_path(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a dataset from a specified path.

        Args:
            path (str): Path to the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Data and target tensors.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at: {path}")

        data = torch.Tensor(np.load(os.path.join(path, "data.npy")))
        targets = torch.Tensor(np.load(os.path.join(path, "targets.npy"))) if os.path.exists(
            os.path.join(path, "targets.npy")) else None
        return data, targets

