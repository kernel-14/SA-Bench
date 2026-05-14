## dataset_loader.py

import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from typing import Tuple, Dict, List

class DatasetLoader:
    """
    DatasetLoader class responsible for loading, preprocessing, and splitting a dataset
    into train/validation/test subsets, and wrapping them in PyTorch DataLoader objects.

    Attributes:
        config (dict): Configuration dictionary loaded from `config.yaml`.
        dataset_name (str): Name of the dataset (e.g., 'licensed_dataset').
        split_ratio (List[float]): Ratios to split the dataset into train/validation/test.
        input_size (int): Size to resize images (default is 512x512).
        augmentation (bool): Flag for enabling data augmentation.
        batch_size (int): Number of samples per batch in a DataLoader.
        num_workers (int): Number of workers for parallel data loading.
        shuffle (bool): Flag for shuffling dataset during DataLoader construction.
    """

    def __init__(self, config: dict):
        """
        Initializes the DatasetLoader with configuration.

        Args:
            config (dict): Dictionary containing dataset and training settings.
        """
        self.config = config
        self.dataset_name: str = config['dataset'].get('dataset_name', "dataset_name_default")
        self.split_ratio: List[float] = config['dataset'].get('split_ratio', [0.8, 0.1, 0.1])
        self.input_size: int = config['dataset'].get('input_size', 512)
        self.augmentation: bool = config['dataset'].get('augmentation', True)
        self.batch_size: int = config['training'].get('batch_size', 40)
        self.num_workers: int = config['dataset'].get('num_workers', 4)
        self.shuffle: bool = config['dataset'].get('shuffle', True)

        self._validate_split_ratio()

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Main method to load, preprocess, and split the dataset and return DataLoader objects.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test DataLoaders.
        """
        # Step 1: Load the dataset
        dataset = self._load_dataset()

        # Step 2: Split the dataset into train, val, test sets
        train_set, val_set, test_set = self._split_dataset(dataset)

        # Step 3: Apply transformations to subsets
        transforms_dict = self._get_transforms(augmentation=self.augmentation)
        train_set.transform = transforms_dict['train']
        val_set.transform = transforms_dict['val']
        test_set.transform = transforms_dict['test']

        # Step 4: Wrap datasets in DataLoader instances
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader

    def _load_dataset(self) -> Dataset:
        """
        Loads the raw dataset based on the dataset name specified in the config.

        Returns:
            Dataset: PyTorch Dataset object loaded with paired text-image samples.
        """
        if self.dataset_name == "licensed_dataset":
            # Replace with logic to load a licensed dataset; Placeholder for directory paths
            dataset_path = "./data/licensed_dataset"
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"[ERROR] Dataset not found in path: {dataset_path}")
            
            # Placeholder: Simulated PyTorch dataset that includes text-image pairs
            dataset = torch.utils.data.DatasetFolder(
                root=dataset_path,
                loader=lambda x: x,
                extensions=(".jpg", ".png"),  # Example extensions for images
            )
            print(f"[INFO] Successfully loaded dataset: {self.dataset_name}.")
            return dataset
        else:
            raise ValueError(f"[ERROR] Unsupported dataset: {self.dataset_name}")
    
    def _split_dataset(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Splits the dataset into train, validation, and test subsets based on split ratios.

        Args:
            dataset (Dataset): Entire loaded dataset.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: Train, validation, and test dataset splits.
        """
        total_size = len(dataset)
        train_size = int(self.split_ratio[0] * total_size)
        val_size = int(self.split_ratio[1] * total_size)
        test_size = total_size - train_size - val_size

        if train_size + val_size + test_size != total_size:
            print("[WARNING] Split sizes do not sum to total dataset size; adjusting splits.")

        train_set, val_set, test_set = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.get('general', {}).get('seed', 42))
        )
        print(f"[INFO] Dataset split into Train: {train_size}, Validation: {val_size}, Test: {test_size}.")
        return train_set, val_set, test_set

    def _get_transforms(self, augmentation: bool) -> Dict[str, transforms.Compose]:
        """
        Defines the preprocessing and augmentation transformations.

        Args:
            augmentation (bool): Whether to apply augmentations for training.

        Returns:
            Dict[str, transforms.Compose]: Transformations for train, val, and test splits.
        """
        train_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.RandomHorizontalFlip() if augmentation else None,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        val_test_transforms = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        print(f"[INFO] Data augmentations {'enabled' if augmentation else 'disabled'}.")
        return {'train': train_transforms, 'val': val_test_transforms, 'test': val_test_transforms}

    def _validate_split_ratio(self) -> None:
        """
        Validates the provided split ratios and ensures they sum to 1.
        """
        total_ratio = sum(self.split_ratio)
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"[ERROR] Dataset split ratios do not sum to 1: {self.split_ratio}")
        print("[INFO] Split ratios validated successfully.")
