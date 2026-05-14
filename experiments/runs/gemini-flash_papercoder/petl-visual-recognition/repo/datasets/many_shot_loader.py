## datasets/many_shot_loader.py

import os
import abc
import math
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets import CIFAR100, ImageFolder
from typing import Dict, Any, Optional

# Assuming ImageTransforms is located in datasets/transforms.py
from datasets.transforms import ImageTransforms
# Assuming BaseDatasetLoader is defined in datasets/vtab_loader.py
# To avoid circular dependency, we might define BaseDatasetLoader here or in a common base file.
# For now, let's assume it's directly accessible, or we'll copy its definition if truly necessary.
# Based on the file list and logic analysis, BaseDatasetLoader should be available from vtab_loader.py for now.
from datasets.vtab_loader import BaseDatasetLoader # Re-importing to ensure availability in current context


class TransformedSubset(Dataset):
    """
    A wrapper for `torch.utils.data.Subset` that allows applying a transform
    to each item retrieved from the subset.
    This is necessary because `Subset` does not directly support a `transform` attribute.
    """
    def __init__(self, subset: Subset, transform: Optional[Any] = None) -> None:
        """
        Args:
            subset (Subset): The original Subset instance.
            transform (Optional[Any]): The transform to apply to the data.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        """
        Retrieves an item from the subset and applies the transform if available.

        Args:
            index (int): The index of the item within the subset.

        Returns:
            Any: The transformed data item.
        """
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        """
        Returns the number of items in the subset.
        """
        return len(self.subset)


class ManyShotLoader(BaseDatasetLoader):
    """
    Loads and manages data for many-shot datasets (e.g., CIFAR-100, RESISC, Clevr-Distance)
    as specified in the paper. Handles 90/10 train/validation split and provides
    loaders for training, validation, and original test set evaluation.
    Applies dataset-specific data augmentations for training.
    """

    def __init__(self, config: Dict[str, Any], dataset_name: str) -> None:
        """
        Initializes the ManyShotLoader.

        Args:
            config (Dict[str, Any]): The full configuration dictionary from config.yaml.
            dataset_name (str): The name of the specific many-shot dataset
                                (e.g., 'cifar100', 'resisc45', 'clevr_distance').
        """
        self.config: Dict[str, Any] = config
        self.dataset_name: str = dataset_name
        self.split_seed: int = self.config['seed'] # Use global seed for deterministic splits

        # Retrieve base path for the dataset from the configuration
        self.base_path: str = self.config['datasets']['base_paths'].get(dataset_name)
        if not self.base_path:
            raise ValueError(f"Dataset base path for '{dataset_name}' not found in config.yaml.")
        
        self.batch_size: int = self.config['training']['batch_size']
        # Determine number of workers. Default to half CPU cores if not specified in config.
        self.num_workers: int = self.config['training'].get('num_workers', os.cpu_count() // 2 if os.cpu_count() else 0)
        
        # Initialize ImageTransforms to get specific augmentation pipelines
        self.image_transforms_manager = ImageTransforms(self.config)
        
        # Internal dataset objects (raw, before transforms are applied dynamically)
        self._train_dataset_raw: Optional[Subset] = None
        self._val_dataset_raw: Optional[Subset] = None
        self._test_dataset_raw: Optional[Dataset] = None
        self._num_classes: int = 0

        # Prepare the data splits immediately upon initialization
        self._prepare_data_splits()

    def _prepare_data_splits(self) -> None:
        """
        Loads the specified full-size dataset and performs a 90/10 train/validation split.
        The original test set is loaded separately.
        """
        full_train_dataset: Optional[Dataset] = None
        
        # --- Dataset-Specific Loading ---
        if self.dataset_name == 'cifar100':
            full_train_dataset = CIFAR100(root=self.base_path, train=True, download=True, transform=None)
            self._test_dataset_raw = CIFAR100(root=self.base_path, train=False, download=True, transform=None)
            self._num_classes = 100
        elif self.dataset_name in ['resisc45', 'clevr_distance']:
            train_dir = os.path.join(self.base_path, 'train')
            test_dir = os.path.join(self.base_path, 'test')

            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"Training data for '{self.dataset_name}' not found at: {train_dir}")
            if not os.path.exists(test_dir):
                raise FileNotFoundError(f"Test data for '{self.dataset_name}' not found at: {test_dir}")
            
            full_train_dataset = ImageFolder(root=train_dir, transform=None)
            self._test_dataset_raw = ImageFolder(root=test_dir, transform=None)
            self._num_classes = len(full_train_dataset.classes)
        else:
            raise ValueError(f"Unsupported many-shot dataset: {self.dataset_name}")

        if full_train_dataset is None or len(full_train_dataset) == 0:
            raise RuntimeError(f"No training data loaded for {self.dataset_name}. Check dataset path and integrity.")

        # --- 90/10 Train/Validation Split ---
        total_train_size: int = len(full_train_dataset)
        val_size: int = math.floor(0.10 * total_train_size)
        train_size: int = total_train_size - val_size

        if train_size == 0 or val_size == 0:
            raise ValueError(f"Invalid split sizes for dataset '{self.dataset_name}'. "
                             f"Calculated train: {train_size}, val: {val_size}. "
                             f"Total samples: {total_train_size}. Check dataset integrity.")

        generator = torch.Generator().manual_seed(self.split_seed)
        self._train_dataset_raw, self._val_dataset_raw = random_split(
            full_train_dataset, [train_size, val_size], generator=generator
        )

    def load_train_data(self) -> DataLoader:
        """
        Loads and returns a DataLoader for the training split, configured with
        dataset-specific training transformations.
        """
        if self._train_dataset_raw is None:
            raise RuntimeError("Training dataset not initialized. "
                               "Ensure _prepare_data_splits was called successfully.")

        train_transforms = self.image_transforms_manager.get_many_shot_train_transforms(self.dataset_name)
        transformed_train_dataset = TransformedSubset(self._train_dataset_raw, transform=train_transforms)

        return DataLoader(
            transformed_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def load_val_data(self) -> DataLoader:
        """
        Loads and returns a DataLoader for the validation split, configured with
        evaluation transformations.
        """
        if self._val_dataset_raw is None:
            raise RuntimeError("Validation dataset not initialized. "
                               "Ensure _prepare_data_splits was called successfully.")

        eval_transforms = self.image_transforms_manager.get_many_shot_eval_transforms()
        transformed_val_dataset = TransformedSubset(self._val_dataset_raw, transform=eval_transforms)

        return DataLoader(
            transformed_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def load_test_data(self) -> DataLoader:
        """
        Loads and returns a DataLoader for the original test set, configured with
        evaluation transformations.
        """
        if self._test_dataset_raw is None:
            raise RuntimeError("Test dataset not initialized. "
                               "Ensure _prepare_data_splits was called successfully.")

        eval_transforms = self.image_transforms_manager.get_many_shot_eval_transforms()
        
        # Apply transform directly to the base dataset object
        # Note: torchvision datasets often allow direct assignment to .transform
        # For Subset, we need the TransformedSubset wrapper.
        self._test_dataset_raw.transform = eval_transforms

        return DataLoader(
            self._test_dataset_raw,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_num_classes(self) -> int:
        """
        Returns the number of classes for the current many-shot dataset.
        """
        if self._num_classes == 0:
            raise RuntimeError("Number of classes not determined. Dataset might not have been loaded correctly.")
        return self._num_classes

