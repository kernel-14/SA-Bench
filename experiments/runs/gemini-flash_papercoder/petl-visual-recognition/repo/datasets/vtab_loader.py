## datasets/vtab_loader.py

import os
import abc
import math
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from typing import Dict, Any, Optional

# Assuming ImageTransforms is located in datasets/transforms.py
from datasets.transforms import ImageTransforms


class BaseDatasetLoader(abc.ABC):
    """
    Abstract Base Class for all dataset loaders.
    Defines the common interface for loading training, validation, and test data,
    and retrieving the number of classes.
    """

    @abc.abstractmethod
    def load_train_data(self) -> DataLoader:
        """Loads and returns the DataLoader for the training dataset."""
        pass

    @abc.abstractmethod
    def load_val_data(self) -> DataLoader:
        """Loads and returns the DataLoader for the validation dataset."""
        pass

    @abc.abstractmethod
    def load_test_data(self) -> DataLoader:
        """Loads and returns the DataLoader for the test dataset."""
        pass

    @abc.abstractmethod
    def get_num_classes(self) -> int:
        """Returns the number of classes in the dataset."""
        pass


class VTABLoader(BaseDatasetLoader):
    """
    Loads and manages data for VTAB-1K tasks as specified in the paper.
    Handles data splitting for hyperparameter tuning (80/20 train/val from 1000 samples)
    and provides loaders for final training (all 1000 samples) and original test set evaluation.
    No data augmentation is applied for VTAB-1K tasks.
    """

    def __init__(self, config: Dict[str, Any], task_name: str, split_seed: int) -> None:
        """
        Initializes the VTABLoader.

        Args:
            config (Dict[str, Any]): The full configuration dictionary from config.yaml.
            task_name (str): The name of the specific VTAB-1K task (e.g., 'Caltech101').
            split_seed (int): The random seed to use for deterministic data splitting.
        """
        self.config: Dict[str, Any] = config
        self.task_name: str = task_name
        self.split_seed: int = split_seed

        # Retrieve base path for VTAB-1K datasets from the configuration
        self.vtab1k_base_path: str = self.config['datasets']['base_paths']['vtab1k']
        # Construct the full directory path for the specific task's data
        self.task_data_path: str = os.path.join(self.vtab1k_base_path, self.task_name)

        # Initialize ImageTransforms. For VTAB-1K, the paper states no data augmentation.
        # So, training and evaluation transforms are the same (standard eval-like transforms).
        image_transforms_manager = ImageTransforms(self.config)
        self.train_transforms = image_transforms_manager.get_vtab_transforms()
        self.eval_transforms = image_transforms_manager.get_vtab_transforms()

        # Internal dataset objects
        self._train_dataset_hp_tuning: Optional[Subset] = None
        self._val_dataset_hp_tuning: Optional[Subset] = None
        self._train_dataset_final: Optional[ImageFolder] = None  # Full 1000-shot training dataset
        self._test_dataset: Optional[ImageFolder] = None
        self._num_classes: int = 0

        # Prepare the data splits immediately upon initialization
        self._prepare_data_splits()

        # Determine the number of classes from the loaded dataset
        if self._train_dataset_final and hasattr(self._train_dataset_final, 'classes'):
            self._num_classes = len(self._train_dataset_final.classes)
        else:
            raise ValueError(f"Failed to load training data for task {self.task_name} or "
                             "determine number of classes. Check dataset path and structure.")

    def _prepare_data_splits(self) -> None:
        """
        Loads the specified VTAB-1K task dataset and creates the necessary
        train/validation/test splits according to the paper's methodology.
        - `self._train_dataset_final`: The dataset containing all 1000 training images.
        - `self._train_dataset_hp_tuning`, `self._val_dataset_hp_tuning`: An 80/20 split
          derived from the 1000 training images, used for hyperparameter tuning.
        - `self._test_dataset`: The original VTAB-1K test set for final evaluation.
        """
        # Define paths assuming a standard VTAB-1K directory structure
        # e.g., <vtab1k_base_path>/<task_name>/train_1000/class1/...
        train_1000_path: str = os.path.join(self.task_data_path, 'train_1000')
        test_path: str = os.path.join(self.task_data_path, 'test')

        if not os.path.exists(train_1000_path):
            raise FileNotFoundError(f"1000-shot training data for task '{self.task_name}' "
                                    f"not found at: {train_1000_path}. "
                                    "Please ensure VTAB-1K data is correctly structured.")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data for task '{self.task_name}' "
                                    f"not found at: {test_path}. "
                                    "Please ensure VTAB-1K data is correctly structured.")

        # Load the full 1000-shot training dataset
        self._train_dataset_final = ImageFolder(root=train_1000_path, transform=self.train_transforms)

        # Create 80/20 train/validation split for hyperparameter tuning from the 1000 samples
        total_train_samples: int = len(self._train_dataset_final)
        train_size_hp_tuning: int = math.floor(0.8 * total_train_samples)
        val_size_hp_tuning: int = total_train_samples - train_size_hp_tuning

        if total_train_samples == 0:
            raise ValueError(f"No training samples found for task '{self.task_name}' at {train_1000_path}.")
        if train_size_hp_tuning == 0 or val_size_hp_tuning == 0:
            raise ValueError(f"Invalid split sizes for task '{self.task_name}'. "
                             f"Calculated train: {train_size_hp_tuning}, val: {val_size_hp_tuning}. "
                             f"Total samples: {total_train_samples}. Check dataset integrity.")

        # Ensure deterministic splitting using the provided split_seed
        generator = torch.Generator().manual_seed(self.split_seed)
        self._train_dataset_hp_tuning, self._val_dataset_hp_tuning = random_split(
            self._train_dataset_final,
            [train_size_hp_tuning, val_size_hp_tuning],
            generator=generator
        )

        # Load the original test dataset
        self._test_dataset = ImageFolder(root=test_path, transform=self.eval_transforms)

    def load_train_data(self) -> DataLoader:
        """
        Loads and returns a DataLoader for the training data.
        For VTAB-1K, this corresponds to the full 1000 training samples
        used after hyperparameter tuning.
        """
        if self._train_dataset_final is None:
            raise RuntimeError("Training dataset (final 1000 samples) not initialized. "
                               "Ensure _prepare_data_splits was called successfully.")

        batch_size: int = self.config['training']['batch_size']
        # Use half of available CPU cores for workers, default to 0 if count is None or 0
        num_workers: int = os.cpu_count() // 2 if os.cpu_count() else 0
        
        return DataLoader(self._train_dataset_final, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def load_val_data(self) -> DataLoader:
        """
        Loads and returns a DataLoader for the validation data.
        For VTAB-1K, this corresponds to the 20% split from the 1000 training samples
        used for hyperparameter tuning.
        """
        if self._val_dataset_hp_tuning is None:
            raise RuntimeError("Validation dataset (HP tuning) not initialized. "
                               "Ensure _prepare_data_splits was called successfully.")

        batch_size: int = self.config['training']['batch_size']
        num_workers: int = os.cpu_count() // 2 if os.cpu_count() else 0
        
        return DataLoader(self._val_dataset_hp_tuning, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def load_test_data(self) -> DataLoader:
        """
        Loads and returns a DataLoader for the test data.
        For VTAB-1K, this corresponds to the original test set provided with the benchmark.
        """
        if self._test_dataset is None:
            raise RuntimeError("Test dataset not initialized. "
                               "Ensure _prepare_data_splits was called successfully.")

        batch_size: int = self.config['training']['batch_size']
        num_workers: int = os.cpu_count() // 2 if os.cpu_count() else 0
        
        return DataLoader(self._test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def get_num_classes(self) -> int:
        """
        Returns the number of classes for the current VTAB-1K task.
        """
        if self._num_classes == 0:
            raise RuntimeError("Number of classes not determined. Dataset might not have been loaded correctly.")
        return self._num_classes

