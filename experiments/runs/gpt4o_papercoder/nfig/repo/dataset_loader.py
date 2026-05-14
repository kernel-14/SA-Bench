## dataset_loader.py

import os
import yaml
from typing import Dict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DatasetLoader:
    """Handles the loading and preprocessing of datasets for training, validation, and testing."""

    def __init__(self, config: Dict):
        """
        Initializes the DatasetLoader with configurations and preprocessing.

        Args:
            config (Dict): Parsed configuration dictionary from config.yaml.
                          Contains dataset paths, resolution, and batch size.
        """
        self.config = config

        # Fetch dataset paths and settings from config
        self.train_split = self.config["dataset"].get("train_split", "")
        self.val_split = self.config["dataset"].get("val_split", "")
        self.test_split = self.config["dataset"].get("test_split", "")
        self.resolution = self.config["dataset"].get("resolution", 256)
        self.batch_size = self.config["training"].get("batch_size", 768)

        # Check for missing paths
        if not os.path.exists(self.train_split):
            raise FileNotFoundError(f"Training dataset path '{self.train_split}' does not exist.")
        if not os.path.exists(self.val_split):
            raise FileNotFoundError(f"Validation dataset path '{self.val_split}' does not exist.")
        if not os.path.exists(self.test_split):
            raise FileNotFoundError(f"Test dataset path '{self.test_split}' does not exist.")
        
        # Define preprocessing transformations for train, validation, and test datasets
        # Normalization stats are standard for ImageNet
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(self.resolution),  # Random cropping for augmentation
            transforms.RandomHorizontalFlip(),             # Horizontal flip for augmentation
            transforms.ToTensor(),                         # Convert to PyTorch tensor
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution)),  # Resize to fixed resolution
            transforms.ToTensor(),                                 # Convert to PyTorch tensor
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)  # Normalize
        ])

    def load_train_data(self) -> DataLoader:
        """
        Loads the training dataset into a PyTorch DataLoader.

        Returns:
            DataLoader: DataLoader object containing training data batches.
        """
        train_dataset = datasets.ImageFolder(root=self.train_split, transform=self.transform_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle data for training
            num_workers=4,  # Use 4 workers for data loading
            pin_memory=True  # Optimize performance for GPU
        )
        return train_loader

    def load_val_data(self) -> DataLoader:
        """
        Loads the validation dataset into a PyTorch DataLoader.

        Returns:
            DataLoader: DataLoader object containing validation data batches.
        """
        val_dataset = datasets.ImageFolder(root=self.val_split, transform=self.transform_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=4,  # Use 4 workers for data loading
            pin_memory=True  # Optimize performance for GPU
        )
        return val_loader

    def load_test_data(self) -> DataLoader:
        """
        Loads the test dataset into a PyTorch DataLoader.

        Returns:
            DataLoader: DataLoader object containing test data batches.
        """
        test_dataset = datasets.ImageFolder(root=self.test_split, transform=self.transform_val)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for test data
            num_workers=4,  # Use 4 workers for data loading
            pin_memory=True  # Optimize performance for GPU
        )
        return test_loader
