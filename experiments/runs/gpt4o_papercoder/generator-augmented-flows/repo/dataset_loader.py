import os
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DatasetLoader:
    """
    Handles dataset loading, preprocessing, and preparing DataLoader objects
    for training, validation, and testing splits.
    """

    def __init__(self, config: dict):
        """
        Initializes the DatasetLoader class with the given configuration.

        Args:
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.config = config
        self.dataset_name = config["dataset"].get("name", "CIFAR-10")  # Default: CIFAR-10
        self.resolution = config["dataset"].get("resolution", 32)  # Resolution for resizing images
        self.batch_size = config["training"].get("batch_size", 128)  # Default batch size: 128
        self.use_gpu = config["hardware"].get("use_gpu", True)  # Use GPU if available
        self.pin_memory = self.use_gpu  # Pin memory for DataLoader if using GPU
        self.preprocessing = config["dataset"].get("preprocessing", {})
        self.num_workers = 4  # Default number of workers for DataLoader

    def load_data(self) -> Dict[str, DataLoader]:
        """
        Loads the dataset, applies preprocessing, and returns DataLoader objects for
        training, validation, and testing splits.

        Returns:
            dict: A dictionary with keys "train", "val", "test", each mapping to a DataLoader.
        """
        # Define the dataset and its specific preprocessing pipeline
        train_transforms, test_transforms = self._construct_transforms()

        # Load datasets
        train_dataset = self._get_dataset(self.dataset_name, split="train", transform=train_transforms)
        test_dataset = self._get_dataset(self.dataset_name, split="test", transform=test_transforms)

        # Construct DataLoaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

        # Return loaders
        return {"train": train_loader, "val": test_loader, "test": test_loader}

    def _construct_transforms(self) -> (transforms.Compose, transforms.Compose):
        """
        Constructs the preprocessing pipeline for the dataset.

        Returns:
            tuple: A tuple containing train_transforms and test_transforms.
        """
        # Get preprocessing options
        normalize = self.preprocessing.get("normalize", True)
        resize = self.preprocessing.get("resize", True)

        # Define normalization values per dataset
        if self.dataset_name == "CIFAR-10":
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        elif self.dataset_name == "ImageNet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            # Default normalization for CelebA, LSUN Church
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        # Train-specific transforms (with optional resizing)
        train_transforms = [
            transforms.Resize((self.resolution, self.resolution)) if resize else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
        # Test-specific transforms (same as train minus augmentation)
        test_transforms = train_transforms.copy()

        # Add scaling to [-1, 1]
        scale_transform = transforms.Lambda(lambda x: x * 2 - 1)
        train_transforms.append(scale_transform)
        test_transforms.append(scale_transform)

        # Add normalization
        if normalize:
            norm_transform = transforms.Normalize(mean=mean, std=std)
            train_transforms.append(norm_transform)
            test_transforms.append(norm_transform)

        # Final transform pipelines
        return transforms.Compose(train_transforms), transforms.Compose(test_transforms)

    def _get_dataset(self, dataset_name: str, split: str, transform: transforms.Compose):
        """
        Maps the dataset name to its respective loader in torchvision.datasets
        and initializes it.

        Args:
            dataset_name (str): Name of the dataset (e.g., CIFAR-10, ImageNet).
            split (str): Either 'train' or 'test'.
            transform (transforms.Compose): Preprocessing pipeline for the dataset.

        Returns:
            torch.utils.data.Dataset: The dataset object.
        """
        root_dir = os.path.expanduser("~/datasets")  # Default dataset storage path
        is_train = split == "train"

        # Map dataset names to torchvision loaders
        if dataset_name == "CIFAR-10":
            return datasets.CIFAR10(
                root=root_dir, train=is_train, transform=transform, download=True
            )
        elif dataset_name == "ImageNet":
            # Note: ImageNet requires manual download
            return datasets.ImageNet(
                root=os.path.join(root_dir, "imagenet"), split="train" if is_train else "val", transform=transform
            )
        elif dataset_name == "CelebA":
            return datasets.CelebA(
                root=root_dir, split="train" if is_train else "test", transform=transform, download=True
            )
        elif dataset_name == "LSUN":
            return datasets.LSUN(
                root=root_dir,
                classes=["church_outdoor_train" if is_train else "church_outdoor_val"],
                transform=transform,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
