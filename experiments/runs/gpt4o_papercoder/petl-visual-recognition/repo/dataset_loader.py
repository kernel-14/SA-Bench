## dataset_loader.py
import os
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CIFAR100
from torch.utils.data import random_split, Dataset
from typing import Dict, Tuple
from utils import set_random_seed

class DatasetLoader:
    """Class to load datasets for low-shot, many-shot, and distribution-shift tasks."""
    
    def __init__(self, config: Dict):
        """
        Initialize the DatasetLoader with configuration settings.
        Args:
            config (Dict): Configuration dictionary loaded from 'config.yaml'.
        """
        self.config = config
        self.low_shot_config = config['datasets']['low_shot']
        self.many_shot_config = config['datasets']['many_shot']
        self.dist_shift_config = config['datasets']['distribution_shift']
        self.seed = config.get('training', {}).get('seed', 42)  # Default seed
        self.imagenet_norm_mean = (0.485, 0.456, 0.406)
        self.imagenet_norm_std = (0.229, 0.224, 0.225)
        
        # Set random seed for reproducibility
        set_random_seed(self.seed)
    
    def _normalize_transform(self) -> T.Compose:
        """Return a torchvision transform for normalizing images."""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.imagenet_norm_mean, std=self.imagenet_norm_std)
        ])
    
    def _augment_transform(self) -> T.Compose:
        """Return a torchvision transform for augmenting and normalizing images."""
        return T.Compose([
            T.RandomResizedCrop(size=224),  # As per ImageNet size
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=self.imagenet_norm_mean, std=self.imagenet_norm_std)
        ])
    
    def _train_val_split(
        self,
        dataset: Dataset,
        train_ratio: float
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and validation sets based on the given ratio.
        Args:
            dataset (Dataset): The full dataset to split.
            train_ratio (float): The proportion of training samples (0-1).
        Returns:
            Tuple[Dataset, Dataset]: Train and validation datasets.
        """
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))
    
    def load_vtab_dataset(self, task_name: str) -> Dict[str, Dataset]:
        """Load a VTAB-1K dataset (low-shot regime) for a specific task.
        Args:
            task_name (str): The name of the VTAB-1K task to load.
        Returns:
            Dict[str, Dataset]: Dict with 'train', 'val', and 'test' datasets.
        """
        # Assuming VTAB datasets are stored in a specific folder
        dataset_path = os.path.join(self.low_shot_config.get('source', 'VTAB-1K'), task_name)
        
        # Apply basic normalization (no augmentation for low-shot)
        base_transform = self._normalize_transform()
        dataset = ImageFolder(root=dataset_path, transform=base_transform)
        
        # Split into train and validation
        splits = self.low_shot_config.get('splits', {'train': 80, 'val': 20})  # Default to 80/20 split
        train_split = splits['train'] / 100.0  # Convert percentage to proportion
        train_set, val_set = self._train_val_split(dataset, train_split)
        
        # Leave the original `test` dataset untouched
        test_set = ImageFolder(
            root=os.path.join(dataset_path, 'test'),
            transform=base_transform
        )
        
        return {"train": train_set, "val": val_set, "test": test_set}
    
    def load_imagenet_distribution_shifts(self) -> Dict[str, Dataset]:
        """Load the ImageNet-1K dataset for fine-tuning and shifted datasets for robustness evaluation.
        Returns:
            Dict[str, Dataset]: Dict containing 'target' for base ImageNet and shifted datasets.
        """
        datasets = {}
        target_path = self.dist_shift_config['target']
        shifted_versions = self.dist_shift_config['shifted_versions']
        samples_per_class = self.dist_shift_config.get("samples_per_class", 100)
        
        # ImageNet-1K target dataset
        datasets['target'] = ImageFolder(
            root=target_path,
            transform=self._augment_transform()  # Fine-tuning with augmentation
        )
        
        # Shifted datasets
        for shift_name in shifted_versions:
            datasets[shift_name] = ImageFolder(
                root=os.path.join(target_path, shift_name),
                transform=self._normalize_transform()  # Normalized for evaluation
            )
        
        return datasets
    
    def load_many_shot_dataset(self, dataset_name: str) -> Dict[str, Dataset]:
        """Load datasets for the many-shot experimental regime.
        Args:
            dataset_name (str): Name of the dataset (e.g., 'CIFAR-100', 'RESISC').
        Returns:
            Dict[str, Dataset]: Dict with 'train', 'val', and 'test' datasets.
        """
        if dataset_name == 'CIFAR-100':
            dataset = CIFAR100(
                root='./data',
                train=True,
                transform=self._augment_transform(),  # Include augmentations for train
                download=True
            )
            test_set = CIFAR100(
                root='./data',
                train=False,
                transform=self._normalize_transform(),  # Normalize for test
                download=True
            )
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not implemented.")
        
        # Split train into train/validation sets (90/10 split)
        train_split = 0.9  # Default split proportion for many-shot
        train_set, val_set = self._train_val_split(dataset, train_split)
        
        return {"train": train_set, "val": val_set, "test": test_set}
