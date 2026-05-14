import os
import abc
import math
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
from typing import Dict, Any, Optional, List
from collections import defaultdict
import random

# Assuming ImageTransforms is located in datasets/transforms.py
from datasets.transforms import ImageTransforms
# Assuming BaseDatasetLoader is defined in datasets/vtab_loader.py
from datasets.vtab_loader import BaseDatasetLoader


class RobustnessLoader(BaseDatasetLoader):
    """
    Loads and manages data for the "Robustness to Distribution Shifts" experiments
    as specified in the paper. This includes:
    - 100-shot ImageNet-1K for training.
    - ImageNet-1K test data as the target distribution.
    - ImageNet-V2, ImageNet-R, ImageNet-S, and ImageNet-A for out-of-distribution evaluation.
    Applies strong data augmentation for training and standard evaluation transforms.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the RobustnessLoader.

        Args:
            config (Dict[str, Any]): The full configuration dictionary from config.yaml.
        """
        self.config: Dict[str, Any] = config
        self.split_seed: int = self.config['seed']  # Use global seed for deterministic operations

        # Retrieve dataset base paths from the configuration
        self.imagenet_path: str = self.config['datasets']['base_paths']['imagenet']
        self.imagenet_v2_path: str = self.config['datasets']['base_paths']['imagenet_v2']
        self.imagenet_r_path: str = self.config['datasets']['base_paths']['imagenet_r']
        self.imagenet_s_path: str = self.config['datasets']['base_paths']['imagenet_s']
        self.imagenet_a_path: str = self.config['datasets']['base_paths']['imagenet_a']

        # Initialize ImageTransforms to get specific augmentation pipelines
        self.image_transforms_manager = ImageTransforms(self.config)

        self.batch_size: int = self.config['training']['batch_size']
        # Determine number of workers. Default to half CPU cores if not specified in config.
        self.num_workers: int = self.config['training'].get('num_workers', os.cpu_count() // 2 if os.cpu_count() else 0)

        self._num_classes: int = 1000  # ImageNet-1K has 1000 classes

    def load_train_data(self) -> DataLoader:
        """
        Loads the 100-shot ImageNet-1K training data.
        Selects 100 images per class deterministically using the split_seed.

        Returns:
            DataLoader: A DataLoader for the 100-shot ImageNet-1K training set.
        """
        train_transforms = self.image_transforms_manager.get_robustness_train_transforms()
        full_imagenet_train_dataset = ImageFolder(
            root=os.path.join(self.imagenet_path, 'train'),
            transform=train_transforms
        )

        # Implement 100-shot sampling
        selected_indices: List[int] = []
        class_indices = defaultdict(list)

        # Group indices by class
        for idx, (_, label) in enumerate(full_imagenet_train_dataset.samples):
            class_indices[label].append(idx)

        # Ensure deterministic sampling
        rng = random.Random(self.split_seed)

        # Select 100 samples per class
        for label in sorted(class_indices.keys()):
            indices_for_class = class_indices[label]
            if len(indices_for_class) > 100:
                selected_from_class = rng.sample(indices_for_class, 100)
            else:
                # If a class has fewer than 100 samples, take all of them
                selected_from_class = indices_for_class
            selected_indices.extend(selected_from_class)

        # Create a Subset containing only the 100-shot samples
        hundred_shot_train_subset = Subset(full_imagenet_train_dataset, selected_indices)

        return DataLoader(
            hundred_shot_train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def load_val_data(self) -> DataLoader:
        """
        Returns None, as the robustness study typically does not use a separate
        validation set from the target distribution. The target test set is used
        for evaluating performance on the target distribution.
        """
        return None

    def load_test_data(self) -> DataLoader:
        """
        Returns the DataLoader for the target distribution test set (ImageNet-1K).
        This method is part of the BaseDatasetLoader interface and delegates
        to load_test_data_target.
        """
        return self.load_test_data_target()

    def load_test_data_target(self) -> DataLoader:
        """
        Loads the ImageNet-1K test data (often referred to as 'val' in ImageNet
        for public benchmarks) as the target distribution for evaluation.

        Returns:
            DataLoader: A DataLoader for the ImageNet-1K test set.
        """
        eval_transforms = self.image_transforms_manager.get_robustness_eval_transforms()
        imagenet_test_dataset = ImageFolder(
            root=os.path.join(self.imagenet_path, 'val'), # ImageNet's val folder is commonly used as test set
            transform=eval_transforms
        )

        return DataLoader(
            imagenet_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def load_test_data_ood(self) -> Dict[str, DataLoader]:
        """
        Loads the ImageNet-V2, ImageNet-R, ImageNet-S, and ImageNet-A test data
        for out-of-distribution (OOD) evaluation.

        Returns:
            Dict[str, DataLoader]: A dictionary where keys are OOD dataset names
                                   and values are their corresponding DataLoaders.
        """
        ood_data_loaders: Dict[str, DataLoader] = {}
        eval_transforms = self.image_transforms_manager.get_robustness_eval_transforms()

        # Define OOD datasets and their paths
        datasets_info = {
            'imagenet_v2': self.imagenet_v2_path,
            'imagenet_r': self.imagenet_r_path,
            'imagenet_s': self.imagenet_s_path,
            'imagenet_a': self.imagenet_a_path
        }

        for name, path in datasets_info.items():
            if not os.path.exists(path):
                print(f"Warning: OOD dataset path not found for {name} at {path}. Skipping this dataset.")
                continue
            
            ood_dataset = ImageFolder(root=path, transform=eval_transforms)
            ood_loader = DataLoader(
                ood_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            ood_data_loaders[name] = ood_loader

        return ood_data_loaders

    def get_num_classes(self) -> int:
        """
        Returns the number of classes for the ImageNet-1K task (1000).
        """
        return self._num_classes

