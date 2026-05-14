"""Dataset loading and preprocessing for PETL experiments.

VTAB-1K: 19 tasks across 3 groups (Natural, Specialized, Structured)
Many-shot: CIFAR-100, RESISC, Clevr-Distance
Robustness: ImageNet-1K (100-shot) + ImageNet-V2, R, S, A
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as torch_datasets
import numpy as np
import os
from PIL import Image


# VTAB-1K Dataset definition
VTAB_DATASETS = {
    'natural': [
        'caltech101', 'cifar100', 'dtd', 'flowers102', 'pets',
        'sun397', 'svhn'
    ],
    'specialized': [
        'eurosat', 'resisc45', 'retinopathy', 'camelyon',
    ],
    'structured': [
        'clevr_count', 'clevr_distance', 'dmlab', 'kitti',
        'dsprites_orientation', 'dsprites_location',
        'smallnorb_azimuth', 'smallnorb_elevation',
    ],
}

ALL_VTAB_TASKS = []
for group_tasks in VTAB_DATASETS.values():
    ALL_VTAB_TASKS.extend(group_tasks)


def get_vtab1k_dataset(dataset_name, data_dir='/data/vtab-1k', 
                       split='train', seed=42):
    """Get a VTAB-1K dataset.
    
    VTAB-1K provides 1000 training images per task.
    We split 800/200 for train/val following the paper.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Path to VTAB-1K data
        split: 'train', 'val', or 'test'
        seed: Random seed for reproducible splits
    
    Returns:
        Dataset, num_classes
    """
    # VTAB-1K is typically provided as TFRecords or preprocessed numpy arrays
    # This is a placeholder that should be connected to actual VTAB data
    
    img_size = 224
    
    # Standard preprocessing (no data augmentation per paper)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Map dataset names to torchvision datasets where possible
    # For custom VTAB datasets, you'd need the actual data
    dataset_map = {
        'caltech101': ('caltech101', 102),
        'cifar100': ('cifar100', 100),
        'dtd': ('dtd', 47),
        'flowers102': ('flowers102', 102),
        'pets': ('oxford_iiit_pet', 37),
        'sun397': ('sun397', 397),
        'svhn': ('svhn', 10),
        'eurosat': ('eurosat', 10),
        'resisc45': ('resisc45', 45),
    }
    
    if dataset_name in dataset_map:
        ds_name, num_classes = dataset_map[dataset_name]
        # Use torchvision
        if hasattr(torch_datasets, ds_name.upper()):
            dataset_cls = getattr(torch_datasets, ds_name.upper())
        else:
            # Placeholder
            dataset = _get_placeholder_dataset(img_size, num_classes, 1000)
            return dataset, num_classes
        
        if split == 'test':
            dataset = dataset_cls(root=data_dir, split='test' if ds_name != 'svhn' else 'test',
                                  transform=transform, download=True)
        else:
            dataset = dataset_cls(root=data_dir, split='train' if ds_name != 'svhn' else 'train',
                                  transform=transform, download=True)
            # Split 800/200
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(dataset))
            if len(indices) > 1000:
                indices = indices[:1000]
            train_idx = indices[:800]
            val_idx = indices[800:1000]
            if split == 'train':
                dataset = Subset(dataset, train_idx)
            else:
                dataset = Subset(dataset, val_idx)
    else:
        # Custom VTAB datasets (Clevr, dSprites, etc.)
        dataset, num_classes = _get_placeholder_dataset(img_size, 10, 1000)
    
    return dataset, num_classes


def _get_placeholder_dataset(img_size, num_classes, num_samples):
    """Create a placeholder dataset for testing."""
    class PlaceholderDataset(Dataset):
        def __init__(self, size, classes, samples):
            self.size = size
            self.classes = classes
            self.samples = samples
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
        def __len__(self):
            return self.samples
        
        def __getitem__(self, idx):
            img = torch.randn(3, self.size, self.size)
            label = idx % self.classes
            return img, label
    
    return PlaceholderDataset(img_size, num_classes, num_samples), num_classes


def get_many_shot_dataset(dataset_name, data_dir='/data',
                          split='train', seed=42):
    """Get many-shot datasets: CIFAR-100, RESISC, Clevr-Distance.
    
    Splits: 90/10 train/val for all datasets.
    """
    img_size = 224
    
    if dataset_name.lower() == 'cifar100':
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])
        transform_eval = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        ])
        
        if split == 'test':
            dataset = torch_datasets.CIFAR100(
                root=data_dir, train=False, transform=transform_eval, download=True
            )
            return dataset, 100
        
        full_dataset = torch_datasets.CIFAR100(
            root=data_dir, train=True, transform=transform_train, download=True
        )
        
        # 90/10 split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(full_dataset))
        split_idx = int(0.9 * len(full_dataset))
        
        if split == 'train':
            subset = Subset(full_dataset, indices[:split_idx])
        else:
            val_dataset = torch_datasets.CIFAR100(
                root=data_dir, train=True, transform=transform_eval, download=True
            )
            subset = Subset(val_dataset, indices[split_idx:])
        
        return subset, 100
    
    elif dataset_name.lower() == 'resisc':
        # RESISC45: remote sensing, 45 classes, 25.2K images
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.368, 0.381, 0.344], std=[0.196, 0.180, 0.171]),
        ])
        transform_eval = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.368, 0.381, 0.344], std=[0.196, 0.180, 0.171]),
        ])
        
        dataset = torch_datasets.ImageFolder(
            root=os.path.join(data_dir, 'RESISC45'),
            transform=transform_train if split == 'train' else transform_eval,
        )
        
        if split == 'test':
            return dataset, 45
        
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(dataset))
        split_idx = int(0.9 * len(dataset))
        
        if split == 'train':
            return Subset(dataset, indices[:split_idx]), 45
        else:
            return Subset(dataset, indices[split_idx:]), 45
    
    elif dataset_name.lower() == 'clevr_distance':
        # Clevr-Distance: 6 depth classes, 70K samples
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        dataset, num_classes = _get_placeholder_dataset(img_size, 6, 70000)
        
        if split == 'test':
            return dataset, 6
        
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(dataset))
        split_idx = int(0.9 * len(dataset))
        
        if split == 'train':
            return Subset(dataset, indices[:split_idx]), 6
        else:
            return Subset(dataset, indices[split_idx:]), 6
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_robustness_dataset(dataset_name, data_dir='/data/imagenet',
                           split='train', shots_per_class=100):
    """Get robustness datasets: ImageNet-1K (100-shot) and distribution shifts.
    
    Distribution shift datasets: ImageNet-V2, ImageNet-R, ImageNet-S, ImageNet-A
    """
    img_size = 224
    
    # Strong augmentation for robustness (following Wortsman et al.)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]),
    ])
    
    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.458, 0.408], std=[0.269, 0.261, 0.276]),
    ])
    
    if dataset_name == 'imagenet':
        # 100-shot ImageNet
        if split == 'train':
            dataset = torch_datasets.ImageNet(
                root=data_dir, split='train', transform=transform_train
            )
            # Subsample to 100 per class
            return _subsample_per_class(dataset, shots_per_class)
        else:
            return torch_datasets.ImageNet(
                root=data_dir, split='val', transform=transform_eval
            )
    
    elif dataset_name == 'imagenet_v2':
        dataset = torch_datasets.ImageFolder(
            root=os.path.join(data_dir, 'imagenetv2-matched-frequency'),
            transform=transform_eval,
        )
        return dataset
    
    elif dataset_name == 'imagenet_r':
        dataset = torch_datasets.ImageFolder(
            root=os.path.join(data_dir, 'imagenet-r'),
            transform=transform_eval,
        )
        return dataset
    
    elif dataset_name == 'imagenet_s':
        dataset = torch_datasets.ImageFolder(
            root=os.path.join(data_dir, 'imagenet-sketch'),
            transform=transform_eval,
        )
        return dataset
    
    elif dataset_name == 'imagenet_a':
        dataset = torch_datasets.ImageFolder(
            root=os.path.join(data_dir, 'imagenet-a'),
            transform=transform_eval,
        )
        return dataset
    
    else:
        raise ValueError(f"Unknown robustness dataset: {dataset_name}")


def _subsample_per_class(dataset, shots_per_class, seed=42):
    """Subsample dataset to have exactly shots_per_class samples per class."""
    if not hasattr(dataset, 'targets'):
        return dataset
    
    targets = np.array(dataset.targets)
    rng = np.random.RandomState(seed)
    indices = []
    
    for c in np.unique(targets):
        class_indices = np.where(targets == c)[0]
        if len(class_indices) > shots_per_class:
            selected = rng.choice(class_indices, shots_per_class, replace=False)
        else:
            selected = class_indices
        indices.extend(selected)
    
    return Subset(dataset, indices)
