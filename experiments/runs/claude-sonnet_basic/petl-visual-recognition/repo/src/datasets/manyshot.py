"""
Many-shot dataset utilities.
Datasets used for many-shot evaluation:
- CIFAR-100: Natural image dataset, 50K training images, 100 classes
- RESISC45: Remote sensing dataset, 25.2K training samples, 45 classes
- Clevr-Distance: Synthetic depth classification, 70K samples, 6 classes
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image
import numpy as np


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_cifar100_transforms(train=True, img_size=224):
    """Get transforms for CIFAR-100.
    Apply horizontal flipping for CIFAR-100 as per paper.
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transform


def get_resisc_transforms(train=True, img_size=224):
    """Get transforms for RESISC45.
    Apply horizontal and vertical flipping for RESISC as per paper.
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transform


def get_clevr_transforms(train=True, img_size=224):
    """Get transforms for Clevr-Distance.
    No augmentation for Clevr as per paper.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


class ImageFolderWithSplit(Dataset):
    """Dataset that loads from a directory with train/val split."""
    
    def __init__(self, root, split='train', transform=None, val_ratio=0.1, seed=42):
        self.root = root
        self.split = split
        self.transform = transform
        
        # Load all samples
        self.dataset = datasets.ImageFolder(root)
        
        # Create train/val split (90/10)
        total = len(self.dataset)
        val_size = int(total * val_ratio)
        train_size = total - val_size
        
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = random_split(
            range(total), [train_size, val_size], generator=generator
        )
        
        if split == 'train':
            self.indices = list(train_indices)
        else:
            self.indices = list(val_indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_manyshot_dataset(data_dir, dataset_name, split='train', img_size=224, 
                          batch_size=64, num_workers=4, shuffle=None):
    """
    Get a many-shot dataset dataloader.
    
    Args:
        data_dir: Root directory containing dataset
        dataset_name: 'cifar100', 'resisc45', or 'clevr_distance'
        split: 'train', 'val', or 'test'
        img_size: Image size for resizing
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    if dataset_name == 'cifar100':
        transform = get_cifar100_transforms(train=(split == 'train'), img_size=img_size)
        if split == 'test':
            dataset = datasets.CIFAR100(data_dir, train=False, transform=transform, download=True)
        else:
            full_dataset = datasets.CIFAR100(data_dir, train=True, transform=transform, download=True)
            # 90/10 split
            total = len(full_dataset)
            val_size = int(total * 0.1)
            train_size = total - val_size
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
            dataset = train_dataset if split == 'train' else val_dataset
    
    elif dataset_name == 'resisc45':
        transform = get_resisc_transforms(train=(split == 'train'), img_size=img_size)
        dataset_dir = os.path.join(data_dir, 'NWPU-RESISC45')
        dataset = ImageFolderWithSplit(dataset_dir, split=split, transform=transform)
    
    elif dataset_name == 'clevr_distance':
        transform = get_clevr_transforms(train=(split == 'train'), img_size=img_size)
        dataset_dir = os.path.join(data_dir, 'clevr_distance')
        if split == 'test':
            dataset = datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)
        else:
            dataset = ImageFolderWithSplit(os.path.join(dataset_dir, 'train'), split=split, transform=transform)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    
    return loader
