"""
VTAB-1K dataset utilities.
VTAB-1K consists of 19 classification tasks from 3 groups:
- Natural: Caltech101, CIFAR-100, DTD, Flowers102, Pets, SVHN, Sun397
- Specialized: Camelyon, EuroSAT, Resisc45, Retinopathy
- Structured: Clevr-Count, Clevr-Distance, DMLab, KITTI, dSpr-Loc, dSpr-Ori, sNORB-Azim, sNORB-Elev

Following the original VTAB-1K paper, we use 1000 training images per task.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json


# VTAB-1K dataset names and their groups
VTAB_DATASETS = {
    'natural': [
        'caltech101',
        'cifar100',
        'dtd',
        'flowers102',
        'pets',
        'svhn',
        'sun397',
    ],
    'specialized': [
        'camelyon',
        'eurosat',
        'resisc45',
        'retinopathy',
    ],
    'structured': [
        'clevr_count',
        'clevr_distance',
        'dmlab',
        'kitti',
        'dsprites_loc',
        'dsprites_ori',
        'smallnorb_azimuth',
        'smallnorb_elevation',
    ],
}

# Number of classes per dataset
VTAB_NUM_CLASSES = {
    'caltech101': 102,
    'cifar100': 100,
    'dtd': 47,
    'flowers102': 102,
    'pets': 37,
    'svhn': 10,
    'sun397': 397,
    'camelyon': 2,
    'eurosat': 10,
    'resisc45': 45,
    'retinopathy': 5,
    'clevr_count': 8,
    'clevr_distance': 6,
    'dmlab': 6,
    'kitti': 4,
    'dsprites_loc': 16,
    'dsprites_ori': 16,
    'smallnorb_azimuth': 18,
    'smallnorb_elevation': 9,
}

# ImageNet mean and std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_vtab_transforms(train=True, img_size=224):
    """Get transforms for VTAB-1K datasets.
    
    Note: Following the original VTAB-1K paper and most PEFT papers,
    we do NOT apply data augmentation as it's challenging to identify
    augmentations that uniformly benefit all 19 datasets.
    """
    if train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
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


class VTABDataset(Dataset):
    """VTAB-1K dataset wrapper."""
    
    def __init__(self, data_dir, dataset_name, split='train', transform=None):
        """
        Args:
            data_dir: Root directory containing VTAB-1K data
            dataset_name: Name of the dataset (e.g., 'caltech101')
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        
        # Load data list
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load image paths and labels from data directory."""
        split_file = os.path.join(self.data_dir, self.dataset_name, f'{self.split}.txt')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        samples = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ')
                    img_path = parts[0]
                    label = int(parts[1])
                    full_path = os.path.join(self.data_dir, self.dataset_name, img_path)
                    samples.append((full_path, label))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Return a black image if loading fails
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_vtab_dataset(data_dir, dataset_name, split='train', img_size=224, batch_size=64, 
                     num_workers=4, shuffle=None):
    """
    Get a VTAB-1K dataset dataloader.
    
    Args:
        data_dir: Root directory containing VTAB-1K data
        dataset_name: Name of the dataset
        split: 'train', 'val', or 'test'
        img_size: Image size for resizing
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        shuffle: Whether to shuffle (default: True for train, False otherwise)
    
    Returns:
        DataLoader
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    transform = get_vtab_transforms(train=(split == 'train'), img_size=img_size)
    dataset = VTABDataset(data_dir, dataset_name, split=split, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    
    return loader


def get_all_vtab_datasets(data_dir, split='train', img_size=224, batch_size=64, num_workers=4):
    """Get all 19 VTAB-1K datasets."""
    loaders = {}
    for group, datasets in VTAB_DATASETS.items():
        for dataset_name in datasets:
            try:
                loader = get_vtab_dataset(data_dir, dataset_name, split, img_size, batch_size, num_workers)
                loaders[dataset_name] = loader
            except FileNotFoundError:
                print(f"Warning: Dataset {dataset_name} not found, skipping.")
    return loaders
