"""
ImageNet dataset utilities for robustness evaluation.
Includes ImageNet-1K and distribution shift variants:
- ImageNet-V2: New test set with original labeling protocol
- ImageNet-R: Renditions for 200 ImageNet classes
- ImageNet-S: Sketch images for 1K ImageNet classes
- ImageNet-A: Natural adversarial examples for 200 ImageNet classes
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_imagenet_transforms(train=True, img_size=224):
    """Get transforms for ImageNet datasets."""
    if train:
        # Strong augmentation following [107] (CLIP paper setup)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transform


class FewShotImageNet(Dataset):
    """ImageNet with few-shot sampling (100 shots per class)."""
    
    def __init__(self, root, num_shots=100, transform=None, seed=42):
        self.root = root
        self.num_shots = num_shots
        self.transform = transform
        
        # Load full dataset
        full_dataset = datasets.ImageFolder(root)
        self.classes = full_dataset.classes
        self.class_to_idx = full_dataset.class_to_idx
        
        # Sample num_shots per class
        import random
        random.seed(seed)
        
        class_samples = {}
        for img_path, label in full_dataset.samples:
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(img_path)
        
        self.samples = []
        for label, paths in class_samples.items():
            sampled = random.sample(paths, min(num_shots, len(paths)))
            for path in sampled:
                self.samples.append((path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_imagenet_dataset(data_dir, split='train', num_shots=None, img_size=224,
                          batch_size=64, num_workers=4, shuffle=None):
    """
    Get an ImageNet dataset dataloader.
    
    Args:
        data_dir: Root directory containing ImageNet data
        split: 'train' or 'val'
        num_shots: If specified, sample this many shots per class
        img_size: Image size for resizing
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader
    """
    if shuffle is None:
        shuffle = (split == 'train')
    
    transform = get_imagenet_transforms(train=(split == 'train'), img_size=img_size)
    
    if num_shots is not None and split == 'train':
        dataset = FewShotImageNet(
            os.path.join(data_dir, 'train'),
            num_shots=num_shots,
            transform=transform
        )
    else:
        dataset_dir = os.path.join(data_dir, split)
        dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )
    
    return loader


def get_distribution_shift_datasets(data_dir, img_size=224, batch_size=64, num_workers=4):
    """
    Get all distribution shift datasets for robustness evaluation.
    
    Returns:
        dict of DataLoaders for each distribution shift dataset
    """
    transform = get_imagenet_transforms(train=False, img_size=img_size)
    
    datasets_dict = {}
    
    # ImageNet-V2
    v2_dir = os.path.join(data_dir, 'imagenet-v2')
    if os.path.exists(v2_dir):
        datasets_dict['imagenet_v2'] = DataLoader(
            datasets.ImageFolder(v2_dir, transform=transform),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    
    # ImageNet-R
    r_dir = os.path.join(data_dir, 'imagenet-r')
    if os.path.exists(r_dir):
        datasets_dict['imagenet_r'] = DataLoader(
            datasets.ImageFolder(r_dir, transform=transform),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    
    # ImageNet-S (Sketch)
    s_dir = os.path.join(data_dir, 'imagenet-sketch')
    if os.path.exists(s_dir):
        datasets_dict['imagenet_s'] = DataLoader(
            datasets.ImageFolder(s_dir, transform=transform),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    
    # ImageNet-A
    a_dir = os.path.join(data_dir, 'imagenet-a')
    if os.path.exists(a_dir):
        datasets_dict['imagenet_a'] = DataLoader(
            datasets.ImageFolder(a_dir, transform=transform),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    
    return datasets_dict
