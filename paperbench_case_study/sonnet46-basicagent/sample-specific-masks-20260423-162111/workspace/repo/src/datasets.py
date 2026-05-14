"""
Dataset loading utilities for SMM (Sample-specific Multi-channel Masks).
Supports: CIFAR10, CIFAR100, SVHN, GTSRB, Flowers102, DTD, UCF101,
          Food101, SUN397, EuroSAT, OxfordPets, StanfordCars
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

IMAGENETNORMALIZE = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

DATASET_INFO = {
    'CIFAR10':     {'num_classes': 10,  'img_size': 32},
    'CIFAR100':    {'num_classes': 100, 'img_size': 32},
    'SVHN':        {'num_classes': 10,  'img_size': 32},
    'GTSRB':       {'num_classes': 43,  'img_size': 32},
    'Flowers102':  {'num_classes': 102, 'img_size': 128},
    'DTD':         {'num_classes': 47,  'img_size': 128},
    'UCF101':      {'num_classes': 101, 'img_size': 128},
    'Food101':     {'num_classes': 101, 'img_size': 128},
    'SUN397':      {'num_classes': 397, 'img_size': 128},
    'EuroSAT':     {'num_classes': 10,  'img_size': 128},
    'OxfordPets':  {'num_classes': 37,  'img_size': 128},
    'StanfordCars':{'num_classes': 196, 'img_size': 128},
}


def get_transforms(model_name='ResNet'):
    """
    Returns train and test transforms.
    For ViT_B32, imgsize=384; otherwise imgsize=224.
    """
    if model_name == 'ViT_B32':
        imgsize = 384
    else:
        imgsize = 224

    train_preprocess = transforms.Compose([
        transforms.Resize((imgsize + 32, imgsize + 32)),
        transforms.RandomCrop(imgsize),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    test_preprocess = transforms.Compose([
        transforms.Resize((imgsize, imgsize)),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    return train_preprocess, test_preprocess


def get_dataset(dataset_name, data_root, model_name='ResNet'):
    """
    Load train and test datasets for the given dataset name.
    Returns (train_dataset, test_dataset, num_classes).
    """
    train_transform, test_transform = get_transforms(model_name)
    root = os.path.join(data_root, dataset_name)
    os.makedirs(root, exist_ok=True)

    if dataset_name == 'CIFAR10':
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
        test_ds  = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    elif dataset_name == 'CIFAR100':
        train_ds = datasets.CIFAR100(root=root, train=True, download=True, transform=train_transform)
        test_ds  = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)

    elif dataset_name == 'SVHN':
        train_ds = datasets.SVHN(root=root, split='train', download=True, transform=train_transform)
        test_ds  = datasets.SVHN(root=root, split='test',  download=True, transform=test_transform)

    elif dataset_name == 'GTSRB':
        train_ds = datasets.GTSRB(root=root, split='train', download=True, transform=train_transform)
        test_ds  = datasets.GTSRB(root=root, split='test',  download=True, transform=test_transform)

    elif dataset_name == 'Flowers102':
        train_ds = datasets.Flowers102(root=root, split='train', download=True, transform=train_transform)
        test_ds  = datasets.Flowers102(root=root, split='test',  download=True, transform=test_transform)

    elif dataset_name == 'DTD':
        train_ds = datasets.DTD(root=root, split='train', download=True, transform=train_transform)
        test_ds  = datasets.DTD(root=root, split='test',  download=True, transform=test_transform)

    elif dataset_name == 'UCF101':
        # UCF101 requires annotation files; use ImageFolder if pre-organized
        train_ds = datasets.ImageFolder(root=os.path.join(root, 'train'), transform=train_transform)
        test_ds  = datasets.ImageFolder(root=os.path.join(root, 'test'),  transform=test_transform)

    elif dataset_name == 'Food101':
        train_ds = datasets.Food101(root=root, split='train', download=True, transform=train_transform)
        test_ds  = datasets.Food101(root=root, split='test',  download=True, transform=test_transform)

    elif dataset_name == 'SUN397':
        # SUN397 from torchvision
        full_ds = datasets.SUN397(root=root, download=True, transform=train_transform)
        # Split following Chen et al. (2023): use first 15888 for train, rest for test
        n_train = 15888
        indices = list(range(len(full_ds)))
        train_ds = Subset(full_ds, indices[:n_train])
        full_ds_test = datasets.SUN397(root=root, download=True, transform=test_transform)
        test_ds = Subset(full_ds_test, indices[n_train:n_train + 19850])

    elif dataset_name == 'EuroSAT':
        full_ds = datasets.EuroSAT(root=root, download=True, transform=train_transform)
        # Split: 13500 train, 8100 test (following paper)
        n_train = 13500
        indices = list(range(len(full_ds)))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_ds = Subset(full_ds, indices[:n_train])
        full_ds_test = datasets.EuroSAT(root=root, download=True, transform=test_transform)
        test_ds = Subset(full_ds_test, indices[n_train:n_train + 8100])

    elif dataset_name == 'OxfordPets':
        train_ds = datasets.OxfordIIITPet(root=root, split='trainval', download=True, transform=train_transform)
        test_ds  = datasets.OxfordIIITPet(root=root, split='test',     download=True, transform=test_transform)

    elif dataset_name == 'StanfordCars':
        train_ds = datasets.StanfordCars(root=root, split='train', download=True, transform=train_transform)
        test_ds  = datasets.StanfordCars(root=root, split='test',  download=True, transform=test_transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_classes = DATASET_INFO[dataset_name]['num_classes']
    return train_ds, test_ds, num_classes


def get_dataloaders(dataset_name, data_root, batch_size=256, num_workers=4, model_name='ResNet'):
    """Returns (train_loader, test_loader, num_classes)."""
    train_ds, test_ds, num_classes = get_dataset(dataset_name, data_root, model_name)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes
