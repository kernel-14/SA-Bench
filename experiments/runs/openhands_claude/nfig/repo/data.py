import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_imagenet_transforms(
    image_size: int = 256,
    is_train: bool = True,
) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def build_imagenet_dataset(
    data_root: str,
    image_size: int = 256,
    split: str = "train",
) -> datasets.ImageFolder:
    """
    Build ImageNet dataset.

    Args:
        data_root: path to ImageNet root (contains 'train' and 'val' subdirs)
        image_size: target image size
        split: 'train' or 'val'

    Returns:
        ImageFolder dataset
    """
    is_train = split == "train"
    split_dir = os.path.join(data_root, split)
    transform = build_imagenet_transforms(image_size, is_train)
    return datasets.ImageFolder(split_dir, transform=transform)


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 8,
    is_train: bool = True,
    distributed: bool = False,
) -> Tuple[DataLoader, Optional[torch.utils.data.distributed.DistributedSampler]]:
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=is_train
        )
        shuffle = False
    else:
        shuffle = is_train

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    return loader, sampler
