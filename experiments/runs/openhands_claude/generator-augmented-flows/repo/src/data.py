import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder


class ImageDataset(Dataset):
    """Generic image dataset wrapper with preprocessing."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]
        return img


def get_cifar10(root: str, train: bool = True) -> Dataset:
    """
    CIFAR-10 dataset at 32x32 resolution.
    Pixel values linearly scaled to [-1, 1].
    """
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]) if train else T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = CIFAR10(root=root, train=train, download=True, transform=transform)
    return ImageDataset(dataset)


def get_imagenet(root: str, resolution: int = 32, train: bool = True) -> Dataset:
    """
    ImageNet dataset resized to resolution x resolution.
    Preprocessing: resize smaller side, center crop, scale to [-1, 1].
    """
    split = "train" if train else "val"
    data_dir = os.path.join(root, split)

    transforms_list = [
        T.Resize(resolution),
        T.CenterCrop(resolution),
    ]
    if train:
        transforms_list.insert(0, T.RandomHorizontalFlip())
    transforms_list += [
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    transform = T.Compose(transforms_list)
    dataset = ImageFolder(root=data_dir, transform=transform)
    return ImageDataset(dataset)


def get_celeba(root: str, resolution: int = 64, train: bool = True) -> Dataset:
    """
    CelebA dataset at 64x64 resolution.
    Preprocessing: resize smaller side to resolution, center crop, scale to [-1, 1].
    """
    split = "train" if train else "valid"
    transforms_list = [
        T.Resize(resolution),
        T.CenterCrop(resolution),
    ]
    if train:
        transforms_list.insert(0, T.RandomHorizontalFlip())
    transforms_list += [
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    transform = T.Compose(transforms_list)
    dataset = torchvision.datasets.CelebA(
        root=root, split=split, download=True, transform=transform
    )
    return ImageDataset(dataset)


def get_lsun_church(root: str, resolution: int = 64, train: bool = True) -> Dataset:
    """
    LSUN Church dataset at 64x64 resolution.
    Preprocessing: resize smaller side to resolution, center crop, scale to [-1, 1].
    """
    classes = ["church_outdoor_train"] if train else ["church_outdoor_val"]
    transforms_list = [
        T.Resize(resolution),
        T.CenterCrop(resolution),
    ]
    if train:
        transforms_list.insert(0, T.RandomHorizontalFlip())
    transforms_list += [
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    transform = T.Compose(transforms_list)
    dataset = torchvision.datasets.LSUN(root=root, classes=classes, transform=transform)
    return ImageDataset(dataset)


def get_dataset(name: str, root: str, resolution: int, train: bool = True) -> Dataset:
    """
    Factory function to get a dataset by name.

    Args:
        name: dataset name ('cifar10', 'imagenet', 'celeba', 'lsun_church')
        root: path to dataset root directory
        resolution: target image resolution
        train: whether to load training or validation split

    Returns:
        Dataset instance
    """
    name = name.lower()
    if name == "cifar10":
        return get_cifar10(root, train=train)
    elif name == "imagenet":
        return get_imagenet(root, resolution=resolution, train=train)
    elif name == "celeba":
        return get_celeba(root, resolution=resolution, train=train)
    elif name in ("lsun_church", "lsun"):
        return get_lsun_church(root, resolution=resolution, train=train)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader for the given dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


class InfiniteDataLoader:
    """Wraps a DataLoader to provide infinite iteration."""

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self._iterator = iter(dataloader)

    def __next__(self) -> torch.Tensor:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            return next(self._iterator)

    def __iter__(self):
        return self
