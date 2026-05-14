"""
Dataset loading and preprocessing for CIFAR-10, ImageNet, CelebA, and LSUN Church.
Images are resized, center-cropped, and linearly scaled to [-1, 1].
"""
import os
from typing import Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image


DATASETS = ["cifar10", "imagenet", "celeba", "lsun_church"]


def get_transform(image_size: int) -> transforms.Compose:
    """Standard preprocessing: resize, center crop, and scale to [-1, 1]."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])


class ImageDataset(Dataset):
    """Generic dataset wrapper for image folders."""

    def __init__(
        self,
        root: str,
        image_size: int,
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.image_size = image_size
        self.transform = transform or get_transform(image_size)
        self.image_files = self._get_image_files()

    def _get_image_files(self):
        """Get list of image paths."""
        files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.PNG", "*.webp"]:
            import glob
            files.extend(glob.glob(os.path.join(self.root, "**", ext), recursive=True))
        return sorted(files)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def get_dataset(
    name: str,
    root: str = "./data",
    image_size: int = 32,
    train: bool = True,
) -> Dataset:
    """
    Get dataset by name.

    Args:
        name: Dataset name (cifar10, imagenet, celeba, lsun_church)
        root: Root directory for data storage
        image_size: Target image size
        train: Whether to load training or test set

    Returns:
        PyTorch Dataset
    """
    transform = get_transform(image_size)

    if name == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=True,
        )

    elif name == "imagenet":
        split = "train" if train else "val"
        data_root = os.path.join(root, "imagenet")
        if not os.path.exists(data_root):
            os.makedirs(data_root, exist_ok=True)
        return torchvision.datasets.ImageNet(
            root=data_root,
            split=split,
            transform=transform,
        )

    elif name == "celeba":
        data_root = os.path.join(root, "celeba")
        if not os.path.exists(data_root):
            os.makedirs(data_root, exist_ok=True)
        return torchvision.datasets.CelebA(
            root=data_root,
            split="train" if train else "test",
            transform=transform,
            download=True,
        )

    elif name == "lsun_church":
        data_root = os.path.join(root, "lsun")
        if not os.path.exists(data_root):
            os.makedirs(data_root, exist_ok=True)
        return torchvision.datasets.LSUN(
            root=data_root,
            classes=["church_outdoor_train"] if train else ["church_outdoor_val"],
            transform=transform,
        )

    else:
        raise ValueError(f"Unknown dataset: {name}")


def create_dataloader(
    name: str,
    root: str = "./data",
    image_size: int = 32,
    batch_size: int = 512,
    num_workers: int = 4,
    train: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader for the specified dataset."""
    dataset = get_dataset(name, root, image_size, train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )
