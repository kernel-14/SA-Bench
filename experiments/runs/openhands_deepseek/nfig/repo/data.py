"""
Data loading and preprocessing for ImageNet dataset.
As described in Section 4.1 of the paper.
"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_imagenet_loaders(
    data_path: str,
    image_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create ImageNet train and validation dataloaders.
    Dataset: ILSVRC 2012 subset of ImageNet with 1000 classes.
    """
    train_transform = get_train_transform(image_size)
    val_transform = get_val_transform(image_size)

    train_dataset = datasets.ImageNet(
        root=data_path,
        split="train",
        transform=train_transform,
    )

    val_dataset = datasets.ImageNet(
        root=data_path,
        split="val",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader


class ImageNetSubset(Dataset):
    """Subset of ImageNet for quick experimentation."""

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        image_size: int = 256,
        max_samples: Optional[int] = None,
    ):
        self.transform = (
            get_train_transform(image_size)
            if split == "train"
            else get_val_transform(image_size)
        )
        full_dataset = datasets.ImageNet(root=data_path, split=split)
        self.dataset = full_dataset
        if max_samples is not None:
            import random
            indices = random.sample(range(len(full_dataset)), min(max_samples, len(full_dataset)))
            self.dataset = torch.utils.data.Subset(full_dataset, indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.dataset[idx]
        img = self.transform(img) if hasattr(self.dataset[idx], '__iter__') else img
        if isinstance(self.dataset, torch.utils.data.Subset):
            img, label = self.dataset[idx]
            img = self.transform(img)
        return img, label


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert normalized tensor back to [0, 1] range."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean


class ImageNetDataset:
    """Wrapper for ImageNet dataset with consistent interface."""

    def __init__(
        self,
        data_path: str,
        image_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_loader, self.val_loader = get_imagenet_loaders(
            data_path=data_path,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    @property
    def num_classes(self) -> int:
        return 1000
