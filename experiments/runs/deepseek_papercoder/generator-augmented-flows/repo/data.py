"""
data.py – Dataset loading and preprocessing for consistency model training.

This module provides a DataModule class that abstracts the specifics of
different image datasets used in the paper: CIFAR‑10, ImageNet, CelebA,
and LSUN Church. It applies the standard preprocessing chain described in
the paper: resize the smaller side to the target resolution, center crop
to a square, and linearly scale pixel values to [-1, 1].

The module is self‑contained; it does not depend on other project files.
"""

import os
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class DataModule:
    """
    Handles dataset instantiation, preprocessing, and data loader creation.

    Args:
        dataset_name (str): Identifier of the dataset: "cifar10", "imagenet",
            "celeba", "lsun_church".
        batch_size (int): Number of samples per batch.
        resolution (int): Target image resolution (images are square after
            cropping).
        data_dir (str, optional): Root directory where datasets are stored or
            will be downloaded. Defaults to "./data".

    Attributes:
        dataset_name (str): Name of the dataset.
        batch_size (int): Batch size used for the training loader.
        resolution (int): Side length of the square images.
        data_dir (str): Directory containing the dataset.
        transform (T.Compose): Preprocessing pipeline applied to each image.
        dataset (torch.utils.data.Dataset): The underlying PyTorch dataset object.
        train_loader (DataLoader): DataLoader yielding batches of transformed images
            for training. It shuffles every epoch and drops the last incomplete batch.
        data_shape (Tuple[int, int, int]): Shape of a single image (C, H, W).
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        resolution: int,
        data_dir: str = "./data",
    ) -> None:
        self.dataset_name: str = dataset_name
        self.batch_size: int = batch_size
        self.resolution: int = resolution
        self.data_dir: str = data_dir

        # --------------------------------------------------
        # Preprocessing transforms (identical for all datasets)
        # Resize the smaller side to the target resolution,
        # then centre‑crop to a square, convert to tensor
        # (range [0, 1]), and linearly scale to [-1, 1].
        # --------------------------------------------------
        self.transform: T.Compose = T.Compose(
            [
                T.Resize(resolution, interpolation=InterpolationMode.BILINEAR),
                T.CenterCrop(resolution),
                T.ToTensor(),                          # [0, 255] -> [0.0, 1.0]
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1.0, 1.0]
            ]
        )

        # --------------------------------------------------
        # Instantiate the PyTorch Dataset
        # --------------------------------------------------
        self.dataset: torch.utils.data.Dataset

        if self.dataset_name == "cifar10":
            self.dataset = dset.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transform,
            )
        elif self.dataset_name == "imagenet":
            # ImageFolder expects a directory containing subdirectories for each class
            # The user must place the ImageNet training data under <data_dir>/imagenet/train
            imagenet_path = os.path.join(self.data_dir, "imagenet", "train")
            if not os.path.isdir(imagenet_path):
                raise FileNotFoundError(
                    f"ImageNet training data not found at {imagenet_path}. "
                    "Please download ImageNet and place it accordingly."
                )
            self.dataset = dset.ImageFolder(root=imagenet_path, transform=self.transform)
        elif self.dataset_name == "celeba":
            # CelebA is expected to be at <data_dir>/CelebA
            celeba_path = os.path.join(self.data_dir, "CelebA")
            if not os.path.isdir(celeba_path):
                raise FileNotFoundError(
                    f"CelebA dataset not found at {celeba_path}. "
                    "Please download it and place it accordingly."
                )
            self.dataset = dset.CelebA(
                root=self.data_dir,
                split="train",
                download=False,
                transform=self.transform,
            )
        elif self.dataset_name == "lsun_church":
            # LSUN church_outdoor training set
            # Expects the standard LMDB format in <data_dir>/lsun/church_outdoor_train
            lsun_path = os.path.join(self.data_dir, "lsun")
            if not os.path.isdir(lsun_path):
                raise FileNotFoundError(
                    f"LSUN dataset not found at {lsun_path}. "
                    "Please download LSUN Church and place it accordingly."
                )
            self.dataset = dset.LSUN(
                root=self.data_dir,
                classes="church_outdoor_train",
                transform=self.transform,
            )
        else:
            raise ValueError(
                f"Unknown dataset_name '{self.dataset_name}'. "
                "Choose from 'cifar10', 'imagenet', 'celeba', 'lsun_church'."
            )

        # --------------------------------------------------
        # Create training DataLoader
        # --------------------------------------------------
        # The paper uses shuffling; drop_last=True ensures constant batch size.
        # num_workers=4 and pin_memory=True are sensible defaults; they can be
        # tuned per machine.
        self.train_loader: DataLoader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        # Image shape (C, H, W) – used by other modules
        self.data_shape: Tuple[int, int, int] = (3, self.resolution, self.resolution)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the DataLoader for training.

        The loader provides batches of transformed images in a shuffled order,
        dropping the last incomplete batch if necessary.

        Returns:
            DataLoader: The training dataloader.
        """
        return self.train_loader

    def get_data_shape(self) -> Tuple[int, int, int]:
        """
        Returns the shape of a single image (channels, height, width).

        Returns:
            Tuple[int, int, int]: The shape as (3, resolution, resolution).
        """
        return self.data_shape

