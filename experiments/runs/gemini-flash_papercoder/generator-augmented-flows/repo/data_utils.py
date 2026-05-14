# data_utils.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Dict, Any

class DataUtils:
    """
    Utility class for handling dataset loading, preprocessing, and statistics calculation.
    """

    @staticmethod
    def get_dataloader(
        dataset_name: str,
        resolution: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Loads the specified dataset, applies necessary transformations, and returns a DataLoader.

        Args:
            dataset_name (str): Name of the dataset (e.g., "cifar10", "celeba").
            resolution (int): Target image resolution (e.g., 32, 64).
            batch_size (int): Batch size for the DataLoader.
            num_workers (int): Number of worker processes for data loading.
            pin_memory (bool): Whether to pin CPU memory for faster data transfer to GPU.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            torch.utils.data.DataLoader: Configured DataLoader for the dataset.

        Raises:
            ValueError: If an unsupported dataset_name is provided.
        """
        # Define common transformations for all datasets
        # 1. Resize the smaller side to 'resolution' and then center crop to 'resolution x resolution'.
        # 2. Convert to Tensor.
        # 3. Normalize pixel values from [0, 1] to [-1, 1].
        transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(), # Converts to [0, 1] range
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes to [-1, 1]
        ])

        dataset_name_lower = dataset_name.lower().replace(" ", "").replace("-", "")

        if dataset_name_lower == "cifar10":
            dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        elif dataset_name_lower == "imagenet":
            # For ImageNet, assume a pre-downloaded dataset structure, e.g., ImageFolder.
            # The root path should point to the ImageNet training data directory.
            # This requires manual download and structuring of ImageNet.
            raise NotImplementedError(
                f"ImageNet dataset '{dataset_name}' loading requires specific path setup for pre-downloaded data "
                "and is not directly downloadable via torchvision.datasets without a specific version or license."
            )
        elif dataset_name_lower == "celeba":
            dataset = datasets.CelebA(root="./data", split="train", download=True, transform=transform)
        elif dataset_name_lower == "lsunchurch": # This handles both "lsun_church" and "lsunchurch"
            # LSUN loading can be complex. Assuming a specific category, e.g., 'church_outdoor_train'.
            # 'root' needs to be the base path for LSUN.
            dataset = datasets.LSUN(root="./data", classes=['church_outdoor_train'], transform=transform)
        elif dataset_name_lower == "ffhq":
            # FFHQ is not directly in torchvision.datasets. Requires custom dataset or pre-download.
            # This is typically used for the ECT setting, which is not the primary focus of the reproduction plan.
            raise NotImplementedError(
                f"FFHQ dataset '{dataset_name}' loading requires a custom dataset class or pre-downloaded structure."
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True # Ensure all batches have the same size, important for some training setups
        )
        return dataloader

    @staticmethod
    def calculate_sigma_d_squared(dataloader: DataLoader, device: torch.device) -> float:
        """
        Calculates the empirical population variance of the data distribution (sigma_d^2)
        across all pixel values and channels in the dataset.

        The pixel values are assumed to be in the range [-1, 1].

        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            device (torch.device): The device (CPU/GPU) to perform calculations on.

        Returns:
            float: The empirical population variance (sigma_d^2) of the data.
        """
        sum_val = torch.tensor(0.0, device=device)
        sum_sq_val = torch.tensor(0.0, device=device)
        total_pixels = 0

        print(f"Calculating sigma_d^2 for {len(dataloader.dataset)} images...")
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Calculating data variance")):
            images = images.to(device)
            # Flatten the batch of images into a 1D tensor of pixel values
            # Each image (C, H, W) becomes (C * H * W)
            # A batch (B, C, H, W) becomes (B * C * H * W)
            flat_pixels = images.view(-1)

            sum_val += torch.sum(flat_pixels)
            sum_sq_val += torch.sum(flat_pixels ** 2)
            total_pixels += flat_pixels.numel()

        if total_pixels == 0:
            raise ValueError("No data found in dataloader to calculate variance.")
        
        # Calculate population variance: E[X^2] - (E[X])^2
        mean = sum_val / total_pixels
        sigma_d_squared = (sum_sq_val / total_pixels) - (mean ** 2)

        print(f"Calculated sigma_d^2: {sigma_d_squared.item():.6f}")
        return sigma_d_squared.item()

