"""
Dataset classes for MoE-POT pre-training and fine-tuning.

Handles the 6 pre-training datasets:
- FNO-NS (1e-5, 1e-3): Navier-Stokes with different viscosities
- PDEBench-CNS (0.1, 0.01): Compressible Navier-Stokes
- PDEBench-SWE: Shallow Water Equations
- PDEBench-DR: Diffusion-Reaction
- CFDBench: Computational Fluid Dynamics with irregular geometries

Data preprocessing follows DPOT [15]:
- Spatial resolution standardized to H=128
- Channel padding to max channels across datasets
- Mask channel for irregular geometries
- Balanced sampling across datasets
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class PDEDataset(Dataset):
    """
    Generic PDE dataset that loads pre-processed trajectory data.

    Each sample is a trajectory of T+1 frames: (u^0, ..., u^{T-1}, u^T).
    During training, the model takes (u^0, ..., u^{T-1}) as input and
    predicts u^T.

    Args:
        data_path: Path to the .npy or .npz data file.
        num_input_timesteps: Number of input timesteps T.
        target_resolution: Target spatial resolution H (default 128).
        max_channels: Maximum number of channels for padding.
        dataset_name: Name identifier for this dataset.
        split: 'train' or 'test'.
        train_size: Number of training samples.
        test_size: Number of test samples.
    """

    def __init__(
        self,
        data_path: str,
        num_input_timesteps: int = 10,
        target_resolution: int = 128,
        max_channels: int = 4,
        dataset_name: str = "unknown",
        split: str = "train",
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.num_input_timesteps = num_input_timesteps
        self.target_resolution = target_resolution
        self.max_channels = max_channels
        self.dataset_name = dataset_name
        self.split = split

        # Load data
        self.data = self._load_data(data_path)
        # data shape: (N, T_total, C, H, W) or (N, T_total, H, W) for single-channel

        # Normalize to (N, T_total, C, H, W)
        if self.data.ndim == 4:
            self.data = self.data[:, :, np.newaxis, :, :]

        N, T_total, C, H, W = self.data.shape

        # Split train/test
        if split == "train":
            n = train_size if train_size is not None else int(0.8 * N)
            self.data = self.data[:n]
        else:
            n = test_size if test_size is not None else int(0.2 * N)
            self.data = self.data[-n:]

        self.N, self.T_total, self.C, self.H, self.W = self.data.shape

    def _load_data(self, path: str) -> np.ndarray:
        """Load data from file."""
        if path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".npz"):
            data = np.load(path)
            # Try common keys
            for key in ["data", "u", "solution", "x"]:
                if key in data:
                    return data[key]
            # Use first key
            return data[list(data.keys())[0]]
        elif path.endswith(".h5") or path.endswith(".hdf5"):
            import h5py
            with h5py.File(path, "r") as f:
                for key in ["data", "u", "solution"]:
                    if key in f:
                        return f[key][:]
                return f[list(f.keys())[0]][:]
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def _preprocess_sample(self, u: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single trajectory.

        Args:
            u: (T_total, C, H, W) numpy array.

        Returns:
            Preprocessed tensor (T_total, max_channels, target_H, target_W).
        """
        T, C, H, W = u.shape
        target_H = target_W = self.target_resolution

        # Convert to tensor
        u_tensor = torch.from_numpy(u.astype(np.float32))

        # Resize spatial dimensions if needed
        if H != target_H or W != target_W:
            u_flat = u_tensor.reshape(T * C, 1, H, W)
            u_flat = torch.nn.functional.interpolate(
                u_flat, size=(target_H, target_W), mode="bilinear", align_corners=False
            )
            u_tensor = u_flat.reshape(T, C, target_H, target_W)

        # Pad channels to max_channels
        if C < self.max_channels:
            pad = torch.ones(T, self.max_channels - C, target_H, target_W)
            u_tensor = torch.cat([u_tensor, pad], dim=1)
        elif C > self.max_channels:
            u_tensor = u_tensor[:, :self.max_channels]

        return u_tensor

    def __len__(self) -> int:
        # Each sample can generate multiple (input, target) pairs
        return self.N * max(1, self.T_total - self.num_input_timesteps)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Determine which trajectory and which starting timestep
        n_windows = max(1, self.T_total - self.num_input_timesteps)
        traj_idx = idx // n_windows
        t_start = idx % n_windows

        u = self.data[traj_idx]  # (T_total, C, H, W)
        u_processed = self._preprocess_sample(u)  # (T_total, max_C, H, W)

        # Input: T frames starting at t_start
        t_end = t_start + self.num_input_timesteps
        u_input = u_processed[t_start:t_end]  # (T, max_C, H, W)

        # Target: next frame
        if t_end < self.T_total:
            u_target = u_processed[t_end]  # (max_C, H, W)
        else:
            u_target = u_processed[-1]

        return {
            "input": u_input,
            "target": u_target,
            "dataset_name": self.dataset_name,
        }


class MixedPDEDataset(Dataset):
    """
    Mixed dataset combining multiple PDE datasets with balanced sampling.

    Implements the balanced sampling strategy from the paper:
        p_k = w_k / (K * |D_k| * sum_k w_k)

    where w_k is the importance weight for dataset k.

    Args:
        datasets: List of PDEDataset instances.
        weights: Importance weights for each dataset (default: uniform).
    """

    def __init__(
        self,
        datasets: List[PDEDataset],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.K = len(datasets)

        if weights is None:
            weights = [1.0] * self.K
        self.weights = weights

        # Compute dataset sizes
        self.sizes = [len(d) for d in datasets]
        self.total_size = sum(self.sizes)

        # Compute cumulative indices for indexing
        self.cumulative_sizes = []
        cumsum = 0
        for size in self.sizes:
            self.cumulative_sizes.append(cumsum)
            cumsum += size

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes):
            if i + 1 < self.K and idx >= self.cumulative_sizes[i + 1]:
                continue
            dataset_idx = i
            break

        local_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

    def get_sampling_weights(self) -> torch.Tensor:
        """
        Compute per-sample sampling weights for balanced sampling.

        Returns:
            weights: (total_size,) tensor of sampling weights.
        """
        sample_weights = []
        for k, (dataset, w_k) in enumerate(zip(self.datasets, self.weights)):
            # p_k = w_k / (K * |D_k| * sum_k w_k)
            p_k = w_k / (self.K * len(dataset) * sum(self.weights))
            # Each sample in dataset k gets weight p_k
            sample_weights.extend([p_k] * len(dataset))
        return torch.tensor(sample_weights, dtype=torch.float32)

    def create_balanced_sampler(self) -> WeightedRandomSampler:
        """Create a WeightedRandomSampler for balanced dataset sampling."""
        weights = self.get_sampling_weights()
        return WeightedRandomSampler(
            weights=weights,
            num_samples=self.total_size,
            replacement=True,
        )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for mixed PDE batches."""
    inputs = torch.stack([item["input"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    dataset_names = [item["dataset_name"] for item in batch]
    return {
        "input": inputs,
        "target": targets,
        "dataset_names": dataset_names,
    }


def create_dataloaders(
    dataset_configs: List[Dict],
    num_input_timesteps: int = 10,
    target_resolution: int = 128,
    max_channels: int = 4,
    batch_size: int = 20,
    num_workers: int = 4,
    split: str = "train",
) -> DataLoader:
    """
    Create a DataLoader for mixed PDE pre-training.

    Args:
        dataset_configs: List of dicts with keys:
            - path: str, path to data file
            - name: str, dataset name
            - train_size: int, number of training samples
            - test_size: int, number of test samples
            - weight: float, importance weight (default 1.0)
        num_input_timesteps: Number of input timesteps T.
        target_resolution: Target spatial resolution.
        max_channels: Maximum number of channels.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        split: 'train' or 'test'.

    Returns:
        DataLoader for the mixed dataset.
    """
    datasets = []
    weights = []

    for config in dataset_configs:
        dataset = PDEDataset(
            data_path=config["path"],
            num_input_timesteps=num_input_timesteps,
            target_resolution=target_resolution,
            max_channels=max_channels,
            dataset_name=config.get("name", "unknown"),
            split=split,
            train_size=config.get("train_size"),
            test_size=config.get("test_size"),
        )
        datasets.append(dataset)
        weights.append(config.get("weight", 1.0))

    mixed_dataset = MixedPDEDataset(datasets, weights=weights)

    if split == "train":
        sampler = mixed_dataset.create_balanced_sampler()
        loader = DataLoader(
            mixed_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            mixed_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return loader


# Default dataset configurations for the 6 pre-training datasets
DEFAULT_PRETRAIN_DATASETS = [
    {
        "name": "NS_1e-5",
        "path": "data/fno_ns_1e-5.npy",
        "train_size": 1000,
        "test_size": 200,
        "weight": 1.0,
    },
    {
        "name": "NS_1e-3",
        "path": "data/fno_ns_1e-3.npy",
        "train_size": 1000,
        "test_size": 200,
        "weight": 1.0,
    },
    {
        "name": "CNS_0.1_0.01",
        "path": "data/pdebench_cns_0.1_0.01.npy",
        "train_size": 9000,
        "test_size": 200,
        "weight": 1.0,
    },
    {
        "name": "SWE",
        "path": "data/pdebench_swe.npy",
        "train_size": 900,
        "test_size": 60,
        "weight": 1.0,
    },
    {
        "name": "DR",
        "path": "data/pdebench_dr.npy",
        "train_size": 900,
        "test_size": 60,
        "weight": 1.0,
    },
    {
        "name": "CFDBench",
        "path": "data/cfdbench.npy",
        "train_size": 9000,
        "test_size": 1000,
        "weight": 1.0,
    },
]
