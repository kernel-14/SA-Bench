"""
Dataset loading and preprocessing for the heterogeneous PDE training corpus.

All datasets are standardized to c3p128 (3 channels, 128×128 spatial, float16)
and stored as HDF5 files with trajectories of length 4.

Dataset families (12 distinct PDE systems):
  FNO-v:     fno_v3, fno_v4, fno_v5
  PDEArena:  pa_ns, pa_nsc, pa_swe
  PDEBench:  pb_cns_low, pb_cns_high, pb_swe
  The Well:  w_am, w_gs, w_swe, w_rb, w_sf, w_tr, w_ve
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from config import DatasetConfig


# ---------------------------------------------------------------------------
# Single-dataset HDF5 loader
# ---------------------------------------------------------------------------

class PDEDataset(Dataset):
    """Loads trajectories from a single HDF5 file.

    Expected HDF5 layout:
        /data  — float16 array of shape (N, T, C, H, W)
                 N: number of trajectories
                 T: trajectory length (≥ traj_len)
                 C: channels (3)
                 H, W: spatial (128)

    Returns windows of length traj_len sampled from each trajectory.
    """

    def __init__(
        self,
        path: str,
        traj_len: int = 4,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.path = path
        self.traj_len = traj_len
        self.normalize = normalize

        with h5py.File(path, "r") as f:
            data = f["data"]
            N, T, C, H, W = data.shape
            self.N = N
            self.T = T
            self.C = C
            self.H = H
            self.W = W

            # Compute per-channel mean and std for normalization
            if normalize:
                sample = data[:min(1000, N)].astype(np.float32)
                self.mean = sample.mean(axis=(0, 1, 3, 4), keepdims=False)  # (C,)
                self.std = sample.std(axis=(0, 1, 3, 4), keepdims=False) + 1e-8
            else:
                self.mean = np.zeros(C, dtype=np.float32)
                self.std = np.ones(C, dtype=np.float32)

        # Build index: (trajectory_idx, start_timestep)
        n_windows = T - traj_len + 1
        all_indices = [
            (i, s) for i in range(N) for s in range(n_windows)
        ]

        # Split
        rng = random.Random(42)
        rng.shuffle(all_indices)
        n_train = int(len(all_indices) * train_ratio)
        n_val = int(len(all_indices) * val_ratio)

        if split == "train":
            self.indices = all_indices[:n_train]
        elif split == "val":
            self.indices = all_indices[n_train:n_train + n_val]
        elif split == "test":
            self.indices = all_indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        traj_idx, start = self.indices[idx]
        with h5py.File(self.path, "r") as f:
            frames = f["data"][traj_idx, start:start + self.traj_len]  # (T, C, H, W)

        frames = frames.astype(np.float32)
        if self.normalize:
            frames = (frames - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        return torch.from_numpy(frames)  # (traj_len, C, H, W)


# ---------------------------------------------------------------------------
# Uniform-sampling multi-dataset wrapper (DPOT-style equal probability)
# ---------------------------------------------------------------------------

class UniformPDEDataset(Dataset):
    """Samples uniformly across all sub-datasets regardless of their sizes.

    Each __getitem__ call:
      1. Picks a dataset uniformly at random.
      2. Picks a random sample from that dataset.

    This matches the "equal probability" sampling in DPOT [13].
    """

    def __init__(self, datasets: List[Dataset]):
        super().__init__()
        self.datasets = datasets
        # Virtual length: max dataset size × number of datasets
        self._len = max(len(d) for d in datasets) * len(datasets)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> torch.Tensor:
        ds = self.datasets[idx % len(self.datasets)]
        inner_idx = torch.randint(len(ds), (1,)).item()
        return ds[inner_idx]


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_dataset(
    cfg: DatasetConfig,
    split: str = "train",
) -> Dataset:
    """Build the unified PDE dataset from all available sub-datasets.

    Scans cfg.root_dir for HDF5 files named <dataset_name>.h5.
    Missing files are silently skipped.
    """
    sub_datasets = []
    for name in cfg.datasets:
        path = os.path.join(cfg.root_dir, f"{name}.h5")
        if not os.path.exists(path):
            continue
        ds = PDEDataset(
            path=path,
            traj_len=cfg.traj_len,
            split=split,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            normalize=True,
        )
        sub_datasets.append(ds)

    if not sub_datasets:
        raise FileNotFoundError(
            f"No HDF5 dataset files found in {cfg.root_dir}. "
            "Expected files named <dataset_name>.h5"
        )

    if split == "train":
        return UniformPDEDataset(sub_datasets)
    else:
        return ConcatDataset(sub_datasets)


def build_dataloader(
    cfg: DatasetConfig,
    split: str = "train",
    batch_size: int = 256,
    num_workers: Optional[int] = None,
) -> DataLoader:
    dataset = build_dataset(cfg, split)
    nw = cfg.num_workers if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=nw,
        pin_memory=cfg.pin_memory,
        drop_last=(split == "train"),
        persistent_workers=(nw > 0),
    )


# ---------------------------------------------------------------------------
# Kolmogorov turbulence dataset (for few-shot finetuning)
# ---------------------------------------------------------------------------

class KolmogorovDataset(Dataset):
    """Kolmogorov flow dataset at Re=222 (u, v velocity fields).

    Expected HDF5 layout:
        /u  — float32 array (N, T, H, W)
        /v  — float32 array (N, T, H, W)

    Stacks u and v as channels, pads to 3 channels with zeros.
    Resizes to 128×128 if needed.
    """

    def __init__(
        self,
        path: str,
        traj_len: int = 4,
        split: str = "train",
        n_train: int = 200,
        n_test: int = 500,
        normalize: bool = True,
    ):
        super().__init__()
        self.path = path
        self.traj_len = traj_len

        with h5py.File(path, "r") as f:
            u = f["u"][:]  # (N, T, H, W)
            v = f["v"][:]  # (N, T, H, W)

        N, T, H, W = u.shape
        # Stack channels: (N, T, 2, H, W), pad to 3 channels
        data = np.stack([u, v, np.zeros_like(u)], axis=2).astype(np.float32)

        # Resize to 128×128 if needed
        if H != 128 or W != 128:
            import torch.nn.functional as F_
            data_t = torch.from_numpy(data).reshape(N * T, 3, H, W)
            data_t = F_.interpolate(data_t, size=(128, 128), mode="bilinear", align_corners=False)
            data = data_t.reshape(N, T, 3, 128, 128).numpy()

        if normalize:
            mean = data.mean(axis=(0, 1, 3, 4))
            std = data.std(axis=(0, 1, 3, 4)) + 1e-8
            data = (data - mean[None, None, :, None, None]) / std[None, None, :, None, None]

        # Build windows
        n_windows = T - traj_len + 1
        all_windows = [
            data[i, s:s + traj_len]
            for i in range(N)
            for s in range(n_windows)
        ]

        rng = random.Random(42)
        rng.shuffle(all_windows)

        if split == "train":
            self.windows = all_windows[:n_train * n_windows]
        elif split == "test":
            self.windows = all_windows[-(n_test * n_windows):]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.windows[idx])  # (traj_len, 3, 128, 128)


# ---------------------------------------------------------------------------
# Collate function: split trajectory into list of frames
# ---------------------------------------------------------------------------

def collate_traj(batch: List[torch.Tensor]) -> List[torch.Tensor]:
    """Stack batch and split into per-frame tensors.

    Args:
        batch: list of (traj_len, C, H, W) tensors

    Returns:
        frames: list of traj_len tensors, each (B, C, H, W)
    """
    stacked = torch.stack(batch, dim=0)  # (B, traj_len, C, H, W)
    return [stacked[:, t] for t in range(stacked.shape[1])]
