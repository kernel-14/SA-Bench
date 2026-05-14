"""Dataset loading, preprocessing, and flow marching sampling.

Supports 12 PDE families from:
  - FNO-v (v5, v4, v3)
  - PDEArena (NS, NSC, SWE)
  - PDEBench (CNSL, CNSH, SWE)
  - The Well (AM, GS, SWE, RB, SF, TR, VE)

All data is unified to c3p128 format with float16 precision.
"""

import os
import random
from typing import Dict, Optional, Tuple, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class UniformPDEDataset(Dataset):
    """Dataset wrapper for a single PDE family in c3p128 format.

    Expects data stored as .h5 files with 'trajectories' dataset
    of shape [N_traj, T, C, H, W].
    """

    def __init__(
        self,
        data_path: str,
        name: str,
        trajectory_length: int = 5,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        self.name = name
        self.trajectory_length = trajectory_length

        with h5py.File(data_path, "r") as f:
            data = f["trajectories"][:]

        n_traj = len(data)
        indices = list(range(n_traj))
        random.seed(42)
        random.shuffle(indices)

        train_end = int(train_ratio * n_traj)
        val_end = train_end + int(val_ratio * n_traj)

        if split == "train":
            indices = indices[:train_end]
        elif split == "val":
            indices = indices[train_end:val_end]
        elif split == "test":
            indices = indices[val_end:]

        self.data = torch.from_numpy(data[indices]).float()

    def __len__(self) -> int:
        return len(self.data) * (self.data.shape[1] - self.trajectory_length + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        traj_idx = idx // (self.data.shape[1] - self.trajectory_length + 1)
        start = idx % (self.data.shape[1] - self.trajectory_length + 1)
        return self.data[traj_idx, start:start + self.trajectory_length]


class MultiPDEDataset(Dataset):
    """Uniformly samples from multiple PDE datasets.

    Following DPOT practice: datasets are sampled with equal probabilities.
    """

    def __init__(
        self,
        datasets: List[UniformPDEDataset],
        samples_per_epoch: int = 1_000_000,
    ):
        super().__init__()
        self.datasets = datasets
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
        ds = random.choice(self.datasets)
        local_idx = random.randint(0, len(ds) - 1)
        return ds[local_idx]


class FlowMarchingDataset(Dataset):
    """Produces flow marching training samples from trajectory chunks.

    For each trajectory chunk of length T+1 (T=4 future frames),
    returns:
    - x0, x1, x2, x3, x4: states at physical timesteps
    - k0, k1, k2, k3: bridge parameters (sampled uniformly)
    - t0, t1, t2, t3: flow times (sampled uniformly)
    - x_{s, t_s}^{k_s}: interpolated noisy states for s=0,1,2,3
    """

    def __init__(
        self,
        source_dataset: Dataset,
        num_frames: int = 4,
        k_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        self.source_dataset = source_dataset
        self.num_frames = num_frames
        self.k_range = k_range

    def __len__(self) -> int:
        return len(self.source_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.source_dataset[idx]  # [T, C, H, W], T = num_frames + 1
        x = traj[:self.num_frames + 1]  # [5, C, H, W]

        batch = {}
        for s in range(self.num_frames):
            batch[f"x{s}"] = x[s]
            batch[f"x{s+1}"] = x[s + 1]
            t = random.random()
            k = random.uniform(*self.k_range)
            batch[f"t{s}"] = torch.tensor(t)
            batch[f"k{s}"] = torch.tensor(k)

            z = torch.randn_like(x[s])
            x_t_k = t * x[s + 1] + k * (1 - t) * x[s] + (1 - t) * (1 - k) * z
            batch[f"x_tk_{s}"] = x_t_k

        return batch


def build_dataloaders(
    data_config,
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    """Build dataloader for the specified split across all PDE datasets."""
    data_root = data_config.data_root
    datasets = []

    for ds_name in data_config.datasets:
        data_path = os.path.join(data_root, f"{ds_name}.h5")
        if not os.path.exists(data_path):
            continue
        ds = UniformPDEDataset(
            data_path=data_path,
            name=ds_name,
            trajectory_length=data_config.trajectory_length,
            split=split,
            train_ratio=data_config.train_split,
            val_ratio=data_config.val_split,
        )
        datasets.append(ds)

    if not datasets:
        raise FileNotFoundError(f"No datasets found in {data_root}")

    multi_ds = MultiPDEDataset(datasets)
    flow_ds = FlowMarchingDataset(multi_ds, num_frames=data_config.trajectory_length - 1)

    return DataLoader(
        flow_ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def prepare_flow_marching_batch(
    batch: Dict[str, torch.Tensor],
    device: str = "cuda",
    num_frames: int = 4,
) -> Dict[str, torch.Tensor]:
    """Move batch to device and organize into training format.

    Returns dict with:
    - x0, x1, x2, x3, x4: clean states [B, C, H, W]
    - y0, y1, y2, y3: noisy latent states [B, C_lat, H_lat, W_lat]
    - t0...t3: flow times [B]
    - k0...k3: bridge params [B]
    """
    out = {}
    for key, val in batch.items():
        out[key] = val.to(device)
    return out


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_l2re(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2 Relative Error: ||pred - target||_2 / ||target||_2."""
    diff = torch.norm(pred - target, p=2, dim=(-3, -2, -1))
    norm = torch.norm(target, p=2, dim=(-3, -2, -1))
    return (diff / (norm + 1e-8)).mean()


def compute_vrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Variance-Normalized RMSE.

    VRMSE = RMSE / std(target), computed per-channel and averaged.
    """
    se = (pred - target).pow(2).mean(dim=(-2, -1))  # [B, C]
    var = target.var(dim=(-2, -1), unbiased=False)  # [B, C]
    rmse = se.sqrt()
    vrmse = (rmse / (var.sqrt() + 1e-8)).mean()
    return vrmse
