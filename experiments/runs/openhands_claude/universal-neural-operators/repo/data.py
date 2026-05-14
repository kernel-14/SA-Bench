from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# Normalizer utilities
# ---------------------------------------------------------------------------

class UnitGaussianNormalizer:
    """Normalize data to zero mean and unit variance (per-channel)."""

    def __init__(self, x: torch.Tensor, eps: float = 1e-5) -> None:
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True) + eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


class RangeNormalizer:
    """Normalize data to [0, 1] range (per-channel)."""

    def __init__(self, x: torch.Tensor, eps: float = 1e-5) -> None:
        self.min = x.min(dim=0, keepdim=True).values
        self.max = x.max(dim=0, keepdim=True).values
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min.to(x.device)) / (self.max.to(x.device) - self.min.to(x.device) + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.max.to(x.device) - self.min.to(x.device) + self.eps) + self.min.to(x.device)


# ---------------------------------------------------------------------------
# Base PDEBench dataset
# ---------------------------------------------------------------------------

class PDEBenchDataset(Dataset):
    """
    Base class for PDEBench HDF5 datasets.
    PDEBench stores data as (n_samples, *spatial, n_vars) or (n_samples, T, *spatial, n_vars).
    """

    def __init__(
        self,
        file_path: str,
        input_keys: List[str],
        output_keys: List[str],
        n_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.file_path = file_path
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.normalize = normalize

        self._load(n_samples)

    def _load(self, n_samples: Optional[int]) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Burgers' equation dataset (1-D, from PDEBench)
# File: 1D/Burgers/Train/1D_Burgers_Sols_Nu{nu}.hdf5
# Keys: tensor (n_samples, T, x)
# ---------------------------------------------------------------------------

class BurgersDataset(Dataset):
    """
    1-D Burgers' equation dataset from PDEBench.
    Input: initial condition u(x, t=0)
    Output: solution u(x, t=T)

    HDF5 structure: 'tensor' of shape (N, T, x)
    """

    def __init__(
        self,
        file_path: str,
        t_in: int = 10,
        t_out: int = 40,
        n_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.t_in = t_in
        self.t_out = t_out
        self.normalize = normalize

        with h5py.File(file_path, "r") as f:
            data = torch.tensor(np.array(f["tensor"]), dtype=torch.float32)

        if n_samples is not None:
            data = data[:n_samples]

        self.n_samples = data.shape[0]
        # data: (N, T, x)
        self.inputs = data[:, :t_in, :]   # (N, t_in, x)
        self.outputs = data[:, t_out, :]  # (N, x)

        if normalize:
            self.in_normalizer = UnitGaussianNormalizer(self.inputs)
            self.out_normalizer = UnitGaussianNormalizer(self.outputs)
            self.inputs = self.in_normalizer.encode(self.inputs)
            self.outputs = self.out_normalizer.encode(self.outputs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs: (t_in, x) → (t_in, x) as channels
        # outputs: (x,) → (1, x)
        x = self.inputs[idx]   # (t_in, x)
        y = self.outputs[idx].unsqueeze(0)  # (1, x)
        return x, y


# ---------------------------------------------------------------------------
# Gray-Scott reaction-diffusion dataset (2-D, from PDEBench)
# File: 2D/Gray_Scott/Train/2D_Gray_Scott_Sols_{params}.hdf5
# Keys: 'tensor' of shape (N, T, H, W, 2) — two species u, v
# ---------------------------------------------------------------------------

class GrayScottDataset(Dataset):
    """
    2-D Gray-Scott reaction-diffusion dataset from PDEBench.
    Input: initial conditions for both species (u, v) at t=0
    Output: solution (u, v) at t=T
    """

    def __init__(
        self,
        file_path: str,
        t_in: int = 1,
        t_out: int = -1,
        n_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.normalize = normalize

        with h5py.File(file_path, "r") as f:
            data = torch.tensor(np.array(f["tensor"]), dtype=torch.float32)

        if n_samples is not None:
            data = data[:n_samples]

        self.n_samples = data.shape[0]
        # data: (N, T, H, W, 2)
        T = data.shape[1]
        t_out_idx = t_out if t_out >= 0 else T - 1

        # Input: first t_in timesteps, both species → (N, 2*t_in, H, W)
        inp = data[:, :t_in, :, :, :]  # (N, t_in, H, W, 2)
        inp = inp.permute(0, 1, 4, 2, 3)  # (N, t_in, 2, H, W)
        self.inputs = inp.reshape(self.n_samples, t_in * 2, data.shape[2], data.shape[3])

        # Output: both species at t_out → (N, 2, H, W)
        self.outputs = data[:, t_out_idx, :, :, :].permute(0, 3, 1, 2)  # (N, 2, H, W)

        if normalize:
            self.in_normalizer = UnitGaussianNormalizer(self.inputs)
            self.out_normalizer = UnitGaussianNormalizer(self.outputs)
            self.inputs = self.in_normalizer.encode(self.inputs)
            self.outputs = self.out_normalizer.encode(self.outputs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


# ---------------------------------------------------------------------------
# Navier-Stokes dataset (2-D, from PDEBench)
# File: 2D/NS_Incom/Train/NS_Re{Re}_N{N}_T{T}.hdf5
# Keys: 'Vx', 'Vy', 'P' of shape (N, T, H, W)
# ---------------------------------------------------------------------------

class NavierStokesDataset(Dataset):
    """
    2-D incompressible Navier-Stokes dataset from PDEBench.
    Input: velocity fields (Vx, Vy) and pressure P at t=0..t_in
    Output: velocity fields and pressure at t=t_out
    """

    def __init__(
        self,
        file_path: str,
        t_in: int = 10,
        t_out: int = 20,
        n_samples: Optional[int] = None,
        normalize: bool = True,
        include_pressure: bool = True,
    ) -> None:
        self.normalize = normalize
        self.include_pressure = include_pressure

        with h5py.File(file_path, "r") as f:
            vx = torch.tensor(np.array(f["Vx"]), dtype=torch.float32)
            vy = torch.tensor(np.array(f["Vy"]), dtype=torch.float32)
            if include_pressure and "P" in f:
                p = torch.tensor(np.array(f["P"]), dtype=torch.float32)
            else:
                p = None

        if n_samples is not None:
            vx = vx[:n_samples]
            vy = vy[:n_samples]
            if p is not None:
                p = p[:n_samples]

        self.n_samples = vx.shape[0]
        T = vx.shape[1]
        t_out_idx = min(t_out, T - 1)

        # Build input tensor: (N, n_vars * t_in, H, W)
        fields_in = [vx[:, :t_in], vy[:, :t_in]]
        if p is not None:
            fields_in.append(p[:, :t_in])
        self.inputs = torch.cat(fields_in, dim=1)  # (N, n_vars*t_in, H, W)

        # Build output tensor: (N, n_vars, H, W)
        fields_out = [vx[:, t_out_idx], vy[:, t_out_idx]]
        if p is not None:
            fields_out.append(p[:, t_out_idx])
        self.outputs = torch.stack(fields_out, dim=1)  # (N, n_vars, H, W)

        if normalize:
            self.in_normalizer = UnitGaussianNormalizer(self.inputs)
            self.out_normalizer = UnitGaussianNormalizer(self.outputs)
            self.inputs = self.in_normalizer.encode(self.inputs)
            self.outputs = self.out_normalizer.encode(self.outputs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


# ---------------------------------------------------------------------------
# Heat equation dataset (2-D, from PDEBench)
# File: 2D/Heat/Train/2D_Heat_Sols_{params}.hdf5
# Keys: 'tensor' of shape (N, T, H, W)
# ---------------------------------------------------------------------------

class HeatDataset(Dataset):
    """
    2-D heat equation dataset from PDEBench.
    Input: initial temperature field
    Output: temperature field at t=T
    """

    def __init__(
        self,
        file_path: str,
        t_in: int = 1,
        t_out: int = -1,
        n_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.normalize = normalize

        with h5py.File(file_path, "r") as f:
            data = torch.tensor(np.array(f["tensor"]), dtype=torch.float32)

        if n_samples is not None:
            data = data[:n_samples]

        self.n_samples = data.shape[0]
        T = data.shape[1]
        t_out_idx = t_out if t_out >= 0 else T - 1

        self.inputs = data[:, :t_in, :, :]   # (N, t_in, H, W)
        self.outputs = data[:, t_out_idx, :, :].unsqueeze(1)  # (N, 1, H, W)

        if normalize:
            self.in_normalizer = UnitGaussianNormalizer(self.inputs)
            self.out_normalizer = UnitGaussianNormalizer(self.outputs)
            self.inputs = self.in_normalizer.encode(self.inputs)
            self.outputs = self.out_normalizer.encode(self.outputs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


# ---------------------------------------------------------------------------
# Advection dataset (1-D, from PDEBench)
# File: 1D/Advection/Train/1D_Advection_Sols_beta{beta}.hdf5
# Keys: 'tensor' of shape (N, T, x)
# ---------------------------------------------------------------------------

class AdvectionDataset(Dataset):
    """
    1-D advection equation dataset from PDEBench.
    Input: initial condition u(x, t=0)
    Output: solution u(x, t=T)
    """

    def __init__(
        self,
        file_path: str,
        t_in: int = 1,
        t_out: int = -1,
        n_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.normalize = normalize

        with h5py.File(file_path, "r") as f:
            data = torch.tensor(np.array(f["tensor"]), dtype=torch.float32)

        if n_samples is not None:
            data = data[:n_samples]

        self.n_samples = data.shape[0]
        T = data.shape[1]
        t_out_idx = t_out if t_out >= 0 else T - 1

        self.inputs = data[:, :t_in, :]   # (N, t_in, x)
        self.outputs = data[:, t_out_idx, :].unsqueeze(1)  # (N, 1, x)

        if normalize:
            self.in_normalizer = UnitGaussianNormalizer(self.inputs)
            self.out_normalizer = UnitGaussianNormalizer(self.outputs)
            self.inputs = self.in_normalizer.encode(self.inputs)
            self.outputs = self.out_normalizer.encode(self.outputs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


# ---------------------------------------------------------------------------
# Reaction-diffusion with advection (extended input set scenario)
# Adds advection velocity field as additional input channel
# ---------------------------------------------------------------------------

class ReactionDiffusionAdvectionDataset(Dataset):
    """
    Extended reaction-diffusion dataset with advection term.
    Used for the 'input function set extension' experiment.

    Combines Gray-Scott data with an advection velocity field
    as an additional input channel.
    """

    def __init__(
        self,
        rd_file_path: str,
        adv_file_path: str,
        t_in: int = 1,
        t_out: int = -1,
        n_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.normalize = normalize

        # Load reaction-diffusion data
        with h5py.File(rd_file_path, "r") as f:
            rd_data = torch.tensor(np.array(f["tensor"]), dtype=torch.float32)

        # Load advection data (used as additional forcing/velocity field)
        with h5py.File(adv_file_path, "r") as f:
            adv_data = torch.tensor(np.array(f["tensor"]), dtype=torch.float32)

        if n_samples is not None:
            rd_data = rd_data[:n_samples]
            adv_data = adv_data[:n_samples]

        self.n_samples = min(rd_data.shape[0], adv_data.shape[0])
        rd_data = rd_data[:self.n_samples]
        adv_data = adv_data[:self.n_samples]

        T_rd = rd_data.shape[1]
        t_out_idx = t_out if t_out >= 0 else T_rd - 1

        # RD input: (N, t_in, H, W, 2) → (N, 2*t_in, H, W)
        rd_inp = rd_data[:, :t_in, :, :, :].permute(0, 1, 4, 2, 3)
        H, W = rd_inp.shape[3], rd_inp.shape[4]
        rd_inp = rd_inp.reshape(self.n_samples, t_in * 2, H, W)

        # Advection input: (N, t_in, x) → interpolate to 2D if needed
        # For simplicity, treat as a 1D field broadcast to 2D
        adv_inp = adv_data[:, :t_in, :]  # (N, t_in, x)
        if adv_inp.shape[-1] != H:
            adv_inp = torch.nn.functional.interpolate(
                adv_inp.unsqueeze(1), size=(t_in, H), mode="bilinear", align_corners=False
            ).squeeze(1)
        adv_inp = adv_inp.unsqueeze(-1).expand(-1, -1, -1, W)  # (N, t_in, H, W)

        # Concatenate: (N, 2*t_in + t_in, H, W)
        self.inputs = torch.cat([rd_inp, adv_inp], dim=1)
        self.outputs = rd_data[:, t_out_idx, :, :, :].permute(0, 3, 1, 2)  # (N, 2, H, W)

        if normalize:
            self.in_normalizer = UnitGaussianNormalizer(self.inputs)
            self.out_normalizer = UnitGaussianNormalizer(self.outputs)
            self.inputs = self.in_normalizer.encode(self.inputs)
            self.outputs = self.out_normalizer.encode(self.outputs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


# ---------------------------------------------------------------------------
# Heat equation with convection (extended input set scenario)
# ---------------------------------------------------------------------------

class HeatConvectionDataset(Dataset):
    """
    Heat equation extended with convection term.
    Adds a velocity field as additional input channel.
    Used for the 'input function set extension' experiment.
    """

    def __init__(
        self,
        heat_file_path: str,
        velocity_file_path: str,
        t_in: int = 1,
        t_out: int = -1,
        n_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.normalize = normalize

        with h5py.File(heat_file_path, "r") as f:
            heat_data = torch.tensor(np.array(f["tensor"]), dtype=torch.float32)

        with h5py.File(velocity_file_path, "r") as f:
            vel_data = torch.tensor(np.array(f["tensor"]), dtype=torch.float32)

        if n_samples is not None:
            heat_data = heat_data[:n_samples]
            vel_data = vel_data[:n_samples]

        self.n_samples = min(heat_data.shape[0], vel_data.shape[0])
        heat_data = heat_data[:self.n_samples]
        vel_data = vel_data[:self.n_samples]

        T = heat_data.shape[1]
        t_out_idx = t_out if t_out >= 0 else T - 1

        heat_inp = heat_data[:, :t_in, :, :]  # (N, t_in, H, W)
        H, W = heat_inp.shape[2], heat_inp.shape[3]

        # Velocity field as additional input
        vel_inp = vel_data[:, :t_in, :, :]  # (N, t_in, H, W) or needs reshape
        if vel_inp.shape[2] != H or vel_inp.shape[3] != W:
            vel_inp = torch.nn.functional.interpolate(
                vel_inp.view(-1, 1, vel_inp.shape[2], vel_inp.shape[3]),
                size=(H, W), mode="bilinear", align_corners=False
            ).view(self.n_samples, t_in, H, W)

        self.inputs = torch.cat([heat_inp, vel_inp], dim=1)  # (N, 2*t_in, H, W)
        self.outputs = heat_data[:, t_out_idx, :, :].unsqueeze(1)  # (N, 1, H, W)

        if normalize:
            self.in_normalizer = UnitGaussianNormalizer(self.inputs)
            self.out_normalizer = UnitGaussianNormalizer(self.outputs)
            self.inputs = self.in_normalizer.encode(self.inputs)
            self.outputs = self.out_normalizer.encode(self.outputs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


# ---------------------------------------------------------------------------
# Multi-physics dataset: combines multiple physics datasets for pre-training
# ---------------------------------------------------------------------------

class MultiPhysicsDataset(Dataset):
    """
    Combines multiple physics datasets for simultaneous pre-training.
    Returns (inputs, outputs, physics_name) tuples.
    """

    def __init__(self, datasets: Dict[str, Dataset]) -> None:
        self.datasets = datasets
        self.physics_names = list(datasets.keys())
        self.lengths = [len(d) for d in datasets.values()]
        self.cumulative = np.cumsum([0] + self.lengths)
        self.total = sum(self.lengths)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        for i, (start, end) in enumerate(zip(self.cumulative[:-1], self.cumulative[1:])):
            if start <= idx < end:
                local_idx = idx - start
                x, y = list(self.datasets.values())[i][local_idx]
                return x, y, self.physics_names[i]
        raise IndexError(f"Index {idx} out of range")


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "burgers_1d": BurgersDataset,
    "gray_scott_2d": GrayScottDataset,
    "navier_stokes_2d": NavierStokesDataset,
    "heat_2d": HeatDataset,
    "advection_1d": AdvectionDataset,
    "rd_advection_2d": ReactionDiffusionAdvectionDataset,
    "heat_convection_2d": HeatConvectionDataset,
}


def build_dataset(
    dataset_type: str,
    file_path: str,
    split: str = "train",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> Dataset:
    """
    Build and split a dataset.

    Args:
        dataset_type: key in DATASET_REGISTRY
        file_path: path to HDF5 file (or primary file for multi-file datasets)
        split: 'train', 'val', or 'test'
        train_ratio: fraction for training
        val_ratio: fraction for validation
        seed: random seed for splitting
        **kwargs: passed to dataset constructor
    """
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_REGISTRY.keys())}")

    full_dataset = DATASET_REGISTRY[dataset_type](file_path, **kwargs)
    n = len(full_dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test], generator=generator)

    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    return splits[split]


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def get_n_in_out(dataset_type: str, t_in: int = 1) -> Tuple[int, int]:
    """Return (n_in, n_out) for a given dataset type."""
    mapping = {
        "burgers_1d": (t_in, 1),
        "gray_scott_2d": (2 * t_in, 2),
        "navier_stokes_2d": (3 * t_in, 3),
        "heat_2d": (t_in, 1),
        "advection_1d": (t_in, 1),
        "rd_advection_2d": (3 * t_in, 2),
        "heat_convection_2d": (2 * t_in, 1),
    }
    return mapping[dataset_type]
