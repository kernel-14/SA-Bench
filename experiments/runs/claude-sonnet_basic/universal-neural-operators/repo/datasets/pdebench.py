"""
PDEBench dataset interface.

PDEBench (Takamoto et al., 2022) provides a comprehensive benchmark for scientific ML.
This module provides an interface to load PDEBench data for the multi-physics
pretraining scenario described in the paper.

The paper uses PDEBench for the "General multi-physics learning" scenario:
- Pretrain on advection and Burgers' equation
- Fine-tune on reaction-diffusion

PDEBench data can be downloaded from: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Optional, Tuple, List


class PDEBenchDataset(Dataset):
    """
    Interface for PDEBench datasets.
    
    Supports loading HDF5 files from PDEBench for:
    - 1D Advection
    - 1D Burgers
    - 1D Reaction-Diffusion
    
    If PDEBench data is not available, falls back to synthetic data generation.
    """

    SUPPORTED_EQUATIONS = [
        "advection",
        "burgers",
        "reaction_diffusion",
    ]

    def __init__(
        self,
        equation: str,
        data_path: Optional[str] = None,
        split: str = "train",
        n_samples: Optional[int] = None,
        nx: int = 256,
        seed: int = 42,
    ):
        """
        Args:
            equation: One of 'advection', 'burgers', 'reaction_diffusion'
            data_path: Path to PDEBench HDF5 file. If None, uses synthetic data.
            split: 'train', 'val', or 'test'
            n_samples: Number of samples to use (None = all)
            nx: Spatial resolution
            seed: Random seed for synthetic data
        """
        assert equation in self.SUPPORTED_EQUATIONS, \
            f"Equation must be one of {self.SUPPORTED_EQUATIONS}"
        
        self.equation = equation
        self.split = split
        
        if data_path is not None and os.path.exists(data_path):
            self.inputs, self.outputs = self._load_pdebench(data_path, split, n_samples)
        else:
            print(f"PDEBench data not found at {data_path}. Using synthetic data.")
            self.inputs, self.outputs = self._generate_synthetic(equation, n_samples or 1000, nx, seed)
        
        self.inputs = torch.FloatTensor(self.inputs)
        self.outputs = torch.FloatTensor(self.outputs)

    def _load_pdebench(self, data_path: str, split: str, n_samples: Optional[int]):
        """Load data from PDEBench HDF5 file."""
        try:
            import h5py
            with h5py.File(data_path, 'r') as f:
                # PDEBench format: data[split]['u'] contains the solution
                if split in f:
                    u = f[split]['u'][:]
                else:
                    # Try loading all data and splitting
                    u = f['u'][:]
                    n_total = len(u)
                    if split == 'train':
                        u = u[:int(0.8 * n_total)]
                    elif split == 'val':
                        u = u[int(0.8 * n_total):int(0.9 * n_total)]
                    else:
                        u = u[int(0.9 * n_total):]
                
                if n_samples is not None:
                    u = u[:n_samples]
                
                # u shape: (n_samples, nt+1, nx) or (n_samples, nx, nt+1)
                if u.ndim == 3:
                    if u.shape[1] > u.shape[2]:
                        # (n_samples, nt+1, nx) -> use first and last time steps
                        u0 = u[:, 0, :]
                        u_final = u[:, -1, :]
                    else:
                        u0 = u[:, :, 0]
                        u_final = u[:, :, -1]
                
                n_samples_actual = len(u0)
                nx = u0.shape[-1]
                
                # Create inputs with initial condition
                inputs = np.zeros((n_samples_actual, 2, nx))
                inputs[:, 0] = u0
                # Second channel: placeholder for parameter (will be set per equation)
                inputs[:, 1] = 1.0
                
                outputs = u_final[:, np.newaxis, :]
                
                return inputs, outputs
        except Exception as e:
            print(f"Error loading PDEBench data: {e}. Falling back to synthetic.")
            return self._generate_synthetic(self.equation, n_samples or 1000, 256, 42)

    def _generate_synthetic(self, equation: str, n_samples: int, nx: int, seed: int):
        """Generate synthetic data as fallback."""
        if equation == "advection":
            from .advection import generate_advection_data
            return generate_advection_data(n_samples, nx, seed=seed)
        elif equation == "burgers":
            from .burgers import generate_burgers_data
            inputs, outputs, _ = generate_burgers_data(n_samples, nx, seed=seed)
            return inputs, outputs
        elif equation == "reaction_diffusion":
            from .reaction_diffusion import generate_reaction_diffusion_data
            return generate_reaction_diffusion_data(n_samples, nx, seed=seed)
        else:
            raise ValueError(f"Unknown equation: {equation}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    @property
    def n_input(self):
        return self.inputs.shape[1]

    @property
    def n_output(self):
        return self.outputs.shape[1]


class MultiPhysicsDataset(Dataset):
    """
    Combined dataset for multi-physics pretraining.
    
    Combines multiple PDE datasets for simultaneous training.
    Uses adapter-based approach where each physics has its own lifting/projection.
    """

    def __init__(self, datasets: List[Dataset], physics_ids: Optional[List[int]] = None):
        """
        Args:
            datasets: List of datasets for different physics
            physics_ids: Optional list of physics IDs (defaults to 0, 1, 2, ...)
        """
        self.datasets = datasets
        self.physics_ids = physics_ids or list(range(len(datasets)))
        
        # Compute cumulative sizes
        self.cumulative_sizes = []
        total = 0
        for d in datasets:
            total += len(d)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                dataset_idx = i
                break
        
        # Compute local index
        if dataset_idx > 0:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        else:
            local_idx = idx
        
        inputs, outputs = self.datasets[dataset_idx][local_idx]
        physics_id = self.physics_ids[dataset_idx]
        
        return inputs, outputs, physics_id
