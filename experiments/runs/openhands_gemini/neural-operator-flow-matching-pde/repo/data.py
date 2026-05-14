
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
import numpy as np
import os
import random

from config import DataConfig

class DummyPDEDataset(Dataset):
    """
    A dummy dataset that simulates PDE trajectory data.
    Generates random data with specified spatial resolution, channels, and trajectory length.
    """
    def __init__(self,
                 num_samples: int,
                 spatial_resolution: int = DataConfig.SPATIAL_RESOLUTION,
                 num_channels: int = DataConfig.NUM_CHANNELS,
                 trajectory_length: int = DataConfig.TRAJECTORY_LENGTH,
                 precision: str = DataConfig.PRECISION):
        super().__init__()
        self.num_samples = num_samples
        self.spatial_resolution = spatial_resolution
        self.num_channels = num_channels
        self.trajectory_length = trajectory_length
        
        if precision == "float16":
            self.dtype = torch.float16
        elif precision == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unsupported precision: {precision}. Choose 'float16' or 'float32'.")

        print(f"Initialized DummyPDEDataset with {num_samples} samples, "
              f"resolution={spatial_resolution}x{spatial_resolution}, channels={num_channels}, "
              f"trajectory_length={trajectory_length}, dtype={self.dtype}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a single trajectory of states.
        Shape: (trajectory_length, num_channels, spatial_resolution, spatial_resolution)
        """
        # Simulate a trajectory of `trajectory_length` states
        trajectory = torch.randn(
            self.trajectory_length,
            self.num_channels,
            self.spatial_resolution,
            self.spatial_resolution,
            dtype=self.dtype
        )
        return trajectory

class HeterogeneousPDEDataLoader:
    """
    Manages loading and sampling from multiple PDE datasets.
    Simulates the heterogeneous dataset described in the paper by combining multiple
    DummyPDEDatasets and sampling from them with equal probabilities.
    """
    def __init__(self,
                 dataset_configs: Dict[str, Dict],
                 batch_size: int,
                 num_workers: int = 0,
                 seed: int = 42):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.datasets: Dict[str, Dataset] = {}
        self.data_loaders: Dict[str, DataLoader] = {}
        self.dataset_names = list(dataset_configs.keys())

        print("Initializing HeterogeneousPDEDataLoader with dummy datasets...")
        for name, config in dataset_configs.items():
            print(f"  - Creating dummy dataset for {name} with {config['num_samples']} samples.")
            dataset = DummyPDEDataset(**config)
            self.datasets[name] = dataset
            self.data_loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size // len(self.dataset_names), # Distribute batch size among datasets
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
        self.data_loader_iterators = {name: iter(loader) for name, loader in self.data_loaders.items()}
        print("HeterogeneousPDEDataLoader initialized.")

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        """
        Samples a batch from one of the datasets with equal probability.
        The batch will contain a trajectory of 4 states: x_0, x_1, x_2, x_3.
        For training FMT, we typically need (x_0, x_1), (x_1, x_2), (x_2, x_3).
        """
        # Sample a dataset uniformly
        sampled_dataset_name = random.choice(self.dataset_names)
        
        try:
            batch = next(self.data_loader_iterators[sampled_dataset_name])
        except StopIteration:
            # If a dataloader is exhausted, re-initialize its iterator
            self.data_loaders[sampled_dataset_name] = DataLoader(
                self.datasets[sampled_dataset_name],
                batch_size=self.batch_size // len(self.dataset_names),
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
            self.data_loader_iterators[sampled_dataset_name] = iter(self.data_loaders[sampled_dataset_name])
            batch = next(self.data_loader_iterators[sampled_dataset_name])
        
        return batch # batch is (sub_batch_size, trajectory_length, C, H, W)

    def get_train_dataloader(self):
        # This class itself acts as the training dataloader by implementing __iter__ and __next__
        return self

    def get_eval_dataloader(self, dataset_name: str, batch_size: int = None):
        """
        Returns a DataLoader for evaluation on a specific dataset.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found.")
        
        eval_batch_size = batch_size if batch_size is not None else self.batch_size
        return DataLoader(
            self.datasets[dataset_name],
            batch_size=eval_batch_size,
            shuffle=False, # No need to shuffle for evaluation
            num_workers=self.num_workers,
            pin_memory=True
        )

# Example Usage (to be used in train.py)
def get_pde_data_loader(batch_size: int, num_workers: int) -> HeterogeneousPDEDataLoader:
    # This dictionary would typically be generated from parsing actual dataset directories
    # Here, we create dummy configurations for 12 PDE families, simulating the paper's description
    pde_dataset_configs = {
        "FNO-v5": {"num_samples": 15400, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "FNO-v4": {"num_samples": 368000, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "FNO-v3": {"num_samples": 184000, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "PA-NS": {"num_samples": 48000, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "PA-NSC": {"num_samples": 120000, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "PA-SWE": {"num_samples": 470000, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "PB-CNS": {"num_samples": 598000, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "PB-SWE": {"num_samples": 77600, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "W-GS": {"num_samples": 92200, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "W-AM": {"num_samples": 13400, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "W-SWE": {"num_samples": 96400, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
        "W-RB": {"num_samples": 266600, "spatial_resolution": 128, "num_channels": 3, "trajectory_length": 4, "precision": "float16"},
    }
    
    # Adjust total samples to reflect actual paper numbers (approx 2.5M trajectories)
    # The dummy num_samples are illustrative, for a true reproduction, this would be based on actual data sizes.
    # The current numbers sum up to ~2.5M.

    return HeterogeneousPDEDataLoader(
        dataset_configs=pde_dataset_configs,
        batch_size=batch_size,
        num_workers=num_workers
    )

