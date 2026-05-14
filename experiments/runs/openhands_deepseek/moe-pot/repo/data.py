import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Optional, Tuple
import scipy.io as sio
import h5py


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_INFO: Dict[str, dict] = {
    "NS_1e-5": {
        "source": "FNO",
        "pde": "Navier-Stokes",
        "viscosity": 1e-5,
        "n_train": 1000,
        "n_test": 200,
        "n_channels": 1,
        "resolution": (64, 64),
        "time_steps": 20,
        "params": {"nu": 1e-5}
    },
    "NS_1e-3": {
        "source": "FNO",
        "pde": "Navier-Stokes",
        "viscosity": 1e-3,
        "n_train": 1000,
        "n_test": 200,
        "n_channels": 1,
        "resolution": (64, 64),
        "time_steps": 20,
        "params": {"nu": 1e-3}
    },
    "CNS_0.1_0.01": {
        "source": "PDEBench",
        "pde": "CompressibleNavierStokes",
        "n_train": 9000,
        "n_test": 200,
        "n_channels": 3,  # velocity (2), pressure, density
        "resolution": (128, 128),
        "time_steps": 21,
        "params": {"eta": 0.1, "zeta": 0.01}
    },
    "SWE": {
        "source": "PDEBench",
        "pde": "ShallowWater",
        "n_train": 900,
        "n_test": 60,
        "n_channels": 1,
        "resolution": (128, 128),
        "time_steps": 101,
        "params": {}
    },
    "DR": {
        "source": "PDEBench",
        "pde": "DiffusionReaction",
        "n_train": 900,
        "n_test": 60,
        "n_channels": 2,
        "resolution": (128, 128),
        "time_steps": 101,
        "params": {}
    },
    "CFDBench": {
        "source": "CFDBench",
        "pde": "IncompressibleNavierStokes",
        "n_train": 9000,
        "n_test": 1000,
        "n_channels": 3,  # u, v, p
        "resolution": (128, 128),
        "time_steps": 21,
        "irregular": True,
        "params": {}
    },
    "NS_1e-4": {
        "source": "FNO",
        "pde": "Navier-Stokes",
        "viscosity": 1e-4,
        "n_train": 2000,
        "n_test": 200,
        "n_channels": 1,
        "resolution": (64, 64),
        "time_steps": 20,
        "params": {"nu": 1e-4}
    },
    "CNS_1_0.01": {
        "source": "PDEBench",
        "pde": "CompressibleNavierStokes",
        "n_train": 2000,
        "n_test": 200,
        "n_channels": 3,
        "resolution": (128, 128),
        "time_steps": 21,
        "params": {"eta": 1.0, "zeta": 0.01}
    },
    "PDEArena": {
        "source": "PDEArena",
        "pde": "IncompressibleNavierStokes",
        "n_train": 2000,
        "n_test": 200,
        "n_channels": 3,
        "resolution": (128, 128),
        "time_steps": 21,
        "params": {}
    },
}


def _interpolate_2d(data: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Simple bilinear interpolation using numpy.

    data: (..., H, W) or (..., C, H, W)
    """
    import scipy.interpolate

    shape = data.shape
    if data.ndim == 2:
        h, w = shape
        y = np.linspace(0, 1, h)
        x = np.linspace(0, 1, w)
        y_new = np.linspace(0, 1, target_h)
        x_new = np.linspace(0, 1, target_w)
        interp = scipy.interpolate.RegularGridInterpolator(
            (y, x), data, bounds_error=False, fill_value=0.0)
        yy, xx = np.meshgrid(y_new, x_new, indexing='ij')
        return interp(np.stack([yy, xx], axis=-1))

    elif data.ndim == 3:
        c, h, w = shape
        out = np.zeros((c, target_h, target_w), dtype=data.dtype)
        for i in range(c):
            out[i] = _interpolate_2d(data[i], target_h, target_w)
        return out

    elif data.ndim == 4:
        n, c, h, w = shape
        out = np.zeros((n, c, target_h, target_w), dtype=data.dtype)
        for i in range(n):
            out[i] = _interpolate_2d(data[i], target_h, target_w)
        return out

    raise ValueError(f"Unsupported ndim: {data.ndim}")


class PDEDataset(Dataset):
    """Generic PDE dataset for loading and preprocessing.

    Handles:
    - Spatial resolution standardization (to H=128)
    - Channel padding to max across datasets
    - Mask channel for irregular geometries
    - Noise injection during pre-training
    """

    def __init__(self, name: str, split: str = "train",
                 target_resolution: int = 128,
                 max_channels: int = 8,
                 num_timesteps_in: int = 10,
                 add_noise: bool = False,
                 noise_eps: float = 1e-4,
                 data_root: str = "./data"):
        super().__init__()
        self.name = name
        self.split = split
        self.target_resolution = target_resolution
        self.max_channels = max_channels
        self.num_timesteps_in = num_timesteps_in
        self.add_noise = add_noise
        self.noise_eps = noise_eps
        self.data_root = data_root

        self.info = DATASET_INFO[name]
        self.n_channels = self.info["n_channels"]
        self.irregular = self.info.get("irregular", False)

        if split == "train":
            self.n_samples = self.info["n_train"]
        elif split == "test":
            self.n_samples = self.info["n_test"]
        else:
            # For fine-tuning, use train split
            self.n_samples = self.info["n_train"]

        self.data = self._load_data()

    def _load_data(self) -> np.ndarray:
        """Load data from disk. Returns (N, C, H, W, T) or (N, H, W, T)."""
        data_path = os.path.join(self.data_root, self.name, f"{self.split}.npy")
        if os.path.exists(data_path):
            return np.load(data_path)
        # Fallback: generate synthetic data for testing
        return self._generate_synthetic()

    def _generate_synthetic(self) -> np.ndarray:
        """Generate synthetic PDE data for development/testing."""
        src_h, src_w = self.info["resolution"]
        time_steps = self.info["time_steps"]
        n = min(self.n_samples, 100)  # smaller for synthetic

        data = np.random.randn(n, self.n_channels, src_h, src_w, time_steps).astype(np.float32)
        if self.irregular:
            mask = np.ones((n, 1, src_h, src_w, time_steps), dtype=np.float32)
            data = np.concatenate([data, mask], axis=1)
        return data

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load sample: (C, H, W, T_total)
        sample = self.data[idx]

        C, H_src, W_src, T_total = sample.shape

        # Resize spatial dimensions if needed
        if H_src != self.target_resolution or W_src != self.target_resolution:
            # Reshape to (C*T, H, W) for interpolation
            sample_flat = sample.reshape(-1, H_src, W_src)
            sample_flat = _interpolate_2d(
                sample_flat, self.target_resolution, self.target_resolution)
            sample = sample_flat.reshape(C, self.target_resolution,
                                          self.target_resolution, T_total)

        # Pad channels to max_channels
        if C < self.max_channels:
            pad = np.zeros((self.max_channels - C, self.target_resolution,
                            self.target_resolution, T_total), dtype=sample.dtype)
            sample = np.concatenate([sample, pad], axis=0)

        # Ensure we have enough timesteps
        T_in = self.num_timesteps_in
        assert T_total >= T_in + 1, f"Not enough timesteps: {T_total} < {T_in + 1}"

        # Randomly select a starting timestep
        max_start = T_total - T_in - 1
        t_start = random.randint(0, max_start)

        # Input: T_in frames, Output: next frame
        u_in = sample[..., t_start:t_start + T_in]  # (C_max, H, W, T_in)
        u_out = sample[..., t_start + T_in]          # (C_max, H, W)

        # Add noise during pre-training
        if self.add_noise:
            noise_scale = self.noise_eps * np.linalg.norm(u_in)
            noise = np.random.randn(*u_in.shape).astype(u_in.dtype) * noise_scale
            u_in = u_in + noise

        return {
            "input": torch.from_numpy(u_in.copy()).float(),
            "target": torch.from_numpy(u_out.copy()).float(),
            "dataset": self.name,
        }


class BalancedDatasetSampler(Sampler):
    """Balanced sampling across multiple datasets.

    Sampling probability for dataset k:
        p_k = w_k / (K * |D_k| * sum_j w_j)
    """

    def __init__(self, datasets: List[PDEDataset],
                 weights: Optional[Dict[str, float]] = None,
                 total_samples: Optional[int] = None):
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.dataset_sizes = [len(d) for d in datasets]

        if weights is None:
            weights = {d.name: 1.0 for d in datasets}

        K = self.num_datasets
        w_sum = sum(weights[d.name] for d in datasets)

        self.probs = []
        for d in datasets:
            wk = weights.get(d.name, 1.0)
            pk = wk / (K * len(d) * w_sum)
            self.probs.append(pk)

        # Normalize probabilities
        total_p = sum(self.probs)
        self.probs = [p / total_p for p in self.probs]

        self.total_samples = total_samples or sum(self.dataset_sizes)

    def __iter__(self):
        for _ in range(self.total_samples):
            # Select dataset
            d_idx = np.random.choice(self.num_datasets, p=self.probs)
            # Select sample within dataset
            s_idx = np.random.randint(0, self.dataset_sizes[d_idx])
            yield (d_idx, s_idx)

    def __len__(self):
        return self.total_samples


def collate_multi_dataset(batch: List[Tuple[int, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    """Collate function for multi-dataset balanced sampling.

    Each element is (d_idx, sample_dict) from BalancedDatasetSampler iteration.
    """
    inputs = []
    targets = []
    names = []
    for item in batch:
        if isinstance(item, tuple):
            _, sample = item
        else:
            sample = item
        inputs.append(sample["input"])
        targets.append(sample["target"])
        names.append(sample["dataset"])

    return {
        "input": torch.stack(inputs, dim=0),
        "target": torch.stack(targets, dim=0),
        "dataset": names,
    }


def create_multi_dataset_loaders(dataset_names: List[str],
                                  split: str = "train",
                                  target_resolution: int = 128,
                                  max_channels: int = 8,
                                  num_timesteps_in: int = 10,
                                  batch_size: int = 20,
                                  add_noise: bool = False,
                                  noise_eps: float = 1e-4,
                                  dataset_weights: Optional[Dict[str, float]] = None,
                                  num_workers: int = 4,
                                  data_root: str = "./data") -> DataLoader:
    """Create a DataLoader that samples from multiple PDE datasets with
    balanced sampling.

    Returns a DataLoader yielding dicts with keys:
        input: (B, C_max, H, W, T_in)
        target: (B, C_max, H, W)
        dataset: list of dataset names
    """
    datasets = []
    for name in dataset_names:
        ds = PDEDataset(
            name=name,
            split=split,
            target_resolution=target_resolution,
            max_channels=max_channels,
            num_timesteps_in=num_timesteps_in,
            add_noise=add_noise,
            noise_eps=noise_eps,
            data_root=data_root,
        )
        datasets.append(ds)

    sampler = BalancedDatasetSampler(
        datasets,
        weights=dataset_weights,
        total_samples=sum(len(d) for d in datasets),
    )

    # Build a combined dataset for sampling
    class CombinedDataset(Dataset):
        def __init__(self, datasets, sampler):
            self.datasets = datasets
            self.sampler = sampler

        def __len__(self):
            return len(self.sampler)

        def __getitem__(self, idx):
            # Use the sampler's deterministic ordering based on epoch
            d_idx, s_idx = list(self.sampler)[idx]
            sample = self.datasets[d_idx][s_idx]
            return sample

    class MultiDatasetWrapper(Dataset):
        """Wrapper that uses the sampler to create epoch-consistent indexing."""
        def __init__(self, datasets, sampler):
            self.datasets = datasets
            self.sampler = sampler
            self._indices = list(self.sampler)

        def __len__(self):
            return len(self._indices)

        def __getitem__(self, idx):
            d_idx, s_idx = self._indices[idx]
            return self.datasets[d_idx][s_idx]

    wrapper = MultiDatasetWrapper(datasets, sampler)

    loader = DataLoader(
        wrapper,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader


def create_single_dataset_loader(dataset_name: str,
                                  split: str = "train",
                                  target_resolution: int = 128,
                                  max_channels: int = 8,
                                  num_timesteps_in: int = 10,
                                  batch_size: int = 20,
                                  add_noise: bool = False,
                                  noise_eps: float = 1e-4,
                                  num_workers: int = 4,
                                  data_root: str = "./data") -> DataLoader:
    """Create a DataLoader for a single PDE dataset."""
    ds = PDEDataset(
        name=dataset_name,
        split=split,
        target_resolution=target_resolution,
        max_channels=max_channels,
        num_timesteps_in=num_timesteps_in,
        add_noise=add_noise,
        noise_eps=noise_eps,
        data_root=data_root,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader
