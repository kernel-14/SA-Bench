"""Dataset loading and preprocessing for MoE-POT.

Implements the data pipeline described in Appendix B.1:
- Spatial resolution standardization to H=128 via interpolation
- Channel padding to max_channels with constant fill (1.0)
- Mask channel for irregular geometries (CFDBench)
- Balanced sampling across datasets with importance weights
- Noise injection during pre-training
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from config import DATASET_CONFIGS, DatasetConfig


TARGET_SPATIAL_SIZE = 128
MAX_CHANNELS = 4   # maximum number of physical channels across all datasets


class PDEDataset(Dataset):
    """Generic PDE dataset loader.

    Loads pre-saved .npy or .pt trajectory files and returns windows of
    T+1 consecutive frames: (u^0, ..., u^{T-1}) as input and u^T as target.

    Expected file layout (one of):
      <root>/<name>/train.npy  shape: (N, total_T, C, H, W)
      <root>/<name>/train.pt   same shape as tensor

    Args:
        root:         path to dataset root directory
        cfg:          DatasetConfig for this dataset
        split:        'train' or 'test'
        num_input_frames: T (number of input frames)
        max_channels: pad channels to this value
        target_size:  spatial size to resize to
        dataset_id:   integer label used for interpretability experiments
    """

    def __init__(
        self,
        root: str,
        cfg: DatasetConfig,
        split: str = "train",
        num_input_frames: int = 10,
        max_channels: int = MAX_CHANNELS,
        target_size: int = TARGET_SPATIAL_SIZE,
        dataset_id: int = 0,
        has_mask: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.num_input_frames = num_input_frames
        self.max_channels = max_channels
        self.target_size = target_size
        self.dataset_id = dataset_id
        self.has_mask = has_mask

        data_path = Path(root) / cfg.name / f"{split}.npy"
        pt_path = Path(root) / cfg.name / f"{split}.pt"

        if pt_path.exists():
            data = torch.load(pt_path, map_location="cpu")
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            self.data = data.float()
        elif data_path.exists():
            arr = np.load(str(data_path))
            self.data = torch.from_numpy(arr).float()
        else:
            raise FileNotFoundError(
                f"Dataset file not found at {data_path} or {pt_path}. "
                "Please download and place the dataset files."
            )

        # data shape: (N, total_T, C, H, W)
        if self.data.ndim == 4:
            # (N, total_T, H, W) → add channel dim
            self.data = self.data.unsqueeze(2)

        self.N, self.total_T, self.C, self.H, self.W = self.data.shape

        # Number of valid windows per trajectory
        self.windows_per_traj = max(1, self.total_T - num_input_frames)

    def __len__(self) -> int:
        return self.N * self.windows_per_traj

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx = idx // self.windows_per_traj
        frame_start = idx % self.windows_per_traj

        frames = self.data[traj_idx, frame_start : frame_start + self.num_input_frames + 1]
        # frames: (T+1, C, H, W)

        input_frames = frames[: self.num_input_frames]   # (T, C, H, W)
        target_frame = frames[self.num_input_frames]     # (C, H, W)

        # Resize spatial dimensions if needed
        if self.H != self.target_size or self.W != self.target_size:
            input_frames = self._resize(input_frames, self.target_size)
            target_frame = self._resize(target_frame.unsqueeze(0), self.target_size).squeeze(0)

        # Pad channels to max_channels
        input_frames = self._pad_channels(input_frames, self.max_channels)
        target_frame = self._pad_channels(target_frame.unsqueeze(0), self.max_channels).squeeze(0)

        return {
            "input": input_frames,    # (T, max_C, H, W)
            "target": target_frame,   # (max_C, H, W)
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.long),
        }

    @staticmethod
    def _resize(x: torch.Tensor, size: int) -> torch.Tensor:
        """Resize spatial dims of (T, C, H, W) or (C, H, W) tensor."""
        if x.ndim == 3:
            return F.interpolate(
                x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
            ).squeeze(0)
        T, C, H, W = x.shape
        x_flat = x.view(T * C, 1, H, W)
        x_resized = F.interpolate(x_flat, size=(size, size), mode="bilinear", align_corners=False)
        return x_resized.view(T, C, size, size)

    @staticmethod
    def _pad_channels(x: torch.Tensor, max_channels: int) -> torch.Tensor:
        """Pad channel dimension with 1.0 to reach max_channels.

        x shape: (T, C, H, W) or (C, H, W)
        """
        if x.ndim == 3:
            C = x.shape[0]
            if C < max_channels:
                pad = torch.ones(*x.shape[:-3], max_channels - C, *x.shape[-2:])
                x = torch.cat([x, pad], dim=0)
        else:
            T, C, H, W = x.shape
            if C < max_channels:
                pad = torch.ones(T, max_channels - C, H, W)
                x = torch.cat([x, pad], dim=1)
        return x[:max_channels] if x.ndim == 3 else x[:, :max_channels]


class MixedPDEDataset(Dataset):
    """Concatenated dataset from multiple PDE sources with balanced sampling.

    Implements the balanced sampling strategy from Appendix B.1:
        p_k = w_k / (K * |D_k| * Σ_k w_k)

    Returns samples with dataset_id labels for interpretability analysis.
    """

    def __init__(
        self,
        datasets: List[PDEDataset],
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative = np.cumsum([0] + self.lengths)
        self.total = sum(self.lengths)

        if weights is None:
            weights = [1.0] * len(datasets)
        self.weights = weights

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        for i, (start, end) in enumerate(
            zip(self.cumulative[:-1], self.cumulative[1:])
        ):
            if start <= idx < end:
                return self.datasets[i][idx - start]
        raise IndexError(f"Index {idx} out of range for MixedPDEDataset of size {self.total}")

    def get_sampler_weights(self) -> torch.Tensor:
        """Compute per-sample weights for WeightedRandomSampler.

        p_k ∝ w_k / |D_k|  (uniform within each dataset, scaled by importance weight)
        """
        K = len(self.datasets)
        total_weight = sum(self.weights)
        sample_weights = []
        for i, (dataset, w_k) in enumerate(zip(self.datasets, self.weights)):
            n_k = len(dataset)
            per_sample_weight = w_k / (K * n_k * total_weight + 1e-8)
            sample_weights.extend([per_sample_weight] * n_k)
        return torch.tensor(sample_weights, dtype=torch.float32)


def build_pretrain_datasets(
    data_root: str,
    split: str = "train",
    num_input_frames: int = 10,
    dataset_names: Optional[List[str]] = None,
) -> MixedPDEDataset:
    """Build the mixed pre-training dataset from 6 PDE sources."""
    if dataset_names is None:
        dataset_names = list(DATASET_CONFIGS.keys())

    datasets = []
    weights = []
    for i, name in enumerate(dataset_names):
        cfg = DATASET_CONFIGS[name]
        has_mask = name == "cfdbench"
        try:
            ds = PDEDataset(
                root=data_root,
                cfg=cfg,
                split=split,
                num_input_frames=num_input_frames,
                dataset_id=i,
                has_mask=has_mask,
            )
            datasets.append(ds)
            weights.append(cfg.weight)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if not datasets:
        raise RuntimeError(f"No datasets found under {data_root}")

    return MixedPDEDataset(datasets, weights)


def build_single_dataset(
    data_root: str,
    dataset_name: str,
    split: str = "train",
    num_input_frames: int = 10,
    dataset_id: int = 0,
) -> PDEDataset:
    """Build a single-dataset loader for fine-tuning or evaluation."""
    from config import DATASET_CONFIGS, DOWNSTREAM_CONFIGS

    all_cfgs = {**DATASET_CONFIGS, **DOWNSTREAM_CONFIGS}
    if dataset_name not in all_cfgs:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    cfg = all_cfgs[dataset_name]
    has_mask = dataset_name == "cfdbench"
    return PDEDataset(
        root=data_root,
        cfg=cfg,
        split=split,
        num_input_frames=num_input_frames,
        dataset_id=dataset_id,
        has_mask=has_mask,
    )


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
) -> DataLoader:
    """Build a DataLoader, optionally with balanced sampling."""
    sampler = None
    if use_weighted_sampler and isinstance(dataset, MixedPDEDataset):
        sample_weights = dataset.get_sampler_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def inject_noise(
    u_frames: torch.Tensor,
    noise_scale: float = 0.01,
) -> torch.Tensor:
    """Inject small-scale Gaussian noise into input frames.

    ε ~ N(0, ε_scale * ||u^{<t}|| * I)

    Applied only during pre-training (not fine-tuning or inference).

    Args:
        u_frames:    (B, T, C, H, W)
        noise_scale: ε in the paper (default 0.01)

    Returns:
        noisy_frames: (B, T, C, H, W)
    """
    B, T, C, H, W = u_frames.shape
    noisy = u_frames.clone()
    for t in range(T):
        u_lt = u_frames[:, :t + 1]  # (B, t+1, C, H, W)
        norm = u_lt.norm(dim=(1, 2, 3, 4), keepdim=True)  # (B, 1, 1, 1, 1)
        std = noise_scale * norm.squeeze(-1)               # (B, 1, 1, 1)
        eps = torch.randn_like(u_frames[:, t]) * std
        noisy[:, t] = u_frames[:, t] + eps
    return noisy
