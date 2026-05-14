## data_loader.py
"""Data loading and preprocessing for MoE‑POT experiments.

This module provides:

* ``PDEFrameDataset``: a map‑style PyTorch ``Dataset`` that reads
  pre‑processed HDF5 files and returns (input_frames, target_frame,
  mask, channels_per_task).
* ``DatasetLoader``: a manager that downloads, pre‑processes, caches,
  and constructs balanced ``DataLoader`` objects for pre‑training,
  fine‑tuning, and downstream tasks.

The pre‑processing pipeline standardises spatial resolution, unifies
channel counts across heterogeneous PDE datasets, and extracts
autoregressive sliding‑window pairs.  A weighted‑random sampler is
used to avoid bias from dataset size discrepancies during pre‑training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import interpolate
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from config import Config
from utils import Utils

# ---------------------------------------------------------------------------
# Dataset-level metadata
# ---------------------------------------------------------------------------
_PHYSICAL_CHANNELS: Dict[str, int] = {
    "FNO-NS-1e-5": 1,  # vorticity
    "FNO-NS-1e-3": 1,
    "PDEBench-CNS-0.1-0.01": 4,  # ρ, u, v, p
    "PDEBench-SWE": 1,  # water depth
    "PDEBench-DR": 2,  # two density fields
    "CFDBench": 3,  # u, v, p (mask channel is added separately)
}

_HAS_MASK_CHANNEL: Dict[str, bool] = {
    "CFDBench": True,
}

# Valid channel indices after padding to Cmax=5.
# For CFDBench: physical channels 0‑2, mask at 4 → valid = [0,1,2,4].
# Others: only physical channels, no mask.
_VALID_CHANNELS: Dict[str, List[int]] = {
    name: (
        [i for i in range(_PHYSICAL_CHANNELS[name])]
        if not _HAS_MASK_CHANNEL.get(name, False)
        else [i for i in range(_PHYSICAL_CHANNELS[name])] + [4]  # mask channel ind 4
    )
    for name in _PHYSICAL_CHANNELS
}

# Default list of datasets used for pre‑training (from the paper)
_DEFAULT_PRETRAIN_DATASETS: List[str] = [
    "FNO-NS-1e-5",
    "FNO-NS-1e-3",
    "PDEBench-CNS-0.1-0.01",
    "PDEBench-SWE",
    "PDEBench-DR",
    "CFDBench",
]


class PDEFrameDataset(Dataset):
    """PyTorch Dataset that reads a pre‑processed HDF5 file.

    Each sample returned is a tuple:
        input_frames : Tensor of shape ``(T_in, Cmax, H, W)``
        target_frame : Tensor of shape ``(Cmax, H, W)``
        mask         : Tensor of shape ``(Cmax, H, W)`` (1 = valid, 0 = invalid)
        channels_per_task : int, the number of original physical channels
                            (excluding padding and mask channel).
    """

    def __init__(self, h5_path: str) -> None:
        """Initialise from an HDF5 file created by ``DatasetLoader._preprocess_single_dataset``."""
        self._file = h5py.File(h5_path, "r")
        self._inputs = self._file["inputs"]  # (N, T_in, Cmax, H, W)
        self._targets = self._file["targets"]  # (N, Cmax, H, W)
        self._masks = self._file["masks"]  # (N, Cmax, H, W)
        self._length = self._inputs.shape[0]
        self.channels_per_task: int = int(
            self._file.attrs.get("channels_per_task", 0)
        )
        self.valid_channels: List[int] = list(
            self._file.attrs.get("valid_channels", [])
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self._length}.")

        inp = torch.from_numpy(self._inputs[idx]).float()
        tgt = torch.from_numpy(self._targets[idx]).float()
        msk = torch.from_numpy(self._masks[idx]).float()
        return inp, tgt, msk, self.channels_per_task


class DatasetLoader:
    """Manages downloading, pre‑processing, and DataLoader construction.

    Parameters
    ----------
    config : Config
        Global configuration object.
    """

    def __init__(self, config: Config) -> None:
        Utils.set_seed(config.seed)
        self.config = config
        self.data_root = Path(config.data_root)
        self.cache_dir = self.data_root / "preprocessed"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use the dataset list specified in config, otherwise fall back to
        # the paper’s default six pre‑training datasets.
        if not hasattr(self.config, "dataset_list"):
            self.config.dataset_list = _DEFAULT_PRETRAIN_DATASETS

        # Dictionaries are kept as class variables for internal lookup
        self._physical_channels = _PHYSICAL_CHANNELS
        self._has_mask_channel = _HAS_MASK_CHANNEL
        self._valid_channels = _VALID_CHANNELS

    # ------------------------------------------------------------------
    # Raw data retrieval (stubs)
    # ------------------------------------------------------------------
    def _download_and_cache(self, dataset_name: str) -> str:
        """Ensure raw data exists and return the raw directory path.

        If the raw directory is missing, a ``FileNotFoundError`` is
        raised with instructions to manually download the data.
        """
        raw_dir = self.data_root / "raw" / dataset_name
        if raw_dir.is_dir():
            return str(raw_dir)

        raise FileNotFoundError(
            f"Raw data for '{dataset_name}' not found at {raw_dir}.\n"
            "Please download the dataset manually from the official source "
            "and place it in that directory. The expected layout is:\n"
            f"  {raw_dir}/\n"
            "      train/   (trajectory files in .npy, .npz, or .h5)\n"
            "      test/    (same structure)\n"
            "See the reproduction plan for download links."
        )

    def _load_raw_data(self, dataset_name: str, split: str) -> List[np.ndarray]:
        """Load raw trajectory arrays for a dataset and data split.

        Returns a list of numpy arrays, each of shape
        ``(time_steps, H_orig, W_orig, C_orig)``.
        """
        raw_dir = Path(self._download_and_cache(dataset_name))
        split_dir = raw_dir / split

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory {split_dir} does not exist."
            )

        trajectories: List[np.ndarray] = []
        # Accept .npy, .npz, and .h5 files (common formats)
        for ext in ("*.npy", "*.npz"):
            for file_path in sorted(split_dir.glob(ext)):
                if ext == "*.npz":
                    with np.load(file_path) as data:
                        # Attempt common keys
                        arr = data.get("arr_0") or data.get("data")
                        if arr is None:
                            raise KeyError(f"No 'arr_0' or 'data' key found in {file_path}")
                        arr = np.asarray(arr)
                else:
                    arr = np.load(file_path)

                # Ensure shape is (T, H, W, C) – add singleton channel dim if needed
                if arr.ndim == 3:
                    arr = arr[..., np.newaxis]  # (T, H, W) -> (T, H, W, 1)
                elif arr.ndim != 4:
                    raise ValueError(
                        f"Unexpected array shape {arr.shape} in {file_path}"
                    )
                trajectories.append(arr)

        if not trajectories:
            raise FileNotFoundError(
                f"No trajectory data files were found in {split_dir}."
            )
        return trajectories

    # ------------------------------------------------------------------
    # Pre‑processing
    # ------------------------------------------------------------------
    def _preprocess_single_dataset(self, dataset_name: str, split: str) -> str:
        """Pre‑process a single PDE dataset and write an HDF5 cache file.

        Returns the path to the cached file.
        """
        cache_file = self.cache_dir / f"{dataset_name}_{split}.h5"
        if cache_file.exists():
            return str(cache_file)

        traj_list = self._load_raw_data(dataset_name, split)
        H_tgt, W_tgt = self.config.spatial_resolution  # (128, 128)
        Cmax = self.config.max_channels
        T_in = self.config.input_frames

        phys_channels = self._physical_channels[dataset_name]
        has_mask = self._has_mask_channel.get(dataset_name, False)
        valid_channels = self._valid_channels[dataset_name]

        all_inputs: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_masks: List[np.ndarray] = []

        # Iterate over raw trajectories
        for traj in traj_lines if (traj_lines := traj_list) is not None:  # (reorder simple)
            pass

        for traj in traj_list:
            T, H_orig, W_orig, C_orig = traj.shape

            # ---- Spatial resizing ----
            # Convert to torch tensor (T, C, H, W) for interpolate
            traj_t = torch.from_numpy(traj).float().permute(0, 3, 1, 2)
            traj_t = interpolate(
                traj_t,
                size=(H_tgt, W_tgt),
                mode="bilinear",
                align_corners=False,
            )
            # Back to (T, H, W, C)
            traj_resized = traj_t.permute(0, 2, 3, 1).numpy()

            # ---- Channel padding ----
            # Create array of shape (T, Cmax, H, W) filled with constant 1.
            frames_padded = np.ones((T, Cmax, H_tgt, W_tgt), dtype=np.float32)
            # Insert physical channels
            for c in range(phys_channels):
                frames_padded[:, c] = traj_resized[..., c]

            # Handle geometric mask for CFDBench
            if has_mask:
                # Assume mask is the last channel of original data
                mask_raw = traj_resized[..., -1]  # (T, H, W)
                # Insert mask at the designated index (config.mask_channel_index)
                frames_padded[:, self.config.mask_channel_index] = mask_raw
                spatial_mask = mask_raw[0]  # shape (H, W), assume constant over time
            else:
                spatial_mask = np.ones((H_tgt, W_tgt), dtype=np.float32)

            # Build full channel‑wise mask: 1.0 for valid channels (times
            # spatial mask), 0.0 otherwise.
            full_mask = np.zeros((Cmax, H_tgt, W_tgt), dtype=np.float32)
            for ch in valid_channels:
                full_mask[ch] = spatial_mask

            # ---- Extract sliding‑window pairs ----
            if T < T_in + 1:
                print(
                    f"Warning: trajectory length {T} is too short for "
                    f"T_in={T_in} in {dataset_name}/{split}. Skipping."
                )
                continue

            for start in range(0, T - T_in):
                input_frames = frames_padded[start : start + T_in]  # (T_in, Cmax, H, W)
                target_frame = frames_padded[start + T_in]          # (Cmax, H, W)
                # Mask is identical for all samples from this trajectory
                all_inputs.append(input_frames)
                all_targets.append(target_frame)
                all_masks.append(full_mask.copy())

        if not all_inputs:
            raise RuntimeError(
                f"No valid sequences could be extracted for {dataset_name}/{split}."
            )

        # ---- Save to HDF5 ----
        inputs_arr = np.stack(all_inputs, axis=0)  # (N, T_in, Cmax, H, W)
        targets_arr = np.stack(all_targets, axis=0)  # (N, Cmax, H, W)
        masks_arr = np.stack(all_masks, axis=0)

        with h5py.File(cache_file, "w") as f:
            f.create_dataset("inputs", data=inputs_arr, dtype=np.float32)
            f.create_dataset("targets", data=targets_arr, dtype=np.float32)
            f.create_dataset("masks", data=masks_arr, dtype=np.float32)
            f.attrs["channels_per_task"] = phys_channels
            f.attrs["valid_channels"] = valid_channels

        return str(cache_file)

    # ------------------------------------------------------------------
    # Balanced sampler
    # ------------------------------------------------------------------
    def _balanced_sampler(
        self, datasets: List[PDEFrameDataset]
    ) -> WeightedRandomSampler:
        """Create a ``WeightedRandomSampler`` that balances datasets equally.

        Each dataset receives equal total weight regardless of its size,
        following the paper’s balanced sampling strategy with importance
        weights ``w_k = 1`` (configurable).  The per‑sample weight is set
        to ``1 / (K * |D_k|)`` so that the sum of weights inside dataset
        *k* is ``1/K``.
        """
        K = len(datasets)
        sample_weights: List[float] = []
        for ds in datasets:
            size = len(ds)
            weight = 1.0 / (K * size)  # equal total weight per dataset
            sample_weights.extend([weight] * size)

        total_samples = sum(len(ds) for ds in datasets)
        return WeightedRandomSampler(
            sample_weights,
            num_samples=total_samples,
            replacement=True,
        )

    # ------------------------------------------------------------------
    # Public loading interfaces
    # ------------------------------------------------------------------
    def load_pretrain_data(self) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """Build balanced training DataLoader and per‑dataset validation loaders.

        Returns
        -------
        train_loader : DataLoader
            DataLoader for the combined pre‑training datasets with
            balanced sampling.
        val_loaders : Dict[str, DataLoader]
            Mapping from dataset name to its test‑set DataLoader.
        """
        dataset_names = self.config.dataset_list
        train_datasets: List[PDEFrameDataset] = []
        val_loaders: Dict[str, DataLoader] = {}

        for name in dataset_names:
            train_file = self._preprocess_single_dataset(name, "train")
            val_file = self._preprocess_single_dataset(name, "test")
            train_ds = PDEFrameDataset(train_file)
            val_ds = PDEFrameDataset(val_file)
            train_datasets.append(train_ds)

            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            val_loaders[name] = val_loader

        # Combine training datasets
        concat_dataset = ConcatDataset(train_datasets)
        sampler = self._balanced_sampler(train_datasets)

        train_loader = DataLoader(
            concat_dataset,
            batch_size=self.config.pretrain_batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.train_loader = train_loader
        self.val_loaders = val_loaders
        return train_loader, val_loaders

    def load_single_task(self, task_name: str, split: str) -> DataLoader:
        """Return a DataLoader for a single dataset (fine‑tuning / downstream).

        Parameters
        ----------
        task_name : str
            Name of the dataset (must match keys in ``_PHYSICAL_CHANNELS``
            or be a known downstream dataset).
        split : str
            ``'train'`` or ``'test'``.
        """
        # If the task is a downstream dataset not seen during pre‑processing,
        # the same pipeline still works because we genericly load from raw.
        h5_file = self._preprocess_single_dataset(task_name, split)
        ds = PDEFrameDataset(h5_file)

        shuffle = split == "train"
        # For evaluation (test) we keep batch size 1; for training, use
        # the pre‑training batch size (adjustable if memory constrained).
        batch_size = (
            self.config.pretrain_batch_size if split == "train" else 1
        )

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=shuffle,
        )
        return loader

