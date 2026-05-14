## dataset.py
"""
PyTorch Dataset for loading preprocessed PDE trajectories from an HDF5 file.

The HDF5 file contains one group per sub‑dataset, each with:
    - 'data'   : array of shape (N, seq_len=4, channels=3, height=128, width=128), dtype float16
    - Attributes (per sub‑dataset):
        - 'train_indices', 'val_indices', 'test_indices'  (optional list of integers)
        - Normalisation parameters:
            * For minmax: 'min' (3,), 'max' (3,) float32
            * For zscore: 'mean' (3,), 'std' (3,) float32

If no split indices are present, the dataset will randomly partition the data
using the provided val_split and test_split ratios with a fixed seed.
The class also provides sample weights for balanced training across sub‑datasets.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import normalize_field  # see project‑level utils

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------

Tensor = torch.Tensor
IndexType = List[Tuple[str, int]]  # (sub_dataset_name, local_idx)

# -----------------------------------------------------------------------------
# Helper to derive global normalisation parameters from per‑sub‑dataset stats
# -----------------------------------------------------------------------------

def _compute_global_stats(
    norm_params_per_sub: Dict[str, Dict[str, Union[np.ndarray, List[float]]]],
    method: str,
) -> Dict[str, np.ndarray]:
    """
    Compute global min/max or mean/std across all sub‑datasets from their
    stored per‑channel statistics. Returns a dict compatible with normalize_field.
    """
    # We'll gather per‑channel arrays and element‑wise aggregate.
    if method == "minmax":
        all_mins = [np.asarray(params["min"]) for params in norm_params_per_sub.values()]
        all_maxs = [np.asarray(params["max"]) for params in norm_params_per_sub.values()]
        global_min = np.min(np.stack(all_mins), axis=0)
        global_max = np.max(np.stack(all_maxs), axis=0)
        return {"min": global_min, "max": global_max}
    elif method == "zscore":
        raise NotImplementedError(
            "Global z‑score normalisation not yet implemented; "
            "per‑dataset stats must be sufficient."
        )
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

# -----------------------------------------------------------------------------
# HD5Dataset
# -----------------------------------------------------------------------------

class HD5Dataset(Dataset):
    """
    Dataset of short PDE trajectories stored in a single HDF5 file.

    Each sample is a tuple (normalized_frames, dataset_id) where
    normalized_frames is a float32 tensor of shape (4, 3, 128, 128).
    dataset_id is an integer identifier for the originating sub‑dataset.

    Args:
        data_config: A dictionary containing the keys:
            - h5_path: path to the HDF5 file (str)
            - sub_datasets: list of sub‑dataset group names (List[str])
            - seq_len: length of trajectory – expected 4 (int)
            - image_size: spatial resolution – expected 128 (int)
            - channels: number of physical fields – expected 3 (int)
            - normalize: 'minmax' or 'zscore' (str)
            - val_split: fraction held out for validation (float)
            - test_split: fraction held out for test (float)
            - use_equal_sampling: whether to compute sample weights (bool)
        split: one of 'train', 'val', 'test'.
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        split: str = "train",
    ) -> None:
        super().__init__()

        # Validate split
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of 'train','val','test', got '{split}'")

        # Extract configuration
        self.h5_path: str = data_config["h5_path"]
        self.sub_dataset_names: List[str] = data_config["sub_datasets"]
        self.seq_len: int = data_config.get("seq_len", 4)
        self.image_size: int = data_config.get("image_size", 128)
        self.channels: int = data_config.get("channels", 3)
        self.normalize_method: str = data_config["normalize"]
        self.val_split: float = data_config["val_split"]
        self.test_split: float = data_config["test_split"]
        self.use_equal_sampling: bool = data_config.get("use_equal_sampling", True)

        # Per‑worker file handles
        self._file: Optional[h5py.File] = None
        self._pid: Optional[int] = None  # PID that opened the current file handle

        # Mappings
        self.dataset_ids: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.sub_dataset_names)
        }
        self.norm_params: Dict[str, Dict[str, np.ndarray]] = {}
        self.samples: IndexType = []  # list of (sub_dataset_name, local_index)

        # Build sample list and load per‑sub‑dataset metadata
        # Use a temporary file handle in the main process to read attributes.
        with h5py.File(self.h5_path, "r") as f:
            for ds_name in self.sub_dataset_names:
                if ds_name not in f:
                    raise KeyError(f"Sub‑dataset '{ds_name}' not found in HDF5 file.")
                grp = f[ds_name]
                data_shape = grp["data"].shape
                if len(data_shape) != 5 or data_shape[1] != self.seq_len:
                    raise RuntimeError(
                        f"Expected data shape (N, {self.seq_len}, {self.channels}, "
                        f"{self.image_size}, {self.image_size}), got {data_shape}."
                    )
                total_samples = data_shape[0]

                # --- Spatial/channel dimensions check for sanity ---
                c_, h_, w_ = data_shape[2], data_shape[3], data_shape[4]
                if c_ != self.channels or h_ != self.image_size or w_ != self.image_size:
                    raise RuntimeError(
                        f"Data shape mismatch: expected channels={self.channels}, "
                        f"height={self.image_size}, width={self.image_size}; "
                        f"got ({c_}, {h_}, {w_})."
                    )

                # --- Load split indices ---
                train_idx, val_idx, test_idx = self._get_split_indices(
                    grp, total_samples, ds_name
                )
                # Choose the indices for the requested split
                if split == "train":
                    indices = train_idx
                elif split == "val":
                    indices = val_idx
                else:  # test
                    indices = test_idx

                # Extend samples
                for local_idx in indices:
                    self.samples.append((ds_name, local_idx))

                # --- Load normalisation parameters ---
                norm_params: Dict[str, Union[np.ndarray, List[float]]] = {}
                if self.normalize_method == "minmax":
                    if "min" not in grp.attrs or "max" not in grp.attrs:
                        raise KeyError(
                            f"Sub‑dataset '{ds_name}' missing 'min'/'max' attributes "
                            f"for minmax normalisation."
                        )
                    norm_params["min"] = np.asarray(grp.attrs["min"], dtype=np.float32)
                    norm_params["max"] = np.asarray(grp.attrs["max"], dtype=np.float32)
                elif self.normalize_method == "zscore":
                    if "mean" not in grp.attrs or "std" not in grp.attrs:
                        raise KeyError(
                            f"Sub‑dataset '{ds_name}' missing 'mean'/'std' attributes "
                            f"for z‑score normalisation."
                        )
                    norm_params["mean"] = np.asarray(grp.attrs["mean"], dtype=np.float32)
                    norm_params["std"] = np.asarray(grp.attrs["std"], dtype=np.float32)
                else:
                    raise ValueError(f"Unknown normalisation method: {self.normalize_method}")
                self.norm_params[ds_name] = norm_params

        # Compute global normalisation stats for the whole dataset (used by normalize())
        self.global_stats: Dict[str, np.ndarray] = _compute_global_stats(
            self.norm_params, self.normalize_method
        )

        # Compute sample weights for balanced sampling
        if self.use_equal_sampling and split == "train":
            # Count samples per sub‑dataset in the training split
            counts: Dict[str, int] = {}
            for ds_name, _ in self.samples:
                counts[ds_name] = counts.get(ds_name, 0) + 1
            total_sub = len(self.sub_dataset_names)
            self.sample_weights: Optional[torch.Tensor] = torch.zeros(len(self.samples), dtype=torch.float64)
            for i, (ds_name, _) in enumerate(self.samples):
                self.sample_weights[i] = 1.0 / (total_sub * counts[ds_name])
        else:
            self.sample_weights = None

        self._len: int = len(self.samples)

    def _get_split_indices(
        self,
        grp: h5py.Group,
        total: int,
        ds_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads pre‑computed split indices if available; otherwise generates
        deterministic random splits using a fixed seed (42) combined with the
        sub‑dataset name.
        """
        if (
            "train_indices" in grp
            and "val_indices" in grp
            and "test_indices" in grp
        ):
            train = np.asarray(grp["train_indices"][:])
            val = np.asarray(grp["val_indices"][:])
            test = np.asarray(grp["test_indices"][:])
            # Quick validation
            assert len(train) + len(val) + len(test) == total, (
                f"Sum of split indices != total for {ds_name}"
            )
            return train, val, test

        # Generate splits deterministically
        seed = hash(ds_name) % 2**32  # use a fixed seed derived from name
        rng = np.random.RandomState(seed=abs(seed) % (2**31 - 1) + 1)
        indices = np.arange(total)
        rng.shuffle(indices)

        n_train = int(total * (1.0 - self.val_split - self.test_split))
        n_val = int(total * self.val_split)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        return train_idx, val_idx, test_idx

    def _get_file(self) -> h5py.File:
        """Return a per‑process (lazy) HDF5 file handle."""
        current_pid = os.getpid()
        if self._pid != current_pid:
            # Close old handle if any (should not happen in typical fork scenario)
            if self._file is not None:
                self._file.close()
            self._file = h5py.File(self.h5_path, "r")
            self._pid = current_pid
        return self._file

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        if idx >= self._len:
            raise IndexError(f"Index {idx} out of range [0, {self._len-1}]")

        ds_name, local_idx = self.samples[idx]
        file_handle = self._get_file()
        grp = file_handle[ds_name]

        # Read trajectory data (float16)
        raw_data = grp["data"][local_idx]  # shape (4, 3, 128, 128)
        # Convert to tensor
        data = torch.from_numpy(raw_data).float()  # upcast to float32 for operations

        # Apply per‑sub‑dataset normalization
        norm_params = self.norm_params[ds_name]
        data = normalize_field(data, method=self.normalize_method, stats=norm_params)

        dataset_id = self.dataset_ids[ds_name]
        return data, dataset_id

    def normalize(self, data: Tensor) -> Tensor:
        """
        Normalize a physical field tensor using the global statistics
        computed over all sub‑datasets.
        """
        return normalize_field(data, method=self.normalize_method, stats=self.global_stats)

