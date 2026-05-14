# dataset.py
# ============================================================================
# Purpose: Implement a PyTorch Dataset that loads pre‑generated SC‑FNO data from
#          an HDF5 file and provides samples for training, validation, and testing.
#          The data on disk is assumed to have been written by DataGenerator with
#          the exact dataset names: 'p', 'u_input', 'u_true', 'J_true'.
# ============================================================================

import os
import random
from typing import Optional, List, Tuple

import h5py
import torch
from torch.utils.data import Dataset

from config import Config  # only needed for construction


class PDEDataset(Dataset):
    """PyTorch Dataset for SC‑FNO training, validation and testing.

    Each sample consists of:
        p       : parameter vector (n_params,)
        u_input : initial solution segment   (..., M)
        u_true  : target solution evolution  (..., N_time-M)
        J_true  : Jacobian of target w.r.t. parameters (n_params, ..., N_time-M)

    The HDF5 file is read entirely into memory at instantiation for fast access.
    Train / val / test splits are determined by a reproducible random permutation
    of the dataset indices, following the ratios given in the configuration.
    """

    def __init__(
        self,
        config: Config,
        split: str = "train",
        n_train_samples: Optional[int] = None,
        file_path: Optional[str] = None,
    ) -> None:
        """Create a dataset instance.

        Args:
            config:           Global configuration object (frozen dataclass).
            split:            Which subset to load; one of 'train', 'val', 'test'.
            n_train_samples:  If not None and split == 'train', restrict the
                              training set to this number of samples (used for
                              data‑volume experiments).
            file_path:        Explicit path to the HDF5 file.  When None, a default
                              path is constructed from config.global_params.data_dir
                              and config.equation.

        Raises:
            FileNotFoundError: If the HDF5 file does not exist.
            KeyError:          If the expected dataset groups are missing.
            ValueError:        If the requested split is invalid.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Determine the HDF5 file path.
        # ------------------------------------------------------------------
        if file_path is None:
            data_dir = config.global_params["data_dir"]
            eq_name = config.equation
            file_path = os.path.join(data_dir, f"{eq_name}_data.h5")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"HDF5 data file not found: {file_path}")

        # ------------------------------------------------------------------
        # 2. Load all data into memory.
        # ------------------------------------------------------------------
        with h5py.File(file_path, "r") as hf:
            # The keys are 'p', 'u_input', 'u_true', 'J_true' as stored by DataGenerator.
            p_np = hf["p"][:]           # shape (N, n_params)
            u_in_np = hf["u_input"][:]  # shape varies with equation
            u_t_np = hf["u_true"][:]
            J_t_np = hf["J_true"][:]

        # Convert to torch tensors (float32, CPU).
        self.p = torch.from_numpy(p_np).float()
        self.u_input = torch.from_numpy(u_in_np).float()
        self.u_true = torch.from_numpy(u_t_np).float()
        self.J_true = torch.from_numpy(J_t_np).float()

        # ------------------------------------------------------------------
        # 3. Shuffle and split indices.
        # ------------------------------------------------------------------
        n_samples = self.p.shape[0]
        seed = config.global_params["seed"]
        rng = random.Random(seed)
        indices = list(range(n_samples))
        rng.shuffle(indices)

        # Ratios from configuration
        data_cfg = config.data_params
        train_ratio = data_cfg["train_ratio"]
        val_ratio = data_cfg["val_ratio"]

        # Compute cut points (int floors, remainder goes to test)
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)

        # Select the appropriate slice
        if split == "train":
            self.indices = indices[:train_end]
            # Optional sub‑sampling for data‑volume experiments
            if n_train_samples is not None:
                self.indices = self.indices[:n_train_samples]
        elif split == "val":
            self.indices = indices[train_end:val_end]
        elif split == "test":
            self.indices = indices[val_end:]
        else:
            raise ValueError(f"Invalid split name: '{split}'.  Must be 'train', 'val', or 'test'.")

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Retrieve a single sample.

        Args:
            idx: Index within the current split (0‑based).

        Returns:
            Tuple of four tensors: (p, u_input, u_true, J_true).
            All tensors reside on CPU.
        """
        # Resolve the global index from the (potentially shuffled) list
        real_idx = self.indices[idx]
        return (
            self.p[real_idx],
            self.u_input[real_idx],
            self.u_true[real_idx],
            self.J_true[real_idx],
        )

