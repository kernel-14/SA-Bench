```python
## dataset.py
"""
Dataset classes for the Wavelet Diffusion Neural Operator (WDNO) reproduction.

- PDEBenchDataset      : Loads 1D Burgers, Advection, and Compressible Navier–Stokes data.
- IncompressibleFluidDataset: Loads 2D incompressible fluid and ERA5 data.
- MultiResolutionDataset: Wraps a base dataset and a WaveletTransform to produce
  wavelet‑domain samples for Base‑Resolution (BRM) and Super‑Resolution (SRM) training.
"""

import os
import glob
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import ptwt          # for 1D wavelet transforms
from wavelet_utils import WaveletTransform   # 2D/3D transforms

logger = logging.getLogger(__name__)


###############################################################################
# Helpers
###############################################################################

def _list_files_sorted(directory: str, pattern: str = "*.h5") -> List[str]:
    """Return a deterministic list of matching file paths."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        # fallback to .npz
        files = glob.glob(os.path.join(directory, "*.npz"))
    return sorted(files)


def _downsample_to_size(
    x: torch.Tensor, target_size: Union[Tuple[int, ...], List[int]], mode: str = "area"
) -> torch.Tensor:
    """
    Downsample a tensor to the given spatial size using interpolation.
    Handles 1D, 2D, and 3D inputs (with leading batch and channel dims added internally).

    Args:
        x: Tensor of shape (..., spatial_dims).  The spatial_dims are the last
            `len(target_size)` dimensions.
        target_size: Tuple of target spatial sizes (e.g., (H, W) or (T, H, W)).
        mode: Interpolation mode ('area' recommended for downsampling).

    Returns:
        Tensor with the same number of leading dimensions, spatial dims replaced by target_size.
    """
    # Save origin shape to reconstruct
    orig_shape = x.shape
    spatial_ndim = len(target_size)
    # We need to add batch and channel dims: treat as (1, 1, ...)
    for _ in range(4 - spatial_ndim):
        x = x.unsqueeze(0)  # now shape (1, 1, ... )
    # Ensure x has 4 dims (N, C, D, H) for 3D, (N, C, H, W) for 2D, (N, C, W) for 1D
    if x.dim() == 3:   # 1D spatial
        x = x.unsqueeze(2)   # (1,1,1,W) -> (1,1,1,W) actually already 4d? we added until 4d.
    # Now x is 4D. Perform interpolation.
    x = F.interpolate(x, size=target_size, mode=mode)
    # Remove the extra dimensions we added, preserving batch dim? We added a batch dim of 1,
    # we need to remove it and any channel dim if original had none.
    # We'll just reshape to (*leading_dims, *target_size) using original leading dims.
    leading_dims = orig_shape[:-spatial_ndim]
    out = x.reshape(leading_dims + tuple(target_size))
    return out


def _wavedec1d(
    x: torch.Tensor, wavelet: str, mode: str = "periodization"
) -> List[torch.Tensor]:
    """
    1D discrete wavelet transform using ptwt.
    Returns [cA, cD] as tensors of half length.
    """
    coeffs = ptwt.wavedec(x, wavelet, level=1, mode=mode)
    # wavedec returns [cA_n, cD_n, cD_{n-1}, ...] for level n. For level=1: [cA1, cD1]
    return [coeffs[0], coeffs[1]]


def _match_wavelet_1d_to_target(
    coeffs_1d: List[torch.Tensor], target_shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    Take 1D wavelet coefficients (each of length L) and repeat them to match a
    target 2D/3D wavelet tensor shape.

    The target shape is expected to be (num_subbands, T_w, H_w) or (num_subbands, T_w, H_w, W_w).
    We assume the 1D signal originally lived on the spatial axis (last dimension of target).
    The length of each coefficient should match the last dimension size of target.
    We repeat along the missing dimensions.
    """
    # target_shape: (num_channels, *spatial_dims)
    if len(target_shape) < 2:
        raise ValueError(f"target_shape must have at least 2 dims (channel, spatial), got {target_shape}")
    # number of spatial dims
    spatial_dims = target_shape[1:]
    num_coeff = len(coeffs_1d)   # should be 2
    # Build list of tensors: each coeff is 1D of length L
    out_list = []
    for coeff in coeffs_1d:
        # coeff shape (L,)
        L = coeff.shape[0]
        # we need to repeat to match each spatial dim
        # start with shape (1, L) -> we will view as (1, L) and expand
        t = coeff.view(1, 1, -1)  # (1, 1, L)
        # Now we need to expand to target spatial dims: target_shape[1:] should be e.g., (T_w, L)
        # But note: target_shape spatial dims might be e.g., (T_w, X_w) where X_w should equal L.
        # We'll check.
        if L != spatial_dims[-1]:
            raise ValueError(f"1D coeff length {L} does not match last spatial dim of target {spatial_dims[-1]}")
        # We'll expand by repeating along the preceding spatial dims.
        # For each spatial dim before the last, we repeat.
        for dim_size in spatial_dims[:-1]:
            t = t.repeat(1, dim_size, 1)   # now shape (1, dim_size, L) for first, then (1, T_w, dim_size, L)...
        # Now t has shape (1, *spatial_dims)
        # Add channel dim by stacking multiple coeffs later.
        out_list.append(t.squeeze(0))   # remove batch, keep (*spatial_dims)
    # Stack along channel dim: shape (num_coeff, *spatial_dims)
    out = torch.stack(out_list, dim=0)
    return out


###############################################################################
# PDEBenchDataset
###############################################################################

class PDEBenchDataset(Dataset):
    """
    Dataset for 1D PDEBench‑like data: Burgers (sim/ctrl), Advection, Compressible NS.

    Expects HDF5 (.h5) or NumPy (.npz) files inside a root directory.
    Each file should contain a single trajectory with datasets/arrays:
        - Burgers simulation: 'u' (81,120), 'f' (80,120), 'u0' (120,)
        - Burgers control   : 'u' (81,120), 'f' (80,120), 'u0' (120,), 'uT' (120,)
        - Advection         : 'data' (T, spatial) from PDEBench. We extract first frame as u0.
        - CFD (Navier‑Stokes): 'density' (81,120)

    The class creates a unified 'state' tensor (the variable to be learned/generated)
    of shape (1, T, X) where T is the number of time steps (may be 81 for state or 80 for force).
    All other fields are kept as condition inputs.

    Args:
        config (Dict[str, Any]): Full configuration dictionary (from Config.get_all_configs()).
        split (str): 'train', 'val', or 'test'.
    """

    def __init__(self, config: Dict[str, Any], split: str) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.experiment = config["experiment"]

        # Determine data root
        data_cfg = config["data"]
        if self.experiment in ("burgers_1d_sim", "burgers_1d_ctrl"):
            self.root = data_cfg["burgers_root"]
        elif self.experiment == "advection_1d":
            self.root = data_cfg["advection_root"]
        elif self.experiment == "cfd_1d":
            self.root = data_cfg["cfd_root"]
        else:
            raise ValueError(f"Unsupported experiment for PDEBenchDataset: {self.experiment}")

        # Split sizes
        self.num_train = data_cfg.get("num_train", 40000)
        self.num_val   = data_cfg.get("num_val", 2000)
        self.num_test  = data_cfg.get("num_test", 50)

        # Load all files with deterministic ordering
        seed = config["seed"]
        rng = np.random.RandomState(seed)
        all_files = _list_files_sorted(self.root, "*.h5")
        if not all_files:
            raise FileNotFoundError(f"No .h5 or .npz files found in {self.root}")
        # Ensure reproducibility by using the same random permutation as in data generation?
        # We'll simply use the sorted list and apply a fixed shuffle (seed) to split.
        rng.shuffle(all_files)

        # Select files for the requested split
        if split == "train":
            start, end = 0, self.num_train
        elif split == "val":
            start, end = self.num_train, self.num_train + self.num_val
        elif split == "test":
            start, end = self.num_train + self.num_val, self.num_train + self.num_val + self.num_test
        else:
            raise ValueError(f"Unknown split: {split}")
        self.files = all_files[start:end]

        # We will load all trajectories into memory (small size)
        self.samples: List[Dict[str, torch.Tensor]] = []
        self._load_trajectories()

        # Derived shapes
        if self.experiment in ("burgers_1d_sim", "burgers_1d_ctrl"):
            # state is u for simulation, f for control? We'll decide based on task.
            if "ctrl" in self.experiment:
                self.state_shape = self.samples[0]["f"].shape   # (80, 120)
                self.spatial_res = self.state_shape[1]
            else:
                self.state_shape = self.samples[0]["u"].shape   # (81, 120)
                self.spatial_res = self.state_shape[1]
        elif self.experiment == "advection_1d":
            self.state_shape = self.samples[0]["u"].shape       # (81, 120)
            self.spatial_res = self.state_shape[1]
        elif self.experiment == "cfd_1d":
            self.state_shape = self.samples[0]["u"].shape       # (81, 120)
            self.spatial_res = self.state_shape[1]

    def _load_h5(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Load one trajectory from an HDF5 file."""
        with h5py.File(file_path, "r") as f:
            data = {}
            if self.experiment in ("burgers_1d_sim", "burgers_1d_ctrl"):
                data["u"] = torch.from_numpy(f["u"][:]).float()
                data["f"] = torch.from_numpy(f["f"][:]).float()
                data["u0"] = torch.from_numpy(f["u0"][:]).float()
                if "ctrl" in self.experiment:
                    data["uT"] = torch.from_numpy(f["uT"][:]).float()
            elif self.experiment == "advection_1d":
                # PDEBench advection: 'data' shape (T, X, 1)
                arr = f["data"][:].astype(np.float32)  # shape (T, X, 1)
                # squeeze last dim
                arr = np.squeeze(arr, axis=-1)  # (T, X)
                # we need 81 timesteps, 120 spatial -> resize
                T, X = arr.shape
                if T != 81 or X != 120:
                    # Use simple interpolation to resize
                    from scipy.interpolate import RegularGridInterpolator
                    old_t = np.linspace(0, 1, T)
                    old_x = np.linspace(0, 1, X)
                    new_t = np.linspace(0, 1, 81)
                    new_x = np.linspace(0, 1, 120)
                    interp = RegularGridInterpolator((old_t, old_x), arr)
                    tt, xx = np.meshgrid(new_t, new_x, indexing='ij')
                    arr = interp(np.stack([tt, xx], axis=-1))
                    arr = arr.astype(np.float32)
                data["u"] = torch.from_numpy(arr)
                data["u0"] = data["u"][0]   # first frame
            elif self.experiment == "cfd_1d":
                arr = f["density"][:].astype(np.float32)  # shape (T, X)
                if arr.shape != (81, 120):
                    # resize as above
                    from scipy.interpolate import RegularGridInterpolator
                    old_t = np.linspace(0, 1, arr.shape[0])
                    old_x = np.linspace(0, 1, arr.shape[1])
                    new_t = np.linspace(0, 1, 81)
                    new_x = np.linspace(0, 1, 120)
                    interp = RegularGridInterpolator((old_t, old_x), arr)
                    tt, xx = np.meshgrid(new_t, new_x, indexing='ij')
                    arr = interp(np.stack([tt, xx], axis=-1)).astype(np.float32)
                data["u"] = torch.from_numpy(arr)
                data["u0"] = data["u"][0]
        return data

    def _load_npz(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Load one trajectory from a .npz file."""
        arrs = np.load(file_path)
        data = {}
        for key in arrs.files:
            data[key] = torch.from_numpy(arrs[key]).float()
        return data

    def _load_trajectories(self) -> None:
        """Read all files and store samples."""
        for file in self.files:
            if file.endswith(".h5"):
                sample = self._load_h5(file)
            elif file.endswith(".npz"):
                sample = self._load_npz(file)
            else:
                raise ValueError(f"Unsupported file type: {file}")
            # Ensure all data is float32 and on CPU
            for k in sample:
                sample[k] = sample[k].float()
            # Add 'state' key: the variable to be generated.
            if "ctrl" in self.experiment:
                # control task: state is f
                sample["state"] = sample["f"].unsqueeze(0)   # (1, 80, 120)
            else:
                # simulation task: state is u (including t=0)
                sample["state"] = sample["u"].unsqueeze(0)    # (1, 81, 120)
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


###############################################################################
# IncompressibleFluidDataset
###############################################################################

class IncompressibleFluidDataset(Dataset):
    """
    Dataset for 2D incompressible fluid (sim/ctrl) and ERA5.

    Data files are assumed to be .h5 or .npz, each containing a dictionary with fields:
      - Fluid sim: 'initial_density' (64,64 or 1,64,64), 'control' (T, C, H, W) or (T, H, W*C),
          'density', 'velocity_x', 'velocity_y' each (T, 64, 64),
          'bucket_percentage' (T,).
      - Fluid ctrl: same structure, but the learning target is 'control'.
      - ERA5: 'past' (12, H, W), 'future' (20, H, W).

    The 'state' tensor is constructed as a multi‑channel volume of shape (C, T, H, W)
    by stacking the dynamic fields (e.g., density, vx, vy).  For ERA5 it is simply 'future'.

    Args:
        config (Dict[str, Any]): Full configuration.
        split (str): 'train', 'val', 'test'.
    """

    def __init__(self, config: Dict[str, Any], split: str) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.experiment = config["experiment"]

        data_cfg = config["data"]
        if self.experiment in ("fluid_2d_sim", "fluid_2d_ctrl"):
            self.root = data_cfg["fluid_root"]
        elif self.experiment == "era5":
            self.root = data_cfg["era5_root"]
        else:
            raise ValueError(f"Unsupported experiment for IncompressibleFluidDataset: {self.experiment}")

        self.num_train = data_cfg.get("num_train", 40000)
        self.num_val   = data_cfg.get("num_val", 2000)
        self.num_test  = data_cfg.get("num_test", 50)

        seed = config["seed"]
        rng = np.random.RandomState(seed)
        all_files = _list_files_sorted(self.root, "*.h5")
        if not all_files:
            raise FileNotFoundError(f"No data files found in {self.root}")
        rng.shuffle(all_files)

        if split == "train":
            start, end = 0, self.num_train
        elif split == "val":
            start, end = self.num_train, self.num_train + self.num_val
        elif split == "test":
            start, end = self.num_train + self.num_val, self.num_train + self.num_val + self.num_test
        else:
            raise ValueError(f"Unknown split: {split}")
        self.files = all_files[start:end]

        self.samples: List[Dict[str, torch.Tensor]] = []
        self._load_trajectories()

        # Extract shapes from first sample
        if self.experiment in ("fluid_2d_sim", "fluid_2d_ctrl"):
            first = self.samples[0]
            self.state_shape = first["state"].shape   # (C, T, H, W)
            self.temporal_len = self.state_shape[1]
            self.spatial_res = (self.state_shape[2], self.state_shape[3])
        elif self.experiment == "era5":
            first = self.samples[0]
            self.state_shape = first["state"].shape   # (1, 20, H, W) maybe
            self.temporal_len = self.state_shape[1]
            self.spatial_res = (self.state_shape[2], self.state_shape[3])

    def _load_trajectories(self) -> None:
        for file in self.files:
            if file.endswith(".h5"):
                sample = self._load_h5(file)
            elif file.endswith(".npz"):
                arrs = np.load(file)
                sample = {k: torch.from_numpy(arrs[k]).float() for k in arrs.files}
            else:
                raise ValueError(f"Unsupported file type: {file}")

            # Build 'state' tensor
            if self.experiment in ("fluid_2d_sim", "fluid_2d_ctrl"):
                # Stack density, velocity_x, velocity_y
                # Ensure each field shape is (T, H, W)
                density = sample["density"].unsqueeze(1) if sample["density"].dim() == 2 else sample["density"]
                vx = sample["velocity_x"].unsqueeze(1) if sample["velocity_x"].dim() == 2 else sample["velocity_x"]
                vy = sample["velocity_y"].unsqueeze(1) if sample["velocity_y"].dim() == 2 else sample["velocity_y"]
                # stack along channel dim -> (C, T, H, W) after transpose
                # each is (T, H, W) after unsqueeze? Actually they are (T, H, W). We'll stack as new channel.
                state = torch.stack([density, vx, vy], dim=0)  # (3, T, H, W) — but need channel first
                # make sure channel dim is 0: if they originally were (T, H, W), stack(0) will be (3, T, H, W) correct.
                sample["state"] = state
                # Ensure other fields have consistent shapes
                # initial_density: (H, W) -> (1, H, W)
                if "initial_density" in sample and sample["initial_density"].dim() == 2:
                    sample["initial_density"] = sample["initial_density"].unsqueeze(0)
                # control: assume shape (T, C, H, W) or (T, ...). We'll keep as is.
            elif self.experiment == "era5":
                # state = future (20, H, W) as (1, 20, H, W)
                future = sample["future"].unsqueeze(0)  # (1,20,H,W)
                sample["state"] = future
                # past: (12, H, W) -> maybe keep as (12, H, W) or add channel dim
                sample["past"] = sample["past"] if sample["past"].dim() == 3 else sample["past"].unsqueeze(0)

            self.samples.append(sample)

    def _load_h5(self, file_path: str) -> Dict[str, torch.Tensor]:
        with h5py.File(file_path, "r") as f:
            data = {}
            for key in f.keys():
                arr = f[key][:]
                if isinstance(arr, np.ndarray):
                    data[key] = torch.from_numpy(arr).float()
                else:
                    logger.warning(f"Skipping non‑array dataset {key} in {file_path}")
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


###############################################################################
# MultiResolutionDataset
###############################################################################

class MultiResolutionDataset(Dataset):
    """
    Dataset wrapper that applies wavelet transforms and resolution scaling,
    producing samples ready for the diffusion models.

    - In 'brm' mode, it returns the wavelet coefficients of the state at the
      original (highest) resolution together with the wavelet‑transformed
      high‑resolution condition fields.
    - In 'srm' mode, it returns wavelet coefficients of a high‑resolution state
      (at a finer scale), the upsampled wavelet coefficients of a low‑resolution
      state (at a coarser scale), and the wavelet‑transformed condition fields at
      the high‑resolution scale.

    Args:
        base_dataset: Instance of PDEBenchDataset or IncompressibleFluidDataset.
        scales: List of spatial scaling factors (e.g., [1.0, 0.5, 0.25]). Sorted internally.
        wavelet_transform: Configured WaveletTransform (2D for 1D cases, 3D for 2D cases).
        mode: 'brm' or 'srm'.
        state_keys: List of keys in the base sample that together form the state tensor
                    (default: ['state']). The state tensor must have shape (C, T, ...).
        cond_keys_high: Keys of condition fields that should be kept at high resolution
                        (for BRM the original resolution, for SRM the target high scale).
        time_scale_with_spatial: If True, the temporal dimension is scaled by the same
                                 factor as the spatial dimensions.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        scales: List[float],
        wavelet_transform: WaveletTransform,
        mode: str,
        state_keys: Optional[List[str]] = None,
        cond_keys_high: Optional[List[str]] = None,
        time_scale_with_spatial: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("brm", "srm"):
            raise ValueError(f"mode must be 'brm' or 'srm', got {mode}")

        self.base = base_dataset
        self.wavelet_transform = wavelet_transform
        self.mode = mode
        self.state_keys = state_keys if state_keys is not None else ["state"]
        self.cond_keys_high = cond_keys_high if cond_keys_high is not None else []
        self.time_scale_with_spatial = time_scale_with_spatial

        # Parse wavelet info
        self.wavelet_ndim = wavelet_transform.ndim
        if self.wavelet_ndim == 2:
            self.n_subbands = 4
        elif self.wavelet_ndim == 3:
            self.n_subbands = 8
        else:
            raise ValueError(f"Unsupported wavelet ndim: {self.wavelet_ndim}")

        # Sort scales descending (largest to smallest)
        self.scales = sorted(scales, reverse=True)
        if len(self.scales) < 2 and mode == "srm":
            raise ValueError("SRM mode requires at least two scales")

        # Get reference state shape from one sample
        sample0 = self.base[0]
        # We expect a single state tensor under self.state_keys[0] (or we can combine multiple,
        # but for simplicity assume one key)
        state0 = sample0[self.state_keys[0]]
        self.orig_state_shape = state0.shape  # (C, T, ...) where T may be 1 for ERA5 or fixed.
        self.ori_C = state0.shape[0]
        self.ori_time = state0.shape[1]
        self.ori_spatial = state0.shape[2:]   # remaining spatial dims

        # Precompute target sizes for each scale
        self.target_sizes: List[Tuple[float, Tuple[int, ...]]] = []
        for s in self.scales:
            if s == 1.0:
                target = state0.shape
            else:
                new_time = round(self.ori_time * s) if time_scale_with_spatial else self.ori_time
                new_spatial = tuple(round(d * s) for d in self.ori_spatial)
                target = (self.ori_C, new_time) + new_spatial
            self.target_sizes.append((s, target))

        # For SRM, create list of adjacent pair indices
        if mode == "srm":
            self.pair_indices = [(i, i+1) for i in range(len(self.scales)-1)]

    def __len__(self) -> int:
        if self.mode == "brm":
            return len(self.base)
        else:
            return len(self.base) * len(self.pair_indices)

    def _get_base_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieve a raw sample from the base dataset."""
        if self.mode == "brm":
            return self.base[idx]
        else:
            base_idx = idx // len(self.pair_indices)
            return self.base[base_idx]

    def _get_scale_pair(self, idx: int) -> Tuple[float, float, Tuple[int, ...], Tuple[int, ...]]:
        """For SRM, return (scale_high, scale_low, size_high, size_low)."""
        pair_idx = idx % len(self.pair_indices)
        i_high, i_low = self.pair_indices[pair_idx]
        scale_high, size_high = self.target_sizes[i_high]
        scale_low, size_low = self.target_sizes[i_low]
        return scale_high, scale_low, size_high[1:], size_low[1:]   # drop C from size

    def _downsample_state(self, state: torch.Tensor, scale: float) -> torch.Tensor:
        """Downsample state tensor to the given scale."""
        if scale == 1.0:
            return state
        _, size = next((s, sz) for s, sz in self.target_sizes if abs(s - scale) < 1e-6)
        target_spatial = size[2:] if state.dim() == 3 else size[1:]  # handle (C,T,X) vs (C,T,H,W)
        # For state, we need to downscale along time and spatial.
        # We'll use _downsample_to_size with full target shape (C, T', *spatial')
        return _downsample_to_size(state, target_spatial)   # function expects spatial dims only; we need to include time. So we'll implement internal method.

    def _downsample_tensor(self, x: torch.Tensor, scale: float, is_1d: bool = False) -> torch.Tensor:
        """
        Generic downsampling to a scale factor relative to the original resolution
        of the base dataset (for condition fields).
        """
        if scale == 1.0:
            return x
        if is_1d:
            # x shape (L,)
            L = x.shape[0]
            new_len = round(L * scale)
            return _downsample_to_size(x, (new_len,))
        else:
            # Assume x has shape (..., spatial_dims). Determine spatial_dims count.
            if x.dim() == 2:
                # 2D tensor (H, W)
                H, W = x.shape
                new_H = round(H * scale)
                new_W = round(W * scale)
                return _downsample_to_size(x, (new_H, new_W))
            elif x.dim() == 3:
                # (T, H, W) or (C, H, W) – we treat last two as spatial; first dim could be time or channel.
                # For condition fields like control which are (C, T, H, W) for fluid, we'll handle specially
                T, H, W = x.shape
                new_H = round(H * scale)
                new_W = round(W * scale)
                if self.time_scale_with_spatial:
                    new_T = round(T * scale)
                else:
                    new_T = T
                return _downsample_to_size(x, (new_T, new_H, new_W))
            else:
                raise ValueError(f"Unsupported tensor dimensionality {x.dim()} for downsampling")

    def _wavelet_transform_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply wavelet transform to a state tensor of shape (C, *spatial).
        Returns wavelet coefficients stacked into (C*K, *spatial_w).
        """
        # state: (C, T, ...) or (C, H, W) depending on ndim
        C = state.shape[0]
        spatial = state.shape[1:]   # remaining dims
        coeffs_per_channel = []
        for c in range(C):
            s = state[c]  # shape (T, X) for 2D, (T, H, W) for 3D
            coeffs = self.wavelet_transform.forward(s)  # list of K tensors, each same spatial_w
            coeffs_per_channel.append(coeffs)
        # For each subband index, stack across channels
        stacked = []
        K = len(coeffs_per_channel[0])
        for k in range(K):
            sub = torch.stack([ch[k] for ch in coeffs_per_channel], dim=0)  # (C, *spatial_w)
            stacked.append(sub)
        # Concatenate along channel dimension -> (C*K, *spatial_w)
        return torch.cat(stacked, dim=0)

    def _transform_condition(
        self, cond: torch.Tensor, target_spatial_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Transform a condition tensor to wavelet coefficients, matching the target
        wavelet spatial shape (which is the state's wavelet spatial dims).
        Handles 1D conditions (u0, uT) by 1D wavelet + repetition,
        and 2D/3D conditions by direct wavelet.
        """
        if cond.dim() == 1:
            # 1D signal -> 1D wavelet
            coeffs = _wavedec1d(cond, self.wavelet_transform.wavelet, self.wavelet_transform.mode)
            # coeffs: [cA, cD] each of half length
            # Build target wavelet tensor shape: (num_subbands, *target_spatial_shape)
            # target_spatial_shape is e.g., (T_w, X_w) for 2D wavelet, (T_w, H_w, W_w) for 3D.
            # Number of subbands we will produce: only 2 (cA, cD) from 1D.
            # However, the state wavelet has K (=4 for 2D, 8 for 3D) subbands.
            # We need to match the number of channels. The paper says "repeat the coefficients"
            # but not clear. It likely means they repeat the 1D coefficients to form a 2D/3D volume
            # that can be concatenated with the state wavelet's subbands as extra channels.
            # But the state wavelet has e.g., 4 channels (cA, cH, cV, cD). How do 2 channels from 1D become
            # part of the condition? The paper: "Since the initial condition and the target state are 1D,
            # we take the 1D wavelet transform, repeat the coefficients, and then concatenate them
            # with the 2D coefficients." This implies they repeat the 1D coefficients to produce
            # a tensor of the same number of channels as the 2D wavelet (4 channels). Possibly they
            # replicate each 1D coefficient across the spatial dimensions to create a 2D array for each
            # of the 4 subbands. I.e., for each of the 4 subbands, they create a 2D array of the same
            # shape as the state wavelet subband, where the values are taken from the corresponding 1D
            # coefficient repeated along the time dimension. That would give a condition with 4 channels.
            # But how to assign which 1D coefficient to which 2D subband? Possibly they treat the 1D
            # approx as the low-pass part and fill all 4 channels with the same? Not clear.
            # A simpler implementation that still yields good results: we can treat the 1D condition
            # as a 2D condition by repeating it over the time dimension first, then applying 2D wavelet,
            # which would produce 4 channels. That seems more principled. However, the paper explicitly
            # says they take 1D wavelet transform, repeat coefficients, and concatenate. 
            # Let's do: take 1D wavelet, get [cA, cD]. We need to produce a tensor with K channels
            # matching target shape (K, T_w, X_w). We can simply repeat each coefficient to fill a 2D array
            # (T_w, X_w) by repeating along the time dimension. Then we have 2 such 2D arrays, but we need
            # K channels. We could replicate each to fill all K channels? Not great.
            # Better: first expand the 1D signal to 2D by repeating over time to match the full state spatial
            # dimensions (before wavelet) and then apply 2D wavelet. That would generate 4 subbands.
            # Since the paper says "repeat the coefficients" after 1D wavelet, we'll do:
            #   coeffs_1d = [cA, cD] each of length L.
            # For each, we create a 2D array of shape (T_w, X_w) by repeating along the time dim.
            # Then we have 2 arrays. To get K channels, we can stack them and then pad with zeros? Or we can
            # treat the 1D condition as providing 2 channels and the state wavelet has K channels, so
            # total condition channels = 2 + (from other conditions). Actually the paper says the condition
            # for 1D simulation is 6 channels: u0 (2 subbands from 1D) + f (4 subbands from 2D) = 6.
            # So they directly use the 2 subbands from 1D as separate channels, not expanded to 4.
            # That makes sense: they take 1D wavelet coefficients, each is 1D; then they repeat them
            # along the time dimension to match the wavelet spatial shape (T_w, X_w). They do not force
            # them to have 4 channels; they have 2 channels that are repeated spatially.
            # So the condition channels will be (2 + 4) = 6, and the state is 4. The denoiser receives
            # 4 + 6 = 10 input channels. That matches Table 18.
            # So we will produce a tensor of shape (2, *target_spatial_shape) by repeating each 1D coeff
            # along the time axis.
            out = _match_wavelet_1d_to_target(coeffs, target_spatial_shape)
            return out
        else:
            # 2D or 3D: apply same wavelet transform as state (multiple channels possible)
            # cond shape (..., H, W) or (..., T, H, W)
            if cond.dim() == 2:
                # single channel 2D
                coeffs = self.wavelet_transform.forward(cond)
                # coeffs list of K tensors each 2D; stack channels
                return torch.stack(coeffs, dim=0)  # (K, H_w, W_w)
            elif cond.dim() == 3:
                # might be multi-channel or (C, H, W) or (T, H, W). We'll assume first dim is channel/time.
                # For simplicity, we'll treat each "channel" of the first dim as a separate 2D slice
                # if ndim==2, or if ndim==3, we'd expect cond to be 3D volume (T, H, W).
                # If wavelet_ndim == 3, we can directly use forward.
                if self.wavelet_ndim == 2:
                    # treat first dim as channel (C) and apply 2D wavelet per channel
                    C = cond.shape[0]
                    coeffs_all = []
                    for c in range(C):
                        coeffs = self.wavelet_transform.forward(cond[c])
                        coeffs_all.append(coeffs)
                    K = len(coeffs_all[0])
                    stacked = [torch.stack([ch[k] for ch in coeffs_all], dim=0) for k in range(K)]
                    return torch.cat(stacked, dim=0)  # (C*K, H_w, W_w)
                else: # ndim==3
                    # cond should be (T, H, W) or (C, T, H, W). We'll assume shape matches 3D wavelet.
                    coeffs = self.wavelet_transform.forward(cond)
                    return torch.stack(coeffs, dim=0)  # (8, T_w, H_w, W_w) maybe
            else:
                raise ValueError(f"Unsupported condition tensor shape {cond.shape}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == "brm":
            sample = self._get_base_sample(idx)
            # Get state tensor
            state = sample[self.state_keys[0]]  # shape (C, T, ...)
            # No downsampling, use highest scale (1.0)
            W_state = self._wavelet_transform_state(state)  # (C*K, *spatial_w)
            # Process condition fields
            cond_wavelets = []
            target_shape = W_state.shape   # will be (C*K, ...)
            for key in self.cond_keys_high:
                cond_raw = sample[key]
                # Transform condition at original resolution (no downsampling)
                W_cond = self._transform_condition(cond_raw, target_shape[1:])
                cond_wavelets.append(W_cond)
            if cond_wavelets:
                W_cond = torch.cat(cond_wavelets, dim=0)
            else:
                W_cond = torch.empty(0)
            return {"W_state": W_state, "W_cond": W_cond}

        else:  # SRM mode
            sample = self._get_base_sample(idx)
            scale_high, scale_low, size_high, size_low = self._get_scale_pair(idx)
            # High-resolution state
            state_raw = sample[self.state_keys[0]]
            state_high = self._downsample_tensor(state_raw, scale_high)   # (C, T_high, ...)
            state_low = self._downsample_tensor(state_raw, scale_low)     # (C, T_low, ...)

            W_high = self._wavelet_transform_state(state_high)
            W_low = self._wavelet_transform_state(state_low)
            # Upsample low to match high wavelet spatial size
            # Use nearest interpolation
            W_low_up = F.interpolate(
                W_low.unsqueeze(0),   # add batch dim
                size=W_high.shape[1:],   # spatial dims only
                mode="nearest"
            ).squeeze(0)

            # Condition: high-resolution parameters at the high scale
            cond_wavelets = []
            target_shape = W_high.shape
            for key in self.cond_keys_high:
                cond_raw = sample[key]
                # Downsample condition to scale_high
                cond_scaled = self._downsample_tensor(cond_raw, scale_high)
                W_cond = self._transform_condition(cond_scaled, target_shape[1:])
                cond_wavelets.append(W_cond)
            if cond_wavelets:
                W_cond_high = torch.cat(cond_wavelets, dim=0)
            else:
                W_cond_high = torch.empty(0)

            return {"W_high": W_high, "W_low_up": W_low_up, "W_cond_high": W_cond_high}


def collate_dict(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.T