```python
## evaluate.py
"""
Evaluation module for the Wavelet Diffusion Neural Operator (WDNO) reproduction.

The ``Evaluator`` class computes standard metrics for simulation, control,
and zero‑shot super‑resolution tasks, strictly following the protocols described
in the paper (WDNO Tables 1‑2, Figure 4, etc.). It relies on a fully‑trained
``WDNO`` instance and a test dataset. For control evaluation, a ground‑truth
solver (not part of this module) is called externally; if unavailable, control
evaluation will be skipped with a warning.
"""

import json
import logging
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Project‑specific imports – adjust if the module layout differs
from config import Config
from models.wdno import WDNO
from dataset import PDEBenchDataset, IncompressibleFluidDataset
from utils import get_device, set_seed

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def _as_float(val: Union[float, torch.Tensor]) -> float:
    """Convert a tensor with a single element to a Python float."""
    if isinstance(val, torch.Tensor):
        return val.item()
    return float(val)


def _normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    """Helper to ensure tensor is on CPU and float32."""
    return x.detach().cpu().float()


def _compute_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask_initial: bool = False,
) -> torch.Tensor:
    """Compute MSE, optionally ignoring the first time frame.

    Args:
        pred: (B, T, ...) or (B, C, T, ...)
        target: same shape as pred.
        mask_initial: if True, the first time step is excluded from the loss.

    Returns:
        Scalar MSE tensor.
    """
    if mask_initial:
        # The masking logic is handled per experiment in the caller.
        pass
    return F.mse_loss(pred, target, reduction='mean')


def _interpolate_to_shape(
    tensor: torch.Tensor,
    target_shape: Tuple[int, ...],
    mode: str = "linear",
) -> torch.Tensor:
    """
    Interpolate a tensor to match `target_shape` spatially.

    Args:
        tensor: Input tensor of shape (B, C, *spatial_dims).
        target_shape: Tuple of target spatial dimensions (e.g., (T, X) or (T, H, W)).
        mode: Interpolation mode ('linear', 'nearest', ...).

    Returns:
        Interpolated tensor with same batch and channel dims, spatial replaced.
    """
    ndim_spat = len(target_shape)
    if ndim_spat == 2:
        # (H, W)
        if tensor.dim() != 4:
            raise ValueError(f"Expected 4‑D tensor for 2D interp, got {tensor.shape}")
        return F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False if mode != 'nearest' else None)
    elif ndim_spat == 3:
        # (T, H, W) – use 3D interpolation
        if tensor.dim() != 5:
            raise ValueError(f"Expected 5‑D tensor for 3D interp, got {tensor.shape}")
        mode_3d = 'trilinear' if mode == 'linear' else mode
        return F.interpolate(tensor, size=target_shape, mode=mode_3d)
    else:
        raise ValueError(f"Unsupported number of spatial dims: {ndim_spat}")


def _wavedec1d(
    x: torch.Tensor, wavelet: str, mode: str = "periodization"
) -> List[torch.Tensor]:
    """1D discrete wavelet transform returning [cA, cD]."""
    import ptwt
    coeffs = ptwt.wavedec(x, wavelet, level=1, mode=mode)
    return [coeffs[0], coeffs[1]]


def _repeat_1d_coeffs_to_target(
    coeffs_1d: List[torch.Tensor],
    target_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Repeat 1D wavelet coefficients to fill a higher‑dimensional grid.

    Args:
        coeffs_1d: list of two tensors [cA, cD], each 1‑D.
        target_shape: target spatial shape, e.g. (H', W') or (T', H', W').

    Returns:
        Tensor of shape (2, *target_shape).
    """
    if len(coeffs_1d) != 2:
        raise ValueError("Expected exactly two 1D coefficients.")
    L = coeffs_1d[0].shape[0]
    if L != target_shape[-1]:
        raise ValueError(f"1D coeff length {L} does not match last dim of target {target_shape[-1]}")
    repeated = []
    for coeff in coeffs_1d:
        view_shape = (1,) + (1,) * (len(target_shape) - 1) + (L,)
        t = coeff.view(*view_shape)
        t = t.expand(1, *target_shape)  # (1, *target_shape)
        repeated.append(t.squeeze(0))   # (*target_shape)
    return torch.stack(repeated, dim=0)


# ------------------------------------------------------------------------------
# Evaluator class
# ------------------------------------------------------------------------------

class Evaluator:
    """
    Computes simulation, control, and super‑resolution metrics for a trained WDNO.

    Args:
        wdno: Trained WDNO instance (with BRM and optionally SRM).
        test_dataset: Dataset providing test samples. Should be a
            ``PDEBenchDataset`` or ``IncompressibleFluidDataset``.
        config: Experiment configuration.
        solver: Optional callable that can compute the final state or objective
            given ``(u0, f)`` in physical space. If not provided and control
            evaluation is requested, a warning is issued and evaluation is skipped.
        highres_dataset: Optional dataset for super‑resolution evaluation at the
            highest resolution. If ``None``, super‑resolution evaluation is skipped.
    """

    def __init__(
        self,
        wdno: WDNO,
        test_dataset: Union[PDEBenchDataset, "IncompressibleFluidDataset"],
        config: Config,
        solver: Optional[Callable[..., torch.Tensor]] = None,
        highres_dataset: Optional[Union[PDEBenchDataset, "IncompressibleFluidDataset"]] = None,
    ) -> None:
        self.wdno = wdno
        self.test_dataset = test_dataset
        self.config = config
        self.solver = solver
        self.highres_dataset = highres_dataset

        self.experiment_name = config.get_experiment_name()
        self.device = get_device(config.get_device())
        self.eval_cfg = config.get_eval_config()
        self.num_samples = self.eval_cfg.get("num_samples", None)
        self.log_interval = self.eval_cfg.get("log_interval", 10)

        # Directory for saving results
        self.results_dir = Path("results") / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set seed for reproducibility during evaluation
        set_seed(config.get_seed())

        # Move models to evaluation device
        self.wdno.brm.to(self.device)
        if self.wdno.srm is not None:
            self.wdno.srm.to(self.device)
        self.wdno.brm.denoiser.eval()
        if self.wdno.srm is not None:
            self.wdno.srm.denoiser.eval()

        # Determine experiment type
        self.is_1d = "1d" in self.experiment_name
        self.is_2d = "2d" in self.experiment_name or "era5" in self.experiment_name
        self.is_control = "ctrl" in self.experiment_name

        # For simulation, we need to know the time dimension index.
        if self.is_1d:
            self.time_dim = 1  # in (B, C=1, T, X)
        else:
            self.time_dim = 2  # in (B, C, T, H, W)

        # Reference sample shape (for spatial dim detection)
        sample0 = self.test_dataset[0]
        self.state_shape_phys = sample0["state"].shape  # (C, T, ...)
        # Number of wavelet subbands
        self.num_subbands = 4 if self.wdno.wavelet_transform.ndim == 2 else 8

    # --------------------------------------------------------------------------
    # Public evaluation methods
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_simulation(self) -> Dict[str, float]:
        """
        Compute MSE, MAE, and Linf for simulation tasks, excluding the initial
        time step (t=0) as specified in the paper.
        """
        logger.info(f"Starting simulation evaluation on '{self.experiment_name}'.")

        dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        total_mse = 0.0
        total_mae = 0.0
        total_linf = 0.0
        count = 0

        for idx, sample in enumerate(dataloader):
            sample = self._dict_to_device(sample, self.device)
            # Build condition dict for WDNO.simulate
            condition = self._build_condition(sample, mode='sim')
            # Ground truth state (full trajectory)
            gt_state = sample["state"]  # (1, T, X) or (C, T, H, W)
            # Exclude initial condition frame (t=0)
            if self.is_1d:
                gt_target = gt_state[:, :, 1:]  # (1, T-1, X)
            else:
                gt_target = gt_state[:, :, 1:]  # (C, T-1, H, W)

            # Generate prediction
            pred_state = self.wdno.simulate(condition, guidance_w=self._get_sim_guidance_weight())
            # Remove initial frame
            if self.is_1d:
                pred_target = pred_state[:, :, 1:]
            else:
                pred_target = pred_state[:, :, 1:]

            # Compute metrics per sample
            mse = F.mse_loss(pred_target, gt_target, reduction='mean').item()
            mae = F.l1_loss(pred_target, gt_target, reduction='mean').item()
            linf = (pred_target - gt_target).abs().max().item()

            total_mse += mse
            total_mae += mae
            total_linf += linf
            count += 1

            if count % self.log_interval == 0 or count == 1:
                logger.info(f"  Sample {count}/{min(len(self.test_dataset), self.num_samples or len(self.test_dataset))}: MSE={mse:.6f}")

            if self.num_samples is not None and count >= self.num_samples:
                break

        if count == 0:
            raise RuntimeError("No samples evaluated.")

        avg_mse = total_mse / count
        avg_mae = total_mae / count
        avg_linf = total_linf / count

        results = {"MSE": avg_mse, "MAE": avg_mae, "Linf": avg_linf, "num_samples": count}
        self._save_metrics(results, "simulation_metrics.json")
        logger.info(f"Simulation metrics: {results}")

        return results

    @torch.no_grad()
    def evaluate_control(self) -> Dict[str, float]:
        """
        Compute the control objective J for control tasks, using the ground‑truth
        solver. If no solver is provided, a warning is logged and an empty dict
        is returned.
        """
        if self.solver is None:
            logger.warning("Control evaluation requires a ground‑truth solver, but none was provided. Skipping.")
            return {}

        logger.info(f"Starting control evaluation on '{self.experiment_name}'.")

        dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        list_J = []
        count = 0

        # Load control‑specific parameters from config
        control_cfg = self.config.get_control_config()
        lambda_guidance = control_cfg["lambda"]
        # Energy penalty weight (alpha). For 1D Burgers', alpha ≈ 0.00002 (as in SAC reward).
        # For 2D fluid, J does not include an explicit energy term in the paper, so alpha=0.
        if self.is_1d:
            alpha = control_cfg.get("alpha", 2e-5)
        else:
            alpha = 0.0

        for idx, sample in enumerate(dataloader):
            sample = self._dict_to_device(sample, self.device)
            condition = self._build_condition(sample, mode='ctrl')

            # Generate the control force sequence
            f_pred = self.wdno.control(condition, lambda_val=lambda_guidance)

            # The solver should return either the final state (1D) or a scalar objective (2D)
            if self.is_1d:
                u0 = sample["u0"]           # (1, X)
                u_star = sample["uT"]       # (1, X)
                u_final_pred = self.solver(u0, f_pred)   # expected (1, X)
                state_error = F.mse_loss(u_final_pred, u_star, reduction='none').mean(dim=1)  # (1,)
                force_energy = (f_pred ** 2).mean(dim=(1, 2))   # (1,)
                J = state_error + alpha * force_energy
                list_J.extend(_as_float(j) for j in J)
            elif self.is_2d:
                # For 2D fluid control, solver returns the missed smoke percentage directly.
                J = self.solver(u0=sample["initial_density"], f=f_pred)  # expected (1,)
                list_J.extend(_as_float(j) for j in J)

            count += 1
            if self.num_samples is not None and count >= self.num_samples:
                break

        if count == 0:
            return {}

        J_array = np.array(list_J)
        mean_J = float(np.mean(J_array))
        std_J = float(np.std(J_array))
        results = {"J_mean": mean_J, "J_std": std_J, "num_samples": count}
        self._save_metrics(results, "control_metrics.json")
        logger.info(f"Control metrics: {results}")
        return results

    @torch.no_grad()
    def evaluate_super_resolution(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate zero‑shot super‑resolution at multiple levels (0x, 1x, ...).

        Returns a nested dict:
            { "0x": {"linear": mse0, "nearest": mse1, "count": ...}, "1x": {...}, ... }
        """
        if self.highres_dataset is None:
            logger.warning("No high‑resolution dataset provided for super‑resolution evaluation. Skipping.")
            return {}

        logger.info(f"Starting super‑resolution evaluation on '{self.experiment_name}'.")

        # Determine scales from config
        data_cfg = self.config.get_data_config()
        scales = data_cfg["super_res_scales"]  # e.g. [1, 0.5, 0.25, 0.125]
        max_level = len(scales) - 1            # number of SR steps (0 = base)

        highres_loader = DataLoader(
            self.highres_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        # Dictionary to accumulate MSE for each level and interpolation mode.
        results: Dict[str, Dict[str, float]] = {
            f"{level}x": {"linear": 0.0, "nearest": 0.0, "count": 0}
            for level in range(max_level + 1)
        }

        count_total = 0
        for idx, sample in enumerate(highres_loader):
            sample = self._dict_to_device(sample, self.device)
            # Build high‑res condition
            condition_high = self._build_condition(sample, mode='sim')
            gt_high = sample["state"]   # full high‑res trajectory (including t=0)

            # Exclude initial frame
            if self.is_1d:
                gt_high_target = gt_high[:, :, 1:]  # (1, T-1, X)
            else:
                gt_high_target = gt_high[:, :, 1:]  # (C, T-1, H, W)

            # Base resolution (level 0) prediction
            base_shape = self._compute_base_spatial(gt_high)          # e.g., (81, 120) for 1D
            cond_base = self._downsample_condition(condition_high, base_shape)
            pred_base = self.wdno.simulate(cond_base, guidance_w=self._get_sim_guidance_weight())
            if self.is_1d:
                pred_base_target = pred_base[:, :, 1:]
            else:
                pred_base_target = pred_base[:, :, 1:]

            # Interpolate and compute MSE for level 0
            for mode in ["linear", "nearest"]:
                pred_interp = _interpolate_to_shape(pred_base_target, gt_high_target.shape[1:], mode)
                mse = F.mse_loss(pred_interp, gt_high_target).item()
                results["0x"][mode] += mse
                results["0x"]["count"] += 1

            # For levels > 0, iteratively apply super‑resolution
            # We need the wavelet coefficients of the high‑res condition
            W_cond_high = self._condition_to_wavelet(condition_high, gt_high_target.shape[1:])
            # Start with the wavelet coefficients of the base prediction
            low_wav = self._state_to_wavelet(pred_base)   # (1, 4, ...) or (1, 8, ...)

            for level in range(1, max_level + 1):
                # One SR step
                wav_high = self.wdno.super_resolve(low_wav, W_cond_high, levels=1)
                phys_high = self.wdno._wavelet_to_physical(wav_high)
                if self.is_1d:
                    phys_high_target = phys_high[:, :, 1:]
                else:
                    phys_high_target = phys_high[:, :, 1:]

                # Evaluate after interpolating to the highest resolution
                for mode in ["linear", "nearest"]:
                    pred_interp = _interpolate_to_shape(phys_high_target, gt_high_target.shape[1:], mode)
                    mse = F.mse_loss(pred_interp, gt_high_target).item()
                    results[f"{level}x"][mode] += mse
                    results[f"{level}x"]["count"] += 1

                # Prepare low_res for next level
                low_wav = wav_high

            count_total += 1
            if self.num_samples is not None and count_total >= self.num_samples:
                break

        # Average MSE over samples
        for level in results:
            cnt = results[level].pop("count")
            if cnt > 0:
                results[level]["linear"] /= cnt
                results[level]["nearest"] /= cnt
            else:
                results[level]["linear"] = float('nan')
                results[level]["nearest"] = float('nan')

        self._save_metrics(results, "super_resolution_metrics.json")
        logger.info(f"Super‑resolution metrics: {results}")
        return results

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    def _build_condition(self, sample: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
        """
        Build a condition dictionary for WDNO.simulate/control from a raw dataset sample.

        Args:
            sample: dict from dataset.
            mode: 'sim' for simulation, 'ctrl' for control.

        Returns:
            dict with keys expected by WDNO (e.g., 'u0', 'f', 'uT', 'initial_density', ...)
        """
        condition = {}
        if self.is_1d:
            condition['u0'] = sample['u0']
            if 'f' in sample and mode != 'ctrl':
                condition['f'] = sample['f']
            if 'uT' in sample and mode == 'ctrl':
                condition['uT'] = sample['uT']
        elif self.is_2d:
            condition['initial_density'] = sample.get('initial_density')
            if mode == 'sim' and 'control' in sample:
                condition['control'] = sample['control']
            if mode == 'ctrl':
                # For control generation, we condition on initial_density only.
                pass
        return condition

    def _get_sim_guidance_weight(self) -> float:
        """Return the classifier‑free guidance weight for simulation."""
        return self.config.get_diffusion_config().get("guidance_weight", 0.0)

    def _dict_to_device(self, d: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in d.items()}

    def _compute_base_spatial(self, gt_high: torch.Tensor) -> Tuple[int, ...]:
        """
        Determine the base spatial shape of the original (non‑super‑resolved) data
        from the high‑resolution ground truth. This is done by dividing the spatial
        dimensions by 2^max_level.
        """
        max_level = len(self.config.get_data_config()["super_res_scales"]) - 1
        factor = 2 ** max_level
        spatial_dims = gt_high.shape[2:]  # skipping batch and channel dims
        base_spatial = tuple(int(s / factor) for s in spatial_dims)
        return base_spatial

    def _downsample_condition(
        self,
        condition: Dict[str, torch.Tensor],
        target_spatial: Tuple[int, ...],
    ) -> Dict[str, torch.Tensor]:
        """
        Downsample each condition tensor to `target_spatial` dimensions.
        Handles 1D signals (u0) and 2D/3D fields.
        """
        downsampled = {}
        for key, val in condition.items():
            if val.dim() == 1:
                # 1D signal – resize length to last spatial dimension
                new_len = target_spatial[-1] if len(target_spatial) > 0 else val.shape[0]
                downsampled[key] = F.interpolate(val.view(1, 1, -1), size=new_len, mode='linear').view(-1)
            elif val.dim() == 2:
                # 2D tensor (H, W)
                downsampled[key] = F.interpolate(
                    val.unsqueeze(0).unsqueeze(0), size=target_spatial, mode='bilinear'
                ).squeeze(0).squeeze(0)
            elif val.dim() == 3:
                # 3D tensor (T, H, W)
                downsampled[key] = F.interpolate(
                    val.unsqueeze(0).unsqueeze(0), size=target_spatial, mode='trilinear'
                ).squeeze(0).squeeze(0)
            else:
                raise NotImplementedError(
                    f"Downsampling for {val.dim()}-D condition '{key}' is not implemented."
                )
        return downsampled

    def _state_to_wavelet(self, state_phys: torch.Tensor) -> torch.Tensor:
        """
        Apply wavelet transform to a physical state tensor and return flat coefficients.

        Args:
            state_phys: shape (B, C, T, X) for 1D or (B, C, T, H, W) for 2D.

        Returns:
            Wavelet coefficients as a tensor of shape (B, C * K, *spatial_w).
        """
        B, C = state_phys.shape[0], state_phys.shape[1]
        wavelet = self.wdno.wavelet_transform
        coeffs_per_channel = []  # will store list of K tensors per (b, c)

        for b in range(B):
            batch_coeffs = []
            for c in range(C):
                s = state_phys[b, c]          # shape (T, ...)
                coeffs = wavelet.forward(s)   # list of K tensors
                batch_coeffs.append(coeffs)
            coeffs_per_channel.append(batch_coeffs)

        K = len(coeffs_per_channel[0][0])  # number of subbands
        # Stack per subband
        stacked = []
        for k in range(K):
            parts = []
            for b in range(B):
                for c in range(C):
                    parts.append(coeffs_per_channel[b][c][k])
            sub = torch.stack(parts, dim=0)          # (B*C, *spatial_w)
            sub = sub.view(B, C, *sub.shape[1:])     # (B, C, *spatial_w)
            stacked.append(sub)
        # Concatenate along channel dim -> (B, C*K, *sp