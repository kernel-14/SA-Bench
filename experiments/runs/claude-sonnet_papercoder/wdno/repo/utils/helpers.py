## utils/helpers.py
"""Utility functions shared across all WDNO modules.

This module is a leaf node in the dependency graph — it imports only from
the standard library, numpy, and PyTorch. No project-internal imports are
used to avoid circular dependencies.

All functions are stateless pure utilities or thin wrappers around
PyTorch/OS primitives.
"""

from __future__ import annotations

import math
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across all RNG sources.

    Covers Python built-in random, NumPy, PyTorch CPU, and PyTorch CUDA
    (including multi-GPU setups used in 2D experiments on 2× A100).

    Args:
        seed: Integer seed value. Paper uses seed=42 (config.yaml:
            experiment.seed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Guidance schedule
# ---------------------------------------------------------------------------


def get_cosine_schedule(
    step: int,
    total_steps: int,
    max_val: float,
    min_val: float = 0.0,
) -> float:
    """Compute cosine-annealed guidance weight λ at a given denoising step.

    The schedule decreases from ``max_val`` at ``step=0`` (most noisy,
    start of denoising) to ``min_val`` at ``step=total_steps`` (cleanest,
    end of denoising). This matches the paper's "cosine" guidance schedule
    described in Table 18 and Table 19.

    Formula:
        lambda_t = min_val + 0.5 * (max_val - min_val) * (1 + cos(pi * step / total_steps))

    At step=0:        lambda_t = max_val  (maximum guidance)
    At step=total:    lambda_t = min_val  (minimum guidance)

    Args:
        step: Current denoising step index. 0 = start of denoising
            (most noisy), total_steps = end (cleanest).
        total_steps: Total number of DDIM sampling steps. From config:
            50 (Burgers), 850 (compressible NS), 100 (2D fluid).
        max_val: Maximum guidance weight (lambda_max). From config:
            inference.burgers.guidance_lambda=120000,
            inference.fluid_2d.guidance_lambda=100.
        min_val: Minimum guidance weight. Defaults to 0.0 so guidance
            fades completely at the end of denoising.

    Returns:
        Scalar float guidance weight for the current step.
    """
    if total_steps <= 0:
        return float(max_val)
    # Clamp step to valid range
    step = max(0, min(step, total_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
    return float(min_val + (max_val - min_val) * cosine_factor)


# ---------------------------------------------------------------------------
# Interpolation utilities for super-resolution evaluation
# ---------------------------------------------------------------------------


def linear_interpolate(x: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """Interpolate a tensor to ``target_shape`` using linear/bilinear/trilinear mode.

    Dispatches to the appropriate ``F.interpolate`` mode based on the
    number of spatial+temporal dimensions inferred from ``x``:

    - 4D input ``[N, C, T, X]``  → ``mode='bilinear'`` (treats T×X as 2D image)
    - 5D input ``[N, C, T, H, W]`` → ``mode='trilinear'`` (treats T×H×W as 3D volume)

    Used in ``SuperResolver.interpolate_to_finest()`` and
    ``Evaluator.evaluate_super_resolution()`` to bring all SR-level outputs
    to the finest resolution for fair MSE comparison (paper Section 4.6,
    Table 16, Table 17).

    Args:
        x: Input tensor of shape ``[N, C, *spatial_temporal_dims]``.
        target_shape: Desired output shape for the non-batch, non-channel
            dimensions, e.g. ``(T_out, X_out)`` for 1D PDE data or
            ``(T_out, H_out, W_out)`` for 2D PDE data.

    Returns:
        Interpolated tensor of shape ``[N, C, *target_shape]``.

    Raises:
        ValueError: If ``x`` is not 4D or 5D, or if ``target_shape`` length
            does not match the spatial+temporal dimensionality of ``x``.
    """
    ndim = x.dim()
    if ndim == 4:
        # [N, C, T, X] — treat as 2D image for bilinear interpolation
        if len(target_shape) != 2:
            raise ValueError(
                f"4D input requires target_shape of length 2, got {len(target_shape)}."
            )
        return F.interpolate(
            x,
            size=tuple(target_shape),
            mode="bilinear",
            align_corners=False,
        )
    elif ndim == 5:
        # [N, C, T, H, W] — treat as 3D volume for trilinear interpolation
        if len(target_shape) != 3:
            raise ValueError(
                f"5D input requires target_shape of length 3, got {len(target_shape)}."
            )
        return F.interpolate(
            x,
            size=tuple(target_shape),
            mode="trilinear",
            align_corners=False,
        )
    else:
        raise ValueError(
            f"linear_interpolate expects 4D or 5D input, got {ndim}D tensor."
        )


def nearest_interpolate(
    x: torch.Tensor, target_shape: Tuple[int, ...]
) -> torch.Tensor:
    """Interpolate a tensor to ``target_shape`` using nearest-neighbor mode.

    Same dispatch logic as ``linear_interpolate`` but uses ``mode='nearest'``.
    Used as the second interpolation baseline in super-resolution evaluation
    (paper Section 4.6, config: super_resolution.eval_interp_modes).

    Args:
        x: Input tensor of shape ``[N, C, *spatial_temporal_dims]``.
        target_shape: Desired output shape for the non-batch, non-channel
            dimensions.

    Returns:
        Interpolated tensor of shape ``[N, C, *target_shape]``.

    Raises:
        ValueError: If ``x`` is not 4D or 5D, or if ``target_shape`` length
            does not match the spatial+temporal dimensionality of ``x``.
    """
    ndim = x.dim()
    if ndim == 4:
        if len(target_shape) != 2:
            raise ValueError(
                f"4D input requires target_shape of length 2, got {len(target_shape)}."
            )
        return F.interpolate(
            x,
            size=tuple(target_shape),
            mode="nearest",
        )
    elif ndim == 5:
        if len(target_shape) != 3:
            raise ValueError(
                f"5D input requires target_shape of length 3, got {len(target_shape)}."
            )
        return F.interpolate(
            x,
            size=tuple(target_shape),
            mode="nearest",
        )
    else:
        raise ValueError(
            f"nearest_interpolate expects 4D or 5D input, got {ndim}D tensor."
        )


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Used for reporting model size (paper Table 11: WDNO has ~140,748,553
    parameters).

    Args:
        model: PyTorch module whose trainable parameters are counted.

    Returns:
        Total number of trainable scalar parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def save_checkpoint(state: dict, path: str) -> None:
    """Save a training checkpoint to disk.

    Creates all parent directories if they do not exist. The ``state`` dict
    convention used by ``Trainer.save_checkpoint`` is::

        {
            'model_state_dict': ...,
            'optimizer_state_dict': ...,
            'scheduler_state_dict': ...,
            'global_step': int,
            'config': dict,
        }

    Args:
        state: Dictionary containing model state, optimizer state,
            scheduler state, global step, and config.
        path: Full file path where the checkpoint will be saved.
            Parent directories are created automatically.
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        make_dirs(parent_dir)
    torch.save(state, path)


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    """Load a checkpoint from disk onto the specified device.

    Args:
        path: Full file path to the saved checkpoint.
        device: Target device for loading tensors, e.g. ``'cuda'``,
            ``'cpu'``, or ``'cuda:0'``. Handles cross-device loading
            (e.g., checkpoint saved on GPU, loaded on CPU).

    Returns:
        The full state dictionary as saved by ``save_checkpoint``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'. "
            "Ensure the path is correct and the file has been saved."
        )
    return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# File system utilities
# ---------------------------------------------------------------------------


def make_dirs(path: str) -> None:
    """Create a directory and all intermediate parents if they do not exist.

    Uses ``exist_ok=True`` so calling this on an existing directory is safe.

    Args:
        path: Directory path to create.
    """
    if path:
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Data normalization
# ---------------------------------------------------------------------------


def normalize_data(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Standardize data to zero mean and unit variance.

    Normalization is applied **before** the wavelet transform in the data
    pipeline, since the wavelet transform is linear and reconstruction error
    is verified on the original scale (paper Appendix A, Table 3).

    Args:
        x: Input tensor of arbitrary shape.
        mean: Mean tensor, broadcastable to ``x``'s shape. Typically
            computed over the training set.
        std: Standard deviation tensor, broadcastable to ``x``'s shape.
        eps: Small constant added to ``std`` to prevent division by zero.
            Defaults to 1e-8.

    Returns:
        Normalized tensor ``(x - mean) / (std + eps)`` with the same shape
        as ``x``.
    """
    return (x - mean) / (std + eps)


def denormalize_data(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Inverse of ``normalize_data``: convert standardized data back to physical units.

    Applied after inverse wavelet transform to get physically meaningful
    predictions before computing evaluation metrics.

    Args:
        x: Normalized tensor of arbitrary shape.
        mean: Mean tensor used during normalization, broadcastable to
            ``x``'s shape.
        std: Standard deviation tensor used during normalization,
            broadcastable to ``x``'s shape.

    Returns:
        Denormalized tensor ``x * std + mean`` with the same shape as ``x``.
    """
    return x * std + mean


# ---------------------------------------------------------------------------
# Diffusion schedule utilities
# ---------------------------------------------------------------------------


def compute_alpha_bar(betas: torch.Tensor) -> torch.Tensor:
    """Compute cumulative product of ``(1 - beta_t)`` values.

    Computes ``alpha_bar_k = prod_{i=1}^{k} alpha_i`` where
    ``alpha_i = 1 - beta_i``, as defined in the paper (Section 2.2):

        alpha_bar_k := prod_{i=1}^{k} alpha_i

    This quantity appears in:
    - Forward diffusion: ``x_k = sqrt(alpha_bar_k) * x_0 + sqrt(1 - alpha_bar_k) * eps``
    - Denoised estimate: ``x0_hat = (x_k - sqrt(1-alpha_bar_k)*eps) / sqrt(alpha_bar_k)``

    Args:
        betas: 1D tensor of length ``K`` containing the noise schedule
            beta values. Produced by ``Diffusion._make_beta_schedule()``.
            Config: ``diffusion.num_timesteps=1000``,
            ``diffusion.beta_schedule=cosine``.

    Returns:
        1D tensor of length ``K`` where element ``k`` is
        ``alpha_bar_{k+1}`` (i.e., the cumulative product up to and
        including timestep ``k+1``).
    """
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)
