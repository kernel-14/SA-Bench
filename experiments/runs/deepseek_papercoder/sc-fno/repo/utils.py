# utils.py
# ============================================================================
# Purpose: Provide common utility functions for reproducibility, error metrics,
#          and random point sampling. These are used across training, evaluation,
#          and inversion modules without depending on project‑specific classes.
# ============================================================================

import random
from typing import Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU/GPU) for full reproducibility.

    Args:
        seed: Integer seed to initialise all random generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behaviour of cuDNN (may slightly impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def relative_l2_error(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Compute the relative L² error between prediction and ground truth.

    The error is computed over all elements (batch, spatial, etc.) as a single
    scalar:  ||pred - true||_2 / (||true||_2 + eps).

    Args:
        pred: Predicted tensor of any shape.
        true: Ground truth tensor of the same shape.

    Returns:
        0‑dim tensor (scalar) representing the relative L² error.
    """
    diff = pred - true
    norm_diff = torch.norm(diff)
    norm_true = torch.norm(true)
    # Prevent division by zero when the true signal is essentially zero.
    return norm_diff / (norm_true + 1e-8)


def r2_score(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Compute the coefficient of determination (R²) as a global metric.

    R² = 1 - sum((true - pred)²) / sum((true - mean(true))²).
    A small epsilon is added to the denominator to avoid division by zero
    (e.g., when true values are constant).

    Args:
        pred: Predicted tensor, any shape.
        true: Ground truth tensor, same shape.

    Returns:
        Scalar tensor containing the R² value.
    """
    # Flatten to 1D to compute over all elements
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    ss_res = torch.sum((true_flat - pred_flat) ** 2)
    ss_tot = torch.sum((true_flat - true_flat.mean()) ** 2)
    # Add epsilon to avoid division‑by‑zero; if true constant, ss_tot ≈ 0.
    return 1.0 - ss_res / (ss_tot + 1e-8)


def sample_random_points(
    shape: Tuple[int, ...],
    n_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate batch‑wise random flat indices for sensitivity point sampling.

    For each sample in a batch, selects `n_points` random spatial‑temporal positions
    (as flat indices into the output domain) without replacement. This implements
    the strategy described in the paper: “randomly select a subset of spatial‑temporal
    points in each epoch”.

    Args:
        shape: Tuple describing the output tensor dimensions for the batch,
               excluding the channel dimension. Typically (B, T_out, S_x) or (B, T_out, S_x, S_y).
        n_points: Number of points to sample per sample. If `n_points` exceeds the number
                  of available points, all points are selected (clamped).

    Returns:
        batch_idx: LongTensor of shape (B, n_points) containing the batch indices.
                   Each row is filled with the corresponding sample index (0 to B‑1).
        flat_idx:  LongTensor of shape (B, n_points) containing random flat indices
                   (0 to total‑points‑per‑sample‑1) for each sample.

    Note:
        Both tensors are on CPU. The caller must move them to the appropriate device
        when indexing GPU tensors.
    """
    B = shape[0]  # batch size
    # Compute total points per sample (product of all dimensions after batch)
    total_points = 1
    for dim in shape[1:]:
        total_points *= dim

    # Clamp n_points to total_points (in case the grid is very small)
    n_actual = min(n_points, total_points)

    # Pre‑allocate tensors for batch indices and flat indices
    batch_idx = torch.zeros(B, n_actual, dtype=torch.long)
    flat_idx = torch.zeros(B, n_actual, dtype=torch.long)

    # Fill each row of batch_idx with the sample index
    for b in range(B):
        batch_idx[b, :] = b
        # Random permutation of all possible flat positions
        perm = torch.randperm(total_points, dtype=torch.long)
        flat_idx[b, :] = perm[:n_actual]

    return batch_idx, flat_idx
