## utils/common.py
"""Foundational utility module for the Robotic World Model (RWM) project.

This module provides stateless, pure utility functions with no circular
dependencies. It is imported by nearly every other module in the project.

All observation indexing constants are defined here following Tables S2-S5
from the paper to ensure consistency across all modules.
"""

import os
import random
import warnings
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Observation slice constants (Tables S2-S5)
# Using slice objects for direct compatibility with both numpy and PyTorch.
# ---------------------------------------------------------------------------

# ANYmal D world model observation space (45-dim, Table S2)
OBS_SLICES_ANYMAL: Dict[str, slice] = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "gravity": slice(6, 9),
    "joint_pos": slice(9, 21),
    "joint_vel": slice(21, 33),
    "joint_torques": slice(33, 45),
}

# ANYmal D policy observation space (48-dim, Table S5)
POLICY_OBS_SLICES_ANYMAL: Dict[str, slice] = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "gravity": slice(6, 9),
    "velocity_command": slice(9, 12),
    "joint_pos": slice(12, 24),
    "joint_vel": slice(24, 36),
    "last_actions": slice(36, 48),
}

# ANYmal D privileged information space (8-dim, Table S3)
PRIV_SLICES_ANYMAL: Dict[str, slice] = {
    "knee_contact": slice(0, 4),
    "foot_contact": slice(4, 8),
}

# Unitree G1 world model observation space (96-dim, Table S2)
OBS_SLICES_G1: Dict[str, slice] = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "gravity": slice(6, 9),
    "joint_pos": slice(9, 38),
    "joint_vel": slice(38, 67),
    "joint_torques": slice(67, 96),
}

# Unitree G1 policy observation space (99-dim, Table S5)
POLICY_OBS_SLICES_G1: Dict[str, slice] = {
    "base_lin_vel": slice(0, 3),
    "base_ang_vel": slice(3, 6),
    "gravity": slice(6, 9),
    "velocity_command": slice(9, 12),
    "joint_pos": slice(12, 41),
    "joint_vel": slice(41, 70),
    "last_actions": slice(70, 99),
}

# Unitree G1 privileged information space (30-dim, Table S3)
PRIV_SLICES_G1: Dict[str, slice] = {
    "body_contact": slice(0, 26),
    "foot_height": slice(26, 28),
    "foot_velocity": slice(28, 30),
}

# Small epsilon for numerical stability in normalization and error metrics
_EPS: float = 1e-8


def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility across all RNG sources.

    The paper reports results averaged over 5 seeds (Tables S10, S11:
    num_seeds=5). This function ensures deterministic behavior for each seed.

    Args:
        seed: Integer seed value. The paper uses seeds 0-4 for 5-seed runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU safety
    # Deterministic CUDA kernels trade some performance for reproducibility.
    # This is the correct tradeoff for research reproduction.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "cuda") -> torch.device:
    """Parse a device string into a torch.device with graceful CPU fallback.

    Args:
        device_str: Device specification string. Supported values:
            - "cuda": Use default CUDA device (cuda:0)
            - "cuda:N": Use CUDA device N
            - "cpu": Use CPU

    Returns:
        A torch.device object pointing to the requested device, or CPU if
        the requested CUDA device is unavailable.

    Raises:
        ValueError: If device_str is not a recognized format.
    """
    if device_str == "cpu":
        return torch.device("cpu")

    if device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            warnings.warn(
                f"Requested device '{device_str}' but CUDA is not available. "
                "Falling back to CPU. Set device='cpu' in config.yaml to "
                "suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")

    raise ValueError(
        f"Unrecognized device string: '{device_str}'. "
        "Expected 'cpu', 'cuda', or 'cuda:N' where N is a device index."
    )


def soft_update(
    target: nn.Module,
    source: nn.Module,
    tau: float = 0.005,
) -> None:
    """Perform Polyak averaging from source network to target network.

    Updates target parameters in-place:
        θ_target ← τ * θ_source + (1 - τ) * θ_target

    Args:
        target: Target network whose parameters will be updated in-place.
        source: Source network providing the new parameter values.
        tau: Interpolation factor in [0, 1]. tau=1.0 performs a hard copy
            (full replacement). tau=0.005 is typical for SAC-style soft
            updates. Default: 0.005.

    Raises:
        ValueError: If target and source have different numbers of parameters.
    """
    target_params = list(target.parameters())
    source_params = list(source.parameters())

    if len(target_params) != len(source_params):
        raise ValueError(
            f"Parameter count mismatch: target has {len(target_params)} "
            f"parameters, source has {len(source_params)} parameters."
        )

    with torch.no_grad():
        for target_param, source_param in zip(target_params, source_params):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )


def explained_variance(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute the fraction of variance in y_true explained by y_pred.

    Standard PPO diagnostic metric for value function quality:
        EV = 1 - Var(y_true - y_pred) / Var(y_true)

    Interpretation:
        - EV ≈ 1.0: Value function is highly accurate.
        - EV ≈ 0.0: Value function explains nothing (predicts mean).
        - EV < 0.0: Value function is worse than predicting the mean.

    Args:
        y_pred: Predicted values from the value function. Any shape; will
            be flattened to 1D. Should be a numpy array (call .cpu().numpy()
            on tensors before passing here).
        y_true: True return values (e.g., GAE returns). Same shape as y_pred.

    Returns:
        Explained variance as a float in [-1.0, 1.0]. Returns 0.0 if
        Var(y_true) == 0 (degenerate case where all returns are identical).
    """
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    var_y = np.var(y_true_flat)
    if var_y < _EPS:
        # Degenerate case: all returns are identical (e.g., early training
        # when rewards are near zero). Return 0.0 to avoid division by zero.
        return 0.0

    ev = float(1.0 - np.var(y_true_flat - y_pred_flat) / var_y)
    # Clip to [-1, 1] for numerical safety
    return float(np.clip(ev, -1.0, 1.0))


def normalize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Normalize a tensor by subtracting mean and dividing by std.

    Supports broadcasting: mean and std can be shape [D] while x is
    [B, D] or [B, T, D].

    Args:
        x: Input tensor to normalize.
        mean: Mean tensor, broadcastable to x.
        std: Standard deviation tensor, broadcastable to x.

    Returns:
        Normalized tensor with same shape as x:
            (x - mean) / (std + eps)
    """
    return (x - mean) / (std + _EPS)


def denormalize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Denormalize a tensor by multiplying by std and adding mean.

    Exact inverse of normalize(). Used to convert world model predictions
    back to physical units for reward computation.

    Args:
        x: Normalized tensor to denormalize.
        mean: Mean tensor used during normalization, broadcastable to x.
        std: Standard deviation tensor used during normalization,
            broadcastable to x.

    Returns:
        Denormalized tensor with same shape as x:
            x * (std + eps) + mean
    """
    return x * (std + _EPS) + mean


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch module to count parameters for.

    Returns:
        Total number of trainable parameters (those with requires_grad=True).
        Excludes frozen layers and registered buffers.

    Example:
        >>> rwm = GRUWorldModel(config)
        >>> print(f"RWM parameters: {count_parameters(rwm):,}")
        RWM parameters: 623,457
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_gaussian(
    mean: torch.Tensor,
    logstd: torch.Tensor,
) -> torch.Tensor:
    """Sample from a Gaussian distribution using the reparameterization trick.

    Implements: sample = mean + eps * exp(logstd), where eps ~ N(0, I).

    This enables gradient flow through mean and logstd, which is essential
    for the autoregressive training objective in Section 3.2 of the paper.
    The reparameterization trick is used in:
        - models/rwm.py: outer autoregression (forecast horizon N steps)
        - training/mbpo_ppo_trainer.py: imagination rollout (T=100 steps)

    Uses log-std parameterization (not log-variance) to match the GRU heads
    which output logstd directly. This keeps the parameterization unconstrained
    (logstd can be any real number, while std must be positive).

    Args:
        mean: Mean of the Gaussian distribution. Shape: [..., D].
        logstd: Log standard deviation. Same shape as mean. The actual
            standard deviation is exp(logstd), which is always positive.

    Returns:
        A sample from N(mean, exp(logstd)^2) with the same shape as mean.
        The sample is on the same device as mean (torch.randn_like handles
        device placement automatically).
    """
    eps = torch.randn_like(mean)
    return mean + eps * torch.exp(logstd)
