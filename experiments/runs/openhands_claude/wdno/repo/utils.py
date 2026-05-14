"""
Shared utilities for WDNO training and evaluation.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error over all elements."""
    return torch.mean((pred - target) ** 2)


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error over all elements."""
    return torch.mean(torch.abs(pred - target))


def compute_linf(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L-infinity error (max absolute error)."""
    return torch.max(torch.abs(pred - target))


def compute_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 error: ||pred - target||_2 / ||target||_2"""
    return torch.norm(pred - target) / (torch.norm(target) + 1e-8)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup."""
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prepare_wavelet_batch_1d(
    u: torch.Tensor,
    f: torch.Tensor,
    wavelet: str = "bior2.4",
    wt_mode: str = "periodization",
    mode: str = "simulation",
    u_target: Optional[torch.Tensor] = None,
) -> tuple:
    """
    Prepare wavelet-transformed batch for 1D experiments.

    For simulation:
      - target: W_u (wavelet of state trajectory)
      - cond: W_u0 (repeated) + W_f

    For control:
      - target: W_f (wavelet of force)
      - cond: W_u0 (repeated) + W_uT

    Args:
        u: state trajectory [batch, T+1, X] (includes t=0)
        f: force [batch, T, X]
        wavelet: wavelet name
        wt_mode: padding mode
        mode: 'simulation' or 'control'
        u_target: target state for control [batch, X]

    Returns:
        (target_wt, cond_wt): wavelet-transformed target and condition
    """
    from pytorch_wavelets import DWTForward, DWT1DForward

    device = u.device
    dwt2d = DWTForward(J=1, wave=wavelet, mode=wt_mode).to(device)
    dwt1d = DWT1DForward(J=1, wave=wavelet, mode=wt_mode).to(device)

    def wt2d(x):
        """x: [B, T, X] → [B, 4, T', X']"""
        x_c = x.unsqueeze(1)  # [B, 1, T, X]
        yl, yh = dwt2d(x_c)
        lh = yh[0][:, :, 0]
        hl = yh[0][:, :, 1]
        hh = yh[0][:, :, 2]
        return torch.cat([yl, lh, hl, hh], dim=1)  # [B, 4, T', X']

    def wt1d_to_2d(x, T_prime, X_prime):
        """x: [B, X] → [B, 4, T', X'] by 1D DWT + repeat"""
        x_c = x.unsqueeze(1)  # [B, 1, X]
        yl, yh = dwt1d(x_c)   # yl: [B, 1, X'], yh[0]: [B, 1, 1, X']
        cD = yh[0][:, :, 0]   # [B, 1, X']
        coeffs = torch.cat([yl, cD, yl, cD], dim=1)  # [B, 4, X']
        # Repeat along T' dimension
        coeffs_2d = coeffs.unsqueeze(2).expand(-1, -1, T_prime, -1)
        return coeffs_2d

    if mode == "simulation":
        # Target: wavelet of u trajectory (excluding t=0)
        u_traj = u[:, 1:]  # [B, T, X]
        w_target = wt2d(u_traj)  # [B, 4, T', X']

        # Condition: W_u0 + W_f
        T_prime, X_prime = w_target.shape[-2], w_target.shape[-1]
        w_u0 = wt1d_to_2d(u[:, 0], T_prime, X_prime)  # [B, 4, T', X']
        w_f = wt2d(f)  # [B, 4, T', X']
        w_cond = torch.cat([w_u0, w_f], dim=1)  # [B, 8, T', X']

        return w_target, w_cond

    else:  # control
        # Target: wavelet of force
        w_target = wt2d(f)  # [B, 4, T', X']

        # Condition: W_u0 + W_uT
        T_prime, X_prime = w_target.shape[-2], w_target.shape[-1]
        w_u0 = wt1d_to_2d(u[:, 0], T_prime, X_prime)  # [B, 4, T', X']
        if u_target is None:
            u_target = u[:, -1]
        w_uT = wt1d_to_2d(u_target, T_prime, X_prime)  # [B, 4, T', X']
        w_cond = torch.cat([w_u0, w_uT], dim=1)  # [B, 8, T', X']

        return w_target, w_cond


def prepare_wavelet_batch_2d(
    state: torch.Tensor,
    cond_2d: torch.Tensor,
    wavelet: str = "bior1.3",
    wt_mode: str = "zero",
) -> tuple:
    """
    Prepare wavelet-transformed batch for 2D experiments.

    Args:
        state: [batch, C, T, H, W] state trajectory
        cond_2d: [batch, C_cond, H, W] 2D condition (e.g. initial density)
        wavelet: wavelet name
        wt_mode: padding mode

    Returns:
        (w_target, w_cond): wavelet-transformed target and condition
    """
    import ptwt
    import pywt

    device = state.device
    w = pywt.Wavelet(wavelet)

    def wt3d(x):
        """x: [B, C, T, H, W] → [B, 8*C, T', H', W']"""
        coeffs = ptwt.wavedec3(x, w, level=1, mode=wt_mode)
        approx = coeffs[0]
        detail_dict = coeffs[1]
        detail_list = [
            detail_dict['aad'], detail_dict['ada'], detail_dict['add'],
            detail_dict['daa'], detail_dict['dad'], detail_dict['dda'],
            detail_dict['ddd'],
        ]
        return torch.cat([approx] + detail_list, dim=1)

    def wt2d_to_3d(x, T_prime, H_prime, W_prime):
        """x: [B, C, H, W] → [B, 8*C, T', H', W'] by 2D DWT + repeat"""
        from pytorch_wavelets import DWTForward
        dwt2d = DWTForward(J=1, wave=wavelet, mode=wt_mode).to(device)
        yl, yh = dwt2d(x)
        lh = yh[0][:, :, 0]
        hl = yh[0][:, :, 1]
        hh = yh[0][:, :, 2]
        coeffs_2d = torch.cat([yl, lh, hl, hh], dim=1)  # [B, 4*C, H', W']
        # Pad to 8*C by repeating
        coeffs_8 = coeffs_2d.repeat(1, 2, 1, 1)  # [B, 8*C, H', W']
        # Add T' dimension
        coeffs_3d = coeffs_8.unsqueeze(2).expand(-1, -1, T_prime, -1, -1)
        return coeffs_3d

    w_target = wt3d(state)  # [B, 8*C, T', H', W']
    T_prime, H_prime, W_prime = w_target.shape[-3], w_target.shape[-2], w_target.shape[-1]
    w_cond = wt2d_to_3d(cond_2d, T_prime, H_prime, W_prime)

    return w_target, w_cond
