"""
Visualization utilities for WDNO.

Provides functions for visualizing:
- 1D PDE simulation results (state trajectories)
- 2D fluid simulation results (density, velocity fields)
- Prediction errors over time
- Super-resolution comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List, Tuple
import torch


def plot_1d_trajectory(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    title: str = "1D PDE Trajectory",
    save_path: Optional[str] = None,
    n_timesteps_to_show: int = 5,
):
    """
    Plot 1D PDE trajectory comparison.
    
    Args:
        u_pred: Predicted trajectory (T, X)
        u_gt: Ground truth trajectory (T, X)
        title: Plot title
        save_path: Path to save figure
        n_timesteps_to_show: Number of time steps to visualize
    """
    T, X = u_gt.shape
    timesteps = np.linspace(0, T - 1, n_timesteps_to_show, dtype=int)
    x = np.linspace(0, 1, X)
    
    fig, axes = plt.subplots(1, n_timesteps_to_show, figsize=(4 * n_timesteps_to_show, 4))
    
    for i, t in enumerate(timesteps):
        ax = axes[i]
        ax.plot(x, u_gt[t], "b-", label="Ground Truth", linewidth=2)
        ax.plot(x, u_pred[t], "r--", label="Prediction", linewidth=2)
        ax.set_title(f"t = {t}")
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("u(t, x)")
            ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_1d_spacetime(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    title: str = "Space-Time Plot",
    save_path: Optional[str] = None,
):
    """
    Plot space-time diagram for 1D PDE.
    
    Args:
        u_pred: Predicted trajectory (T, X)
        u_gt: Ground truth trajectory (T, X)
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    vmin = min(u_gt.min(), u_pred.min())
    vmax = max(u_gt.max(), u_pred.max())
    
    im0 = axes[0].imshow(u_gt, aspect="auto", origin="lower", 
                          vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(u_pred, aspect="auto", origin="lower",
                          vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("x")
    plt.colorbar(im1, ax=axes[1])
    
    error = np.abs(u_pred - u_gt)
    im2 = axes[2].imshow(error, aspect="auto", origin="lower", cmap="hot")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mae_over_time(
    mae_wdno: np.ndarray,
    mae_ddpm: np.ndarray,
    title: str = "MAE Over Time",
    save_path: Optional[str] = None,
):
    """
    Plot MAE comparison over time steps.
    
    Used in ablation study (Figure 6 in paper).
    
    Args:
        mae_wdno: MAE of WDNO at each time step
        mae_ddpm: MAE of DDPM at each time step
        title: Plot title
        save_path: Path to save figure
    """
    T = len(mae_wdno)
    t = np.arange(T)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, mae_wdno, "b-", label="WDNO", linewidth=2)
    ax.plot(t, mae_ddpm, "r--", label="DDPM", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_super_resolution_comparison(
    predictions: List[np.ndarray],
    ground_truth: np.ndarray,
    labels: List[str],
    title: str = "Super-Resolution Comparison",
    save_path: Optional[str] = None,
):
    """
    Plot super-resolution comparison at different levels.
    
    Used in Figure 3 of the paper.
    
    Args:
        predictions: List of predictions at different resolution levels
        ground_truth: Ground truth at highest resolution
        labels: Labels for each prediction
        title: Plot title
        save_path: Path to save figure
    """
    n_preds = len(predictions)
    fig, axes = plt.subplots(3, n_preds + 1, figsize=(4 * (n_preds + 1), 12))
    
    # Ground truth
    axes[0, 0].imshow(ground_truth, aspect="auto", origin="lower", cmap="RdBu_r")
    axes[0, 0].set_title("Ground Truth")
    axes[1, 0].set_visible(False)
    axes[2, 0].set_visible(False)
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        # Interpolate to highest resolution if needed
        if pred.shape != ground_truth.shape:
            from scipy.ndimage import zoom
            zoom_factors = (ground_truth.shape[0] / pred.shape[0],
                           ground_truth.shape[1] / pred.shape[1])
            pred_interp = zoom(pred, zoom_factors)
        else:
            pred_interp = pred
        
        axes[0, i + 1].imshow(pred_interp, aspect="auto", origin="lower", cmap="RdBu_r")
        axes[0, i + 1].set_title(label)
        
        axes[1, i + 1].imshow(ground_truth, aspect="auto", origin="lower", cmap="RdBu_r")
        axes[1, i + 1].set_title("Ground Truth")
        
        error = np.abs(pred_interp - ground_truth)
        axes[2, i + 1].imshow(error, aspect="auto", origin="lower", cmap="hot")
        axes[2, i + 1].set_title(f"Error ({label})")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_2d_fluid(
    density: np.ndarray,
    title: str = "2D Fluid Density",
    save_path: Optional[str] = None,
    n_timesteps: int = 4,
):
    """
    Plot 2D fluid density over time.
    
    Args:
        density: Density field (T, H, W)
        title: Plot title
        save_path: Path to save figure
        n_timesteps: Number of time steps to show
    """
    T, H, W = density.shape
    timesteps = np.linspace(0, T - 1, n_timesteps, dtype=int)
    
    fig, axes = plt.subplots(1, n_timesteps, figsize=(4 * n_timesteps, 4))
    
    for i, t in enumerate(timesteps):
        ax = axes[i]
        im = ax.imshow(density[t], origin="lower", cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"t = {t}")
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mse_vs_super_resolution_level(
    mse_wdno: List[float],
    mse_fno: List[float],
    mse_wno: Optional[List[float]] = None,
    title: str = "MSE vs Super-Resolution Level",
    save_path: Optional[str] = None,
):
    """
    Plot MSE vs super-resolution level.
    
    Used in Figure 4 of the paper.
    
    Args:
        mse_wdno: MSE of WDNO at each SR level
        mse_fno: MSE of FNO at each SR level
        mse_wno: MSE of WNO at each SR level (optional)
        title: Plot title
        save_path: Path to save figure
    """
    levels = list(range(len(mse_wdno)))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(levels, mse_wdno, "b-o", label="WDNO", linewidth=2, markersize=8)
    ax.plot(levels, mse_fno, "r-s", label="FNO (linear)", linewidth=2, markersize=8)
    
    if mse_wno is not None:
        ax.plot(levels, mse_wno, "g-^", label="WNO (linear)", linewidth=2, markersize=8)
    
    ax.set_xlabel("Super-Resolution Level")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
