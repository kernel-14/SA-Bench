"""
Visualization utilities for LUNO uncertainty quantification.

Reproduces Figure 2 (1D predictive uncertainty) and Figure 3 (2D comparison)
from the LUNO paper.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_1d_predictive_uncertainty(
    x: np.ndarray,
    target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    samples: Optional[np.ndarray] = None,
    title: str = "",
    ax_top=None,
    ax_bottom=None,
    color: str = "C0",
    n_eigenfunctions: int = 3,
    cov: Optional[np.ndarray] = None,
) -> Tuple:
    """
    Plot 1D predictive uncertainty (Figure 2 style).

    Top row: target, mean ± 1.96σ, and samples
    Bottom row: spread of predictive distribution, eigenfunctions, covariance heatmap

    Args:
        x: (n_x,) spatial grid
        target: (n_x,) ground truth
        mean: (n_x,) predicted mean
        std: (n_x,) predicted std
        samples: (n_samples, n_x) optional functional samples
        title: plot title
        ax_top, ax_bottom: matplotlib axes (created if None)
        color: line color
        n_eigenfunctions: number of eigenfunctions to plot
        cov: (n_x, n_x) optional covariance matrix for eigenfunction computation
    Returns:
        fig, (ax_top, ax_bottom)
    """
    if ax_top is None or ax_bottom is None:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 6))
    else:
        fig = ax_top.get_figure()

    # Top: target, mean, confidence band, samples
    ax_top.plot(x, target, "k-", linewidth=1.5, label="Target", zorder=5)
    ax_top.plot(x, mean, color=color, linewidth=1.5, label="Mean", zorder=4)
    ax_top.fill_between(
        x,
        mean - 1.96 * std,
        mean + 1.96 * std,
        alpha=0.3,
        color=color,
        label="±1.96σ",
    )
    if samples is not None:
        for s in samples[:4]:
            ax_top.plot(x, s, color=color, alpha=0.3, linewidth=0.8)

    ax_top.set_title(title)
    ax_top.legend(fontsize=8)
    ax_top.set_xlabel("x")

    # Bottom: spread (std), eigenfunctions, covariance heatmap
    ax_bottom.plot(x, std, color=color, linewidth=1.5, label="Std")

    if cov is not None:
        # Compute eigenfunctions
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        for i in range(min(n_eigenfunctions, eigenvectors.shape[1])):
            ef = eigenvectors[:, i] * np.sqrt(max(eigenvalues[i], 0))
            ax_bottom.plot(x, ef, "--", linewidth=1.0, label=f"EF {i+1}", alpha=0.7)

        # Inset: covariance heatmap
        ax_inset = ax_bottom.inset_axes([0.75, 0.55, 0.23, 0.43])
        ax_inset.imshow(cov, aspect="auto", cmap="RdBu_r",
                        vmin=-np.abs(cov).max(), vmax=np.abs(cov).max())
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title("Cov", fontsize=7)

    ax_bottom.legend(fontsize=8)
    ax_bottom.set_xlabel("x")
    ax_bottom.set_ylabel("Spread")

    plt.tight_layout()
    return fig, (ax_top, ax_bottom)


def plot_comparison_figure(
    x: np.ndarray,
    target: np.ndarray,
    methods: Dict[str, Dict],
    figsize: Tuple = (20, 8),
) -> plt.Figure:
    """
    Plot comparison of multiple UQ methods (Figure 2 style, multiple panels).

    Args:
        x: (n_x,) spatial grid
        target: (n_x,) ground truth
        methods: dict mapping method name to dict with keys:
                 "mean", "std", "samples" (optional), "cov" (optional)
        figsize: figure size
    Returns:
        fig
    """
    n_methods = len(methods)
    fig, axes = plt.subplots(2, n_methods, figsize=figsize)

    colors = [f"C{i}" for i in range(n_methods)]

    for j, (name, data) in enumerate(methods.items()):
        ax_top = axes[0, j] if n_methods > 1 else axes[0]
        ax_bottom = axes[1, j] if n_methods > 1 else axes[1]

        plot_1d_predictive_uncertainty(
            x=x,
            target=target,
            mean=data["mean"],
            std=data["std"],
            samples=data.get("samples"),
            title=name,
            ax_top=ax_top,
            ax_bottom=ax_bottom,
            color=colors[j],
            cov=data.get("cov"),
        )

    plt.tight_layout()
    return fig


def plot_2d_comparison(
    target: np.ndarray,
    ensemble_data: Dict,
    luno_la_data: Dict,
    figsize: Tuple = (16, 8),
) -> plt.Figure:
    """
    Plot 2D comparison between ensemble and LUNO-LA (Figure 3 style).

    Args:
        target: (n_x, n_y) ground truth
        ensemble_data: dict with "mean", "std", "sample", "residual", "null_space_residual"
        luno_la_data: dict with "mean", "std", "sample", "residual"
        figsize: figure size
    Returns:
        fig
    """
    fig, axes = plt.subplots(2, 8, figsize=figsize)

    def plot_field(ax, data, title, cmap="RdBu_r", vmin=None, vmax=None):
        if vmin is None:
            vmin = -np.abs(data).max()
        if vmax is None:
            vmax = np.abs(data).max()
        im = ax.imshow(data.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    # Ensemble row
    plot_field(axes[0, 0], target, "Target")
    plot_field(axes[0, 1], ensemble_data["residual"], "Residual")
    plot_field(axes[0, 2], ensemble_data["std"], "Std", cmap="viridis", vmin=0)
    plot_field(axes[0, 3], np.abs(ensemble_data["residual"]) / (ensemble_data["std"] + 1e-8),
               "|res|/σ", cmap="hot", vmin=0)
    plot_field(axes[0, 4], ensemble_data["sample"], "Sample")
    # Null space residual (ensemble-specific)
    if "null_space_residual" in ensemble_data:
        plot_field(axes[0, 5], ensemble_data["null_space_residual"], "Null-space res.")
    axes[0, 0].set_ylabel("Ensemble", fontsize=9)

    # LUNO-LA row
    plot_field(axes[1, 0], target, "Target")
    plot_field(axes[1, 1], luno_la_data["residual"], "Residual")
    plot_field(axes[1, 2], luno_la_data["std"], "Std", cmap="viridis", vmin=0)
    plot_field(axes[1, 3], np.abs(luno_la_data["residual"]) / (luno_la_data["std"] + 1e-8),
               "|res|/σ", cmap="hot", vmin=0)
    plot_field(axes[1, 4], luno_la_data["sample"], "Sample")
    axes[1, 0].set_ylabel("LUNO-LA", fontsize=9)

    # Hide unused axes
    for j in range(5, 8):
        axes[0, j].set_visible(False)
        axes[1, j].set_visible(False)

    plt.suptitle("Ensemble vs LUNO-LA: Predictive Uncertainty", fontsize=11)
    plt.tight_layout()
    return fig


def plot_autoregressive_rollout(
    results: Dict[str, Dict[str, List[float]]],
    metric: str = "nll",
    figsize: Tuple = (10, 5),
) -> plt.Figure:
    """
    Plot autoregressive rollout performance (Figure 4 style).

    Args:
        results: dict mapping method -> {"rmse": [...], "nll": [...], "chi2": [...]}
        metric: which metric to plot
        figsize: figure size
    Returns:
        fig
    """
    fig, ax = plt.subplots(figsize=figsize)

    method_colors = {
        "Input Perturbations": "C0",
        "Ensemble": "C1",
        "Sample-Iso": "C2",
        "LUNO-Iso": "C3",
        "Sample-LA": "C4",
        "LUNO-LA": "C5",
    }

    for method, data in results.items():
        values = data[metric]
        steps = np.arange(1, len(values) + 1)
        color = method_colors.get(method, "gray")
        ax.plot(steps, values, color=color, linewidth=1.5, label=method)

    ax.set_xlabel("Rollout step")
    metric_labels = {"rmse": "RMSE", "nll": "NLL", "chi2": "χ²"}
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_title(f"Autoregressive rollout: {metric_labels.get(metric, metric)}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig
