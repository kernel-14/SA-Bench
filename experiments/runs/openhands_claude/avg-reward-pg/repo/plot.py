"""
Plotting utilities for reproducing Figures 1 and 2 of the paper.

Figure 1(a): Average reward vs. iteration for different (S, A) sizes.
Figure 1(b): Average reward vs. iteration for different reward variances.
Figure 2:    Average reward vs. iteration for different transition kernels.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from policy_gradient import PPGResult


# Colour palette consistent with the paper's figures
COLORS = {
    # Experiment 1
    "(3,3)": "#1f77b4",
    "(9,9)": "#ff7f0e",
    "(81,81)": "#2ca02c",
    # Experiment 2
    "no_variance": "#1f77b4",
    "low_variance": "#ff7f0e",
    "high_variance": "#2ca02c",
    "max_variance": "#d62728",
    # Experiment 3
    "uniform": "#1f77b4",
    "non_uniform": "#ff7f0e",
    "deterministic": "#2ca02c",
}

LABELS = {
    "(3,3)": r"$(|S|,|A|) = (3,3)$",
    "(9,9)": r"$(|S|,|A|) = (9,9)$",
    "(81,81)": r"$(|S|,|A|) = (81,81)$",
    "no_variance": "No variance",
    "low_variance": "Low variance",
    "high_variance": "High variance",
    "max_variance": "Max variance",
    "uniform": "Uniform",
    "non_uniform": "Non-uniform",
    "deterministic": "Deterministic",
}


def _setup_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_rewards(
    results: Dict[str, PPGResult],
    title: str,
    output_path: str,
    rho_stars: Optional[Dict[str, float]] = None,
) -> None:
    """
    Plot average reward as a function of iteration number.

    Parameters
    ----------
    results : dict mapping label → PPGResult
    title : str
    output_path : str
    rho_stars : dict mapping label → optimal reward (for reference lines)
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for key, result in results.items():
        color = COLORS.get(key, None)
        label = LABELS.get(key, key)
        iters = np.arange(len(result.rewards))
        ax.plot(iters, result.rewards, label=label, color=color, linewidth=1.8)

    if rho_stars:
        for key, rho_star in rho_stars.items():
            color = COLORS.get(key, "gray")
            ax.axhline(rho_star, color=color, linestyle="--", alpha=0.4, linewidth=1.0)

    _setup_axes(ax, "Iteration", "Average Reward", title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_suboptimality(
    results: Dict[str, PPGResult],
    title: str,
    output_path: str,
    plot_bounds: bool = True,
) -> None:
    """
    Plot suboptimality gap ρ* - ρ^{π_k} as a function of iteration.

    Optionally overlays the theoretical bound from Theorem 1.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for key, result in results.items():
        color = COLORS.get(key, None)
        label = LABELS.get(key, key)
        iters = np.arange(len(result.suboptimality))
        ax.semilogy(
            iters,
            np.maximum(result.suboptimality, 1e-10),
            label=label,
            color=color,
            linewidth=1.8,
        )

        if plot_bounds and result.theoretical_bound:
            bound_iters = np.arange(len(result.theoretical_bound))
            ax.semilogy(
                bound_iters,
                np.maximum(result.theoretical_bound, 1e-10),
                label=f"{label} (bound)",
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
            )

    _setup_axes(ax, "Iteration", r"$\rho^* - \rho^{\pi_k}$", title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_figure1(
    results_exp1: Dict[str, PPGResult],
    results_exp2: Dict[str, PPGResult],
    output_path: str,
    rho_stars_exp1: Optional[Dict[str, float]] = None,
    rho_stars_exp2: Optional[Dict[str, float]] = None,
) -> None:
    """
    Reproduce Figure 1 of the paper (two subplots side by side).

    Left (1a):  Varying (S, A) sizes.
    Right (1b): Varying reward variance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Figure 1(a) ---
    ax = axes[0]
    for key, result in results_exp1.items():
        color = COLORS.get(key, None)
        label = LABELS.get(key, key)
        iters = np.arange(len(result.rewards))
        ax.plot(iters, result.rewards, label=label, color=color, linewidth=1.8)
    if rho_stars_exp1:
        for key, rho_star in rho_stars_exp1.items():
            color = COLORS.get(key, "gray")
            ax.axhline(rho_star, color=color, linestyle="--", alpha=0.4, linewidth=1.0)
    _setup_axes(
        ax,
        "Iteration",
        "Average Reward",
        "Figure 1(a): Varying $(|S|, |A|)$",
    )

    # --- Figure 1(b) ---
    ax = axes[1]
    for key, result in results_exp2.items():
        color = COLORS.get(key, None)
        label = LABELS.get(key, key)
        iters = np.arange(len(result.rewards))
        ax.plot(iters, result.rewards, label=label, color=color, linewidth=1.8)
    if rho_stars_exp2:
        for key, rho_star in rho_stars_exp2.items():
            color = COLORS.get(key, "gray")
            ax.axhline(rho_star, color=color, linestyle="--", alpha=0.4, linewidth=1.0)
    _setup_axes(
        ax,
        "Iteration",
        "Average Reward",
        "Figure 1(b): Varying Reward Variance",
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_figure2(
    results_exp3: Dict[str, PPGResult],
    output_path: str,
    rho_stars: Optional[Dict[str, float]] = None,
) -> None:
    """
    Reproduce Figure 2 of the paper: convergence as a function of C_p.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for key, result in results_exp3.items():
        color = COLORS.get(key, None)
        label = LABELS.get(key, key)
        iters = np.arange(len(result.rewards))
        ax.plot(iters, result.rewards, label=label, color=color, linewidth=1.8)

    if rho_stars:
        for key, rho_star in rho_stars.items():
            color = COLORS.get(key, "gray")
            ax.axhline(rho_star, color=color, linestyle="--", alpha=0.4, linewidth=1.0)

    _setup_axes(
        ax,
        "Iteration",
        "Average Reward",
        r"Figure 2: Convergence as a function of $C_p$",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
