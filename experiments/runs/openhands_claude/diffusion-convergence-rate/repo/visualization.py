"""
Visualization functions for reproducing paper figures.

Implements:
  - Figure 1: Iteration complexity comparison (left: vs L, right: vs epsilon)
  - Figure 2: KL divergence convergence for Gaussian target
  - Figure 3: TV distance comparison for fixed T
"""

import numpy as np
import os
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from theory import (
    our_iteration_complexity,
    benton_2023_iteration_complexity,
    li_yan_2024a_iteration_complexity,
    li_cai_2024_iteration_complexity,
    li_jiao_2024_iteration_complexity,
    gupta_2024_iteration_complexity,
    our_tv_bound,
    benton_2023_tv_bound,
    li_yan_2024a_tv_bound,
    li_cai_2024_tv_bound,
    li_jiao_2024_tv_bound,
    gupta_2024_tv_bound,
)
from experiments import ExperimentResult


# Color scheme matching paper style
METHOD_COLORS = {
    "Ours (Theorem 1)": "#d62728",      # red
    "Benton et al. (2023)": "#1f77b4",  # blue
    "Li & Yan (2024a)": "#2ca02c",      # green
    "Li & Cai (2024)": "#ff7f0e",       # orange
    "Li & Jiao (2024)": "#9467bd",      # purple
    "Gupta et al. (2024)": "#8c564b",   # brown
}

METHOD_LINESTYLES = {
    "Ours (Theorem 1)": "-",
    "Benton et al. (2023)": "--",
    "Li & Yan (2024a)": "-.",
    "Li & Cai (2024)": ":",
    "Li & Jiao (2024)": "--",
    "Gupta et al. (2024)": "-.",
}


def plot_figure1_left(
    d: int = 100,
    epsilon: float = 1.0,
    L_min: float = 0.1,
    L_max: float = 1e4,
    n_points: int = 200,
    output_path: Optional[str] = None,
    log_factor: bool = False,
) -> Optional[object]:
    """
    Figure 1 (left): Iteration complexity as function of L when epsilon = O(1).

    Shows that our result achieves the best complexity across the full range of L.

    Args:
        d: data dimension
        epsilon: target TV distance
        L_min, L_max: range of L values
        n_points: number of L values
        output_path: path to save figure
        log_factor: whether to include log factors

    Returns:
        matplotlib figure if available
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    L_values = np.logspace(np.log10(L_min), np.log10(L_max), n_points)

    fig, ax = plt.subplots(figsize=(7, 5))

    methods = {
        "Ours (Theorem 1)": lambda L: our_iteration_complexity(d, L, epsilon, log_factor),
        "Benton et al. (2023)": lambda L: benton_2023_iteration_complexity(d, epsilon),
        "Li & Yan (2024a)": lambda L: li_yan_2024a_iteration_complexity(d, epsilon),
        "Li & Cai (2024)": lambda L: li_cai_2024_iteration_complexity(d, epsilon),
        "Li & Jiao (2024)": lambda L: li_jiao_2024_iteration_complexity(d, L, epsilon, log_factor),
        "Gupta et al. (2024)": lambda L: gupta_2024_iteration_complexity(d, L, epsilon, log_factor),
    }

    for method, func in methods.items():
        T_vals = np.array([func(L) for L in L_values])
        ax.loglog(
            L_values,
            T_vals,
            label=method,
            color=METHOD_COLORS.get(method, "gray"),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
            linewidth=2.0,
        )

    # Mark sqrt(d) and d
    ax.axvline(x=np.sqrt(d), color="gray", linestyle=":", alpha=0.5, label=f"$\\sqrt{{d}}={np.sqrt(d):.1f}$")
    ax.axvline(x=d, color="gray", linestyle="--", alpha=0.5, label=f"$d={d}$")

    ax.set_xlabel("$L$ (Lipschitz constant)", fontsize=12)
    ax.set_ylabel("Iteration complexity $T$", fontsize=12)
    ax.set_title(f"Iteration complexity vs $L$ ($d={d}$, $\\varepsilon={epsilon}$)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved Figure 1 (left) to {output_path}")

    return fig


def plot_figure1_right(
    d: int = 100,
    L: float = float("inf"),
    T_min: float = 10,
    T_max: float = 1e6,
    n_points: int = 100,
    output_path: Optional[str] = None,
    log_factor: bool = False,
) -> Optional[object]:
    """
    Figure 1 (right): TV distance as function of T when L = infinity.

    Shows improvement over previous results when T <= d^2.

    Args:
        d: data dimension
        L: Lipschitz constant (inf for no smoothness)
        T_min, T_max: range of T values
        n_points: number of T values
        output_path: path to save figure
        log_factor: whether to include log factors

    Returns:
        matplotlib figure if available
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    T_values = np.logspace(np.log10(T_min), np.log10(T_max), n_points)

    fig, ax = plt.subplots(figsize=(7, 5))

    methods = {
        "Ours (Theorem 1)": lambda T: our_tv_bound(d, L, int(T), log_factor),
        "Benton et al. (2023)": lambda T: benton_2023_tv_bound(d, int(T)),
        "Li & Yan (2024a)": lambda T: li_yan_2024a_tv_bound(d, int(T)),
        "Li & Cai (2024)": lambda T: li_cai_2024_tv_bound(d, int(T)),
    }

    for method, func in methods.items():
        tv_vals = np.array([func(T) for T in T_values])
        ax.loglog(
            T_values,
            tv_vals,
            label=method,
            color=METHOD_COLORS.get(method, "gray"),
            linestyle=METHOD_LINESTYLES.get(method, "-"),
            linewidth=2.0,
        )

    # Mark T = d^2
    ax.axvline(x=d**2, color="gray", linestyle=":", alpha=0.5, label=f"$T=d^2={d**2}$")

    ax.set_xlabel("Number of iterations $T$", fontsize=12)
    ax.set_ylabel("TV distance upper bound", fontsize=12)
    ax.set_title(f"TV distance vs $T$ ($d={d}$, $L=\\infty$)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved Figure 1 (right) to {output_path}")

    return fig


def plot_figure2(
    results: Dict[str, ExperimentResult],
    output_path: Optional[str] = None,
) -> Optional[object]:
    """
    Figure 2: KL divergence convergence for Gaussian target (Appendix A).

    Three subplots:
    (a) d=10, k=10
    (b) d=100, k=10
    (c) d=500, k=100

    Blue line: empirical KL divergence
    Black line: theoretical rate O(log^4(T)/T^3)

    Args:
        results: dict from run_all_figure2_experiments
        output_path: path to save figure

    Returns:
        matplotlib figure if available
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    if n_plots == 1:
        axes = [axes]

    subplot_labels = ["(a)", "(b)", "(c)"]

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        T_vals = result.T_values
        kl_vals = result.kl_values

        # Filter out NaN values
        valid = ~np.isnan(kl_vals)
        T_valid = T_vals[valid]
        kl_valid = kl_vals[valid]

        if len(T_valid) == 0:
            continue

        # Empirical KL
        ax.loglog(T_valid, kl_valid, "b-o", linewidth=2, markersize=5, label="Empirical KL")

        # Theoretical rate O(log^4(T)/T^3)
        theoretical = result.theoretical_rate[valid]
        ax.loglog(T_valid, theoretical, "k--", linewidth=2, label=r"$\Theta(\log^4 T / T^3)$")

        ax.set_xlabel("$T$", fontsize=12)
        ax.set_ylabel("KL divergence", fontsize=12)
        ax.set_title(
            f"{subplot_labels[idx]} $d={result.d}$, $k={result.k}$",
            fontsize=12,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Sampling error vs iterations (Gaussian target)", fontsize=13)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved Figure 2 to {output_path}")

    return fig


def plot_figure3(
    d: int = 100,
    T_settings: Optional[Dict[str, int]] = None,
    L_min: float = 0.1,
    L_max: float = 1e4,
    n_points: int = 200,
    output_path: Optional[str] = None,
    log_factor: bool = False,
) -> Optional[object]:
    """
    Figure 3 (Appendix B): TV distance vs L for fixed T.

    Three subplots: T = O(d), T = O(d^{3/2}), T = O(d^2).

    Args:
        d: data dimension
        T_settings: dict mapping label to T value
        L_min, L_max: range of L values
        n_points: number of L values
        output_path: path to save figure
        log_factor: whether to include log factors

    Returns:
        matplotlib figure if available
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    if T_settings is None:
        T_settings = {
            f"$T=O(d)$, $T={d}$": d,
            f"$T=O(d^{{3/2}})$, $T={int(d**1.5)}$": int(d**1.5),
            f"$T=O(d^2)$, $T={d**2}$": d**2,
        }

    L_values = np.logspace(np.log10(L_min), np.log10(L_max), n_points)
    n_plots = len(T_settings)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    subplot_labels = ["left", "middle", "right"]

    for idx, (T_label, T) in enumerate(T_settings.items()):
        ax = axes[idx]

        methods = {
            "Ours (Theorem 1)": lambda L, T=T: our_tv_bound(d, L, T, log_factor),
            "Benton et al. (2023)": lambda L, T=T: benton_2023_tv_bound(d, T),
            "Li & Yan (2024a)": lambda L, T=T: li_yan_2024a_tv_bound(d, T),
            "Li & Cai (2024)": lambda L, T=T: li_cai_2024_tv_bound(d, T),
            "Li & Jiao (2024)": lambda L, T=T: li_jiao_2024_tv_bound(d, L, T, log_factor),
            "Gupta et al. (2024)": lambda L, T=T: gupta_2024_tv_bound(d, L, T, log_factor),
        }

        for method, func in methods.items():
            tv_vals = np.array([func(L) for L in L_values])
            # Clip to [0, 1] for TV distance
            tv_vals = np.clip(tv_vals, 0, 1)
            ax.loglog(
                L_values,
                tv_vals,
                label=method,
                color=METHOD_COLORS.get(method, "gray"),
                linestyle=METHOD_LINESTYLES.get(method, "-"),
                linewidth=2.0,
            )

        ax.axvline(x=np.sqrt(d), color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=d, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

        ax.set_xlabel("$L$", fontsize=12)
        ax.set_ylabel("TV distance", fontsize=12)
        ax.set_title(T_label, fontsize=11)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-6, 2.0)

    plt.suptitle(f"TV distance vs $L$ ($d={d}$)", fontsize=13)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved Figure 3 to {output_path}")

    return fig


def plot_lipschitz_comparison(
    d: int = 100,
    H_values: Optional[List[int]] = None,
    sigma_values: Optional[List[float]] = None,
    T: int = 1000,
    output_path: Optional[str] = None,
) -> Optional[object]:
    """
    Plot comparison of non-uniform vs uniform Lipschitz constants for GMM.

    Args:
        d: data dimension
        H_values: list of number of components
        sigma_values: list of sigma values
        T: number of iterations
        output_path: path to save figure

    Returns:
        matplotlib figure if available
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plot")
        return None

    if H_values is None:
        H_values = [2, 5, 10, 50, 100]
    if sigma_values is None:
        sigma_values = [0.01, 0.1, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: L_nonuniform vs H
    ax = axes[0]
    for sigma in sigma_values:
        L_nonuniform = [np.log(H * (T + d)) for H in H_values]
        ax.semilogy(H_values, L_nonuniform, "-o", label=f"$\\sigma={sigma}$")

    ax.set_xlabel("Number of components $H$", fontsize=12)
    ax.set_ylabel("Non-uniform Lipschitz $L$", fontsize=12)
    ax.set_title("Non-uniform $L = O(\\log(H(T+d)))$", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Ratio L_uniform / L_nonuniform vs sigma
    ax = axes[1]
    sigma_range = np.logspace(-3, 0, 50)
    mu_norm_sq = d  # typical: ||mu||^2 ~ d

    for H in [2, 5, 10]:
        L_nonuniform = np.log(H * (T + d))
        # Uniform Lipschitz lower bound: mu_norm_sq / (4 * sigma_t^4) at tau=0.5
        # sigma_t^2 = 0.5 * sigma^2 + 0.5
        sigma_t_sq = 0.5 * sigma_range**2 + 0.5
        L_uniform_lb = mu_norm_sq / (4.0 * sigma_t_sq**2)
        ratio = L_uniform_lb / L_nonuniform
        ax.loglog(sigma_range, ratio, label=f"$H={H}$")

    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("$\\sigma$ (component std)", fontsize=12)
    ax.set_ylabel("$L_{\\text{uniform}} / L_{\\text{non-uniform}}$", fontsize=12)
    ax.set_title("Ratio of Lipschitz constants (GMM)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved Lipschitz comparison to {output_path}")

    return fig


def save_results_to_csv(results: Dict, output_dir: str):
    """Save experiment results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for name, result in results.items():
        if isinstance(result, ExperimentResult):
            data = np.column_stack([
                result.T_values,
                result.kl_values,
                result.tv_values,
                result.theoretical_rate,
            ])
            header = "T,KL,TV,theoretical_rate"
            np.savetxt(
                os.path.join(output_dir, f"{name}.csv"),
                data,
                delimiter=",",
                header=header,
                comments="",
            )
            print(f"Saved {name} results to {output_dir}/{name}.csv")
