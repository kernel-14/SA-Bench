```python
## plots.py
"""Plots module for reproducing Figures 1, 2, and 3 from
"Instance-dependent Convergence Theory for Diffusion Models".

This module is a pure visualization layer. It takes pre-computed experimental
results (from experiments.py) and theoretical formulas, then renders them into
publication-quality figures matching the paper's Figures 1, 2, and 3.

All heavy computation is delegated upstream. The only external dependency
beyond standard libraries is metrics.Metrics.fit_theoretical_rate (for Figure 2)
and config.Config (for type annotations).

Figure descriptions
-------------------
Figure 1 (left):  Iteration complexity T vs. L for all 5 methods, eps = O(1).
Figure 1 (right): Iteration complexity T vs. eps for L = infinity.
Figure 2:         Empirical KL divergence vs. T with fitted theoretical rate
                  C * log^4(T) / T^3, for three (d, k) configurations.
Figure 3:         TV distance vs. L for fixed T in {O(d), O(d^1.5), O(d^2)}.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from metrics import Metrics

# Use non-interactive backend to avoid display issues in headless environments
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Module-level plot styling constants (matching config.yaml: plotting section)
# ---------------------------------------------------------------------------

# Colors for each method (config.yaml: plotting.colors)
_METHOD_COLORS: Dict[str, str] = {
    "ours": "red",
    "benton2023": "blue",
    "li_yan2024": "green",
    "li_cai2024": "orange",
    "li_jiao2024": "purple",
}

# Line styles for each method (config.yaml: plotting.linestyles)
# Note: (0, 3, 1, 1) is a custom dash-dot pattern for li_jiao2024
_METHOD_LINESTYLES: Dict[str, Any] = {
    "ours": "-",
    "benton2023": "--",
    "li_yan2024": "-.",
    "li_cai2024": ":",
    "li_jiao2024": (0, (3, 1, 1, 1)),  # densely dash-dotted
}

# Legend labels for each method (config.yaml: plotting.labels)
_METHOD_LABELS: Dict[str, str] = {
    "ours": "This work",
    "benton2023": "Benton et al. (2023)",
    "li_yan2024": "Li & Yan (2024a)",
    "li_cai2024": "Li & Cai (2024)",
    "li_jiao2024": "Li & Jiao (2024)",
}

# Ordered list of methods for consistent plot ordering
_METHOD_ORDER: List[str] = [
    "ours",
    "benton2023",
    "li_yan2024",
    "li_cai2024",
    "li_jiao2024",
]

# DPI for saved figures (config.yaml: plotting.dpi)
_DEFAULT_DPI: int = 300

# Figure sizes (config.yaml: plotting.figsize_*)
_FIGSIZE_FIGURE2: Tuple[float, float] = (15.0, 5.0)
_FIGSIZE_FIGURE1: Tuple[float, float] = (12.0, 5.0)
_FIGSIZE_FIGURE3: Tuple[float, float] = (15.0, 5.0)

# Empirical result styling (config.yaml: plotting.empirical_*)
_EMPIRICAL_COLOR: str = "blue"
_THEORETICAL_COLOR: str = "black"
_THEORETICAL_LINESTYLE: str = "--"
_EMPIRICAL_MARKER: str = "o"
_EMPIRICAL_MARKERSIZE: int = 5

# Proxy value for L = infinity (config.yaml: figure1.L_inf_proxy)
_L_INF_PROXY: float = 1.0e12

# Minimum positive value for log-scale plots (avoids log(0))
_LOG_FLOOR: float = 1.0e-20

# Maximum iteration complexity to display (clip for readability)
_T_MAX_DISPLAY: float = 1.0e15


class Plots:
    """Visualization class for reproducing all paper figures.

    Provides methods to generate Figures 1, 2, and 3 from the paper.
    All figures are saved to self.figure_dir in both PDF and PNG formats.

    Attributes:
        figure_dir: Output directory for saved figures.
        dpi: Resolution for saved figures (default 300).
        colors: Dict mapping method names to color strings.
        linestyles: Dict mapping method names to matplotlib linestyle specs.
        labels: Dict mapping method names to legend label strings.
    """

    def __init__(self, figure_dir: str = "figures") -> None:
        """Initialise the Plots object and create the output directory.

        Args:
            figure_dir: Path to the output directory for saved figures.
                Created if it does not exist. Default "figures".

        Raises:
            TypeError: If figure_dir is not a string.
        """
        if not isinstance(figure_dir, str):
            raise TypeError(
                f"figure_dir must be str, got {type(figure_dir).__name__}"
            )

        self.figure_dir: str = figure_dir
        self.dpi: int = _DEFAULT_DPI

        # Create output directory
        os.makedirs(self.figure_dir, exist_ok=True)

        # Styling constants (matching config.yaml)
        self.colors: Dict[str, str] = dict(_METHOD_COLORS)
        self.linestyles: Dict[str, Any] = dict(_METHOD_LINESTYLES)
        self.labels: Dict[str, str] = dict(_METHOD_LABELS)

        # Metrics instance for fit_theoretical_rate
        self.metrics: Metrics = Metrics()

        # Set global matplotlib rcParams for consistent styling
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 12,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 9,
                "figure.dpi": self.dpi,
                "lines.linewidth": 1.8,
            }
        )

    # ------------------------------------------------------------------
    # Private: save figure
    # ------------------------------------------------------------------

    def _save(self, fig: matplotlib.figure.Figure, name: str) -> None:
        """Save a matplotlib figure to both PDF and PNG formats.

        Args:
            fig: The matplotlib Figure object to save.
            name: Base filename (without extension). The figure will be
                saved as {figure_dir}/{name}.pdf and {figure_dir}/{name}.png.
        """
        pdf_path: str = os.path.join(self.figure_dir, name + ".pdf")
        png_path: str = os.path.join(self.figure_dir, name + ".png")

        fig.savefig(pdf_path, dpi=self.dpi, bbox_inches="tight")
        fig.savefig(png_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {pdf_path}")
        print(f"Saved: {png_path}")

    # ------------------------------------------------------------------
    # Figure 2: Empirical convergence (Appendix A)
    # ------------------------------------------------------------------

    def plot_figure2(
        self,
        results: Dict[str, Any],
        configs: List[Config],
    ) -> None:
        """Reproduce Figure 2: empirical KL divergence vs. T with fitted rate.

        Creates a 1x3 subplot figure. Each subplot corresponds to one of the
        three paper configurations (d=10/k=10, d=100/k=10, d=500/k=100).
        Shows empirical KL values (blue line with markers) and the fitted
        theoretical rate C * log^4(T) / T^3 (black dashed line) on log-log axes.

        Args:
            results: Dict mapping config label -> result dict. Each result
                dict must have keys 'T_values' (List[int]), 'kl_values'
                (List[float]), and 'config' (Config).
            configs: List of Config objects in subplot order. Used to
                determine subplot titles and to look up results by label.
        """
        n_configs: int = len(configs)
        if n_configs == 0:
            print("plot_figure2: no configs provided, skipping.")
            return

        fig, axes = plt.subplots(
            1, n_configs, figsize=_FIGSIZE_FIGURE2, squeeze=False
        )
        axes_flat: List[matplotlib.axes.Axes] = [axes[0, i] for i in range(n_configs)]

        subplot_letters: str = "abcdefghij"

        for i, cfg in enumerate(configs):
            ax: matplotlib.axes.Axes = axes_flat[i]

            # Determine result key: prefer cfg.label, fall back to d/k string
            key: str = cfg.label if cfg.label else f"d{cfg.d}_k{cfg.k_active}"

            if key not in results:
                ax.set_title(
                    f"({subplot_letters[i]}) $d={cfg.d}$, $k={cfg.k_active}$ [no data]"
                )
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                )
                continue

            result: Dict[str, Any] = results[key]
            T_values_raw: List[int] = result["T_values"]
            kl_values_raw: List[float] = result["kl_values"]

            # --- Filter valid data points (positive, finite KL) ---
            T_valid: List[int] = []
            kl_valid: List[float] = []
            for T_val, kl_val in zip(T_values_raw, kl_values_raw):
                if (
                    not math.isnan(kl_val)
                    and not math.isinf(kl_val)
                    and kl_val > 0.0
                    and T_val >= 2
                ):
                    T_valid.append(T_val)
                    kl_valid.append(kl_val)

            if len(T_valid) == 0:
                ax.set_title(
                    f"({subplot_letters[i]}) $d={cfg.d}$, $k={cfg.k_active}$ [no valid data]"
                )
                ax.text(
                    0.5,
                    0.5,
                    "No valid data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                )
                continue

            T_arr: np.ndarray = np.array(T_valid, dtype=np.float64)
            kl_arr: np.ndarray = np.array(kl_valid, dtype=np.float64)

            # --- Plot empirical KL ---
            ax.loglog(
                T_arr,
                kl_arr,
                color=_EMPIRICAL_COLOR,
                marker=_EMPIRICAL_MARKER,
                markersize=_EMPIRICAL_MARKERSIZE,
                linewidth=1.8,
                label="Empirical KL",
                zorder=3,
            )

            # --- Fit and plot theoretical rate C * log^4(T) / T^3 ---
            C_fit: float = 1.0
            fitted_curve: np.ndarray = np.ones_like(T_arr)
            try:
                C_fit, fitted_curve = self.metrics.fit_theoretical_rate(
                    T_valid, kl_valid
                )
                ax.loglog(
                    T_arr,
                    fitted_curve,
                    color=_THEORETICAL_COLOR,
                    linestyle=_THEORETICAL_LINESTYLE,
                    linewidth=1.8,
                    label=r"$C \cdot \log^4(T)/T^3$",
                    zorder=2,
                )
            except Exception as exc:
                print(
                    f"plot_figure2: could not fit theoretical rate for {key}: {exc}"
                )

            # --- Axis labels ---
            ax.set_xlabel("$T$")
            ax.set_ylabel("KL divergence")

            # --- Title ---
            ax.set_title(
                f"({subplot_letters[i]}) $d={cfg.d}$, $k={cfg.k_active}$"
            )

            # --- Legend ---
            ax.legend(loc="upper right")

            # --- Grid ---
            ax.grid(True, which="both", alpha=0.3, linestyle=":")

            # --- Annotate fitted constant C ---
            ax.text(
                0.05,
                0.08,
                f"$C = {C_fit:.2e}$",
                transform=ax.transAxes,
                fontsize=9,
                color="black",
                verticalalignment="bottom",
            )

        plt.tight_layout()
        self._save(fig, "figure2")

    # ------------------------------------------------------------------
    # Figure 1: Iteration complexity comparison (Section 1.1)
    # ------------------------------------------------------------------

    def plot_figure1_left(
        self,
        d: int,
        L_values: np.ndarray,
        eps: float = 1.0,
    ) -> None:
        """Reproduce Figure 1 (left): iteration complexity T vs. L, eps = O(1).

        Plots T as a function of L for all 5 methods on log-log axes.
        Methods without L-dependence (Benton, Li-Yan, Li-Cai) appear as
        horizontal lines. Vertical reference lines mark L = sqrt(d) and L = d.

        Args:
            d: Data dimension. From config: figure1.d = 100.
            L_values: Log-spaced array of L values. From config:
                L_min=1.0, L_max=10000.0, L_num_points=200.
            eps: Fixed accuracy level. From config: figure1.eps_fixed = 1.0.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0))

        for method in _METHOD_ORDER:
            T_array: np.ndarray = np.array(
                [
                    min(
                        self._iteration_complexity_dispatch(method, d, float(L), eps),
                        _T_MAX_DISPLAY,
                    )
                    for L in L_values
                ],
                dtype=np.float64,
            )

            # Replace inf with display maximum
            T_array = np.where(np.isinf(T_array), _T_MAX_DISPLAY, T_array)
            T_array = np.clip(T_array, 1.0, _T_MAX_DISPLAY)

            ax.loglog(
                L_values,
                T_array,
                color=self.colors[method],
                linestyle=self.linestyles[method],
                linewidth=2.0,
                label=self.labels[method],
            )

        # --- Vertical reference lines at L = sqrt(d) and L = d ---
        sqrt_d: float = math.sqrt(float(d))
        ax.axvline(
            x=sqrt_d,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=r"$L = \sqrt{d}$",
        )
        ax.axvline(
            x=float(d),
            color="dimgray",
            linestyle="--",
            linewidth=1.2,
            label=r"$L = d$",
        )

        # --- Axis labels and title ---
        ax.set_xlabel("$L$")
        ax.set_ylabel("Iteration complexity $T$")
        ax.set_title(f"$d = {d}$, $\\varepsilon = O(1)$")

        # --- Legend ---
        ax.legend(loc="upper left", fontsize=8)

        # --- Grid ---
        ax.grid(True, which="both", alpha=0.3, linestyle=":")

        plt.tight_layout()
        self._save(fig, "figure1_left")

    def plot_figure1_right(
        self,
        d: int,
        eps_values: np.ndarray,
        L_inf: bool = True,
    ) -> None:
        """Reproduce Figure 1 (right): iteration complexity T vs. eps, L = inf.

        Plots T as a function of eps for all applicable methods on log-log axes.
        For L = infinity, Li & Jiao (2024) has no finite bound and is omitted.

        Args:
            d: Data dimension. From config: figure1.d = 100.
            eps_values: Log-spaced array of eps values. From config:
                eps_min=0.001, eps_max=0.5, eps_num_points=200.
            L_inf: If True, use L = L_INF_PROXY as proxy for L = infinity.
                Default True.
        """
        L_val: float = _L_INF_PROXY if L_inf else float(d)

        fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0))

        for method in _METHOD_ORDER:
            # For L = infinity, li_jiao2024 diverges — skip it
            if L_inf and method == "li_jiao2024":
                continue

            T_array: np.ndarray = np.array(
                [
                    min(
                        self._iteration_complexity_dispatch(
                            method, d, L_val, float(eps)
                        ),
                        _T_MAX_DISPLAY,
                    )
                    for eps in eps_values
                ],
                dtype=np.float64,
            )

            # Replace inf with display maximum
            T_array = np.where(np.isinf(T_array), _T_MAX_DISPLAY, T_array)
            T_array = np.clip(T_array, 1.0, _T_MAX_DISPLAY)

            ax.loglog(
                eps_values,
                T_array,
                color=self.colors[method],
                linestyle=self.linestyles[method],
                linewidth=2.0,
                label=self.labels[method],
            )

        # --- Axis labels and title ---
        ax.set_xlabel("$\\varepsilon$")
        ax.set_ylabel("Iteration complexity $T$")
        title_L: str = "\\infty" if L_inf else f"{L_val:.0f}"
        ax.set_title(f"$d = {d}$, $L = {title_L}$")

        # --- Legend ---
        ax.legend(loc="upper right", fontsize=8)

        # --- Grid ---
        ax.grid(True, which="both", alpha=0.3, linestyle=":")

        plt.tight_layout()
        self._save(fig, "figure1_right")

    # ------------------------------------------------------------------
    # Figure 3: TV distance vs. L for fixed T (Appendix B)
    # ------------------------------------------------------------------

    def plot_figure3(
        self,
        d: int,
        T_cases: List[Tuple[str, float]],
        L_values: Optional[np.ndarray] = None,
    ) -> None:
        """Reproduce Figure 3: TV distance vs. L for fixed T values.

        Creates a 1x3 subplot figure. Each subplot corresponds to one of
        the three T values: T = O(d), T = O(d^1.5), T = O(d^2).
        Shows TV distance bounds for all 5 methods as a function of L.

        Args:
            d: Data dimension. From config: figure3.d = 100.
            T_cases: List of (label, T_value) tuples. From config:
                [('T=O(d)', d), ('T=O(d^1.5)', d^1.5), ('T=O(d^2)', d^2)].
            L_values: Log-spaced array of L values. If None, uses a default
                range from 1.0 to 10000.0 with 200 points (from config:
                figure3.L_min=1.0, figure3.L_max=10000.0, figure3.L_num_points=200).
        """
        if L_values is None:
            L_values = np.logspace(0.0, 4.0, 200)

        n_cases: int = len(T_cases)
        if n_cases == 0:
            print("plot_figure3: no T_cases provided, skipping.")
            return

        fig, axes = plt.subplots(
            1, n_cases, figsize=_FIGSIZE_FIGURE3, squeeze=False
        )
        axes_flat: List[matplotlib.axes.Axes] = [axes[0, i] for i in range(n_cases)]

        subplot_letters: str = "abcdefghij"

        for i, (T_label, T_val) in enumerate(T_cases):
            ax: matplotlib.axes.Axes = axes_flat[i]

            for method in _METHOD_ORDER:
                tv_array: np.ndarray = np.array(
                    [
                        self._tv_dispatch(method, d, float(L), T_val)
                        for L in L_values
                    ],
                    dtype=np.float64,
                )

                # Clip TV to [0, 1] — TV distance is bounded by 1
                tv_array = np.clip(tv_array, 0.0, 1.0)

                # Replace NaN/inf with 1.0 (trivial bound)
                tv_array = np.where(
                    np.isnan(tv_array) | np.isinf(tv_array), 1.0, tv_array
                )

                ax.semilogx(
                    L_values,
                    tv_array,
                    color=self.colors[method],
                    linestyle=self.linestyles[method],
                    linewidth=2.0,
                    label=self.labels[method],
                )

            # --- Axis labels ---
            ax.set_xlabel("$L$")
            if i == 0:
                ax.set_ylabel("TV distance $\\varepsilon$")

            # --- Title ---
            ax.set_title(f"({subplot_letters[i]}) {T_label}")

            # --- Y-axis limits ---
            ax.set_ylim([0.0, 1.1])

            # --- Reference line at TV = 1 (trivial bound) ---
            ax.axhline(
                y=1.0,
                color="black",
                linestyle=":",
                linewidth=1.0,
                alpha=0.5,
            )

            # --- Legend (only on first subplot) ---
            if i == 0:
                ax.legend(loc="upper left", fontsize=8)

            # --- Grid ---
            ax.grid(True, which="both", alpha=0.3, linestyle=":")

        plt.tight_layout()
        self._save(fig, "figure3")

    # ------------------------------------------------------------------
    # Private: iteration complexity helpers
    # ------------------------------------------------------------------

    def _iteration_complexity_dispatch(
        self, method: str, d: int, L: float, eps: float
    ) -> float:
        """Dispatch to the appropriate iteration complexity formula.

        Args:
            method: One of the keys in _METHOD_ORDER.
            d: Data dimension.
            L: Lipschitz constant.
            eps: Target accuracy.

        Returns:
            Iteration complexity as a float. Returns float('inf') for
            methods that have no finite bound (e.g., li_jiao2024 with L=inf).
        """
        if method == "ours":
            return self._iteration_complexity_ours(d, L, eps)
        elif method == "benton2023":
            return self._iteration_complexity_benton(d, eps)
        elif method == "li_yan2024":
            return self._iteration_complexity_li_yan(d, eps)
        elif method == "li_cai2024":
            return self._iteration_complexity_li_cai(d, eps)
        elif method == "li_jiao2024":
            return self._iteration_complexity_li_jiao(d, L, eps)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _iteration_complexity_ours(
        self, d: int, L: float, eps: float
    ) -> float:
        """Iteration complexity for this work (Theorem 1, Section 1.1).

        Formula (omitting log factors):
            min(d, d^(2/3) * L^(1/3), d^(1/3) * L) * eps^(-2/3)

        For L = infinity: min reduces to d, giving d * eps^(-2/3).

        Args:
            d: Data dimension.
            L: Non-uniform Lipschitz constant. Use large value for L=inf.
            eps: Target accuracy in TV distance.

        Returns:
            Iteration complexity as a float.
        """
        d_f: float = float(d)
        L_f: float = float(L)
        eps_f: float = float(eps)

        if math.isinf(L_f) or L_f > 1.0e30:
            # L = infinity: min(d, inf, inf) = d
            complexity_factor: float = d_f
        else:
            term1: float = d_f
            term2: float = (d_f ** (2.0 / 3.0)) * (L_f ** (1.0 / 3.0))
            term3: float = (d_f ** (1.0 / 3.0)) * L_f
            complexity_factor = min(term1, term2, term3)

        return complexity_factor * (eps_f ** (-2.0 / 3.0))

    def _iteration_complexity_benton(self, d: int, eps: float) -> float:
        """Iteration complexity for Benton et al. (2023).

        Formula: d * eps^(-2)

        Args:
            d: Data dimension.
            eps: Target accuracy in TV distance.

        Returns:
            Iteration complexity as a float.
        """
        return float(d) * (float(eps) ** (-2.0))

    def _iteration_complexity_li_yan(self, d: int, eps: float) -> float:
        """Iteration complexity for Li & Yan (2024a).

        Formula: d * eps^(-1)

        Args:
            d: Data dimension.
            eps: Target accuracy in TV distance.

        Returns:
            Iteration complexity as a float.
        """
        return float(d) * (float(eps) ** (-1.0))

    def _iteration_complexity_li_cai(self, d: int, eps: float) -> float:
        """Iteration complexity for Li & Cai (2024).

        Formula: d^(5/4) * eps^(-1/2)

        Args:
            d: Data dimension.
            eps: Target accuracy in TV distance.

        Returns:
            Iteration complexity as a float.
        """
        return (float(d) ** (5.0 / 4.0)) * (float(eps) ** (-0.5))

    def _iteration_complexity_li_jiao(
        self, d: int, L: float, eps: float
    ) -> float:
        """Iteration complexity for Li & Jiao (2024).

        Formula: d^(1/3) * L * eps^(-2/3)

        For L = infinity: returns float('inf') (no finite bound without
        smoothness assumption).

        Args:
            d: Data dimension.
            L: Lipschitz constant. Use large value for L=inf.
            eps: Target accuracy in TV distance.

        Returns:
            Iteration complexity as a float. Returns float('inf') for L=inf.
        """
        L_f: float = float(L)
        if math.isinf(L_f) or L_f > 1.0e30:
            return float("inf")
        return (float(d) ** (1.0 / 3.0)) * L_f * (float(eps) ** (-2.0 / 3.0))

    # ------------------------------------------------------------------
    # Private: TV distance bound helpers
    # ------------------------------------------------------------------

    def _tv_dispatch(
        self, method: str, d: int, L: float, T: float
    ) -> float:
        """Dispatch to the appropriate TV distance bound formula.

        Args:
            method: One of the keys in _METHOD_ORDER.
            d: Data dimension.
            L: Lipschitz constant.
            T: Number of iterations.

        Returns:
            TV distance bound as a float in [0, 1].
        """
        if method == "ours":
            return self._tv_ours(d, L, T)
        elif method == "benton2023":
            return self._tv_benton(d, T)
        elif method == "li_yan2024":
            return self._tv_li_yan(d, T)
        elif method == "li_cai2024":
            return self._tv_li_cai(d, T)
        elif method == "li_jiao2024":
            return self._tv_li_jiao(d, L, T)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _tv_ours(self, d: int, L: float, T: float) -> float:
        """TV distance bound for this work (Theorem 1, Eq. 13).

        Formula (omitting log factors):
            TV = (min(d^(3/2), d * L^(1/2), d^(1/2) * L^(3/2)) / T^(3/2))

        Derived from: TV <= C * min{d^(3/2), d*L^(1/2), d^(1/2)*L^(3/2)} * log^4(T) / T^(3/2)
        (Theorem 1, ignoring log factors for comparison plot).

        For L = infinity: min reduces to d^(3/2), giving (d^(3/2) / T^(3/2)).

        Args:
            d: Data dimension.
            L: Non-uniform Lipschitz constant.
            T: Number of iterations.

        Returns:
            TV distance bound as a float, clamped to [0, 1].
        """
        d_f: float = float(d)
        L_f: float = float(L)
        T_f: float = float(T)

        if T_f <= 0.0:
            return 1.0

        if math.isinf(L_f) or L_f > 1.0e30:
            # L = infinity: min(d^(3/2), inf, inf) = d^(3/2)
            complexity_factor: float = d_f ** (3.0 / 2.0)
        else:
            term1: float = d_f ** (3.0 / 2.0)
            term2: float = d_f * (L_f ** (1.0 / 2.0))
            term3: float = (d_f ** (1.0 / 2.0)) * (L_f ** (3.0 / 2.0))
            complexity_factor = min(term1, term2, term3)

        tv: float = complexity_factor / (T_f ** (3.0 / 2.0))
        return min(max(tv, 0.0), 1.0)

    def _tv_benton(self, d: int, T: float) -> float:
        """TV distance bound for Benton et al. (2023).

        Formula: TV = sqrt(d