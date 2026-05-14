## utils/visualization.py
"""Visualization utilities for reproducing paper figures and result tables.

This module provides the Visualizer class for generating all figures and
tables from "Train for the Worst, Plan for the Best: Understanding Token
Ordering in Masked Diffusions". It is a pure reporting utility with no
dependencies on model training or inference logic.

Figures produced:
    - Fig. 2 (left): IsoFLOP scaling law curves (plot_scaling_laws)
    - Fig. 2 (right): Per-position error imbalance (plot_error_imbalance)
    - Fig. 3: Generative perplexity vs. entropy (plot_gen_ppl)

Tables produced:
    - Tables 1-5: Accuracy/perplexity comparisons (save_results_table)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use non-interactive backend to avoid display issues in headless environments.
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level style constants
# ---------------------------------------------------------------------------

# Figure DPI for raster outputs.
_FIGURE_DPI: int = 150

# Font sizes.
_TITLE_FONTSIZE: int = 13
_LABEL_FONTSIZE: int = 11
_TICK_FONTSIZE: int = 9
_LEGEND_FONTSIZE: int = 9
_ANNOTATION_FONTSIZE: int = 8

# Line widths.
_LINE_WIDTH_MAIN: float = 2.0
_LINE_WIDTH_SAMPLE: float = 1.2

# Bar chart settings.
_BAR_WIDTH: float = 0.35
_BAR_ALPHA: float = 0.85

# Shaded region alpha for error imbalance plot.
_REGION_ALPHA: float = 0.15

# Color palette — aligned with paper's visual style.
# ARM (identity) → orange; MDM (uniform) → blue; closer → green;
# much_closer → red; vanilla → gray; adaptive → cornflowerblue.
_COLOR_MAP: Dict[str, str] = {
    "ARM": "#E87722",           # orange
    "identity": "#E87722",      # orange (alias)
    "MDM": "#1F77B4",           # blue
    "uniform": "#1F77B4",       # blue (alias)
    "uniform_random": "#1F77B4",
    "closer": "#2CA02C",        # green
    "much_closer": "#D62728",   # red
    "vanilla": "#7F7F7F",       # gray
    "adaptive": "#1F77B4",      # blue (matches paper Fig. 3 "Adaptive MDM (Blue)")
    "top_probability": "#FF7F0E",
    "top_margin": "#1F77B4",
    "arm_no_order": "#BCBD22",
    "arm_with_order": "#17BECF",
    # Fallback for unknown keys.
    "default": "#9467BD",
}

# Linestyle map for permutation families.
_LINESTYLE_MAP: Dict[str, str] = {
    "ARM": "-",
    "identity": "-",
    "MDM": "-",
    "uniform": "-",
    "uniform_random": "-",
    "closer": "--",
    "much_closer": ":",
}

# Marker map for permutation families.
_MARKER_MAP: Dict[str, str] = {
    "ARM": "o",
    "identity": "o",
    "MDM": "s",
    "uniform": "s",
    "uniform_random": "s",
    "closer": "^",
    "much_closer": "D",
}


class Visualizer:
    """Generates figures and result tables for the masked diffusion paper.

    All outputs are written to ``output_dir``.  Each plot method saves both
    a PDF (for publication quality) and a PNG (for quick inspection).

    Attributes:
        output_dir: Directory where all output files are written.
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        """Initializes the Visualizer and creates the output directory.

        Also sets global matplotlib style defaults so all plots share a
        consistent visual appearance.

        Args:
            output_dir: Directory for saving figures and tables.  Created
                (including parents) if it does not already exist.
        """
        self.output_dir: str = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._configure_matplotlib()
        logger.info("Visualizer initialized. Output directory: '%s'.", output_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot_scaling_laws(
        self,
        results: Dict[str, List[Tuple[float, float]]],
        title: str = "IsoFLOP Scaling Laws: π-Learner vs. ARM vs. MDM",
    ) -> None:
        """Plots IsoFLOP scaling law curves (Figure 2, left of the paper).

        Each curve corresponds to a permutation type.  The expected visual
        pattern is: ARM (identity) achieves the lowest (best) validation loss,
        MDM (uniform random) achieves the highest (worst), and interpolating
        permutations fall in between.

        Args:
            results: Dict mapping permutation type label to a list of
                ``(log_flops, val_loss)`` tuples — one tuple per IsoFLOP
                point.  Expected keys include ``'ARM'``, ``'MDM'``,
                ``'pi_closer_0'``, ``'pi_closer_1'``, ``'pi_closer_2'``,
                ``'pi_much_closer_0'``, etc.
            title: Title string for the plot.
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        # Track which family labels have already been added to the legend
        # so that individual permutation samples within a family share one
        # legend entry.
        legend_added: Dict[str, bool] = {}

        for key, points in results.items():
            if not points:
                logger.warning("No data points for key '%s'; skipping.", key)
                continue

            family: str = self._get_permutation_family(key)
            color: str = self._get_color(family)
            linestyle: str = _LINESTYLE_MAP.get(family, "-")
            marker: str = _MARKER_MAP.get(family, "o")

            # Sort by log_flops for a clean line.
            sorted_points: List[Tuple[float, float]] = sorted(points, key=lambda p: p[0])
            log_flops_vals: List[float] = [p[0] for p in sorted_points]
            val_loss_vals: List[float] = [p[1] for p in sorted_points]

            # Determine label and line width.
            is_primary: bool = key in ("ARM", "MDM", "identity", "uniform", "uniform_random")
            lw: float = _LINE_WIDTH_MAIN if is_primary else _LINE_WIDTH_SAMPLE
            alpha: float = 1.0 if is_primary else 0.65

            # Only add a legend entry for the first curve in each family.
            label: Optional[str] = None
            if family not in legend_added:
                label = self._get_family_display_name(family)
                legend_added[family] = True

            ax.plot(
                log_flops_vals,
                val_loss_vals,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=lw,
                alpha=alpha,
                markersize=5,
                label=label,
            )

        ax.set_xlabel("log(FLOPs)", fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel(r"$-\log p_\theta(x)$", fontsize=_LABEL_FONTSIZE)
        ax.set_title(title, fontsize=_TITLE_FONTSIZE)
        ax.tick_params(labelsize=_TICK_FONTSIZE)
        ax.legend(fontsize=_LEGEND_FONTSIZE, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")

        fig.tight_layout()
        self._save_figure(fig, "scaling_laws")

    def plot_error_imbalance(
        self,
        errors: Dict[str, np.ndarray],
        n_latent: int,
        n_obs: int,
        title: str = "Error Imbalance Across Masking Problems",
    ) -> None:
        """Plots per-position prediction error (Figure 2, right of the paper).

        Shows that MDM's prediction error is higher for latent positions than
        for observation positions, reproducing the key result of Section 3.3.

        Args:
            errors: Dict mapping a label (e.g. ``'model'``, ``'proxy'``, or
                ``'error'``) to a ``np.ndarray`` of shape ``[N + P]`` with
                one error value per sequence position.  When the key
                ``'error'`` is present it is plotted as the MSE difference
                between model and proxy.  Otherwise all arrays are plotted.
            n_latent: Number of latent token positions (first N positions).
            n_obs: Number of observation token positions (next P positions).
            title: Title string for the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 4))

        n_positions: int = n_latent + n_obs
        x_positions: np.ndarray = np.arange(n_positions)

        # Shaded background regions.
        ax.axvspan(
            -0.5,
            n_latent - 0.5,
            alpha=_REGION_ALPHA * 2,
            color="#D62728",
            label="Latent positions (harder)",
        )
        ax.axvspan(
            n_latent - 0.5,
            n_positions - 0.5,
            alpha=_REGION_ALPHA,
            color="#2CA02C",
            label="Observation positions (easier)",
        )

        # Plot error arrays.
        for label, error_array in errors.items():
            if error_array is None or len(error_array) == 0:
                logger.warning("Empty error array for key '%s'; skipping.", label)
                continue

            # Trim or pad to n_positions if necessary.
            arr: np.ndarray = np.asarray(error_array, dtype=float)
            if len(arr) > n_positions:
                arr = arr[:n_positions]
            elif len(arr) < n_positions:
                arr = np.pad(arr, (0, n_positions - len(arr)), constant_values=np.nan)

            color: str = self._get_color(label)
            display_label: str = self._get_error_display_name(label)
            ax.plot(
                x_positions,
                arr,
                color=color,
                linewidth=_LINE_WIDTH_MAIN,
                label=display_label,
                alpha=0.9,
            )

        # Vertical separator between latent and observation regions.
        ax.axvline(
            x=n_latent - 0.5,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )

        ax.set_xlabel("Position index", fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel("MSE Error", fontsize=_LABEL_FONTSIZE)
        ax.set_title(title, fontsize=_TITLE_FONTSIZE)
        ax.tick_params(labelsize=_TICK_FONTSIZE)
        ax.set_xlim(-0.5, n_positions - 0.5)
        ax.legend(fontsize=_LEGEND_FONTSIZE, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        # Annotate region labels on the x-axis.
        ax.text(
            n_latent / 2,
            ax.get_ylim()[0],
            f"Latent\n(N={n_latent})",
            ha="center",
            va="bottom",
            fontsize=_ANNOTATION_FONTSIZE,
            color="#D62728",
            alpha=0.8,
        )
        ax.text(
            n_latent + n_obs / 2,
            ax.get_ylim()[0],
            f"Observation\n(P={n_obs})",
            ha="center",
            va="bottom",
            fontsize=_ANNOTATION_FONTSIZE,
            color="#2CA02C",
            alpha=0.8,
        )

        fig.tight_layout()
        self._save_figure(fig, "error_imbalance")

    def save_results_table(
        self,
        results: Dict[str, Any],
        filename: str = "results",
    ) -> None:
        """Saves a results dictionary as CSV, LaTeX, and a printed table.

        Handles both flat dicts (e.g., puzzle accuracy per method) and
        two-level nested dicts (e.g., LLaDA results per strategy × task, or
        NAE-SAT results per (N,P) instance × strategy).

        Accuracy values in [0, 1] are automatically formatted as percentages
        to match the paper's reporting style (e.g., ``"89.49%"``).

        Args:
            results: Flat or two-level nested dict of results.  Examples:

                Flat (puzzle)::

                    {'vanilla': 0.0688, 'top_margin': 0.8949}

                Nested by (N,P) instance (Table 1)::

                    {'(25,275)': {'vanilla': 0.7806, 'adaptive': 0.9376}}

                Nested by strategy × task (Table 4)::

                    {'vanilla': {'HumanEval-Single': 0.318, 'Math': 0.285}}

            filename: Base filename (without extension).  Outputs are saved
                as ``{output_dir}/{filename}.csv`` and
                ``{output_dir}/{filename}.tex``.
        """
        df: pd.DataFrame = self._results_to_dataframe(results)

        if df.empty:
            logger.warning("Empty results dict; no table saved for '%s'.", filename)
            return

        # Format numeric columns: values in [0, 1] → percentage strings.
        df_display: pd.DataFrame = self._format_dataframe_percentages(df)

        # Save CSV (raw numeric values for downstream processing).
        csv_path: str = os.path.join(self.output_dir, filename + ".csv")
        df.to_csv(csv_path)
        logger.info("Saved results CSV: '%s'.", csv_path)

        # Save LaTeX table.
        tex_path: str = os.path.join(self.output_dir, filename + ".tex")
        try:
            df_display.to_latex(tex_path, escape=False, index=True)
            logger.info("Saved LaTeX table: '%s'.", tex_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Could not save LaTeX table (%s).", exc)

        # Print formatted table to stdout.
        print(f"\n{'=' * 60}")
        print(f"Results: {filename}")
        print("=" * 60)
        print(df_display.to_string())
        print("=" * 60 + "\n")

    def plot_gen_ppl(
        self,
        vanilla_results: Dict[str, float],
        adaptive_results: Dict[str, float],
        title: str = "Generative Perplexity vs. Entropy: Vanilla vs. Adaptive MDM",
    ) -> None:
        """Plots generative perplexity and entropy comparison (Figure 3).

        Shows that adaptive MDM inference substantially reduces generative
        perplexity while maintaining similar entropy (diversity), reproducing
        the key result of Section 4.2 / Fig. 3 of the paper.

        Args:
            vanilla_results: Dict with keys ``'gen_ppl'`` (float) and
                ``'entropy'`` (float) for vanilla MDM inference.
            adaptive_results: Dict with keys ``'gen_ppl'`` (float) and
                ``'entropy'`` (float) for adaptive (Top Probability Margin)
                MDM inference.
            title: Overall figure title.
        """
        gen_ppl_vanilla: float = float(vanilla_results.get("gen_ppl", 0.0))
        gen_ppl_adaptive: float = float(adaptive_results.get("gen_ppl", 0.0))
        entropy_vanilla: float = float(vanilla_results.get("entropy", 0.0))
        entropy_adaptive: float = float(adaptive_results.get("entropy", 0.0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

        method_labels: List[str] = ["Vanilla", "Adaptive\n(Top Margin)"]
        x_positions: np.ndarray = np.array([0, 1], dtype=float)

        color_vanilla: str = _COLOR_MAP["vanilla"]
        color_adaptive: str = _COLOR_MAP["adaptive"]
        bar_colors: List[str] = [color_vanilla, color_adaptive]

        # ---- Left subplot: Generative Perplexity ----
        gen_ppl_values: List[float] = [gen_ppl_vanilla, gen_ppl_adaptive]
        bars1 = ax1.bar(
            x_positions,
            gen_ppl_values,
            width=_BAR_WIDTH * 2,
            color=bar_colors,
            alpha=_BAR_ALPHA,
            edgecolor="black",
            linewidth=0.8,
        )
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(method_labels, fontsize=_TICK_FONTSIZE)
        ax1.set_ylabel("Generative Perplexity (GenPPL)", fontsize=_LABEL_FONTSIZE)
        ax1.set_title("Generative Perplexity", fontsize=_TITLE_FONTSIZE)
        ax1.tick_params(labelsize=_TICK_FONTSIZE)
        ax1.grid(True, alpha=0.3, linestyle="--", axis="y")

        # Annotate bar values.
        for bar, val in zip(bars1, gen_ppl_values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(gen_ppl_values) * 0.01,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=_ANNOTATION_FONTSIZE,
                fontweight="bold",
            )

        # ---- Right subplot: Entropy ----
        entropy_values: List[float] = [entropy_vanilla, entropy_adaptive]
        bars2 = ax2.bar(
            x_positions,
            entropy_values,
            width=_BAR_WIDTH * 2,
            color=bar_colors,
            alpha=_BAR_ALPHA,
            edgecolor="black",
            linewidth=0.8,
        )
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(method_labels, fontsize=_TICK_FONTSIZE)
        ax2.set_ylabel("Entropy", fontsize=_LABEL_FONTSIZE)
        ax2.set_title("Sample Entropy", fontsize=_TITLE_FONTSIZE)
        ax2.tick_params(labelsize=_TICK_FONTSIZE)
        ax2.grid(True, alpha=0.3, linestyle="--", axis="y")

        # Annotate bar values.
        max_entropy: float = max(entropy_values) if max(entropy_values) > 0 else 1.0
        for bar, val in zip(bars2, entropy_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max_entropy * 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=_ANNOTATION_FONTSIZE,
                fontweight="bold",
            )

        # Add a shared legend.
        from matplotlib.patches import Patch  # noqa: PLC0415
        legend_elements = [
            Patch(facecolor=color_vanilla, alpha=_BAR_ALPHA, edgecolor="black",
                  label="Vanilla MDM"),
            Patch(facecolor=color_adaptive, alpha=_BAR_ALPHA, edgecolor="black",
                  label="Adaptive MDM (Top Margin)"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            fontsize=_LEGEND_FONTSIZE,
            bbox_to_anchor=(0.5, -0.05),
        )

        fig.suptitle(title, fontsize=_TITLE_FONTSIZE, y=1.02)
        fig.tight_layout()
        self._save_figure(fig, "gen_ppl_comparison")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _configure_matplotlib(self) -> None:
        """Sets global matplotlib style defaults for consistent visuals."""
        plt.rcParams.update(
            {
                "figure.dpi": _FIGURE_DPI,
                "savefig.dpi": _FIGURE_DPI,
                "savefig.bbox": "tight",
                "font.size": _TICK_FONTSIZE,
                "axes.titlesize": _TITLE_FONTSIZE,
                "axes.labelsize": _LABEL_FONTSIZE,
                "xtick.labelsize": _TICK_FONTSIZE,
                "ytick.labelsize": _TICK_FONTSIZE,
                "legend.fontsize": _LEGEND_FONTSIZE,
                "lines.linewidth": _LINE_WIDTH_MAIN,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.autolayout": False,
            }
        )

    def _save_figure(self, fig: plt.Figure, base_name: str) -> None:
        """Saves a figure as PDF and PNG, then closes it.

        Args:
            fig: The matplotlib Figure to save.
            base_name: Base filename (without extension).  Files are saved as
                ``{output_dir}/{base_name}.pdf`` and
                ``{output_dir}/{base_name}.png``.
        """
        for ext in ("pdf", "png"):
            path: str = os.path.join(self.output_dir, f"{base_name}.{ext}")
            try:
                fig.savefig(path, bbox_inches="tight")
                logger.info("Saved figure: '%s'.", path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Failed to save figure '%s': %s", path, exc)
        plt.close(fig)

    def _get_color(self, key: str) -> str:
        """Returns the matplotlib color string for a given key.

        Falls back to the ``'default'`` color if the key is not in the map.

        Args:
            key: A permutation family name, method name, or error label.

        Returns:
            A matplotlib color string.
        """
        # Try exact match first, then lowercase.
        color: Optional[str] = _COLOR_MAP.get(key)
        if color is None:
            color = _COLOR_MAP.get(key.lower())
        if color is None:
            color = _COLOR_MAP["default"]
        return color

    def _get_permutation_family(self, key: str) -> str:
        """Extracts the permutation family from a result key.

        Handles keys like ``'pi_closer_0'``, ``'pi_much_closer_2'``,
        ``'ARM'``, ``'MDM'``, ``'identity'``, ``'uniform'``, etc.

        Args:
            key: A result dict key.

        Returns:
            The family string used for color/linestyle/marker lookup.
        """
        key_lower: str = key.lower()

        if key_lower in ("arm", "identity"):
            return "ARM"
        if key_lower in ("mdm", "uniform", "uniform_random"):
            return "MDM"
        if "much_closer" in key_lower:
            return "much_closer"
        if "closer" in key_lower:
            return "closer"
        # Fallback: return the key itself (will use default color).
        return key

    def _get_family_display_name(self, family: str) -> str:
        """Returns a human-readable display name for a permutation family.

        Args:
            family: Internal family string.

        Returns:
            Display name for the legend.
        """
        display_names: Dict[str, str] = {
            "ARM": "ARM (identity, left-to-right)",
            "MDM": "MDM (uniform random π)",
            "closer": "π-learner (closer, L/10 swaps)",
            "much_closer": "π-learner (much-closer, √L swaps)",
        }
        return display_names.get(family, family)

    def _get_error_display_name(self, label: str) -> str:
        """Returns a human-readable display name for an error array label.

        Args:
            label: Key from the ``errors`` dict.

        Returns:
            Display name for the legend.
        """
        display_names: Dict[str, str] = {
            "model": "MDM (trained)",
            "proxy": "Proxy MDM (Bayes-optimal approx.)",
            "error": "MSE Error (model vs. proxy)",
            "mse": "MSE Error",
        }
        return display_names.get(label.lower(), label)

    def _results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Converts a flat or two-level nested results dict to a DataFrame.

        Handles three structural cases:

        1. **Flat dict** (e.g., puzzle accuracy per method):
           ``{'vanilla': 0.07, 'top_margin': 0.89}``
           → single-column DataFrame with method names as index.

        2. **Nested dict, outer = instance, inner = method** (e.g., Table 1):
           ``{'(25,275)': {'vanilla': 0.78, 'adaptive': 0.94}}``
           → DataFrame with instances as rows, methods as columns.

        3. **Nested dict, outer = method, inner = task** (e.g., Table 4):
           ``{'vanilla': {'HumanEval-Single': 0.318, 'Math': 0.285}}``
           → DataFrame with methods as rows, tasks as columns.

        Args:
            results: The results dict to convert.

        Returns:
            A ``pd.DataFrame`` with appropriate index and columns.
        """
        if not results:
            return pd.DataFrame()

        # Detect nesting depth.
        first_value: Any = next(iter(results.values()))

        if isinstance(first_value, dict):
            # Two-level nesting: outer keys → rows, inner keys → columns.
            df: pd.DataFrame = pd.DataFrame.from_dict(results, orient="index")
            df.index.name = "Instance / Method"
            return df
        else:
            # Flat dict: keys → index, values → single column "Value".
            df = pd.DataFrame.from_dict(
                results, orient="index", columns=["Value"]
            )
            df.index.name = "Method"
            return df

    def _format_dataframe_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Formats numeric columns: values in [0, 1] → percentage strings.

        Values outside [0, 1] (e.g., perplexity, loss) are formatted as
        plain floats with 4 decimal places.

        Args:
            df: Input DataFrame with numeric values.

        Returns:
            A copy of the DataFrame with formatted string values.
        """
        df_display: pd.DataFrame = df.copy()

        for col in df_display.columns:
            try:
                numeric_col: pd.Series = pd.to_numeric(df_display[col], errors="coerce")
            except (TypeError, ValueError):
                continue

            if numeric_col.isna().all():
                continue

            # Determine if this column contains probability/accuracy values.
            col_min: float = float(numeric_col.min(skipna=True))
            col_max: float = float(numeric_col.max(skipna=True))
            is_probability: bool = (col_min >= 0.0) and (col_max <= 1.0)

            if is_probability:
                df_display[col] = numeric_col.apply(
                    lambda v: self._format_percentage(v) if not pd.isna(v) else "N/A"
                )
            else:
                df_display[col] = numeric_col.apply(
                    lambda v: f"{v:.4f}" if not pd.isna(v) else "N/A"
                )

        return df_display

    @staticmethod
    def _format_percentage(value: float) -> str:
        """Converts a float in [0, 1] to a percentage string.

        Matches the paper's reporting style, e.g., ``0.8949`` → ``"89.49%"``.

        Args:
            value: A float in [0, 1].

        Returns:
            Percentage string with two decimal places.
        """
        return f"{value * 100:.2f}%"
