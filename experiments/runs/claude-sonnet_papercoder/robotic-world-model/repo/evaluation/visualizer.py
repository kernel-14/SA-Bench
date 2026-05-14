## Code: evaluation/visualizer.py

```python
## evaluation/visualizer.py
"""Visualization module for the Robotic World Model (RWM) project.

This module provides the ``Visualizer`` class, a pure plotting utility that
produces publication-quality figures replicating the paper's key results.
It is consumed by ``evaluation/benchmark.py`` and ``main.py``.

All methods are stateless with respect to model/training state — they accept
pre-computed data (tensors, arrays, dicts) and produce matplotlib figures
saved to disk. The class holds only configuration for consistent styling.

Figures produced:
  - Fig. 3a: Trajectory prediction (solid GT vs dashed predicted)
  - Fig. 3b: Noise robustness (yellow RWM vs grey MLP families)
  - Fig. 4:  Environment benchmark (multi-panel error curves per model)
  - Fig. 5:  Policy training curves (model error + reward, solid GT vs dashed predicted)
  - Fig. S8: Horizon ablation heatmaps (error and training time vs M, N)

Usage:
    viz = Visualizer(save_dir="logs/figures")
    viz.plot_trajectory_prediction(pred, target, labels, title="ANYmal D")
    viz.plot_prediction_error_curves(errors, title="Architecture Comparison")
    viz.plot_noise_robustness(rwm_errors, mlp_errors, noise_levels=[0.01, 0.05, 0.1, 0.2])
    viz.plot_policy_training_curves(model_errors, rewards, title="MBPO-PPO vs Baselines")
    viz.plot_horizon_ablation_heatmap(errors, times, m_values=[8,16,32,64], n_values=[1,2,4,8,16])
    viz.plot_environment_benchmark(results)
"""

import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend to avoid display issues on headless servers.
# Must be set before any other matplotlib imports that trigger GUI initialization.
matplotlib.use("Agg")

# Optional seaborn import for heatmaps — graceful fallback to matplotlib.
try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    sns = None  # type: ignore[assignment]
    SEABORN_AVAILABLE = False
    warnings.warn(
        "seaborn is not installed. Heatmap plots will use matplotlib fallback. "
        "Install with: pip install seaborn==0.13.2",
        UserWarning,
        stacklevel=1,
    )

# Optional torch import for Tensor → numpy conversion.
try:
    import torch
    from torch import Tensor as TorchTensor

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TorchTensor = None  # type: ignore[assignment,misc]
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default figure DPI for saved figures.
_DEFAULT_DPI: int = 150

# Default base font size for all text elements.
_DEFAULT_FONT_SIZE: int = 10

# Moving average window for smoothing training curves (Fig. 5).
_SMOOTHING_WINDOW: int = 10

# Maximum number of observation dimensions to plot in trajectory prediction.
# With D=45 (ANYmal D) or D=96 (G1), plotting all dims is impractical.
_MAX_PLOT_DIMS: int = 12

# Number of columns in the trajectory prediction subplot grid.
_TRAJ_PLOT_COLS: int = 3

# Default figure size (width, height) in inches for single-panel figures.
_DEFAULT_FIGSIZE: Tuple[float, float] = (8.0, 5.0)

# Figure size for two-panel figures (Fig. 5, Fig. S8).
_TWO_PANEL_FIGSIZE: Tuple[float, float] = (12.0, 5.0)

# Figure size per environment panel in the benchmark plot.
_BENCHMARK_PANEL_WIDTH: float = 5.0
_BENCHMARK_PANEL_HEIGHT: float = 4.0

# Colormap for RWM noise curves (yellow family, matching paper Fig. 3b).
_RWM_NOISE_CMAP: str = "YlOrBr"

# Colormap for MLP noise curves (grey family, matching paper Fig. 3b).
_MLP_NOISE_CMAP: str = "Greys"

# Colormap for heatmaps (darker = worse, matching paper Fig. S8).
_HEATMAP_CMAP: str = "YlOrRd"

# Minimum colormap value to avoid too-light colors in noise curves.
_CMAP_MIN: float = 0.35

# Maximum colormap value.
_CMAP_MAX: float = 0.95


class Visualizer:
    """Publication-quality figure generator for RWM evaluation results.

    Produces matplotlib figures replicating the paper's key results. All
    methods save figures to ``self.save_dir`` and close the figure to free
    memory. The class is stateless with respect to model/training state.

    Attributes:
        save_dir: Directory where all figures are saved. Created automatically
            if it does not exist. Corresponds to ``config.log_dir`` in
            ``config.yaml``.
        fig_dpi: DPI for saved figures. Default: 150.
        font_size: Base font size for all text elements. Default: 10.
        model_colors: Dict mapping model/method names to matplotlib color
            strings. Used for consistent coloring across all figures.
        rwm_color: Color for RWM curves in noise robustness plot (yellow,
            matching paper Fig. 3b "yellow curves"). Default: 'goldenrod'.
        mlp_color: Color for MLP curves in noise robustness plot (grey,
            matching paper Fig. 3b "grey curves"). Default: 'grey'.
    """

    def __init__(
        self,
        save_dir: str = "logs/figures",
        fig_dpi: int = _DEFAULT_DPI,
        font_size: int = _DEFAULT_FONT_SIZE,
    ) -> None:
        """Initialize the Visualizer and create the output directory.

        Sets up consistent matplotlib styling and color palettes for all
        figures. The color palette is designed to match the paper's figures
        as closely as possible.

        Args:
            save_dir: Directory path where all figures will be saved.
                Created automatically (including parent directories) if it
                does not exist. Corresponds to ``config.log_dir`` in
                ``config.yaml``. Default: "logs/figures".
            fig_dpi: Resolution (dots per inch) for saved figures. Higher
                values produce sharper images but larger files. Default: 150.
            font_size: Base font size for all text elements (axis labels,
                tick labels, legend text). Title font size is
                ``font_size + 2``. Default: 10.
        """
        # ----------------------------------------------------------------
        # 1. Store configuration
        # ----------------------------------------------------------------
        self.save_dir: str = str(save_dir)
        self.fig_dpi: int = int(fig_dpi)
        self.font_size: int = int(font_size)

        # ----------------------------------------------------------------
        # 2. Create output directory
        # ----------------------------------------------------------------
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # 3. Configure matplotlib style
        # ----------------------------------------------------------------
        # Use a clean academic style. Try seaborn-v0_8-whitegrid first
        # (available in matplotlib >= 3.6), fall back to ggplot.
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            try:
                plt.style.use("seaborn-whitegrid")
            except OSError:
                plt.style.use("ggplot")

        # Update global matplotlib font sizes for consistency.
        plt.rcParams.update({
            "font.size": self.font_size,
            "axes.titlesize": self.font_size + 2,
            "axes.labelsize": self.font_size,
            "xtick.labelsize": self.font_size - 1,
            "ytick.labelsize": self.font_size - 1,
            "legend.fontsize": self.font_size - 1,
            "figure.titlesize": self.font_size + 3,
        })

        # ----------------------------------------------------------------
        # 4. Define model/method color palette
        # ----------------------------------------------------------------
        # Colors chosen for visual distinctiveness and accessibility.
        # Consistent across all figures in the paper.
        self.model_colors: Dict[str, str] = {
            # World model architectures (Fig. 4)
            "RWM-AR": "#1f77b4",       # Strong blue — proposed method, most prominent
            "RWM-TF": "#aec7e8",       # Light blue — teacher-forcing ablation
            "MLP": "#7f7f7f",          # Grey — MLP baseline
            "RSSM": "#ff7f0e",         # Orange — RSSM baseline
            "Transformer": "#2ca02c",  # Green — Transformer baseline
            # Policy optimization methods (Fig. 5)
            "MBPO-PPO": "#1f77b4",     # Blue — proposed method
            "SHAC": "#d62728",         # Red — SHAC baseline
            "DreamerV3": "#9467bd",    # Purple — DreamerV3 baseline
            # Generic fallbacks
            "PPO": "#17becf",          # Cyan — model-free PPO (Table 1)
            "Ground Truth": "#2ca02c", # Green — ground truth reference
        }

        # Special colors for noise robustness plot (Fig. 3b)
        # Paper: "Yellow curves denote RWM ... Grey curves represent the MLP baseline"
        self.rwm_color: str = "goldenrod"   # Yellow, matching paper Fig. 3b
        self.mlp_color: str = "grey"        # Grey, matching paper Fig. 3b

        print(
            f"[Visualizer] Initialized. Figures will be saved to: {self.save_dir}"
        )

    # ----------------------------------------------------------------
    # Private helper methods
    # ----------------------------------------------------------------

    def _to_numpy(self, x: Any) -> np.ndarray:
        """Convert a tensor, list, or array to a numpy float64 array.

        Handles PyTorch tensors (detach + cpu + numpy), lists, and numpy
        arrays uniformly. This avoids repetitive isinstance checks in each
        plotting method.

        Args:
            x: Input data. May be:
                - ``torch.Tensor``: detached, moved to CPU, converted to numpy
                - ``np.ndarray``: returned as float64 array
                - ``list`` or ``tuple``: converted via ``np.array``
                - Any other type: wrapped in ``np.array``

        Returns:
            A numpy array of dtype float64. Shape is preserved from the input.
        """
        if TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64)
        if isinstance(x, np.ndarray):
            return x.astype(np.float64)
        return np.array(x, dtype=np.float64)

    def _smooth(
        self,
        data: np.ndarray,
        window: int = _SMOOTHING_WINDOW,
    ) -> np.ndarray:
        """Apply a moving average to smooth a 1D array.

        Uses ``np.convolve`` with ``mode='same'`` to preserve the array
        length for aligned plotting against the iteration axis. Edge effects
        are corrected by recomputing the average with the actual number of
        contributing values at each boundary position.

        Args:
            data: 1D numpy array to smooth. Must have at least 1 element.
            window: Moving average window size. Default: 10 (matching the
                design specification for training curve smoothing).

        Returns:
            Smoothed 1D numpy array of the same length as ``data``.
            Returns ``data`` unchanged if ``len(data) < window`` or
            ``window <= 1``.
        """
        if len(data) == 0:
            return data
        if window <= 1 or len(data) < window:
            return data.copy()

        # Uniform kernel for moving average
        kernel: np.ndarray = np.ones(window, dtype=np.float64) / window
        smoothed: np.ndarray = np.convolve(data, kernel, mode="same")

        # Correct edge effects from zero-padding in mode='same'.
        # At position i, the actual number of contributing values is:
        #   min(i + window//2 + 1, n) - max(0, i - window//2)
        n: int = len(data)
        half_w: int = window // 2

        for i in range(n):
            start_idx: int = max(0, i - half_w)
            end_idx: int = min(n, i + half_w + 1)
            actual_count: int = end_idx - start_idx
            if actual_count < window and actual_count > 0:
                smoothed[i] = data[start_idx:end_idx].mean()

        return smoothed

    def _save_figure(
        self,
        fig: plt.Figure,
        filename: str,
    ) -> None:
        """Save a figure to disk and close it to free memory.

        Centralizes the save + close pattern used by all plotting methods.
        Handles OSError gracefully with a warning rather than crashing.

        Args:
            fig: Matplotlib figure to save and close.
            filename: Filename (not full path) for the saved figure.
                The full path is ``os.path.join(self.save_dir, filename)``.
                Should include the file extension (e.g., '.png').
        """
        filepath: str = os.path.join(self.save_dir, filename)
        try:
            fig.savefig(
                filepath,
                dpi=self.fig_dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
        except OSError as exc:
            warnings.warn(
                f"[Visualizer] Failed to save figure to '{filepath}': {exc}. "
                "Check that the save directory exists and is writable.",
                UserWarning,
                stacklevel=3,
            )
        finally:
            plt.close(fig)

    def _get_color(self, name: str, default: str = "black") -> str:
        """Look up a model/method color from the palette with a fallback.

        Args:
            name: Model or method name to look up in ``self.model_colors``.
            default: Fallback color if ``name`` is not in the palette.
                Default: "black".

        Returns:
            A matplotlib color string.
        """
        return self.model_colors.get(name, default)

    # ----------------------------------------------------------------
    # Public plotting methods
    # ----------------------------------------------------------------

    def plot_trajectory_prediction(
        self,
        pred: Any,
        target: Any,
        labels: List[str],
        title: str = "Trajectory Prediction",
        start_t: int = 32,
    ) -> None:
        """Plot predicted vs ground truth trajectories over time (Fig. 3a).

        Replicates Fig. 3a from the paper: solid lines for ground truth,
        dashed lines for predictions, with a vertical marker at the timestep
        where autoregressive prediction begins (``start_t=32`` = M from
        ``config.rwm.history_horizon``).

        For high-dimensional observations (D > ``_MAX_PLOT_DIMS``=12), only
        the first ``_MAX_PLOT_DIMS`` dimensions are plotted to keep the figure
        readable. The paper's Fig. 3a shows a representative subset of state
        variables.

        Args:
            pred: Predicted observation trajectories of shape ``[T, D]``.
                T = total trajectory length (history + forecast),
                D = observation feature dimension (45 for ANYmal D, 96 for G1).
                May be a ``torch.Tensor`` or ``np.ndarray``.
            target: Ground truth observation trajectories of shape ``[T, D]``.
                Must have the same shape as ``pred``.
            labels: List of length D containing the name of each observation
                dimension (e.g., ``['base_lin_vel_x', 'base_lin_vel_y', ...]``).
                Used as subplot titles. If shorter than D, remaining dims
                are labeled ``'dim_{i}'``.
            title: Overall figure title. Default: "Trajectory Prediction".
            start_t: Timestep where autoregressive prediction begins.
                Corresponds to ``config.rwm.history_horizon = 32`` (Table S10).
                A vertical dashed line is drawn at this timestep. Default: 32.

        Saves:
            ``{save_dir}/trajectory_prediction.png``
        """
        # ----------------------------------------------------------------
        # 1. Convert to numpy
        # ----------------------------------------------------------------
        pred_np: np.ndarray = self._to_numpy(pred)    # [T, D]
        target_np: np.ndarray = self._to_numpy(target)  # [T, D]

        if pred_np.ndim != 2 or target_np.ndim != 2:
            warnings.warn(
                f"[Visualizer.plot_trajectory_prediction] Expected 2D arrays "
                f"[T, D], got pred.shape={pred_np.shape}, "
                f"target.shape={target_np.shape}. Skipping plot.",
                UserWarning,
                stacklevel=2,
            )
            return

        T: int = pred_np.shape[0]
        D: int = pred_np.shape[1]

        # ----------------------------------------------------------------
        # 2. Select dimensions to plot
        # ----------------------------------------------------------------
        n_plots: int = min(D, _MAX_PLOT_DIMS)

        # Pad labels if shorter than D
        padded_labels: List[str] = list(labels) + [
            f"dim_{i}" for i in range(len(labels), D)
        ]

        # ----------------------------------------------------------------
        # 3. Compute subplot grid dimensions
        # ----------------------------------------------------------------
        n_cols: int = _TRAJ_PLOT_COLS
        n_rows: int = math.ceil(n_plots / n_cols)

        fig_width: float = n_cols * 4.0
        fig_height: float = n_rows * 2.5

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )

        # Time axis: 0 to T-1
        time_axis: np.ndarray = np.arange(T)

        # ----------------------------------------------------------------
        # 4. Plot each dimension
        # ----------------------------------------------------------------
        for plot_idx in range(n_plots):
            row: int = plot_idx // n_cols
            col: int = plot_idx % n_cols
            ax: plt.Axes = axes[row, col]

            dim_label: str = padded_labels[plot_idx] if plot_idx < len(padded_labels) else f"dim_{plot_idx}"

            # Ground truth: solid line
            ax.plot(
                time_axis,
                target_np[:, plot_idx],
                color="#2ca02c",
                linewidth=1.5,
                linestyle="-",
                label="Ground Truth" if plot_idx == 0 else "_nolegend_",
                alpha=0.9,
            )

            # Prediction: dashed line
            ax.plot(
                time_axis,
                pred_np[:, plot_idx],
                color="#1f77b4",
                linewidth=1.5,
                linestyle="--",
                label="Predicted" if plot_idx == 0 else "_nolegend_",
                alpha=0.9,
            )

            # Vertical line at start_t (where prediction begins)
            ax.axvline(
                x=start_t,
                color="black",
                linestyle=":",
                linewidth=0.8,
                alpha=0.6,
                label=f"Prediction start (t={start_t})" if plot_idx == 0 else "_nolegend_",
            )

            # Subplot formatting
            ax.set_title(dim_label, fontsize=self.font_size - 1, pad=3)

            # X-axis label only on bottom row
            if row == n_rows - 1:
                ax.set_xlabel("Timestep", fontsize=self.font_size - 1)

            # Y-axis label only on leftmost column
            if col == 0:
                ax.set_ylabel("Value", fontsize=self.font_size - 1)

            ax.tick_params(labelsize=self.font_size - 2)

        # ----------------------------------------------------------------
        # 5. Hide unused subplots
        # ----------------------------------------------------------------
        for plot_idx in range(n_plots, n_rows * n_cols):
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            axes[row, col].set_visible(False)

        # ----------------------------------------------------------------
        # 6. Add shared legend at the top
        # ----------------------------------------------------------------
        # Create legend handles manually for the first subplot
        legend_handles = [
            plt.Line2D([0], [0], color="#2ca02c", linewidth=1.5, linestyle="-", label="Ground Truth"),
            plt.Line2D([0], [0], color="#1f77b4", linewidth=1.5, linestyle="--", label="Predicted"),
            plt.Line2D([0], [0], color="black", linewidth=0.8, linestyle=":", label=f"Prediction start (t={start_t})"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=3,
            fontsize=self.font_size - 1,
            bbox_to_anchor=(0.5, 1.02),
            frameon=True,
        )

        # ----------------------------------------------------------------
        # 7. Overall title and layout
        # ----------------------------------------------------------------
        fig.suptitle(title, fontsize=self.font_size + 3, y=1.05)
        plt.tight_layout()

        self._save_figure(fig, "trajectory_prediction.png")

    def plot_prediction_error_curves(
        self,
        errors: Dict[str, Any],
        title: str = "Autoregressive Prediction Error",
    ) -> None:
        """Plot relative prediction error curves per model (Fig. 4 single panel).

        Replicates the per-environment panel from Fig. 4: one curve per model,
        x-axis = forecast steps, y-axis = relative prediction error ``e``.
        RWM-AR is plotted with a thicker line to emphasize it as the proposed
        method.

        Args:
            errors: Dict mapping model name (str) to a 1D error curve.
                Each value is a ``Tensor[T]`` or ``np.ndarray[T]`` where T
                is the number of forecast steps. Keys should include model
                names from the paper: ``'RWM-AR'``, ``'RWM-TF'``, ``'MLP'``,
                ``'RSSM'``, ``'Transformer'``.
            title: Figure title. Default: "Autoregressive Prediction Error".

        Saves:
            ``{save_dir}/prediction_errors.png``
        """
        if not errors:
            warnings.warn(
                "[Visualizer.plot_prediction_error_curves] errors dict is empty. "
                "Skipping plot.",
                UserWarning,
                stacklevel=2,
            )
            return

        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

        for model_name, error_curve in errors.items():
            curve_np: np.ndarray = self._to_numpy(error_curve)

            if curve_np.ndim != 1 or len(curve_np) == 0:
                warnings.warn(
                    f"[Visualizer.plot_prediction_error_curves] Error curve for "
                    f"'{model_name}' is not 1D or is empty. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            x_vals: np.ndarray = np.arange(1, len(curve_np) + 1)
            color: str = self._get_color(model_name)

            # RWM-AR gets a thicker, more prominent line
            linewidth: float = 2.5 if model_name == "RWM-AR" else 1.5
            linestyle: str = "--" if model_name == "RWM-TF" else "-"
            zorder: int = 10 if model_name == "RWM-AR" else 5

            ax.plot(
                x_vals,
                curve_np,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                label=model_name,
                zorder=zorder,
                alpha=0.9,
            )

        ax.set_xlabel("Forecast Steps", fontsize=self.font_size)
        ax.set_ylabel("Relative Prediction Error e", fontsize=self.font_size)
        ax.set_title(title, fontsize=self.font_size + 2)
        ax.legend(loc="upper left", fontsize=self.font_size - 1, frameon=True)
        ax.tick_params(labelsize=self.font_size - 1)

        # Start y-axis at 0 for clarity
        ax.set_ylim(bottom=0.0)

        plt.tight_layout()
        self._save_figure(fig, "prediction_errors.png")

    def plot_noise_robustness(
        self,
        rwm_errors: Dict[float, Any],
        mlp_errors: Dict[float, Any],
        noise_levels: List[float],
    ) -> None:
        """Plot noise robustness comparison: RWM (yellow) vs MLP (grey) (Fig. 3b).

        Replicates Fig. 3b from the paper. Each noise level produces one curve
        for RWM (yellow family) and one for MLP (grey family). Higher noise
        levels use darker shades within each family to show the degradation.

        Paper caption: "Yellow curves denote RWM at varying noise levels,
        demonstrating consistent robustness and lower error accumulation across
        forecast steps. Grey curves represent the MLP baseline, which exhibits
        significantly higher error accumulation and reduced robustness to noise."

        Args:
            rwm_errors: Dict mapping noise level (float) to RWM error curve
                (``Tensor[T]`` or ``np.ndarray[T]``). Keys must be a subset
                of ``noise_levels``. From ``Metrics.compute_noise_robustness``.
            mlp_errors: Dict mapping noise level (float) to MLP error curve.
                Same structure as ``rwm_errors``.
            noise_levels: List of noise standard deviations tested. From
                ``config.noise_robustness.noise_levels: [0.01, 0.05, 0.1, 0.2]``.
                Determines the number of curves and the colormap normalization.

        Saves:
            ``{save_dir}/noise_robustness.png``
        """
        if not noise_levels:
            warnings.warn(
                "[Visualizer.plot_noise_robustness] noise_levels is empty. "
                "Skipping plot.",
                UserWarning,
                stacklevel=2,
            )
            return

        fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

        # ----------------------------------------------------------------
        # Build colormaps for RWM (yellow) and MLP (grey) families.
        # Normalize noise levels to [_CMAP_MIN, _CMAP_MAX] for the colormap
        # range to avoid too-light colors at low noise levels.
        # ----------------------------------------------------------------
        n_levels: int = len(noise_levels)

        # Colormap instances
        rwm_cmap = cm.get_cmap(_RWM_NOISE_CMAP)
        mlp_cmap = cm.get_cmap(_MLP_NOISE_CMAP)

        # Map noise level index to colormap value in [_CMAP_MIN, _CMAP_MAX]
        if n_levels == 1:
            cmap_values: List[float] = [(_CMAP_MIN + _CMAP_MAX) / 2.0]
        else:
            cmap_values = [
                _CMAP_MIN + (_CMAP_MAX - _CMAP_MIN) * i / (n_levels - 1)
                for i in range(n_levels)
            ]

        # ----------------------------------------------------------------
        # Plot RWM curves (yellow family)
        # ----------------------------------------------------------------
        for level_idx, noise_level in enumerate(noise_levels):
            if noise_level not in rwm_errors:
                continue

            curve_np: np.ndarray = self._to_numpy(rwm_errors[noise_level])
            if curve_np.ndim != 1 or len(curve_np) == 0:
                continue

            x_vals: np.ndarray = np.arange(1, len(curve_np) + 1)
            color_val: float = cmap_values[level_idx]
            rwm_color_rgba = rwm_cmap(color_val)

            ax.plot(
                x_vals,
                curve_np,
                color=rwm_color_rgba,
                linewidth=1.8,
                linestyle="-",
                alpha=0.85,
                # Only label first and last to avoid legend clutter
                label=f"RWM (σ={noise_level})" if level_idx in (0, n_levels - 1) else "_nolegend_",
            )

        # ----------------------------------------------------------------
        # Plot MLP curves (grey family)
        # ----------------------------------------------------------------
        for level_idx, noise_level in enumerate(noise_levels):
            if noise_level not in mlp_errors:
                continue

            curve_np = self._to_numpy(mlp_errors[noise_level])
            if curve_np.ndim != 1 or len(curve_np) == 0:
                continue

            x_vals = np.arange(1, len(curve_np) + 1)
            color_val = cmap_values[level_idx]
            mlp_color_rgba = mlp_cmap(color_val)

            ax.plot(
                x_vals,
                curve_np,
                color=mlp_color_rgba,
                linewidth=1.8,
                linestyle="-",
                alpha=0.85,
                label=f"MLP (σ={noise_level})" if level_idx in (0, n_levels - 1) else "_nolegend_",
            )

        # ----------------------------------------------------------------
        # Add a clean legend with two family entries + noise level annotation
        # ----------------------------------------------------------------
        # Create custom legend patches for the two model families
        rwm_patch = mpatches.Patch(
            color=rwm_cmap((_CMAP_MIN + _CMAP_MAX) / 2.0),
            label="RWM (yellow family)",
        )
        mlp_patch = mpatches.Patch(
            color=mlp_cmap((_CMAP_MIN + _CMAP_MAX) / 2.0),
            label="MLP (grey family)",
        )

        # Add noise level range annotation
        noise_str: str = f"σ ∈ {{{', '.join(str(n) for n in noise_levels)}}}"
        ax.text(
            0.98, 0.02,
            noise_str,
            transform=ax.transAxes,
            fontsize=self.font_size - 2,
            ha="right",
            va="bottom",
            color="dimgrey",
            style="italic",
        )

        ax.legend(
            handles