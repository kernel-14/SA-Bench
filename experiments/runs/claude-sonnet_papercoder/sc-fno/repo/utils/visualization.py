```python
## utils/visualization.py
"""Visualization utilities for SC-FNO experiments.

Provides a Visualizer class that produces publication-quality figures
reproducing the key visuals from the SC-FNO paper:
  - Figure 3: Solution and sensitivity comparisons (ODE/PDE)
  - Figures 1, 2: Parameter inversion scatter plots
  - Figure 5: R² vs perturbation ratio λ
  - Figure 4: R² vs training set size

All methods save PNG files to save_dir and never call plt.show().
"""

import os
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend — no display required.

# ---------------------------------------------------------------------------
# Module-level style constants (consistent across all figures in the paper)
# ---------------------------------------------------------------------------

# Color and line style conventions matching the paper's figures.
MODEL_COLORS: Dict[str, str] = {
    "fno": "#1f77b4",         # blue
    "sc_fno": "#ff7f0e",      # orange
    "fno_pinn": "#1f77b4",    # blue (dashed)
    "sc_fno_pinn": "#ff7f0e", # orange (dashed)
}

MODEL_LINESTYLES: Dict[str, str] = {
    "fno": "-",
    "sc_fno": "-",
    "fno_pinn": "--",
    "sc_fno_pinn": "--",
}

MODEL_MARKERS: Dict[str, str] = {
    "fno": "o",
    "sc_fno": "s",
    "fno_pinn": "^",
    "sc_fno_pinn": "D",
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "fno": "FNO",
    "sc_fno": "SC-FNO",
    "fno_pinn": "FNO-PINN",
    "sc_fno_pinn": "SC-FNO-PINN",
}

# Default DPI for all saved figures.
_DEFAULT_DPI: int = 150


class Visualizer:
    """Produces and saves all figures for the SC-FNO paper reproduction.

    All plotting methods are stateless beyond self.save_dir. They accept
    pre-computed tensors or numpy arrays, convert them internally, and
    write PNG files to save_dir. plt.show() is never called.

    Attributes:
        save_dir: Directory where all PNG figures are written.

    Example:
        >>> viz = Visualizer("outputs/figures")
        >>> viz.plot_solution_comparison(u_pred, u_true, "PDE1 Test", "pde1")
        >>> viz.plot_inversion_scatter(p_pred, p_true, ["alpha", "beta"],
        ...                            ["FNO", "SC-FNO"], "PDE1 Inversion")
    """

    def __init__(self, save_dir: str = "outputs/figures") -> None:
        """Initializes the Visualizer and creates the output directory.

        Args:
            save_dir: Path to the directory where figures will be saved.
                      Created (including parents) if it does not exist.
                      Sourced from config.yaml key 'figures_dir'.
        """
        self.save_dir: str = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_numpy(self, x: Any) -> np.ndarray:
        """Converts a torch.Tensor or numpy array to a numpy ndarray.

        Handles:
          - torch.Tensor: calls .detach().cpu().numpy()
          - numpy.ndarray: returned as-is
          - Python list / tuple: converted via np.array()
          - Scalar (int, float): wrapped in np.array()

        Args:
            x: Input array-like object.

        Returns:
            A numpy ndarray with the same data.
        """
        # Check for torch.Tensor without a hard import at module level.
        type_name = type(x).__name__
        module_name = getattr(type(x), "__module__", "")

        if module_name == "torch" and type_name == "Tensor":
            return x.detach().cpu().numpy()  # type: ignore[union-attr]

        if isinstance(x, np.ndarray):
            return x

        # Lists, tuples, scalars.
        return np.array(x)

    def _safe_filename(self, title: str) -> str:
        """Converts a human-readable title to a safe filename stem.

        Replaces spaces with underscores, lowercases, and strips characters
        that are unsafe in filenames.

        Args:
            title: Human-readable title string.

        Returns:
            A filesystem-safe filename stem (no extension).
        """
        stem = title.strip().lower().replace(" ", "_")
        # Keep only alphanumeric characters, underscores, and hyphens.
        safe = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in stem)
        # Collapse consecutive underscores.
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_") or "figure"

    def _compute_r2(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Computes the R² (coefficient of determination) score.

        Returns NaN if y_true has zero variance (degenerate case) to avoid
        division by zero rather than crashing.

        Args:
            y_pred: Predicted values, any shape (flattened internally).
            y_true: True values, same shape as y_pred.

        Returns:
            R² score as a Python float, or float('nan') if degenerate.
        """
        y_pred_flat = y_pred.flatten().astype(np.float64)
        y_true_flat = y_true.flatten().astype(np.float64)

        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)

        if ss_tot < 1e-12:
            return float("nan")

        return float(1.0 - ss_res / ss_tot)

    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """Saves a figure to save_dir and closes it to free memory.

        Args:
            fig: The matplotlib Figure object to save.
            filename: Full filename including extension, e.g. "pde1_solution.png".
        """
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=_DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Public plotting methods
    # ------------------------------------------------------------------

    def plot_solution_comparison(
        self,
        u_pred: Any,
        u_true: Any,
        title: str,
        equation_type: str = "pde1",
        coords: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Plots predicted vs true solution paths for ODEs and PDEs.

        Reproduces Figure 3 from the paper. Handles three cases:
          - ODE (1D time): overlaid line plots
          - 1D PDE (space × time): side-by-side heatmaps
          - 2D PDE / PDE3 (spatial only): side-by-side heatmaps

        Args:
            u_pred: Predicted solution. Shape [T] for ODEs, [T, Sx] for 1D
                    PDEs, or [Sx, Sy] for PDE3.
            u_true: True solution. Same shape as u_pred.
            title: Human-readable title used for the figure title and filename.
            equation_type: One of 'ode1', 'ode2', 'pde1', 'pde2', 'pde3',
                           'pde4'. Determines the plot layout.
            coords: Optional dict with keys 't', 'x', 'y' containing 1D
                    coordinate arrays for axis tick labels. If None, integer
                    indices are used.
        """
        u_pred_np = self._to_numpy(u_pred)
        u_true_np = self._to_numpy(u_true)

        is_ode = equation_type in ("ode1", "ode2")
        is_2d_spatial = equation_type == "pde3"

        if is_ode:
            self._plot_solution_ode(u_pred_np, u_true_np, title, coords)
        elif is_2d_spatial:
            self._plot_solution_2d_spatial(u_pred_np, u_true_np, title, coords)
        else:
            # 1D PDE: shape [T, Sx]
            self._plot_solution_1d_pde(u_pred_np, u_true_np, title, coords)

    def _plot_solution_ode(
        self,
        u_pred: np.ndarray,
        u_true: np.ndarray,
        title: str,
        coords: Optional[Dict[str, Any]],
    ) -> None:
        """Plots ODE solution as overlaid line plots (Figure 3a-b style).

        Args:
            u_pred: Predicted solution, shape [T].
            u_true: True solution, shape [T].
            title: Figure title and filename stem.
            coords: Optional dict with key 't' for time axis values.
        """
        t_vals: np.ndarray
        if coords is not None and "t" in coords:
            t_vals = self._to_numpy(coords["t"])
        else:
            t_vals = np.arange(len(u_true))

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(t_vals, u_true.flatten(), color="#1f77b4", linewidth=2.0,
                label="True", zorder=3)
        ax.plot(t_vals, u_pred.flatten(), color="#ff7f0e", linewidth=1.5,
                linestyle="--", label="Predicted", zorder=4)

        ax.set_xlabel("Time $t$", fontsize=12)
        ax.set_ylabel("$u(t)$", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{self._safe_filename(title)}_solution.png"
        self._save_figure(fig, filename)

    def _plot_solution_1d_pde(
        self,
        u_pred: np.ndarray,
        u_true: np.ndarray,
        title: str,
        coords: Optional[Dict[str, Any]],
    ) -> None:
        """Plots 1D PDE solution as side-by-side heatmaps (Figure 3c-d style).

        Args:
            u_pred: Predicted solution, shape [T, Sx].
            u_true: True solution, shape [T, Sx].
            title: Figure title and filename stem.
            coords: Optional dict with keys 't' and 'x' for axis labels.
        """
        # Ensure 2D shape.
        if u_pred.ndim == 1:
            u_pred = u_pred.reshape(1, -1)
        if u_true.ndim == 1:
            u_true = u_true.reshape(1, -1)

        vmin = float(u_true.min())
        vmax = float(u_true.max())

        # Determine axis extents for pcolormesh.
        x_vals: np.ndarray
        t_vals: np.ndarray
        if coords is not None and "x" in coords:
            x_vals = self._to_numpy(coords["x"])
        else:
            x_vals = np.arange(u_true.shape[1])
        if coords is not None and "t" in coords:
            t_vals = self._to_numpy(coords["t"])
        else:
            t_vals = np.arange(u_true.shape[0])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for ax, data, subtitle in zip(axes, [u_true, u_pred], ["True", "Predicted"]):
            im = ax.pcolormesh(
                x_vals, t_vals, data,
                cmap="viridis", vmin=vmin, vmax=vmax, shading="auto"
            )
            ax.set_xlabel("$x$", fontsize=12)
            ax.set_title(subtitle, fontsize=12)

        axes[0].set_ylabel("$t$", fontsize=12)
        fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04,
                     label="$u(x,t)$")
        fig.suptitle(title, fontsize=13, y=1.02)
        plt.tight_layout()

        filename = f"{self._safe_filename(title)}_solution.png"
        self._save_figure(fig, filename)

    def _plot_solution_2d_spatial(
        self,
        u_pred: np.ndarray,
        u_true: np.ndarray,
        title: str,
        coords: Optional[Dict[str, Any]],
    ) -> None:
        """Plots 2D spatial PDE solution as side-by-side heatmaps (Figure 6 style).

        Args:
            u_pred: Predicted vorticity, shape [Sx, Sy].
            u_true: True vorticity, shape [Sx, Sy].
            title: Figure title and filename stem.
            coords: Optional dict with keys 'x' and 'y' for axis labels.
        """
        vmin = float(u_true.min())
        vmax = float(u_true.max())

        x_vals: np.ndarray
        y_vals: np.ndarray
        if coords is not None and "x" in coords:
            x_vals = self._to_numpy(coords["x"])
        else:
            x_vals = np.arange(u_true.shape[1])
        if coords is not None and "y" in coords:
            y_vals = self._to_numpy(coords["y"])
        else:
            y_vals = np.arange(u_true.shape[0])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, data, subtitle in zip(axes, [u_true, u_pred], ["True", "Predicted"]):
            im = ax.pcolormesh(
                x_vals, y_vals, data,
                cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto"
            )
            ax.set_xlabel("$x$", fontsize=12)
            ax.set_ylabel("$y$", fontsize=12)
            ax.set_title(subtitle, fontsize=12)
            ax.set_aspect("equal")

        fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04,
                     label="$\\omega(x,y)$")
        fig.suptitle(title, fontsize=13, y=1.02)
        plt.tight_layout()

        filename = f"{self._safe_filename(title)}_solution.png"
        self._save_figure(fig, filename)

    def plot_sensitivity_comparison(
        self,
        j_pred: Any,
        j_true: Any,
        param_name: str,
        equation_type: str = "pde1",
        title: str = "",
    ) -> None:
        """Plots predicted vs true Jacobian fields ∂u/∂p.

        Reproduces the sensitivity panels in Figure 3. Uses a symmetric
        diverging colormap centered at zero so that unphysical oscillations
        in FNO-predicted sensitivities are visually obvious.

        Args:
            j_pred: Predicted Jacobian for one parameter. Shape [T] for ODEs
                    or [T, Sx] for 1D PDEs.
            j_true: True Jacobian. Same shape as j_pred.
            param_name: Parameter name for axis labels, e.g. 'alpha'.
            equation_type: One of 'ode1', 'ode2', 'pde1', 'pde2', 'pde3',
                           'pde4'. Determines the plot layout.
            title: Optional prefix for the figure title and filename.
        """
        j_pred_np = self._to_numpy(j_pred)
        j_true_np = self._to_numpy(j_true)

        is_ode = equation_type in ("ode1", "ode2")
        full_title = f"{title} ∂u/∂{param_name}" if title else f"∂u/∂{param_name}"

        if is_ode:
            self._plot_sensitivity_ode(j_pred_np, j_true_np, param_name, full_title)
        else:
            self._plot_sensitivity_1d_pde(j_pred_np, j_true_np, param_name, full_title)

    def _plot_sensitivity_ode(
        self,
        j_pred: np.ndarray,
        j_true: np.ndarray,
        param_name: str,
        title: str,
    ) -> None:
        """Plots ODE sensitivity as overlaid line plots.

        Args:
            j_pred: Predicted Jacobian, shape [T].
            j_true: True Jacobian, shape [T].
            param_name: Parameter name for labels.
            title: Figure title and filename stem.
        """
        t_vals = np.arange(len(j_true.flatten()))

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(t_vals, j_true.flatten(), color="#1f77b4", linewidth=2.0,
                label=f"True $\\partial u/\\partial {param_name}$", zorder=3)
        ax.plot(t_vals, j_pred.flatten(), color="#ff7f0e", linewidth=1.5,
                linestyle="--",
                label=f"Predicted $\\partial u/\\partial {param_name}$", zorder=4)

        ax.set_xlabel("Time $t$", fontsize=12)
        ax.set_ylabel(f"$\\partial u/\\partial {param_name}$", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{self._safe_filename(title)}_{param_name}_sensitivity.png"
        self._save_figure(fig, filename)

    def _plot_sensitivity_1d_pde(
        self,
        j_pred: np.ndarray,
        j_true: np.ndarray,
        param_name: str,
        title: str,
    ) -> None:
        """Plots 1D PDE sensitivity as side-by-side heatmaps with symmetric colormap.

        Args:
            j_pred: Predicted Jacobian, shape [T, Sx].
            j_true: True Jacobian, shape [T, Sx].
            param_name: Parameter name for labels.
            title: Figure title and filename stem.
        """
        # Ensure 2D.
        if j_pred.ndim == 1:
            j_pred = j_pred.reshape(1, -1)
        if j_true.ndim == 1:
            j_true = j_true.reshape(1, -1)

        # Symmetric color range centered at zero — makes oscillations obvious.
        abs_max = float(np.abs(j_true).max())
        if abs_max < 1e-12:
            abs_max = 1.0  # Guard against all-zero degenerate case.
        vmin, vmax = -abs_max, abs_max

        t_vals = np.arange(j_true.shape[0])
        x_vals = np.arange(j_true.shape[1])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for ax, data, subtitle in zip(
            axes,
            [j_true, j_pred],
            [f"True $\\partial u/\\partial {param_name}$",
             f"Predicted $\\partial u/\\partial {param_name}$"],
        ):
            im = ax.pcolormesh(
                x_vals, t_vals, data,
                cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto"
            )
            ax.set_xlabel("$x$", fontsize=12)
            ax.set_title(subtitle, fontsize=11)

        axes[0].set_ylabel("$t$", fontsize=12)
        fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04,
                     label=f"$\\partial u/\\partial {param_name}$")
        fig.suptitle(title, fontsize=13, y=1.02)
        plt.tight_layout()

        filename = f"{self._safe_filename(title)}_{param_name}_sensitivity.png"
        self._save_figure(fig, filename)

    def plot_inversion_scatter(
        self,
        p_pred: Any,
        p_true: Any,
        param_names: List[str],
        model_names: List[str],
        title: str = "",
    ) -> None:
        """Plots scatter plots of recovered vs true parameter values.

        Reproduces Figures 1 and 2 from the paper. Creates a grid of subplots
        with rows = parameters and columns = models. Each subplot shows a
        scatter of predicted vs true values with the identity line and R²
        annotation.

        Args:
            p_pred: Predicted parameters, shape [n_models, N_test, n_params].
            p_true: True parameters, shape [N_test, n_params].
            param_names: List of parameter names, length n_params.
                         E.g. ['alpha', 'beta', 'gamma', 'omega'].
            model_names: List of model identifiers, length n_models.
                         E.g. ['fno', 'sc_fno', 'fno_pinn'].
            title: Optional prefix for the figure title and filename.
        """
        p_pred_np = self._to_numpy(p_pred)  # [n_models, N_test, n_params]
        p_true_np = self._to_numpy(p_true)  # [N_test, n_params]

        # Handle the case where p_pred has shape [N_test, n_params] (single model).
        if p_pred_np.ndim == 2:
            p_pred_np = p_pred_np[np.newaxis, :, :]  # → [1, N_test, n_params]

        n_models: int = p_pred_np.shape[0]
        n_params: int = p_true_np.shape[1] if p_true_np.ndim == 2 else 1

        # Ensure p_true is 2D.
        if p_true_np.ndim == 1:
            p_true_np = p_true_np[:, np.newaxis]

        # Clamp param_names and model_names to actual dimensions.
        param_names = list(param_names)[:n_params]
        model_names = list(model_names)[:n_models]

        # Compute global axis limits per parameter for visual consistency.
        # All models for the same parameter share the same xlim/ylim.
        param_limits: List[tuple] = []
        for param_idx in range(n_params):
            all_vals = np.concatenate(
                [p_true_np[:, param_idx]]
                + [p_pred_np[m, :, param_idx] for m in range(n_models)]
            )
            lo = float(np.nanmin(all_vals))
            hi = float(np.nanmax(all_vals))
            margin = (hi - lo) * 0.05 if (hi - lo) > 1e-12 else 0.1
            param_limits.append((lo - margin, hi + margin))

        fig, axes = plt.subplots(
            n_params, n_models,
            figsize=(4 * n_models, 4 * n_params),
            squeeze=False,
        )

        for param_idx, param_name in enumerate(param_names):
            lo, hi = param_limits[param_idx]
            for model_idx, model_name in enumerate(model_names):
                ax = axes[param_idx][model_idx]

                y_pred_vals = p_pred_np[model_idx, :, param_idx]
                y_true_vals = p_true_np[:, param_idx]

                # Scatter plot.
                color = MODEL_COLORS.get(model_name.lower(), "#333333")
                ax.scatter(
                    y_true_vals, y_pred_vals,
                    s=8, alpha=0.5, color=color, linewidths=0,
                )

                # Identity line y = x.
                ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--",
                        linewidth=1.0, zorder=5, label="$y = x$")

                # R² annotation.
                r2 = self._compute_r2(y_pred_vals, y_true_vals)
                if not np.isnan(r2):
                    ax.text(
                        0.05, 0.92,
                        f"$R^2 = {r2:.3f}$",
                        transform=ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  alpha=0.7, edgecolor="none"),
                    )

                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel(f"True ${param_name}$", fontsize=10)
                ax.set_ylabel(f"Predicted ${param_name}$", fontsize=10)

                display_name = MODEL_DISPLAY_NAMES.get(model_name.lower(), model_name)
                ax.set_title(display_name, fontsize=11)
                ax.grid(True, alpha=0.3)

        full_title = title if title else "Parameter Inversion"
        fig.suptitle(full_title, fontsize=14, y=1.01)
        plt.tight_layout()

        filename = f"{self._safe_filename(full_title)}_inversion_scatter.png"
        self._save_figure(fig, filename)

    def plot_r2_vs_perturbation(
        self,
        lambda_vals: List[float],
        r2_dict: Dict[str, List[float]],
        metric: str = "u",
        title: str = "",
    ) -> None:
        """Plots R² as a function of perturbation ratio λ for each model variant.

        Reproduces Figure 5 from the paper. FNO R² drops steeply as λ
        increases while SC-FNO remains relatively stable.

        Args:
            lambda_vals: List of perturbation ratios, e.g. [0.1, 0.2, 0.3, 0.4].
                         Sourced from config.yaml 'perturbation.lambda_values'.
            r2_dict: Dict mapping model name (str) → list of R² values, one
                     per λ value. E.g. {'fno': [0.95, 0.80, 0.65, 0.53],
                                        'sc_fno': [0.98, 0.96, 0.93, 0.91]}.
            metric: Label for the metric being plotted, e.g. 'u' or 'du/dalpha'.
                    Used in the y-axis label and figure title.
            title: Optional prefix for the figure title and filename.
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        # Add a vertical dashed line at λ=0 to mark the training boundary.
        ax.axvline(x=0.0, color="gray", linestyle=":", linewidth=1.0,
                   label="Training range", zorder=1)

        # Add a horizontal reference line at R²=0.
        ax.axhline(y=0.0, color="lightgray", linestyle="-", linewidth=0.8, zorder=1)

        for model_name, r2_values in r2_dict.items():
            color = MODEL_COLORS.get(model_name.lower(), "#333333")
            linestyle = MODEL_LINESTYLES.get(model_name.lower(), "-")
            marker = MODEL_MARKERS.get(model_name.lower(), "o")
            display_name = MODEL_DISPLAY_NAMES.get(model_name.lower(), model_name)

            # Pad or trim r2_values to match lambda_vals length.
            r2_arr = list(r2_values)[: len(lambda_vals)]
            while len(r2_arr) < len(lambda_vals):
                r2_arr.append(float("nan"))

            ax.plot(
                lambda_vals, r2_arr,
                color=color, linestyle=linestyle, marker=marker,
                linewidth=2.0, markersize=7, label=display_name, zorder=3,
            )

        ax.set_xlabel("Perturbation ratio $\\lambda$", fontsize=13)
        ax.set_ylabel(f"$R^2$ ({metric})", fontsize=13)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(left=min(lambda_vals) - 0.02 if lambda_vals else 0.0)

        full_title = (
            f"{title} — $R^2$ vs Perturbation" if title
            else f"$R^2$ vs Perturbation ({metric})"
        )
        ax.set_title(full_title, fontsize=13)
        ax.legend(fontsize=11, loc="lower left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{self._safe_filename(full_title)}_r2_vs_perturbation.png"
        self._save_figure(fig, filename)

    def plot_r2_vs_samples(
        self,
        sample_sizes: List[int],
        r2_dict: Dict[str, List[float]],
        metric: str = "u",
        title: str = "",
    ) -> None:
        """Plots R² vs training set size for each model variant.

        Reproduces Figure 4 from the paper. Uses a log scale on the x-axis
        because sample sizes span two orders of magnitude (100 to 2000).
        Sourced from config.yaml 'data_scaling.sample_sizes'.

        Args:
            sample_sizes: List of training set sizes, e.g. [100, 200, 500, 1000, 2000].
                          Sourced from config.yaml 'data_scaling.sample_sizes'.
            r2_dict: Dict mapping model name (str) → list of R² values, one
                     per sample size. E.g. {'fno': [0.60, 0.72, 0