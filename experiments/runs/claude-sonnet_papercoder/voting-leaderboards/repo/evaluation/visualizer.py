```python
## evaluation/visualizer.py
"""Visualization module for the adversarial leaderboard manipulation paper.

This module provides the Visualizer class, which renders all publication-quality
figures and CSV tables that reproduce the paper's Figures 2, 3, 4, 5, 6 and
all result tables.

Figures produced:
  - Figure 2: PCA scatter plots of BoW features for 3 specific prompts.
  - Figure 3: Heatmap of BoW-based detection accuracy across categories × models.
  - Figure 4: Detection rate vs. votes for naive vs. informed adversary (Scenario 1).
  - Figure 5: Detection rate vs. votes for different noise scales (Scenario 2).
  - Figure 6: Utility loss (avg rank change) vs. noise scale.

Tables produced (CSV):
  - Table 2 / Table 7: Identity-probing detector accuracy.
  - Table 3: Feature comparison on English prompts.
  - Table 4(a/b): High-ranked model simulation results (votes / interactions).
  - Table 5(a/b): Low-ranked model simulation results (votes / interactions).
  - Table 8(a/b): Detector accuracy ablation.
  - Table 9: Non-target strategy ablation.

Design constraints:
  - No direct dependency on Metrics — receives pre-computed DataFrames.
  - Uses 'Agg' matplotlib backend for headless server execution.
  - All figures saved at dpi=150 with bbox_inches='tight'.
  - plt.close(fig) called after every save to prevent memory leaks.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
# Set non-interactive backend before importing pyplot to support headless servers.
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Figure size constants (width, height in inches)
# ---------------------------------------------------------------------------
_FIG_SIZE_PCA: Tuple[int, int] = (18, 5)
_FIG_SIZE_HEATMAP: Tuple[int, int] = (20, 6)
_FIG_SIZE_LINE: Tuple[int, int] = (8, 5)
_FIG_SIZE_UTILITY: Tuple[int, int] = (7, 5)

# ---------------------------------------------------------------------------
# DPI for saved figures
# ---------------------------------------------------------------------------
_SAVE_DPI: int = 150

# ---------------------------------------------------------------------------
# Heatmap color scale bounds (paper Figure 3: "scale: 85% to 100%")
# ---------------------------------------------------------------------------
_HEATMAP_VMIN: float = 85.0
_HEATMAP_VMAX: float = 100.0

# ---------------------------------------------------------------------------
# Canonical category ordering for Figure 3 rows (matches Table 1 / paper order)
# ---------------------------------------------------------------------------
_CATEGORY_ORDER: List[str] = [
    "english",
    "chinese",
    "spanish",
    "indonesian",
    "persian",
    "coding",
    "math",
    "safety",
]

# ---------------------------------------------------------------------------
# Significance level reference line for Figures 4 and 5
# (config.yaml: mitigations.malicious_user_detection.significance_level = 0.01)
# ---------------------------------------------------------------------------
_SIGNIFICANCE_LEVEL: float = 0.01

# ---------------------------------------------------------------------------
# Line styles for adversary types in Figure 4
# ---------------------------------------------------------------------------
_ADVERSARY_LINE_STYLES: Dict[str, str] = {
    "naive": "-",
    "informed": "--",
}

# ---------------------------------------------------------------------------
# Default marker for line plots
# ---------------------------------------------------------------------------
_DEFAULT_MARKER: str = "o"
_SECONDARY_MARKER: str = "s"

# ---------------------------------------------------------------------------
# Number of distinct colors needed (22 models from config.yaml)
# ---------------------------------------------------------------------------
_N_MODEL_COLORS: int = 22


class Visualizer:
    """Renders publication-quality figures and CSV tables for the paper's experiments.

    Provides one plot method per figure in the paper (Figures 2–6) and utility
    methods for saving DataFrames as CSV files. All methods are self-contained:
    they receive pre-computed data, render the figure, save it to disk, and
    close the matplotlib figure to free memory.

    Attributes:
        output_dir: Root directory for all output files. Subdirectories
            'figures/' and 'tables/' are created automatically.
        figures_dir: Path to the figures subdirectory.
        tables_dir: Path to the tables subdirectory.
        model_color_map: Dict mapping model name strings to matplotlib RGBA
            color tuples. Built at init time from a 22-color palette so that
            the same model always gets the same color across all figures.

    Example:
        >>> from evaluation.visualizer import Visualizer
        >>> viz = Visualizer(output_dir="outputs")
        >>> viz.save_table_as_csv(df, "outputs/tables/table2.csv")
        >>> viz.plot_detection_accuracy_heatmap(accuracy_df, "outputs/figures/figure3.png")
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        """Initialize the Visualizer and create output directory structure.

        Creates the output directory and its 'figures/' and 'tables/'
        subdirectories if they do not already exist. Builds the shared
        model color map from a 22-color palette for consistent model
        coloring across all figures.

        Args:
            output_dir: Root directory for all output files. Defaults to
                "outputs" to match config.yaml output_dir. Subdirectories
                'figures/' and 'tables/' are created automatically.

        Example:
            >>> viz = Visualizer(output_dir="outputs")
            >>> os.path.isdir("outputs/figures")
            True
            >>> os.path.isdir("outputs/tables")
            True
        """
        self.output_dir: str = output_dir
        self.figures_dir: str = os.path.join(output_dir, "figures")
        self.tables_dir: str = os.path.join(output_dir, "tables")

        # Create directory structure.
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)

        # Build the shared model color map.
        # With 22 models, tab20 (20 colors) is extended with 2 from tab20b.
        self.model_color_map: Dict[str, Any] = {}
        self._default_colors: List[Any] = self._build_color_palette(_N_MODEL_COLORS)

        logger.info(
            "Visualizer initialized: output_dir='%s', "
            "figures_dir='%s', tables_dir='%s'.",
            self.output_dir,
            self.figures_dir,
            self.tables_dir,
        )

    # -----------------------------------------------------------------------
    # Private helper methods
    # -----------------------------------------------------------------------

    def _build_color_palette(self, n_colors: int) -> List[Any]:
        """Build a list of n_colors distinct RGBA color tuples.

        Uses matplotlib's tab20 colormap (20 colors) extended with colors
        from tab20b to cover up to 40 distinct models. For n_colors <= 20,
        only tab20 is used.

        Args:
            n_colors: Number of distinct colors to generate. Must be positive.

        Returns:
            List of n_colors RGBA tuples from the combined tab20 + tab20b palette.

        Example:
            >>> viz = Visualizer()
            >>> colors = viz._build_color_palette(22)
            >>> len(colors)
            22
        """
        colors: List[Any] = []

        # tab20 provides 20 distinct colors (indices 0.0 to 1.0 in steps of 0.05).
        tab20_cmap = cm.get_cmap("tab20")
        for i in range(min(n_colors, 20)):
            colors.append(tab20_cmap(i / 20.0))

        # If more than 20 colors are needed, extend with tab20b.
        if n_colors > 20:
            tab20b_cmap = cm.get_cmap("tab20b")
            extra_needed: int = n_colors - 20
            for i in range(extra_needed):
                colors.append(tab20b_cmap(i / 20.0))

        return colors

    def _get_model_color_map(self, model_names: List[str]) -> Dict[str, Any]:
        """Build or retrieve a color map for the given model names.

        Assigns colors from self._default_colors to model names in sorted
        alphabetical order for deterministic, consistent coloring across
        all figures. If a model was not seen at init time, it gets a color
        from the extended palette.

        Args:
            model_names: List of model name strings to assign colors to.
                Sorted alphabetically before color assignment.

        Returns:
            Dict mapping model name string to RGBA color tuple.

        Example:
            >>> viz = Visualizer()
            >>> color_map = viz._get_model_color_map(["gpt-4o", "claude-3"])
            >>> len(color_map)
            2
        """
        sorted_names: List[str] = sorted(set(model_names))
        n_needed: int = len(sorted_names)

        # Extend palette if needed.
        if n_needed > len(self._default_colors):
            self._default_colors = self._build_color_palette(n_needed)

        color_map: Dict[str, Any] = {}
        for idx, name in enumerate(sorted_names):
            color_map[name] = self._default_colors[idx % len(self._default_colors)]

        return color_map

    def _ensure_parent_dir(self, file_path: str) -> None:
        """Ensure the parent directory of a file path exists.

        Args:
            file_path: Full path to a file. The parent directory is created
                if it does not already exist.
        """
        parent_dir: str = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def _save_and_close(self, fig: plt.Figure, save_path: str) -> None:
        """Save a matplotlib figure to disk and close it to free memory.

        Args:
            fig: The matplotlib Figure object to save.
            save_path: Full file path for the output image. The parent
                directory is created if it does not exist.
        """
        self._ensure_parent_dir(save_path)
        fig.savefig(save_path, dpi=_SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved to '%s'.", save_path)

    def _shorten_prompt(self, prompt: str, max_chars: int = 60) -> str:
        """Shorten a prompt string for use as a subplot title.

        Args:
            prompt: The full prompt text string.
            max_chars: Maximum number of characters to include before
                truncating with "...". Defaults to 60.

        Returns:
            Shortened prompt string. If the prompt is shorter than max_chars,
            returns it unchanged. Otherwise returns the first max_chars
            characters followed by "...".

        Example:
            >>> viz = Visualizer()
            >>> viz._shorten_prompt("Hello world", 20)
            'Hello world'
            >>> viz._shorten_prompt("A very long prompt that exceeds the limit", 20)
            'A very long prompt t...'
        """
        if len(prompt) <= max_chars:
            return prompt
        return prompt[:max_chars] + "..."

    def _reorder_dataframe_rows(
        self,
        df: pd.DataFrame,
        preferred_order: List[str],
    ) -> pd.DataFrame:
        """Reorder DataFrame rows to match a preferred ordering.

        Rows present in preferred_order appear first (in that order), followed
        by any remaining rows in their original order. Rows in preferred_order
        that are not in the DataFrame are silently skipped.

        Args:
            df: DataFrame to reorder. Index values are matched against
                preferred_order.
            preferred_order: List of index values in the desired order.

        Returns:
            Reordered DataFrame. All original rows are preserved.

        Example:
            >>> df = pd.DataFrame({"a": [1, 2, 3]}, index=["math", "english", "coding"])
            >>> viz._reorder_dataframe_rows(df, ["english", "math", "coding"])
            # Returns df with rows in order: english, math, coding
        """
        # Rows that are in preferred_order and present in the DataFrame.
        ordered_rows: List[str] = [r for r in preferred_order if r in df.index]

        # Remaining rows not in preferred_order.
        remaining_rows: List[str] = [r for r in df.index if r not in preferred_order]

        # Combine: preferred order first, then remaining.
        final_order: List[str] = ordered_rows + remaining_rows

        if not final_order:
            return df

        return df.loc[final_order]

    # -----------------------------------------------------------------------
    # Public plot methods
    # -----------------------------------------------------------------------

    def plot_pca_bow(
        self,
        pca_data: Dict[str, Any],
        save_path: str,
    ) -> None:
        """Plot PCA scatter plots of BoW features for 3 specific prompts.

        Reproduces Figure 2 of the paper: "First two principal components of
        bag-of-words (BoW) features for model responses to three randomly
        selected English prompts. Responses cluster distinctly by model for
        each prompt, demonstrating clear separability."

        Creates a 1×3 subplot figure with one scatter plot per prompt. Each
        point represents one model response, colored by model identity.

        Args:
            pca_data: Dict produced by TrainingBasedDetector.build_pca_visualization_data.
                Expected structure:
                    {
                        "prompts": [str, str, str],  # 3 prompt texts
                        "results": [
                            {
                                "components": np.ndarray of shape (N, 2),
                                "model_labels": List[str],
                                "prompt_text": str
                            },
                            ...  # one entry per prompt (up to 3)
                        ]
                    }
                If "results" has fewer than 3 entries, only available subplots
                are rendered; remaining subplots are left blank.
            save_path: Full file path for the output PNG. Parent directory
                is created if it does not exist.

        Returns:
            None. Saves the figure to save_path and closes the matplotlib figure.

        Example:
            >>> viz = Visualizer(output_dir="outputs")
            >>> viz.plot_pca_bow(pca_data, "outputs/figures/figure2_pca_bow.png")
        """
        logger.info("Plotting PCA BoW visualization (Figure 2) to '%s'.", save_path)

        # --- Extract data ---
        results: List[Dict[str, Any]] = pca_data.get("results", [])
        n_subplots: int = min(len(results), 3)

        if n_subplots == 0:
            logger.warning(
                "plot_pca_bow: pca_data['results'] is empty. "
                "Saving blank figure."
            )
            fig, _ = plt.subplots(1, 3, figsize=_FIG_SIZE_PCA)
            self._save_and_close(fig, save_path)
            return

        # --- Collect all model names across all prompts for consistent coloring ---
        all_model_names: List[str] = []
        for result_entry in results[:3]:
            model_labels: List[str] = result_entry.get("model_labels", [])
            all_model_names.extend(model_labels)

        color_map: Dict[str, Any] = self._get_model_color_map(all_model_names)

        # --- Create figure with 3 subplots ---
        fig, axes = plt.subplots(1, 3, figsize=_FIG_SIZE_PCA)

        # Ensure axes is always a list (handles edge case of n_subplots=1).
        if not hasattr(axes, "__len__"):
            axes = [axes]

        # Track legend handles and labels from the first subplot.
        legend_handles: List[Any] = []
        legend_labels: List[str] = []
        legend_built: bool = False

        for subplot_idx in range(3):
            ax: plt.Axes = axes[subplot_idx]

            if subplot_idx >= n_subplots:
                # No data for this subplot — leave it blank with a label.
                ax.set_visible(False)
                continue

            result_entry: Dict[str, Any] = results[subplot_idx]
            components: np.ndarray = np.array(
                result_entry.get("components", np.zeros((0, 2)))
            )
            model_labels_for_prompt: List[str] = result_entry.get("model_labels", [])
            prompt_text: str = result_entry.get(
                "prompt_text",
                pca_data.get("prompts", [""] * 3)[subplot_idx]
                if subplot_idx < len(pca_data.get("prompts", []))
                else f"Prompt {subplot_idx + 1}",
            )

            # Guard: skip if no data points.
            if components.shape[0] == 0 or len(model_labels_for_prompt) == 0:
                ax.set_title(
                    f"Prompt {subplot_idx + 1}\n(no data)",
                    fontsize=9,
                )
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                continue

            # Guard: ensure components has 2 columns.
            if components.ndim != 2 or components.shape[1] < 2:
                logger.warning(
                    "plot_pca_bow: components for subplot %d has unexpected "
                    "shape %s. Skipping.",
                    subplot_idx,
                    components.shape,
                )
                ax.set_visible(False)
                continue

            # --- Scatter plot per model ---
            unique_models: List[str] = sorted(set(model_labels_for_prompt))

            for model_name in unique_models:
                # Find indices where this model's responses appear.
                model_mask: np.ndarray = np.array(
                    [label == model_name for label in model_labels_for_prompt]
                )

                if not np.any(model_mask):
                    continue

                model_components: np.ndarray = components[model_mask]
                color: Any = color_map.get(model_name, "gray")

                scatter = ax.scatter(
                    model_components[:, 0],
                    model_components[:, 1],
                    c=[color],
                    label=model_name,
                    alpha=0.6,
                    s=20,
                    edgecolors="none",
                )

                # Collect legend handles from the first subplot only.
                if not legend_built:
                    legend_handles.append(scatter)
                    legend_labels.append(model_name)

            legend_built = True

            # --- Axis formatting ---
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)

            # Shorten prompt text for the subplot title.
            short_prompt: str = self._shorten_prompt(prompt_text, max_chars=55)
            ax.set_title(
                f"Prompt {subplot_idx + 1}:\n{short_prompt}",
                fontsize=8,
                pad=6,
            )

        # --- Shared legend below the figure ---
        if legend_handles:
            # Sort legend entries alphabetically by model name.
            sorted_pairs: List[Tuple[Any, str]] = sorted(
                zip(legend_handles, legend_labels), key=lambda x: x[1]
            )
            sorted_handles: List[Any] = [p[0] for p in sorted_pairs]
            sorted_labels: List[str] = [p[1] for p in sorted_pairs]

            # Place legend below the figure with multiple columns.
            n_legend_cols: int = min(6, len(sorted_labels))
            fig.legend(
                sorted_handles,
                sorted_labels,
                loc="lower center",
                ncol=n_legend_cols,
                bbox_to_anchor=(0.5, -0.12),
                fontsize=7,
                markerscale=1.5,
                frameon=True,
            )

        plt.suptitle(
            "PCA of BoW Features for Model Responses (Figure 2)",
            fontsize=12,
            y=1.02,
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)

        self._save_and_close(fig, save_path)

    def plot_detection_accuracy_heatmap(
        self,
        df: pd.DataFrame,
        save_path: str,
    ) -> None:
        """Plot a heatmap of BoW-based detection accuracy across categories and models.

        Reproduces Figure 3 of the paper: "Test accuracy (%) of detectors
        trained to distinguish the target model (specified in each column) from
        other models (scale: 85% to 100%). Prompts featuring domain-specific
        tasks and non-English languages yield the highest detection accuracy.
        Detectors are built using BoW features."

        Args:
            df: DataFrame produced by TrainingBasedDetector.evaluate_all_models_categories
                with:
                  - Index: prompt category names (rows)
                  - Columns: model name strings (columns)
                  - Values: test accuracy percentages (float, 0.0–100.0)
                The color scale is fixed at vmin=85.0, vmax=100.0 per the paper.
                Values below 85% are clipped to the colormap minimum.
            save_path: Full file path for the output PNG. Parent directory
                is created if it does not exist.

        Returns:
            None. Saves the figure to save_path and closes the matplotlib figure.

        Example:
            >>> viz = Visualizer(output_dir="outputs")
            >>> viz.plot_detection_accuracy_heatmap(
            ...     accuracy_df, "outputs/figures/figure3_heatmap.png"
            ... )
        """
        logger.info(
            "Plotting detection accuracy heatmap (Figure 3) to '%s'.", save_path
        )

        if df.empty:
            logger.warning(
                "plot_detection_accuracy_heatmap: input DataFrame is empty. "
                "Saving blank figure."
            )
            fig, ax = plt.subplots(figsize=_FIG_SIZE_HEATMAP)
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            self._save_and_close(fig, save_path)
            return

        # --- Reorder rows to match canonical category order ---
        df_ordered: pd.DataFrame = self._reorder_dataframe_rows(df, _CATEGORY_ORDER)

        # --- Create figure ---
        # Width scales with number of columns (models).
        n_cols: int = len(df_ordered.columns)
        fig_width: float = max(16.0, n_cols * 0.85)
        fig_height: float = max(5.0, len(df_ordered.index) * 0.7)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # --- Seaborn heatmap ---
        # annot=True shows accuracy values in each cell.
        # fmt=".1f" formats to 1 decimal place (e.g., "95.7").
        # vmin/vmax set the color scale to 85–100% per the paper.
        sns.heatmap(
            df_ordered,
            ax=ax,
            vmin=_HEATMAP_VMIN,
            vmax=_HEATMAP_VMAX,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Test Accuracy (%)", "shrink": 0.8},
            annot_kws={"size": 7},
        )

        # --- Axis formatting ---
        # X-axis: model names, rotated 45 degrees for readability.
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=8,
        )

        # Y-axis: category names, horizontal.
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=9,
        )

        ax.set_xlabel("Target Model", fontsize=11, labelpad=10)
        ax.set_ylabel("Prompt Category", fontsize=11, labelpad=10)
        ax.set_title(
            "Detection Accuracy (%) by Model and Prompt Category (BoW Features)\n"
            "Figure 3 — Scale: 85% to 100%",
            fontsize=12,
            pad=12,
        )

        plt.tight_layout()
        self._save_and_close(fig, save_path)

    def plot_detection_rate_vs_votes(
        self,
        df: pd.DataFrame,
        adversary_types: List[str],
        save_path: str,
        title: str = "Detection Rate vs. Number of Adversarial Votes",
        use_log_scale: bool = True,
    ) -> None:
        """Plot detection rate vs. number of adversarial votes.

        Reproduces Figures 4 and 5 of the paper. This single method handles
        both figures because they share the same plot structure (detection rate
        vs. vote count, multiple lines). The caller differentiates them by
        passing different DataFrames and adversary_types lists.

        Figure 4 (Scenario 1): Two lines — "naive" adversary (solid) and
        "informed" adversary (dashed). The naive adversary is detectable but
        the informed adversary evades detection.

        Figure 5 (Scenario 2): Multiple lines for different noise scales.
        Uses a sequential colormap (Blues) to color lines from light (low noise)
        to dark (high noise).

        Args:
            df: DataFrame produced by Metrics.compute_detection_rate_curve with
                columns ['vote_count', 'adversary_type', 'detection_rate'].
                Each row is one (adversary_type, vote_count) data point.
            adversary_types: Ordered list of unique adversary_type values to
                plot. Determines legend order and line style assignment.
                For Figure 4: ["naive", "informed"].
                For Figure 5: noise scale labels like ["noise_0.1", "noise_0.5", ...].
            save_path: Full file path for the output PNG.
            title: Plot title string. Defaults to a generic title; callers
                should pass figure-specific titles.
            use_log_scale: If True, uses a log scale on the x-axis (vote count).
                Defaults to True since vote counts span orders of magnitude
                (10 to 1000).

        Returns:
            None. Saves the figure to save_path and closes the matplotlib figure.

        Example:
            >>> viz = Visualizer(output_dir="outputs")
            >>> viz.plot_detection_rate_vs_votes(
            ...     df=scenario1_df,
            ...     adversary_types=["naive", "informed"],
            ...     save_path="outputs/figures/figure4_scenario1.png",
            ...     title="Scenario 1: Known Benign Distribution",
            ... )
        """
        logger.info(
            "Plotting detection rate vs. votes to '%s' "
            "(adversary_types=%s).",
            save_path,
            adversary_types,
        )

        if df.empty:
            logger.warning(
                "plot_detection_rate_vs_votes: input DataFrame is empty. "
                "Saving blank figure."
            )
            fig, ax = plt.subplots(figsize=_FIG_SIZE_LINE)
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title(title, fontsize=12)
            self._save_and_close(fig, save_path)
            return

        # --- Determine if this is a noise-scale plot (Figure 5) ---
        # Noise-scale plots have adversary_types like "noise_0.1", "noise_0.5", etc.
        is_noise_scale_plot: bool = any(
            str(at).startswith("noise_") for at in adversary_types
        )

        # --- Build color and line style assignments ---
        n_types: int = len(adversary_types)

        if is_noise_scale_plot:
            # Sequential colormap (Blues) for noise scale lines.
            blues_cmap = cm.get_cmap("Blues")
            # Use range [0.3, 0.9] to avoid very light colors.
            line_colors: List[Any] = [
                blues_cmap(0.3 + 0.6 * i / max(n_types - 1, 1))
                for i in range(n_types)
            ]
            line_styles: List[str] = ["-"] * n_types
        else:
            # Distinct colors from tab10 for Figure 4 (few adversary types).
            tab10_cmap = cm.get_cmap("tab10")
            line_colors = [tab10_cmap(i / 10.0) for i in range(n_types)]
            # Use predefined line styles for known adversary types.
            line_styles = [
                _ADVERSARY_LINE_STYLES.get(str(at), "-")
                for at in adversary_types
            ]

        # --- Create figure ---
        fig, ax = plt.subplots(figsize=_FIG_SIZE_LINE)

        # --- Plot one line per adversary type ---
        for type_idx, adversary_type in enumerate(adversary_types):
            # Filter and sort data for this adversary type.
            mask: pd.Series = df["adversary_type"] == adversary_type
            subset: pd.DataFrame = df[mask].sort_values("vote_count")

            if subset.empty:
                logger.debug(
                    "plot_detection_rate_vs_votes: no data for "
                    "adversary_type='%s'. Skipping.",
                    adversary_type,
                )
                continue

            # Drop NaN detection rates.
            subset_clean: pd.DataFrame = subset.dropna(subset=["detection_rate"])

            if subset_clean.empty:
                continue

            color: Any = line_colors[type_idx % len(line_colors)]
            linestyle: str = line_styles[type_idx % len(line_styles)]

            ax.plot(
                subset_clean["vote_count"],
                subset_clean["detection_rate"],
                marker=_DEFAULT_MARKER,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
                label=str(adversary_type),
            )

        # --- Reference line at significance level α = 0.01 ---
        ax.axhline(
            y=_SIGNIFICANCE_LEVEL,
            color="gray",
            linestyle=":",
            alpha=0.7,
            linewidth=1.5,
            label=f"α = {_SIGNIFICANCE_LEVEL}",
        )

        # --- Axis formatting ---
        if use_log_scale:
            ax.set_xscale("log")

        ax.set_xlabel("Number of Adversarial Votes