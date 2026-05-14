## evaluation/edit_distance_analysis.py
"""Edit distance analysis for SCoRe: Self-Correction via Reinforcement Learning.

This module implements the edit distance ratio analysis described in Section 4
of the paper (Figure 4). It measures how much models change their responses
between turn 1 and turn 2, serving as the primary diagnostic for detecting
**behavior collapse** — the failure mode where models learn to make no edits
(or trivial edits) to their first-attempt responses.

Paper context (Section 4, Figure 4):
    "To further understand how these STaR and SFT models edit their responses,
    we measured their edit distance ratios, defined as the edit distance between
    the responses normalized by the total length of both the responses."

    "While the base model sometimes makes substantially large edits to the
    original response, models fine-tuned on D_STaR and D_SFT are overly
    conservative and often make no edits at all. This is akin to a form of
    behavior collapse."

Edit distance ratio formula:
    ratio = editdistance.eval(r1, r2) / (len(r1) + len(r2) + eps)

Thresholds (from config.yaml evaluation.edit_distance_analysis):
    no_edit_threshold: 0.01   — ratios below this are "no edit" (behavior collapse)
    large_edit_threshold: 0.5 — ratios above this are "large edit" (full rewrite)

Typical usage:
    from evaluation.edit_distance_analysis import EditDistanceAnalysis
    from training.rollout_buffer import Trajectory

    analysis = EditDistanceAnalysis()

    # Analyze a batch of trajectories
    stats = analysis.analyze_batch(trajectories)
    print(f"Fraction no-edit: {stats['fraction_no_edit']:.3f}")

    # Plot distribution for a single method
    analysis.plot_distribution(stats['ratios'], label='SCoRe', save_path='plots/score.png')

    # Compare multiple methods (reproduces Figure 4)
    analysis.compare_methods(
        method_ratios={
            'Base Model': base_ratios,
            'STaR': star_ratios,
            'Pair-SFT': pair_sft_ratios,
            'SCoRe': score_ratios,
        },
        save_path='plots/figure4_comparison.png',
    )
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from training.rollout_buffer import Trajectory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — guarded so the module can be imported even if these
# packages are not installed (tests, linting, CI without full deps).
# ---------------------------------------------------------------------------
try:
    import editdistance as _editdistance_module

    _EDITDISTANCE_AVAILABLE: bool = True
except ImportError:
    _EDITDISTANCE_AVAILABLE = False
    logger.warning(
        "editdistance is not installed. Edit distance computation will fall "
        "back to a basic character-level implementation. "
        "Install editdistance==0.8.1 for correct results: "
        "pip install editdistance==0.8.1"
    )

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    _MATPLOTLIB_AVAILABLE: bool = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    logger.warning(
        "matplotlib is not installed. Plot generation will be disabled. "
        "Install matplotlib==3.9.1: pip install matplotlib==3.9.1"
    )

try:
    import seaborn as sns

    _SEABORN_AVAILABLE: bool = True
except ImportError:
    _SEABORN_AVAILABLE = False
    logger.debug(
        "seaborn is not installed. Falling back to matplotlib for plotting. "
        "Install seaborn==0.13.2 for enhanced plot styling."
    )

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Small epsilon to prevent division by zero in ratio computation.
# Consistent with config.yaml score.reward_norm_eps = 1e-8.
_EPS: float = 1e-8

# Thresholds from config.yaml evaluation.edit_distance_analysis
# These are the default values; the class uses them directly.
_NO_EDIT_THRESHOLD: float = 0.01
_LARGE_EDIT_THRESHOLD: float = 0.5

# Maximum response length (in characters) for edit distance computation.
# Prevents O(n*m) slowdown for very long responses. Truncation does not
# materially affect the statistical conclusions since the ratio is normalized.
_MAX_RESPONSE_CHARS: int = 2000

# Color scheme for compare_methods() — matches paper Figure 4 style.
# Keys are method name substrings (case-insensitive matching).
_METHOD_COLORS: Dict[str, str] = {
    "base": "#1f77b4",      # Blue for base model
    "star": "#ff7f0e",      # Orange for STaR
    "pair": "#2ca02c",      # Green for Pair-SFT
    "score": "#d62728",     # Red for SCoRe
}

# Default color for methods not matching any key in _METHOD_COLORS
_DEFAULT_COLOR: str = "#9467bd"  # Purple

# Number of histogram bins for distribution plots
_NUM_BINS: int = 50

# Alpha (transparency) for overlaid histograms in compare_methods()
_HISTOGRAM_ALPHA: float = 0.5

# Figure size for single-method plots
_SINGLE_FIGURE_SIZE: tuple = (8, 5)

# Figure size for comparison plots
_COMPARISON_FIGURE_SIZE: tuple = (10, 6)

# DPI for saved figures
_FIGURE_DPI: int = 150


class EditDistanceAnalysis:
    """Edit distance ratio analysis for self-correction behavior diagnosis.

    Measures how much models change their responses between turn 1 and turn 2.
    A near-zero edit distance ratio indicates behavior collapse (the model
    copies its first attempt). A healthy distribution indicates genuine
    self-correction behavior.

    All methods are effectively stateless — the class exists as a namespace
    consistent with the design specification. The __init__ method accepts
    optional threshold overrides for flexibility, defaulting to the values
    from config.yaml.

    Attributes:
        no_edit_threshold: Ratio below which an edit is classified as "no edit".
            From config.yaml: evaluation.edit_distance_analysis.no_edit_threshold = 0.01.
        large_edit_threshold: Ratio above which an edit is classified as "large edit".
            From config.yaml: evaluation.edit_distance_analysis.large_edit_threshold = 0.5.
    """

    def __init__(
        self,
        no_edit_threshold: float = _NO_EDIT_THRESHOLD,
        large_edit_threshold: float = _LARGE_EDIT_THRESHOLD,
    ) -> None:
        """Initialize EditDistanceAnalysis with configurable thresholds.

        Args:
            no_edit_threshold: Ratio below which an edit is classified as
                "no edit" (behavior collapse indicator). Default 0.01 from
                config.yaml: evaluation.edit_distance_analysis.no_edit_threshold.
            large_edit_threshold: Ratio above which an edit is classified as
                "large edit" (full rewrite indicator). Default 0.5 from
                config.yaml: evaluation.edit_distance_analysis.large_edit_threshold.

        Raises:
            ValueError: If no_edit_threshold >= large_edit_threshold, or if
                either threshold is outside [0.0, 1.0].
        """
        if not (0.0 <= no_edit_threshold < large_edit_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: no_edit_threshold={no_edit_threshold}, "
                f"large_edit_threshold={large_edit_threshold}. "
                "Must satisfy 0.0 <= no_edit_threshold < large_edit_threshold <= 1.0. "
                "These values come from config.yaml: "
                "evaluation.edit_distance_analysis.no_edit_threshold and "
                "evaluation.edit_distance_analysis.large_edit_threshold."
            )

        self.no_edit_threshold: float = no_edit_threshold
        self.large_edit_threshold: float = large_edit_threshold

        logger.debug(
            "EditDistanceAnalysis initialized: "
            "no_edit_threshold=%.3f, large_edit_threshold=%.3f, "
            "editdistance_available=%s, matplotlib_available=%s.",
            self.no_edit_threshold,
            self.large_edit_threshold,
            _EDITDISTANCE_AVAILABLE,
            _MATPLOTLIB_AVAILABLE,
        )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def compute_edit_distance_ratio(
        self, response1: str, response2: str
    ) -> float:
        """Compute the normalized edit distance ratio between two responses.

        Implements the formula from Section 4 of the paper:
            ratio = editdistance.eval(r1, r2) / (len(r1) + len(r2) + eps)

        The normalization by total length ensures comparability across
        responses of different lengths. A ratio near 0 indicates the model
        made no meaningful changes (behavior collapse). A ratio near 1
        indicates a complete rewrite.

        Responses are truncated to _MAX_RESPONSE_CHARS characters before
        computation to prevent O(n*m) slowdown for very long responses.
        This truncation does not materially affect statistical conclusions
        since the ratio is normalized.

        Args:
            response1: The first-attempt response string (turn1_response
                from a Trajectory). May be empty.
            response2: The second-attempt response string (turn2_response
                from a Trajectory). May be empty.

        Returns:
            Float in [0.0, 1.0] representing the normalized edit distance.
            Returns 0.0 if both responses are empty (no edit, trivially).
            Returns 1.0 if one response is empty and the other is not
            (complete replacement).
        """
        # Handle empty inputs
        if not response1 and not response2:
            logger.debug(
                "compute_edit_distance_ratio: Both responses are empty. "
                "Returning 0.0."
            )
            return 0.0

        # Truncate to prevent O(n*m) slowdown for very long responses
        r1: str = response1[:_MAX_RESPONSE_CHARS]
        r2: str = response2[:_MAX_RESPONSE_CHARS]

        # Compute character-level edit distance
        edit_dist: int = self._compute_edit_distance(r1, r2)

        # Normalize by total length with epsilon to prevent division by zero
        total_length: float = float(len(r1) + len(r2)) + _EPS
        ratio: float = float(edit_dist) / total_length

        # Clamp to [0.0, 1.0] for robustness (should always be in range,
        # but floating-point arithmetic can produce tiny violations)
        ratio = max(0.0, min(1.0, ratio))

        logger.debug(
            "compute_edit_distance_ratio: edit_dist=%d, "
            "len(r1)=%d, len(r2)=%d, ratio=%.4f.",
            edit_dist,
            len(r1),
            len(r2),
            ratio,
        )

        return ratio

    def analyze_batch(
        self, trajectories: List[Trajectory]
    ) -> Dict[str, Any]:
        """Compute edit distance statistics over a batch of trajectories.

        Iterates over all trajectories, computes the edit distance ratio
        between turn1_response and turn2_response for each, and aggregates
        statistics. This is the primary method for quantifying behavior
        collapse in a trained model.

        The `fraction_no_edit` field directly measures behavior collapse:
        a high value (e.g., > 0.5) indicates the model has learned to
        copy its first attempt rather than genuinely self-correcting.

        Args:
            trajectories: List of Trajectory objects from
                RolloutBuffer.sample_trajectories() or
                Evaluator._run_two_turn_inference(). Each trajectory must
                have non-None turn1_response and turn2_response string fields.

        Returns:
            Dict with the following keys:
                'ratios' (List[float]): Raw edit distance ratios for all
                    trajectories, in the same order as the input list.
                'mean' (float): Mean edit distance ratio across the batch.
                'std' (float): Standard deviation of edit distance ratios.
                'fraction_no_edit' (float): Fraction of trajectories with
                    ratio < no_edit_threshold (0.01). High values indicate
                    behavior collapse (Section 4, Figure 4a).
                'fraction_large_edit' (float): Fraction of trajectories with
                    ratio > large_edit_threshold (0.5). High values indicate
                    the model is making full rewrites.
            Returns a zero-valued dict with an empty 'ratios' list if
            trajectories is empty.
        """
        if not trajectories:
            logger.debug(
                "analyze_batch: Empty trajectories list. "
                "Returning zero-valued stats dict."
            )
            return {
                "ratios": [],
                "mean": 0.0,
                "std": 0.0,
                "fraction_no_edit": 0.0,
                "fraction_large_edit": 0.0,
            }

        # Compute edit distance ratio for each trajectory
        ratios: List[float] = []
        for i, trajectory in enumerate(trajectories):
            r1: str = trajectory.turn1_response if trajectory.turn1_response else ""
            r2: str = trajectory.turn2_response if trajectory.turn2_response else ""

            ratio: float = self.compute_edit_distance_ratio(r1, r2)
            ratios.append(ratio)

        # Aggregate statistics using numpy
        ratios_array: np.ndarray = np.array(ratios, dtype=np.float64)

        mean_ratio: float = float(np.mean(ratios_array))
        std_ratio: float = float(np.std(ratios_array))

        # Fraction of trajectories with near-zero edit distance (behavior collapse)
        num_no_edit: int = int(np.sum(ratios_array < self.no_edit_threshold))
        fraction_no_edit: float = num_no_edit / len(ratios)

        # Fraction of trajectories with large edit distance (full rewrite)
        num_large_edit: int = int(np.sum(ratios_array > self.large_edit_threshold))
        fraction_large_edit: float = num_large_edit / len(ratios)

        logger.info(
            "analyze_batch: n=%d, mean_ratio=%.4f, std=%.4f, "
            "fraction_no_edit=%.3f (ratio<%.2f), "
            "fraction_large_edit=%.3f (ratio>%.2f).",
            len(trajectories),
            mean_ratio,
            std_ratio,
            fraction_no_edit,
            self.no_edit_threshold,
            fraction_large_edit,
            self.large_edit_threshold,
        )

        return {
            "ratios": ratios,
            "mean": mean_ratio,
            "std": std_ratio,
            "fraction_no_edit": fraction_no_edit,
            "fraction_large_edit": fraction_large_edit,
        }

    def plot_distribution(
        self,
        ratios: List[float],
        label: str,
        save_path: str,
    ) -> None:
        """Plot the histogram of edit distance ratios for a single method.

        Reproduces the style of Figure 4a/4b/4c in the paper for a single
        method. The histogram uses density normalization so distributions
        are comparable across different batch sizes.

        Adds vertical reference lines at no_edit_threshold (0.01) and
        large_edit_threshold (0.5) to visually demarcate the three regions:
        no-edit, moderate-edit, and large-edit.

        If matplotlib is not available, logs a warning and returns without
        saving (graceful degradation — training continues without plots).

        Args:
            ratios: List of edit distance ratio floats in [0.0, 1.0].
                Typically the 'ratios' field from analyze_batch().
            label: Human-readable method name for the plot title and
                filename (e.g., 'Base Model', 'STaR', 'Pair-SFT', 'SCoRe').
            save_path: Full file path for saving the plot (e.g.,
                'outputs/plots/score_edit_dist.png'). The parent directory
                is created if it does not exist.

        Returns:
            None. The plot is saved to save_path and the figure is closed
            to prevent memory leaks.
        """
        if not _MATPLOTLIB_AVAILABLE:
            logger.warning(
                "plot_distribution: matplotlib is not available. "
                "Skipping plot generation for label='%s'. "
                "Install matplotlib==3.9.1 to enable plotting.",
                label,
            )
            return

        if not ratios:
            logger.warning(
                "plot_distribution: Empty ratios list for label='%s'. "
                "Skipping plot generation.",
                label,
            )
            return

        # Ensure the output directory exists
        save_dir: str = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Determine color based on method name
        color: str = self._get_method_color(label)

        # Convert to numpy array for statistics
        ratios_array: np.ndarray = np.array(ratios, dtype=np.float64)
        mean_val: float = float(np.mean(ratios_array))
        std_val: float = float(np.std(ratios_array))
        fraction_no_edit: float = float(
            np.mean(ratios_array < self.no_edit_threshold)
        )

        # Create figure
        fig, ax = plt.subplots(figsize=_SINGLE_FIGURE_SIZE)

        # Bin edges: uniform over [0, 1]
        bin_edges: np.ndarray = np.linspace(0.0, 1.0, _NUM_BINS + 1)

        # Plot histogram with density normalization
        if _SEABORN_AVAILABLE:
            sns.histplot(
                data=ratios_array,
                bins=bin_edges,
                stat="density",
                color=color,
                alpha=0.7,
                ax=ax,
                label=label,
            )
        else:
            ax.hist(
                ratios_array,
                bins=bin_edges,
                density=True,
                color=color,
                alpha=0.7,
                label=label,
            )

        # Add vertical reference lines
        ax.axvline(
            x=self.no_edit_threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"No-edit threshold ({self.no_edit_threshold:.2f})",
        )
        ax.axvline(
            x=self.large_edit_threshold,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"Large-edit threshold ({self.large_edit_threshold:.2f})",
        )

        # Labels and title
        ax.set_xlabel("Edit Distance Ratio", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            f"Edit Distance Distribution: {label}\n"
            f"(n={len(ratios)}, mean={mean_val:.3f}±{std_val:.3f}, "
            f"no-edit={fraction_no_edit:.1%})",
            fontsize=11,
        )
        ax.set_xlim(0.0, 1.0)
        ax.legend(fontsize=9)

        # Apply seaborn styling if available
        if _SEABORN_AVAILABLE:
            sns.despine(ax=ax)

        plt.tight_layout()

        # Save and close
        try:
            fig.savefig(save_path, dpi=_FIGURE_DPI, bbox_inches="tight")
            logger.info(
                "plot_distribution: Saved plot for label='%s' to '%s'.",
                label,
                save_path,
            )
        except Exception as exc:
            logger.error(
                "plot_distribution: Failed to save plot to '%s': %s.",
                save_path,
                exc,
            )
        finally:
            plt.close(fig)

    def compare_methods(
        self,
        method_ratios: Dict[str, List[float]],
        save_path: str,
    ) -> None:
        """Plot overlaid histograms comparing edit distance distributions across methods.

        Directly reproduces Figure 4 of the paper, showing that STaR and
        Pair-SFT models have a large spike near zero (behavior collapse),
        while SCoRe has a healthier distribution indicating genuine
        self-correction behavior.

        Each method's histogram is plotted with 50% transparency so
        overlapping distributions remain visible. Density normalization
        ensures methods with different sample counts are comparable.

        If matplotlib is not available, logs a warning and returns without
        saving (graceful degradation).

        Args:
            method_ratios: Dict mapping method name (str) to list of edit
                distance ratio floats. Expected keys (matching paper Figure 4):
                    'Base Model': ratios from the untuned base model
                    'STaR': ratios from the STaR-trained model
                    'Pair-SFT': ratios from the Pair-SFT-trained model
                    'SCoRe': ratios from the SCoRe-trained model
                Any subset of these keys is valid — the method handles
                arbitrary method names with automatic color assignment.
            save_path: Full file path for saving the comparison plot.
                The parent directory is created if it does not exist.

        Returns:
            None. The plot is saved to save_path and the figure is closed.
        """
        if not _MATPLOTLIB_AVAILABLE:
            logger.warning(
                "compare_methods: matplotlib is not available. "
                "Skipping comparison plot generation. "
                "Install matplotlib==3.9.1 to enable plotting.",
            )
            return

        if not method_ratios:
            logger.warning(
                "compare_methods: Empty method_ratios dict. "
                "Skipping comparison plot generation."
            )
            return

        # Filter out methods with empty ratio lists
        valid_methods: Dict[str, List[float]] = {
            name: ratios
            for name, ratios in method_ratios.items()
            if ratios
        }

        if not valid_methods:
            logger.warning(
                "compare_methods: All method ratio lists are empty. "
                "Skipping comparison plot generation."
            )
            return

        # Ensure the output directory exists
        save_dir: str = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Create figure
        fig, ax = plt.subplots(figsize=_COMPARISON_FIGURE_SIZE)

        # Consistent bin edges across all methods for fair comparison
        bin_edges: np.ndarray = np.linspace(0.0, 1.0, _NUM_BINS + 1)

        # Plot each method's histogram
        legend_handles: List[mpatches.Patch] = []

        for method_name, ratios in valid_methods.items():
            ratios_array: np.ndarray = np.array(ratios, dtype=np.float64)
            color: str = self._get_method_color(method_name)
            mean_val: float = float(np.mean(ratios_array))
            fraction_no_edit: float = float(
                np.mean(ratios_array < self.no_edit_threshold)
            )

            if _SEABORN_AVAILABLE:
                sns.histplot(
                    data=ratios_array,
                    bins=bin_edges,
                    stat="density",
                    color=color,
                    alpha=_HISTOGRAM_ALPHA,
                    ax=ax,
                    label=f"{method_name} (n={len(ratios)}, "
                          f"mean={mean_val:.3f}, "
                          f"no-edit={fraction_no_edit:.1%})",
                )
            else:
                ax.hist(
                    ratios_array,
                    bins=bin_edges,
                    density=True,
                    color=color,
                    alpha=_HISTOGRAM_ALPHA,
                    label=f"{method_name} (n={len(ratios)}, "
                          f"mean={mean_val:.3f}, "
                          f"no-edit={fraction_no_edit:.1%})",
                )

            # Build legend patch for this method
            patch: mpatches.Patch = mpatches.Patch(
                color=color,
                alpha=_HISTOGRAM_ALPHA,
                label=f"{method_name} (n={len(ratios)}, "
                      f"mean={mean_val:.3f}, "
                      f"no-edit={fraction_no_edit:.1%})",
            )
            legend_handles.append(patch)

        # Add vertical reference lines
        ax.axvline(
            x=self.no_edit_threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"No-edit threshold ({self.no_edit_threshold:.2f})",
        )
        ax.axvline(
            x=self.large_edit_threshold,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Large-edit threshold ({self.large_edit_threshold:.2f})",
        )

        # Add threshold lines to legend handles
        no_edit_line = mpatches.Patch(
            color="black",
            alpha=0.7,
            label=f"No-edit threshold ({self.no_edit_threshold:.2f})",
        )
        large_edit_line = mpatches.Patch(
            color="gray",
            alpha=0.7,
            label=f"Large-edit threshold ({self.large_edit_threshold:.2f})",
        )
        legend_handles.extend([no_edit_line, large_edit_line])

        # Labels and title
        ax.set_xlabel("Edit Distance Ratio", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            "Edit Distance Distribution Comparison\n"
            "(Figure 4: SCoRe vs. Baselines)",
            fontsize=12,
        )
        ax.set_xlim(0.0, 1.0)

        # Legend outside the plot area to avoid obscuring the distributions
        ax.legend(
            handles=legend_handles,
            fontsize=8,
            loc="upper right",
            framealpha=0.9,
        )

        # Apply seaborn styling if available
        if _SEABORN_AVAILABLE:
            sns.despine(ax=ax)

        plt.tight_layout()

        # Save and close
        try:
            fig.savefig(save_path, dpi=_FIGURE_DPI, bbox_inches="tight")
            logger.info(
                "compare_methods: Saved comparison plot for %d methods to '%s'.",
                len(valid_methods),
                save_path,
            )
        except Exception as exc:
            logger.error(
                "compare_methods: Failed to save comparison plot to '%s': %s.",
                save_path,
                exc,
            )
        finally:
            plt.close(fig)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _compute_edit_distance(self, s1: str, s2: str) -> int:
        """Compute character-level edit distance between two strings.

        Uses the editdistance library (Levenshtein distance) if available.
        Falls back to a basic dynamic programming implementation if the
        library is not installed.

        Args:
            s1: First string (already truncated to _MAX_RESPONSE_CHARS).
            s2: Second string (already truncated to _MAX_RESPONSE_CHARS).

        Returns:
            Non-negative integer representing the minimum number of single-
            character edits (insertions, deletions, substitutions) required
            to transform s1 into s2.
        """
        if _EDITDISTANCE_AVAILABLE:
            return int(_editdistance_module.eval(s1, s2))
        else:
            # Fallback: basic dynamic programming Levenshtein distance
            return self._levenshtein_fallback(s1, s2)

    @staticmethod
    def _levenshtein_fallback(s1: str, s2: str) -> int:
        """Compute Levenshtein distance via dynamic programming (fallback).

        Used when the editdistance library is not installed. This is an
        O(n*m) implementation where n=len(s1) and m=len(s2). For typical
        LLM responses truncated to _MAX_RESPONSE_CHARS=2000 characters,
        this is at most O(4,000,000) operations — acceptable but slower
        than the C-extension editdistance library.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Non-negative integer Levenshtein distance.
        """
        n: int = len(s1)
        m: int = len(s2)

        # Handle trivial cases
        if n == 0:
            return m
        if m == 0:
            return n

        # Use two rows to save memory (O(min(n,m)) space instead of O(n*m))
        # Ensure s1 is the shorter string for memory efficiency
        if n > m:
            s1, s2 = s2, s1
            n, m = m, n

        # prev_row[j] = edit distance between s1[:0] and s2[:j]
        prev_row: List[int] = list(range(m + 1))
        curr_row: List[int] = [0] * (m + 1)

        for i in range(1, n + 1):
            curr_row[0] = i
            for j in range(1, m + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr_row[j] = prev_row[j - 1]
                else:
                    curr_row[j] = 1 + min(
                        prev_row[j],      # deletion
                        curr_row[j - 1],  # insertion
                        prev_row[j - 1],  # substitution
                    )
            prev_row, curr_row = curr_row, prev_row

        return prev_row[m]

    @staticmethod
    def _get_method_color(method_name: str) -> str:
        """Determine the plot color for a method based on its name.

        Uses case-insensitive substring matching against the _METHOD_COLORS
        dict. Returns _DEFAULT_COLOR if no match is found.

        Args:
            method_name: Human-readable method name (e.g., 'Base Model',
                'STaR', 'Pair-SFT', 'SCoRe').

        Returns:
            Hex color string for use in matplotlib/seaborn plots.
        """
        method_lower: str = method_name.lower()
        for key, color in _METHOD_COLORS.items():
            if key in method_lower:
                return color
        return _DEFAULT_COLOR
