```python
## evaluation/diversity.py
"""Prediction diversity analysis for the PEFT Visual Recognition reproduction study.

This module implements the prediction diversity analysis described in Section 4
of the paper: "Different PEFT Approaches Offer Complementary Information."

The key finding: even though different PEFT methods achieve similar accuracy,
they make substantially different predictions — differing in ~20% of predictions
on DTD/Retinopathy and ~35% on DMLab. This opens opportunities for ensemble
methods and semi-supervised learning.

Paper references:
    Section 4: "Contrary to this expectation, our findings below reveal that
    different PEFT methods acquire distinct and complementary knowledge from
    the same downstream data."
    Figure 3a: Prediction similarity matrices for DTD, Retinopathy, DMLab.
    Figure 1b: Correct prediction overlaps for 5K most confident samples.
    Figure 3b: Wrong prediction overlaps for 5K least confident samples.
    Figure 2: Ranking frequency matrices per VTAB-1K group.

Config references (config.yaml):
    diversity_analysis.top_k_confident: 5000
    diversity_analysis.representative_methods: [lora, houlsby_adapter, ssf]
    diversity_analysis.focus_datasets: [dtd, retinopathy, dmlab]

Typical usage (called by main.py after all methods are trained):
    diversity = DiversityAnalysis(
        predictions=all_predictions,   # Dict[str, Tensor] — method -> (N,) int tensor
        confidences=all_confidences,   # Dict[str, Tensor] — method -> (N,) float tensor
        labels=test_labels,            # (N,) int tensor
    )
    sim_matrix = diversity.prediction_similarity_matrix()
    diversity.plot_similarity_heatmap(sim_matrix, diversity.method_names, "sim.png")

    correct_overlap = diversity.confident_correct_overlap(top_k=5000)
    diversity.plot_venn_diagram(correct_overlap, ['lora', 'houlsby_adapter', 'ssf'], "venn.png")
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method display names mapping internal keys to paper display names.
# Used for axis labels in heatmaps and Venn diagram set labels.
# Matches paper Table 1 and Figure 2 notation exactly.
# ---------------------------------------------------------------------------
METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "linear": "Linear",
    "full": "Full FT",
    "vpt_shallow": "VPT-Shallow",
    "vpt_deep": "VPT-Deep",
    "bitfit": "BitFit",
    "layernorm": "LayerNorm",
    "difffit": "DiffFit",
    "ssf": "SSF",
    "pfeiffer_adapter": "Pfeif. Adapter",
    "houlsby_adapter": "Houl. Adapter",
    "adaptformer": "AdaptFormer",
    "convpass": "ConvPass",
    "repadapter": "RepAdapter",
    "lora": "LoRA",
    "fact_tt": "FacT_TT",
    "fact_tk": "FacT_TK",
}

# ---------------------------------------------------------------------------
# Default representative methods for Venn diagrams.
# config.yaml: diversity_analysis.representative_methods: [lora, houlsby_adapter, ssf]
# Paper: "one method from each PEFT category (LoRA, Adapter, SSF)"
# ---------------------------------------------------------------------------
REPRESENTATIVE_METHODS: List[str] = ["lora", "houlsby_adapter", "ssf"]

# ---------------------------------------------------------------------------
# Default top-k for confidence-based overlap analysis.
# config.yaml: diversity_analysis.top_k_confident: 5000
# Paper: "5K most confident" / "5K least confident"
# ---------------------------------------------------------------------------
DEFAULT_TOP_K: int = 5000

# ---------------------------------------------------------------------------
# Figure size constants for consistent plot dimensions.
# ---------------------------------------------------------------------------
_HEATMAP_FIGSIZE_PER_METHOD: float = 0.75  # inches per method for square heatmap
_HEATMAP_MIN_FIGSIZE: float = 8.0          # minimum figure size in inches
_VENN_FIGSIZE: Tuple[float, float] = (8.0, 6.0)
_RANKING_FIGSIZE_BASE: Tuple[float, float] = (12.0, 8.0)

# ---------------------------------------------------------------------------
# Seaborn style for consistent plot appearance.
# ---------------------------------------------------------------------------
_SEABORN_STYLE: str = "whitegrid"
_HEATMAP_CMAP: str = "Blues"
_RANKING_CMAP: str = "YlOrRd"


class DiversityAnalysis:
    """Prediction diversity analysis for PEFT methods.

    Analyzes how differently PEFT methods predict on the same test set,
    even when they achieve similar accuracy. Provides:
    - Pairwise prediction similarity matrices (Figure 3a)
    - Confidence-based correct/wrong prediction overlap (Figures 1b, 3b)
    - Ranking frequency visualization (Figure 2)

    All tensor operations are performed on CPU to avoid GPU memory issues
    when processing full test sets across 14 methods simultaneously.

    Attributes:
        predictions: Dict mapping method name to (N,) int tensor of predicted
            class indices. All tensors are on CPU.
        confidences: Dict mapping method name to (N,) float tensor of max
            softmax confidence scores. All tensors are on CPU.
        labels: (N,) int tensor of ground-truth class indices. On CPU.
        method_names: Ordered list of method names from predictions.keys().
            Preserves insertion order for consistent matrix indexing.
        num_methods: Number of PEFT methods being compared.
        num_samples: Number of test samples N.
    """

    def __init__(
        self,
        predictions: Dict[str, torch.Tensor],
        confidences: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> None:
        """Initialises DiversityAnalysis with pre-computed predictions.

        Args:
            predictions: Dict mapping method name (e.g., 'lora', 'bitfit') to
                a 1D integer tensor of shape (N,) containing predicted class
                indices for all N test samples. Produced by
                evaluation/metrics.py's compute_predictions() method.
            confidences: Dict mapping method name to a 1D float tensor of
                shape (N,) containing the max softmax probability (confidence
                score) for each sample. Produced by compute_predictions().
                Note: This should be the MAX confidence per sample, i.e.,
                confidences_full.max(dim=1).values where confidences_full
                has shape (N, num_classes).
            labels: 1D integer tensor of shape (N,) containing ground-truth
                class indices. Produced by compute_predictions().

        Raises:
            ValueError: If predictions and confidences have different keys.
            ValueError: If any prediction tensor has a different length than labels.
            ValueError: If predictions is empty.
        """
        # ------------------------------------------------------------------
        # Input validation.
        # ------------------------------------------------------------------
        if len(predictions) == 0:
            raise ValueError(
                "predictions dict is empty. At least one method must be provided."
            )

        if set(predictions.keys()) != set(confidences.keys()):
            pred_keys: set = set(predictions.keys())
            conf_keys: set = set(confidences.keys())
            missing_in_conf: set = pred_keys - conf_keys
            missing_in_pred: set = conf_keys - pred_keys
            raise ValueError(
                f"predictions and confidences must have the same keys. "
                f"Missing in confidences: {missing_in_conf}. "
                f"Missing in predictions: {missing_in_pred}."
            )

        num_samples: int = len(labels)

        for method_name, pred_tensor in predictions.items():
            if len(pred_tensor) != num_samples:
                raise ValueError(
                    f"Prediction tensor for method '{method_name}' has length "
                    f"{len(pred_tensor)}, but labels has length {num_samples}. "
                    "All prediction tensors must have the same length as labels."
                )

        for method_name, conf_tensor in confidences.items():
            if len(conf_tensor) != num_samples:
                raise ValueError(
                    f"Confidence tensor for method '{method_name}' has length "
                    f"{len(conf_tensor)}, but labels has length {num_samples}. "
                    "All confidence tensors must have the same length as labels."
                )

        # ------------------------------------------------------------------
        # Store all tensors on CPU for consistent computation.
        # ------------------------------------------------------------------
        self.predictions: Dict[str, torch.Tensor] = {
            name: tensor.cpu() for name, tensor in predictions.items()
        }
        self.confidences: Dict[str, torch.Tensor] = {
            name: tensor.cpu() for name, tensor in confidences.items()
        }
        self.labels: torch.Tensor = labels.cpu()

        # Preserve insertion order for consistent matrix indexing.
        self.method_names: List[str] = list(predictions.keys())
        self.num_methods: int = len(self.method_names)
        self.num_samples: int = num_samples

        _logger.info(
            "DiversityAnalysis initialised: %d methods, %d test samples. "
            "Methods: %s",
            self.num_methods,
            self.num_samples,
            self.method_names,
        )

    # ------------------------------------------------------------------
    # Core analysis methods
    # ------------------------------------------------------------------

    def prediction_similarity_matrix(self) -> np.ndarray:
        """Computes the pairwise prediction similarity matrix.

        For each pair of methods (i, j), computes the fraction of test samples
        where both methods predict the same class. The diagonal is always 1.0
        (a method agrees with itself on all samples). The matrix is symmetric.

        Paper Figure 3a: "element (i,j) shows the percentage of samples that
        method_i and j predict the same. Although different methods achieve
        similar accuracy, they have diverse predictions."

        Paper finding: "most PEFT methods show diverse predictions in other
        datasets in VTAB-1K. In DTD and Retinopathy, most methods differ in
        about 20% of their predictions, while in DMLab, this difference
        increases to approximately 35%."

        Returns:
            np.ndarray of shape (num_methods, num_methods) with values in [0, 1].
            Element [i, j] = fraction of samples where method i and method j
            predict the same class. Diagonal = 1.0. Matrix is symmetric.

        Note:
            All computations are on CPU tensors. For 14 methods × 50K samples,
            this requires 14×14/2 = 98 pairwise comparisons, each O(N).
        """
        _logger.info(
            "Computing prediction similarity matrix for %d methods × %d samples.",
            self.num_methods,
            self.num_samples,
        )

        # Initialize similarity matrix.
        sim_matrix: np.ndarray = np.zeros(
            (self.num_methods, self.num_methods), dtype=np.float64
        )

        # Compute upper triangle (including diagonal) and mirror.
        for i in range(self.num_methods):
            method_i: str = self.method_names[i]
            pred_i: torch.Tensor = self.predictions[method_i]

            for j in range(i, self.num_methods):
                method_j: str = self.method_names[j]
                pred_j: torch.Tensor = self.predictions[method_j]

                if i == j:
                    # Diagonal: perfect agreement with itself.
                    sim_matrix[i, j] = 1.0
                else:
                    # Fraction of samples where both methods predict the same class.
                    agreement: float = (pred_i == pred_j).float().mean().item()
                    sim_matrix[i, j] = agreement
                    sim_matrix[j, i] = agreement  # Symmetric

                _logger.debug(
                    "Similarity(%s, %s) = %.4f",
                    method_i,
                    method_j,
                    sim_matrix[i, j],
                )

        _logger.info(
            "Prediction similarity matrix computed. "
            "Mean off-diagonal similarity: %.4f",
            float(np.mean(sim_matrix[np.triu_indices(self.num_methods, k=1)])),
        )

        return sim_matrix

    def confident_correct_overlap(
        self,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """Computes correct prediction overlap for the top-k most confident samples.

        For each method, selects the top_k samples with the highest confidence
        scores, finds which of those are correctly predicted, and returns the
        index sets for overlap analysis.

        Paper Figure 1b: "Correct prediction overlaps for the 5K most confident
        data." Paper Figure 3b(a): "correct predictions from the top 5K most
        confident samples for each method."

        Paper finding: "Since they make different predictions in both high and
        low-confidence regimes, this paves the way for new possibilities of
        using different PEFT methods to generate diverse pseudo-labels."

        Args:
            top_k: Number of most confident samples to consider per method.
                Default: 5000 (config.yaml: diversity_analysis.top_k_confident).
                If top_k > num_samples, uses all samples.

        Returns:
            Dict with keys:
            - 'sets': Dict[str, set] — maps method name to set of sample indices
              that are in the top-k most confident AND correctly predicted.
            - 'overlap_counts': Dict — pairwise and triple overlap counts for
              Venn diagram construction. Keys are tuples of method names.
            - 'method_names': List[str] — ordered list of method names.
            - 'top_k': int — the actual top_k used (may be < requested if
              num_samples < top_k).
            - 'mode': str — 'correct_confident' for identification.
        """
        actual_top_k: int = min(top_k, self.num_samples)

        if actual_top_k < top_k:
            _logger.warning(
                "Requested top_k=%d but only %d samples available. "
                "Using top_k=%d.",
                top_k,
                self.num_samples,
                actual_top_k,
            )

        _logger.info(
            "Computing confident correct overlap: top_k=%d, %d methods.",
            actual_top_k,
            self.num_methods,
        )

        # ------------------------------------------------------------------
        # Step 1: For each method, find the set of correctly predicted
        # sample indices among the top-k most confident samples.
        # ------------------------------------------------------------------
        correct_sets: Dict[str, Set[int]] = {}

        for method_name in self.method_names:
            conf: torch.Tensor = self.confidences[method_name]  # (N,)
            pred: torch.Tensor = self.predictions[method_name]  # (N,)

            # Sort by confidence descending; take top-k indices.
            sorted_indices: torch.Tensor = torch.argsort(conf, descending=True)
            top_k_indices: torch.Tensor = sorted_indices[:actual_top_k]  # (top_k,)

            # Find which top-k samples are correctly predicted.
            correct_mask: torch.Tensor = (
                pred[top_k_indices] == self.labels[top_k_indices]
            )  # (top_k,) bool
            correct_indices: torch.Tensor = top_k_indices[correct_mask]  # (<=top_k,)

            # Convert to Python set of integer indices.
            correct_sets[method_name] = set(correct_indices.tolist())

            _logger.debug(
                "Method '%s': %d / %d top-k samples correctly predicted.",
                method_name,
                len(correct_sets[method_name]),
                actual_top_k,
            )

        # ------------------------------------------------------------------
        # Step 2: Compute pairwise and triple overlap counts.
        # ------------------------------------------------------------------
        overlap_counts: Dict[Any, int] = self._compute_overlap_counts(correct_sets)

        return {
            "sets": correct_sets,
            "overlap_counts": overlap_counts,
            "method_names": list(self.method_names),
            "top_k": actual_top_k,
            "mode": "correct_confident",
        }

    def confident_wrong_overlap(
        self,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """Computes wrong prediction overlap for the bottom-k least confident samples.

        For each method, selects the top_k samples with the LOWEST confidence
        scores (least confident), finds which of those are incorrectly predicted,
        and returns the index sets for overlap analysis.

        Paper Figure 3b(b): "wrong predictions, selecting from the 5K least
        confident samples."

        Paper finding: Methods make different mistakes in the low-confidence
        regime, further demonstrating complementary inductive biases.

        Args:
            top_k: Number of least confident samples to consider per method.
                Default: 5000 (config.yaml: diversity_analysis.top_k_confident).
                If top_k > num_samples, uses all samples.

        Returns:
            Dict with same structure as confident_correct_overlap():
            - 'sets': Dict[str, set] — maps method name to set of sample indices
              that are in the bottom-k least confident AND incorrectly predicted.
            - 'overlap_counts': Dict — pairwise and triple overlap counts.
            - 'method_names': List[str] — ordered list of method names.
            - 'top_k': int — the actual top_k used.
            - 'mode': str — 'wrong_unconfident' for identification.
        """
        actual_top_k: int = min(top_k, self.num_samples)

        if actual_top_k < top_k:
            _logger.warning(
                "Requested top_k=%d but only %d samples available. "
                "Using top_k=%d.",
                top_k,
                self.num_samples,
                actual_top_k,
            )

        _logger.info(
            "Computing confident wrong overlap: bottom_k=%d, %d methods.",
            actual_top_k,
            self.num_methods,
        )

        # ------------------------------------------------------------------
        # Step 1: For each method, find the set of incorrectly predicted
        # sample indices among the bottom-k least confident samples.
        # ------------------------------------------------------------------
        wrong_sets: Dict[str, Set[int]] = {}

        for method_name in self.method_names:
            conf: torch.Tensor = self.confidences[method_name]  # (N,)
            pred: torch.Tensor = self.predictions[method_name]  # (N,)

            # Sort by confidence ascending (least confident first); take bottom-k.
            sorted_indices: torch.Tensor = torch.argsort(conf, descending=False)
            bottom_k_indices: torch.Tensor = sorted_indices[:actual_top_k]  # (top_k,)

            # Find which bottom-k samples are incorrectly predicted.
            wrong_mask: torch.Tensor = (
                pred[bottom_k_indices] != self.labels[bottom_k_indices]
            )  # (top_k,) bool
            wrong_indices: torch.Tensor = bottom_k_indices[wrong_mask]  # (<=top_k,)

            # Convert to Python set of integer indices.
            wrong_sets[method_name] = set(wrong_indices.tolist())

            _logger.debug(
                "Method '%s': %d / %d bottom-k samples incorrectly predicted.",
                method_name,
                len(wrong_sets[method_name]),
                actual_top_k,
            )

        # ------------------------------------------------------------------
        # Step 2: Compute pairwise and triple overlap counts.
        # ------------------------------------------------------------------
        overlap_counts: Dict[Any, int] = self._compute_overlap_counts(wrong_sets)

        return {
            "sets": wrong_sets,
            "overlap_counts": overlap_counts,
            "method_names": list(self.method_names),
            "top_k": actual_top_k,
            "mode": "wrong_unconfident",
        }

    # ------------------------------------------------------------------
    # Visualization methods
    # ------------------------------------------------------------------

    def plot_similarity_heatmap(
        self,
        matrix: np.ndarray,
        method_names: List[str],
        save_path: str,
        title: str = "Prediction Similarity Matrix",
        dataset_name: Optional[str] = None,
    ) -> None:
        """Plots and saves the prediction similarity matrix as a heatmap.

        Reproduces Figure 3a and Figure 13 (Appendix C) of the paper.
        Uses seaborn.heatmap with fixed [0, 1] scale, annotations, and
        human-readable method names on axes.

        Args:
            matrix: np.ndarray of shape (num_methods, num_methods) with values
                in [0, 1]. Produced by prediction_similarity_matrix().
            method_names: List of method name strings (internal keys) in the
                same order as the matrix rows/columns. Converted to display
                names using METHOD_DISPLAY_NAMES.
            save_path: File path to save the figure. Parent directory is
                created if it does not exist. Supports .png, .pdf, .svg.
            title: Plot title. Default: "Prediction Similarity Matrix".
            dataset_name: Optional dataset name appended to the title
                (e.g., "DTD", "Retinopathy", "DMLab").
        """
        # ------------------------------------------------------------------
        # Resolve display names for axis labels.
        # ------------------------------------------------------------------
        display_names: List[str] = [
            METHOD_DISPLAY_NAMES.get(name, name) for name in method_names
        ]

        # ------------------------------------------------------------------
        # Compute figure size based on number of methods.
        # ------------------------------------------------------------------
        n: int = len(method_names)
        fig_size: float = max(
            _HEATMAP_MIN_FIGSIZE,
            n * _HEATMAP_FIGSIZE_PER_METHOD,
        )

        # ------------------------------------------------------------------
        # Create figure and heatmap.
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(fig_size + 1.5, fig_size))

        sns.set_style(_SEABORN_STYLE)

        # Determine annotation format based on matrix size.
        # For large matrices (>10 methods), use 2 decimal places.
        # For small matrices, use 3 decimal places.
        fmt_str: str = ".2f" if n > 8 else ".3f"
        annot_fontsize: int = max(6, 10 - n // 3)

        heatmap = sns.heatmap(
            matrix,
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=fmt_str,
            annot_kws={"size": annot_fontsize},
            cmap=_HEATMAP_CMAP,
            xticklabels=display_names,
            yticklabels=display_names,
            square=True,
            linewidths=0.5,
            linecolor="lightgray",
            cbar_kws={"label": "Prediction Agreement", "shrink": 0.8},
        )

        # ------------------------------------------------------------------
        # Formatting.
        # ------------------------------------------------------------------
        full_title: str = title
        if dataset_name is not None:
            full_title = f"{title} — {dataset_name}"
        ax.set_title(full_title, fontsize=14, fontweight="bold", pad=12)

        # Rotate x-axis labels for readability.
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=max(7, 11 - n // 3),
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=max(7, 11 - n // 3),
        )

        ax.set_xlabel("Method", fontsize=11, labelpad=8)
        ax.set_ylabel("Method", fontsize=11, labelpad=8)

        plt.tight_layout()

        # ------------------------------------------------------------------
        # Save and close.
        # ------------------------------------------------------------------
        self._ensure_parent_dir(save_path)

        try:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            _logger.info("Similarity heatmap saved to: %s", save_path)
        except Exception as exc:  # pylint: disable=broad-except
            _logger.error("Failed to save similarity heatmap to %s: %s", save_path, exc)
        finally:
            plt.close(fig)

    def plot_venn_diagram(
        self,
        overlap_data: Dict[str, Any],
        method_names: List[str],
        save_path: str,
        title: Optional[str] = None,
    ) -> None:
        """Plots and saves a 3-way Venn diagram of prediction overlaps.

        Reproduces Figure 1b and Figure 3b of the paper. Uses matplotlib_venn
        to visualize the overlap between three representative PEFT methods.

        Paper: "For demonstration purposes, we select one method from each
        PEFT category (LoRA, Adapter, SSF) and they are FT on CIFAR-100
        in VTAB-1K."

        Args:
            overlap_data: Dict produced by confident_correct_overlap() or
                confident_wrong_overlap(). Must contain 'sets' key mapping
                method names to sets of sample indices.
            method_names: List of exactly 3 method name strings (internal keys)
                to include in the Venn diagram. These must be present in
                overlap_data['sets']. If a method is not found, falls back to
                the first 3 available methods.
                Default representative methods from config.yaml:
                    ['lora', 'houlsby_adapter', 'ssf']
            save_path: File path to save the figure. Parent directory is
                created if it does not exist.
            title: Optional plot title. If None, auto-generated from the
                overlap_data 'mode' field.

        Note:
            Requires matplotlib_venn to be installed:
                pip install matplotlib-venn==0.11.9
        """
        # ------------------------------------------------------------------
        # Step 1: Import matplotlib_venn (optional dependency).
        # ------------------------------------------------------------------
        try:
            from matplotlib_venn import venn3  # type: ignore
        except ImportError as exc:
            _logger.error(
                "matplotlib_venn is required for Venn diagram plotting. "
                "Install with: pip install matplotlib-venn==0.11.9. "
                "Error: %s",
                exc,
            )
            return

        # ------------------------------------------------------------------
        # Step 2: Resolve which 3 methods to plot.
        # ------------------------------------------------------------------
        available_sets: Dict[str, Set[int]] = overlap_data.get("sets", {})

        if not available_sets:
            _logger.error(
                "overlap_data['sets'] is empty. Cannot plot Venn diagram."
            )
            return

        # Filter requested methods to those available in overlap_data.
        valid_methods: List[str] = [
            m for m in method_names if m in available_sets
        ]

        if len(valid_methods) < 3:
            _logger.warning(
                "Only %d of the requested methods %s are available in overlap_data. "
                "Falling back to first 3 available methods: %s",
                len(valid_methods),
                method_names,
                list(available_sets.keys())[:3],
            )
            valid_methods = list(available_sets.keys())[:3]

        if len(valid_methods) < 3:
            _logger.error(
                "Need at least 3 methods for a 3-way Venn diagram, "
                "but only %d are available: %s. Skipping.",
                len(valid_methods),
                valid_methods,
            )
            return

        # Use exactly the first 3 valid methods.
        method_a: str = valid_methods[0]
        method_b: str = valid_methods[1]
        method_c: str = valid_methods[2]

        set_a: Set[int] = available_sets[method_a]
        set_b: Set[int] = available_sets[method_b]
        set_c: Set[int] = available_sets[method_c]

        # ------------------------------------------------------------------
        # Step 3: Compute the 7 subset sizes for venn3.
        # venn3 expects: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
        # where uppercase = in set, lowercase = not in set.
        # ------------------------------------------------------------------
        # Only in A (not B, not C)
        Abc: int = len(set_a - set_b - set_c)
        # Only in B (not A, not C)
        aBc: int = len(set_b - set_a - set_c)
        # In A and B but not C
        ABc: int = len((set_a & set_b) - set_c)
        # Only in C (not A, not B)
        abC: int = len(set_c - set_a - set_b)
        # In A and C but not B
        AbC: int = len((set_a & set_c) - set_b)
        # In B and C but not A
        aBC: int = len((set_b & set_c) - set_a)
        # In all three
        ABC: int = len(set_a & set_b & set_c)

        subsets: Tuple[int, ...] = (Abc, aBc, ABc, abC, AbC, aBC, ABC)

        _logger.info(
            "Venn diagram subsets for (%s, %s, %s): "
            "Abc=%d, aBc=%d, ABc=%d, abC=%d, AbC=%d, aBC=%d, ABC=%d",
            method_a, method_b, method_c,
            Abc, aBc, ABc, abC, AbC, aBC, ABC,
        )

        # ------------------------------------------------------------------
        # Step 4: Resolve display names for set labels.
        # ------------------------------------------------------------------
        label_a: str = METHOD_DISPLAY_NAMES.get(method_a, method_a)
        label_b: str = METHOD_DISPLAY_NAMES.get(method_b, method_b)
        label_c: str = METHOD_DISPLAY_NAMES.get(method_c, method_c)

        # ------------------------------------------------------------------
        # Step 5: Determine title.
        # ------------------------------------------------------------------
        mode: str = overlap_data.get("mode", "")
        top_k_used: int = overlap_data.get("top_k", DEFAULT_TOP_K)

        if title is None:
            if mode == "correct_confident":
                title = (
                    f"Correct Prediction Overlap\n"
                    f"(Top-{top_k_used} Most Confident Samples)"
                )
            elif mode == "wrong_unconfident":
                title = (
                    f"Wrong Prediction Overlap\n"
                    f"(Bottom-{top_k_used} Least Confident Samples)"
                )
            else:
                title = f"Prediction Overlap (Top-{top_k_used} Samples)"

        # ------------------------------------------------------------------
        # Step 6: Create and save the Venn diagram.
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=_VENN_FIGSIZE)

        venn_diagram = venn3(
            subsets=sub