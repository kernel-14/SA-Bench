## Code: evaluation/metrics.py
```python
## evaluation/metrics.py
"""Accuracy computation and evaluation metrics for the PEFT Visual Recognition study.

This module provides the Metrics class — the central accuracy computation hub
used by Trainer (validation), HyperparamSearch (trial evaluation), WiSE
(robustness sweep), DiversityAnalysis (predictions), and main.py (final test
evaluation).

All methods are stateless and operate on tensors or dicts passed in. No
internal state persists between calls.

Paper reference: "Lessons and Insights from a Unifying Study of Parameter-
Efficient Fine-Tuning (PEFT) in Visual Recognition"

Config reference (config.yaml):
    vtab.task_groups: defines the 19 VTAB-1K tasks and their group assignments
    compute.mixed_precision: true — used in compute_predictions for autocast

Typical usage:
    metrics = Metrics()

    # During training (Trainer.validate):
    preds = logits.argmax(dim=1)
    acc = metrics.top1_accuracy(preds, labels)

    # After training (main.py):
    predictions, confidences, labels = metrics.compute_predictions(
        model, test_loader, device='cuda'
    )
    test_acc = metrics.top1_accuracy(predictions, labels)

    # VTAB-1K group reporting:
    group_avgs = metrics.compute_group_avg(task_accs, VTAB_GROUPS)
    rsd = metrics.relative_std_dev(method_accs_for_task)
    rank_matrix = metrics.ranking_frequency(method_accs_dict, num_methods=15)
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VTAB-1K task group definitions.
# Mirrors config.yaml: vtab.task_groups exactly.
# Also defined in datasets/vtab_loader.py — duplicated here to avoid circular
# imports (evaluation/metrics.py is imported by training/trainer.py which
# does not import dataset loaders).
# ---------------------------------------------------------------------------
VTAB_GROUPS: Dict[str, List[str]] = {
    "natural": [
        "caltech101",
        "cifar100",
        "dtd",
        "flowers102",
        "pets",
        "sun397",
        "svhn",
    ],
    "specialized": [
        "camelyon",
        "eurosat",
        "resisc45",
        "retinopathy",
    ],
    "structured": [
        "clevr_count",
        "clevr_distance",
        "dmlab",
        "dsprites_loc",
        "dsprites_ori",
        "smallnorb_azimuth",
        "smallnorb_elevation",
        "kitti",
    ],
}

# Flat list of all 19 VTAB task names in canonical order (matches Table 1).
VTAB_ALL_TASKS: List[str] = (
    VTAB_GROUPS["natural"]
    + VTAB_GROUPS["specialized"]
    + VTAB_GROUPS["structured"]
)

# ---------------------------------------------------------------------------
# Number of classes per VTAB task (config.yaml: vtab.num_classes).
# Used for validation in compute_predictions.
# ---------------------------------------------------------------------------
VTAB_NUM_CLASSES: Dict[str, int] = {
    "caltech101": 102,
    "cifar100": 100,
    "dtd": 47,
    "flowers102": 102,
    "pets": 37,
    "sun397": 397,
    "svhn": 10,
    "camelyon": 2,
    "eurosat": 10,
    "resisc45": 45,
    "retinopathy": 5,
    "clevr_count": 8,
    "clevr_distance": 6,
    "dmlab": 6,
    "dsprites_loc": 16,
    "dsprites_ori": 16,
    "smallnorb_azimuth": 18,
    "smallnorb_elevation": 9,
    "kitti": 4,
}


class Metrics:
    """Stateless evaluation metrics for PEFT experiments.

    Provides accuracy computation, prediction collection, VTAB-1K group
    averaging, relative standard deviation, and ranking frequency matrix
    construction. All methods operate on inputs passed in — no instance
    state is maintained between calls.

    Designed to be instantiated once and reused across all experiments:
        metrics = Metrics()

    Attributes:
        None — this class is stateless.
    """

    def __init__(self) -> None:
        """Initialises the Metrics instance. No state is maintained."""
        pass

    # ------------------------------------------------------------------
    # Core accuracy methods
    # ------------------------------------------------------------------

    def top1_accuracy(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Computes Top-1 classification accuracy.

        Computes the fraction of samples where the predicted class matches
        the ground-truth label. Returns a Python float in [0.0, 1.0].

        The paper reports accuracy as a percentage (e.g., 72.1%). The
        conversion to percentage is the caller's responsibility — this
        method returns the raw fraction (e.g., 0.721).

        Args:
            preds: Predicted class indices, shape (N,), dtype torch.long.
                Typically obtained via ``logits.argmax(dim=1)``.
            labels: Ground-truth class indices, shape (N,), dtype torch.long.
                Must have the same length as preds.

        Returns:
            Top-1 accuracy as a Python float in [0.0, 1.0].
            Returns 0.0 if preds is empty (N=0).

        Raises:
            ValueError: If preds and labels have different lengths.
        """
        if preds.numel() == 0:
            _logger.warning(
                "top1_accuracy called with empty predictions tensor. Returning 0.0."
            )
            return 0.0

        if preds.shape[0] != labels.shape[0]:
            raise ValueError(
                f"preds and labels must have the same length. "
                f"Got preds.shape={preds.shape}, labels.shape={labels.shape}."
            )

        # Move both tensors to CPU for consistent computation.
        # compute_predictions always returns CPU tensors, but callers from
        # Trainer may pass GPU tensors directly.
        preds_cpu: torch.Tensor = preds.cpu()
        labels_cpu: torch.Tensor = labels.cpu()

        # Compute accuracy: fraction of correct predictions.
        correct: torch.Tensor = (preds_cpu == labels_cpu).float()
        accuracy: float = correct.mean().item()

        return accuracy

    def compute_predictions(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs inference on a full dataset split and collects predictions.

        Iterates the DataLoader in evaluation mode (no gradients, model.eval()),
        collects logits, computes softmax confidences and argmax predictions,
        and returns all results as CPU tensors.

        This is the primary inference routine used by:
        - main.py: final test evaluation after training
        - training/wise.py: WiSE robustness sweep (one call per alpha)
        - evaluation/diversity.py: prediction similarity analysis
        - evaluation/ensemble.py: ensemble evaluation (via main.py)

        The returned confidences tensor contains the full softmax distribution
        (N, num_classes), not just the max confidence. This is required by
        DiversityAnalysis.confident_correct_overlap() which ranks samples by
        their max confidence: ``confidences.max(dim=1).values``.

        Paper Section 4: "we selected the correct predictions from the top 5K
        most confident samples for each method" — requires per-sample max
        confidence values.

        Args:
            model: The PEFTModel (or any nn.Module) to evaluate. Will be set
                to eval mode and moved to device. The model is NOT restored to
                its original mode after this call — the caller should call
                model.train() if needed.
            loader: DataLoader returning (images, labels) batches. Images
                should be preprocessed (normalized, resized to 224×224).
            device: Target device for inference. Default: 'cuda'.
                If 'cuda' is specified but unavailable, falls back to 'cpu'
                with a warning.

        Returns:
            A tuple (predictions, confidences, labels) where:
            - predictions: shape (N,), dtype torch.long — predicted class indices.
            - confidences: shape (N, num_classes), dtype torch.float32 — softmax
              probabilities for each class. Used for confidence-based analysis.
            - labels: shape (N,), dtype torch.long — ground-truth class indices.
            All tensors are on CPU.

        Note:
            For large datasets (e.g., ImageNet-1K with 50K samples, 1000 classes),
            the confidences tensor is approximately 50000 × 1000 × 4 bytes ≈ 200MB.
            For VTAB-1K test sets (typically 2K–26K samples), this is negligible.
        """
        # ------------------------------------------------------------------
        # Step 1: Resolve device (fall back to CPU if CUDA unavailable).
        # ------------------------------------------------------------------
        resolved_device: str = device
        if device == "cuda" and not torch.cuda.is_available():
            _logger.warning(
                "CUDA requested but not available. Falling back to CPU for inference."
            )
            resolved_device = "cpu"

        # ------------------------------------------------------------------
        # Step 2: Set model to eval mode and move to device.
        # ------------------------------------------------------------------
        model.to(resolved_device)
        model.eval()

        # ------------------------------------------------------------------
        # Step 3: Determine whether to use mixed precision.
        # config.yaml: compute.mixed_precision: true
        # Only use autocast on CUDA; CPU autocast is not beneficial.
        # ------------------------------------------------------------------
        use_autocast: bool = resolved_device == "cuda"

        # ------------------------------------------------------------------
        # Step 4: Collect predictions, confidences, and labels.
        # ------------------------------------------------------------------
        all_predictions: List[torch.Tensor] = []
        all_confidences: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        num_batches: int = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Handle both (images, labels) and (images, labels, *extra) formats.
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images: torch.Tensor = batch[0]
                    labels: torch.Tensor = batch[1]
                else:
                    raise ValueError(
                        f"Unexpected batch format from DataLoader. "
                        f"Expected (images, labels) tuple, got type {type(batch)}."
                    )

                # Move to device.
                images = images.to(resolved_device, non_blocking=True)
                labels = labels.to(resolved_device, non_blocking=True)

                # Forward pass with optional mixed precision.
                if use_autocast:
                    with torch.cuda.amp.autocast():
                        logits: torch.Tensor = model(images)
                else:
                    logits = model(images)

                # Compute softmax confidences: (B, num_classes)
                confidences: torch.Tensor = torch.softmax(logits, dim=1)

                # Compute predictions: (B,) — argmax is equivalent on logits or probs.
                predictions: torch.Tensor = logits.argmax(dim=1)

                # Accumulate on CPU to avoid GPU memory accumulation.
                all_predictions.append(predictions.cpu())
                all_confidences.append(confidences.cpu())
                all_labels.append(labels.cpu())

                num_batches += 1

        # ------------------------------------------------------------------
        # Step 5: Handle empty loader.
        # ------------------------------------------------------------------
        if num_batches == 0:
            _logger.warning(
                "compute_predictions: DataLoader yielded 0 batches. "
                "Returning empty tensors."
            )
            empty_preds: torch.Tensor = torch.zeros(0, dtype=torch.long)
            empty_confs: torch.Tensor = torch.zeros(0, 0, dtype=torch.float32)
            empty_labels: torch.Tensor = torch.zeros(0, dtype=torch.long)
            return empty_preds, empty_confs, empty_labels

        # ------------------------------------------------------------------
        # Step 6: Concatenate all batches.
        # ------------------------------------------------------------------
        final_predictions: torch.Tensor = torch.cat(all_predictions, dim=0)
        final_confidences: torch.Tensor = torch.cat(all_confidences, dim=0)
        final_labels: torch.Tensor = torch.cat(all_labels, dim=0)

        total_samples: int = final_predictions.shape[0]
        _logger.info(
            "compute_predictions complete: %d samples, %d batches, "
            "predictions shape=%s, confidences shape=%s.",
            total_samples,
            num_batches,
            tuple(final_predictions.shape),
            tuple(final_confidences.shape),
        )

        return final_predictions, final_confidences, final_labels

    # ------------------------------------------------------------------
    # VTAB-1K group reporting methods
    # ------------------------------------------------------------------

    def compute_group_avg(
        self,
        task_accs: Dict[str, float],
        group_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, float]:
        """Computes per-group average accuracy for VTAB-1K reporting.

        Averages per-task accuracies within each group (Natural, Specialized,
        Structured) and computes an overall mean across all available tasks.

        Paper Table 1 reports per-task accuracy and the paper discusses
        group-level patterns in Section 3. The group averages are used in
        Figure 1a (accuracy gain vs. linear probing) and in the analysis
        of Section 3.

        The "overall" mean is computed as the arithmetic mean of all 19
        individual task accuracies (not the mean of group means), since
        groups have different sizes (7, 4, 8 tasks respectively).

        Args:
            task_accs: Dict mapping task name (e.g., "dtd") to accuracy float
                (e.g., 0.722 for 72.2%). Accuracies should be in [0, 1] range.
                Tasks not present in group_map are ignored.
            group_map: Dict mapping group name to list of task names.
                Default: VTAB_GROUPS (from config.yaml: vtab.task_groups).
                Callers can pass a custom group_map for non-VTAB experiments.

        Returns:
            Dict with group names as keys and average accuracy floats as values.
            Always includes an "overall" key with the mean across all tasks
            present in task_accs. Example:
                {
                    "natural": 0.756,
                    "specialized": 0.812,
                    "structured": 0.634,
                    "overall": 0.720,
                }
            Groups with no tasks present in task_accs are omitted from the
            returned dict (not set to 0.0 or None).

        Note:
            Accuracies in task_accs are expected in [0, 1] range (fractions).
            The returned values are also in [0, 1] range. Conversion to
            percentage (×100) is the caller's responsibility.
        """
        if group_map is None:
            group_map = VTAB_GROUPS

        result: Dict[str, float] = {}
        all_available_accs: List[float] = []

        for group_name, task_names in group_map.items():
            group_accs: List[float] = []

            for task_name in task_names:
                if task_name in task_accs:
                    group_accs.append(task_accs[task_name])
                    all_available_accs.append(task_accs[task_name])
                else:
                    _logger.debug(
                        "Task '%s' (group '%s') not found in task_accs. "
                        "Skipping for group average computation.",
                        task_name,
                        group_name,
                    )

            if group_accs:
                group_mean: float = float(np.mean(group_accs))
                result[group_name] = group_mean
                _logger.debug(
                    "Group '%s': %d/%d tasks available, avg=%.4f (%.2f%%)",
                    group_name,
                    len(group_accs),
                    len(task_names),
                    group_mean,
                    group_mean * 100,
                )
            else:
                _logger.warning(
                    "Group '%s': no tasks found in task_accs. "
                    "Group omitted from results.",
                    group_name,
                )

        # Compute overall mean across all available tasks.
        if all_available_accs:
            overall_mean: float = float(np.mean(all_available_accs))
            result["overall"] = overall_mean
            _logger.info(
                "Group averages computed: %d tasks available, overall=%.4f (%.2f%%)",
                len(all_available_accs),
                overall_mean,
                overall_mean * 100,
            )
        else:
            _logger.warning(
                "compute_group_avg: no tasks found in task_accs. "
                "Returning empty dict."
            )

        return result

    def relative_std_dev(self, values: List[float]) -> float:
        """Computes the relative standard deviation (coefficient of variation).

        Computes: std(values) / mean(values) * 100

        Used to reproduce the "Relative Std Dev" row in Table 1, which
        quantifies how similar different PEFT methods are on each task/group.

        Paper: "most PEFT methods perform similarly as the relative standard
        deviations (divided by the means) in all three groups are quite low."
        (Section 3)

        The values in Table 1 are computed across the 14 PEFT methods
        (excluding linear probing and full FT) for each task. The caller
        is responsible for passing only the relevant method accuracies.

        Args:
            values: List of accuracy floats (one per PEFT method) for a given
                task or group. Typically in [0, 1] range (fractions) or
                [0, 100] range (percentages) — the RSD is scale-invariant.
                Must contain at least one value.

        Returns:
            Relative standard deviation as a Python float (percentage).
            Example: 0.81 means the std is 0.81% of the mean.
            Returns 0.0 if len(values) <= 1 (single method, no variation).
            Returns 0.0 if mean is 0 (degenerate case).

        Note:
            Uses population standard deviation (ddof=0), not sample std (ddof=1),
            consistent with the paper's reporting convention.
        """
        if len(values) == 0:
            _logger.warning(
                "relative_std_dev called with empty values list. Returning 0.0."
            )
            return 0.0

        if len(values) == 1:
            return 0.0

        arr: np.ndarray = np.array(values, dtype=np.float64)

        mean_val: float = float(np.mean(arr))
        std_val: float = float(np.std(arr, ddof=0))  # Population std

        if abs(mean_val) < 1e-10:
            _logger.warning(
                "relative_std_dev: mean is approximately zero (%.2e). "
                "Returning 0.0 to avoid division by zero.",
                mean_val,
            )
            return 0.0

        rsd: float = (std_val / abs(mean_val)) * 100.0

        return rsd

    def ranking_frequency(
        self,
        method_accs: Dict[str, List[float]],
        num_methods: int,
    ) -> np.ndarray:
        """Builds the ranking frequency matrix for Figure 2 reproduction.

        For each dataset in the group, ranks all methods by accuracy
        (rank 1 = highest accuracy). Accumulates rank counts into a
        (num_methods × num_methods) matrix where element (i, j) = number
        of times method i ranks j-th across all datasets.

        Paper Figure 2: "Element (i, j) is the number of times method i
        ranks j-th in each group. Methods are ordered by mean ranks
        (in brackets)."

        The row sums of the returned matrix equal the number of datasets
        in the group (e.g., 7 for Natural, 4 for Specialized, 8 for Structured).

        Args:
            method_accs: OrderedDict or regular dict mapping method name to
                a list of per-dataset accuracies. The list length must be
                consistent across all methods (= number of datasets in the group).
                Example for Natural group (7 datasets):
                    {
                        "bitfit":  [0.865, 0.905, 0.703, 0.989, 0.910, 0.912, 0.542],
                        "lora":    [0.857, 0.926, 0.698, 0.991, 0.905, 0.885, 0.555],
                        ...
                    }
                Methods should be in a consistent order (e.g., sorted by mean rank
                for display purposes, though this method does not enforce ordering).
            num_methods: Number of methods being ranked. Must equal
                len(method_accs). Used to set the matrix dimensions.
                Paper Figure 2: 15 methods (14 PEFT + linear probing).

        Returns:
            np.ndarray of shape (num_methods, num_methods) with dtype int32.
            Element [i, j] = number of times method i (in the order of
            method_accs.keys()) ranked j+1-th (1-indexed rank → 0-indexed column).
            Row sums equal the number of datasets in the group.

        Raises:
            ValueError: If num_methods != len(method_accs).
            ValueError: If method accuracy lists have inconsistent lengths.
            ValueError: If method_accs is empty.

        Note:
            Ties are handled using scipy.stats.rankdata with method='min':
            tied methods both receive the lower (better) rank number.
            This matches the paper's intent where ties are rare.
        """
        # ------------------------------------------------------------------
        # Input validation.
        # ------------------------------------------------------------------
        if len(method_accs) == 0:
            raise ValueError(
                "method_accs is empty. Cannot compute ranking frequency matrix."
            )

        if num_methods != len(method_accs):
            raise ValueError(
                f"num_methods ({num_methods}) must equal len(method_accs) "
                f"({len(method_accs)}). "
                "Ensure all methods are included in method_accs."
            )

        # Extract method names in consistent order.
        method_names: List[str] = list(method_accs.keys())

        # Validate that all methods have the same number of datasets.
        dataset_counts: List[int] = [len(accs) for accs in method_accs.values()]
        if len(set(dataset_counts)) > 1:
            raise ValueError(
                f"Inconsistent number of datasets across methods: {dataset_counts}. "
                "All methods must have accuracy lists of the same length."
            )

        num_datasets: int = dataset_counts[0]

        if num_datasets == 0:
            _logger.warning(
                "ranking_frequency: all methods have 0 datasets. "
                "Returning zero matrix."
            )
            return np.zeros((num_methods, num_methods), dtype=np.int32)

        # ------------------------------------------------------------------
        # Build the ranking frequency matrix.
        # ------------------------------------------------------------------
        rank_matrix: np.ndarray = np.zeros(
            (num_methods, num_methods), dtype=np.int32
        )

        for dataset_idx in range(num_datasets):
            # Extract accuracy of each method for this dataset.
            accs_for_dataset: np.ndarray = np.array(
                [method_accs[method_names[i]][dataset_idx] for i in range(num_methods)],
                dtype=np.float64,
            )

            # Rank methods by accuracy in descending order.
            # scipy.stats.rankdata ranks in ascending order by default.
            # We negate to rank in descending order (rank 1 = highest accuracy).
            # method='min': tied methods both get the lower (better) rank.
            ranks: np.ndarray = stats.rankdata(
                -accs_for_dataset, method="min"
            ).astype(np.int32)
            # ranks[i] is the rank of method i (1-indexed, 1 = best).

            # Accumulate rank counts.
            for method_idx in range(num_methods):
                rank_value: int = int(ranks[method_idx])  # 1-indexed rank

                # Convert to 0-indexed column: rank 1 → column 0.
                col_idx: int = rank_value - 1

                # Guard against out-of-bounds (can happen with ties when
                # rank > num_methods due to min-rank tie handling).
                if 0 <= col_idx < num_methods:
                    rank_matrix[method_idx, col_idx] += 1
                else:
                    _logger.warning(
                        "Rank %d for method '%s' on dataset %d is out of bounds "
                        "[0, %d). Skipping.",
                        rank_value,
                        method_names[method_idx],
                        dataset_idx,
                        num_methods,
                    )

        # ------------------------------------------------------------------
        # Validate: row sums should equal num_datasets.
        # ------------------------------------------------------------------
        row_sums: np.ndarray = rank_matrix.sum(axis=1)
        expected_sum: int = num_datasets

        if not np.all(row_sums == expected_sum):
            _logger.warning(
                "ranking_frequency: some row sums (%s) do not equal "
                "num_datasets=%d. This may indicate ties that pushed ranks "
                "beyond num_methods. Matrix may be slightly inaccurate.",
                row_sums.tolist(),
                expected_sum,
            )

        _logger.info(
            "Ranking frequency matrix computed: shape=%s, num_datasets=%d, "
            "num_methods=%d.",
            rank_matrix.shape,
            num_datasets,
            num_methods,
        )

        return rank_matrix

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------

    def compute_all_vtab_group_stats(
        self,
        all_method_task_accs: Dict[str, Dict[str, float]],
        group_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Computes group averages for all methods in a VTAB-1K experiment.

        Convenience wrapper that calls compute_group_avg for each method
        and assembles the results into a nested dict.

        Args:
            all_method_task_accs: Dict mapping method name to per-task accuracy
                dict. Example:
                    {
                        "lora": {"dtd": 0.698, "caltech101": 0.926, ...},
                        "bitfit": {"dtd": 0.703, "caltech101": 0.905, ...},
                        ...
                    }
            group_map: Group definition dict. Default: VTAB_GROUPS.

        Returns:
            Dict mapping method name to group average dict. Example:
                {
                    "lora": {"natural": 0.756, "specialized": 0.812, ..., "overall": 0.720},
                    "bitfit": {"natural": 0.754, ...},
                    ...
                }
        """
        if group_map is None:
            group_map = VTAB_GROUPS

        results: Dict[str, Dict[str, float]] = {}

        for method_name, task_accs in all_method_task_accs.items():
            results[method_name] = self.compute_group_avg(task_accs, group_map)

        return results

    def compute_vtab_relative_std_devs(
        self,
        all_method_task_accs: Dict[str, Dict[str, float]],
        group_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, float]:
        """Computes relative standard deviation per task across all methods.

        Reproduces the "Relative Std Dev" row in Table 1. For each task,
        computes the RSD across all methods' accuracies on that task.

        Args:
            all_method_task_accs: Dict mapping method name to per-task accuracy
                dict. Should include only the 14 PEFT methods (not linear
                probing or full FT) to match Table 1's convention.
            group_map: Group definition dict. Default: VTAB_GROUPS.
                Used to determine which tasks to include.

        Returns:
            Dict mapping task name to RSD float (percentage). Example:
                {"caltech101": 0.81, "dtd": 1.13, "clevr_count": 2.67, ...}
            Also includes group-level RSDs:
                {"natural_rsd": 1.24, "specialized_rsd": 0.83, ...}
        """
        if group_map is None:
            group_map = VTAB_GROUPS

        # Collect all task names from the group map.
        all_tasks: List[str] = []
        for task_list in group_map.values():
            all_tasks.extend(task_list)

        task_rsds: Dict[str, float] = {}

        for task_name in all_tasks:
            # Collect accuracies for this task across all methods.
            task_method_accs: List[float] = []
            for method_name, task_accs in all_method_task_accs.items():
                if task_name in task_accs:
                    task_method_accs.append(task_accs[task_name])

            if task_method_accs:
                task_rsds[task_name] = self.relative_std_dev(task_method_accs)
            else:
                _logger.debug(
                    "Task '%s' not found in any method's results. Skipping RSD.",
                    task_name,
                )

        # Compute group-level RSDs (RSD of group averages across methods).
        for group_name, task_list in group_map.items():
            group_method_avgs: List[float] = []

            for method_name, task_accs in all_method_task_accs.items():
                group_accs: List[float] = [
                    task_accs[t] for t in task_list if t in task_accs
                ]
                if group_accs:
                    group_method_avgs.append(float(np.mean(group_accs)))

            if group_method_avgs:
                task_rsds[f"{group_name}_rsd"] = self.relative_std_dev(
                    group_method_avgs
                )

        return task_rsds

    def build_ranking_frequency_per_group(
        self,
        all_method_task_accs: Dict[str, Dict[str, float]],
        group_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Builds ranking frequency matrices for all VTAB-1K groups.

        Convenience wrapper that calls ranking_frequency for each group
        separately, as shown in Figure 2 of the paper.

        Args:
            all_method_task_accs: Dict mapping method name to per-task accuracy