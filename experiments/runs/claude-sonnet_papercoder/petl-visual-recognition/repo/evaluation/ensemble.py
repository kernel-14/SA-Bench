## evaluation/ensemble.py
"""Ensemble evaluation for the PEFT Visual Recognition reproduction study.

This module implements the ensemble evaluation described in Section 4 of the paper:
"Different PEFT Approaches Offer Complementary Information."

The core insight: despite similar individual accuracies, different PEFT methods
make diverse predictions — and this diversity can be exploited via ensemble
methods to consistently improve over any single method.

Paper: "The most straightforward approach is ensemble, e.g., majority vote over
methods. Figure 4 demonstrates the ensemble performance gain over all the PEFT
methods in each dataset, where we use the worst PEFT method as the baseline."

Config references (config.yaml):
    ensemble.strategy: average_logits
    ensemble.baseline: worst_method

Typical usage (called by main.py after all methods are trained):
    evaluator = EnsembleEvaluator(
        logits=all_logits,    # Dict[str, Tensor] — method -> (N, C) float tensor
        labels=test_labels,   # (N,) long tensor
    )
    ensemble_acc = evaluator.ensemble_accuracy()
    method_accs = evaluator.per_method_accuracy()
    gain = evaluator.gain_over_worst()

    # Batch evaluation across all VTAB-1K datasets (for Figure 4):
    results_df = evaluator.compute_all_gains(
        all_dataset_logits=all_logits_dict,
        all_labels=all_labels_dict,
    )
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    """Ensemble evaluation for multiple PEFT methods on a single test set.

    Implements majority vote and average-logits ensemble strategies, computes
    per-method accuracy, ensemble accuracy, and the gain of the ensemble over
    the worst single method.

    All tensor operations are performed on CPU. Input logits are expected as
    raw pre-softmax logits (not probabilities), consistent with the paper's
    "average logits" strategy (config.yaml: ensemble.strategy: average_logits).

    Attributes:
        logits: Dict mapping method name to (N, C) float tensor of raw logits.
            All tensors are on CPU.
        labels: (N,) long tensor of ground-truth class indices. On CPU.
        method_names: Sorted list of method name strings for deterministic ordering.
        num_samples: Number of test samples N.
        num_classes: Number of output classes C.
        num_methods: Number of PEFT methods being evaluated.
    """

    def __init__(
        self,
        logits: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> None:
        """Initialises EnsembleEvaluator with pre-computed logits and labels.

        Args:
            logits: Dict mapping method name (e.g., 'lora', 'bitfit') to a
                2D float tensor of shape (N, C) containing raw pre-softmax
                logits for all N test samples and C classes. Produced by
                collecting model outputs during evaluation. All tensors must
                have the same shape.
            labels: 1D long tensor of shape (N,) containing ground-truth
                class indices. Produced by evaluation/metrics.py's
                compute_predictions() method.

        Raises:
            ValueError: If logits dict is empty.
            ValueError: If any logit tensor does not have shape (N, C).
            ValueError: If labels does not have shape (N,).
            ValueError: If logit tensors have inconsistent shapes.
        """
        # ------------------------------------------------------------------
        # Input validation.
        # ------------------------------------------------------------------
        if len(logits) == 0:
            raise ValueError(
                "logits dict is empty. At least one method must be provided "
                "for ensemble evaluation."
            )

        # Move labels to CPU and validate shape.
        labels_cpu: torch.Tensor = labels.cpu()
        if labels_cpu.dim() != 1:
            raise ValueError(
                f"labels must be a 1D tensor of shape (N,), "
                f"got shape {tuple(labels_cpu.shape)}."
            )

        num_samples: int = labels_cpu.shape[0]

        # Validate and move all logit tensors to CPU.
        logits_cpu: Dict[str, torch.Tensor] = {}
        num_classes: Optional[int] = None

        for method_name, logit_tensor in logits.items():
            tensor_cpu: torch.Tensor = logit_tensor.cpu()

            if tensor_cpu.dim() != 2:
                raise ValueError(
                    f"Logit tensor for method '{method_name}' must be 2D "
                    f"with shape (N, C), got shape {tuple(tensor_cpu.shape)}."
                )

            if tensor_cpu.shape[0] != num_samples:
                raise ValueError(
                    f"Logit tensor for method '{method_name}' has {tensor_cpu.shape[0]} "
                    f"samples, but labels has {num_samples} samples. "
                    "All logit tensors must have the same number of samples as labels."
                )

            if num_classes is None:
                num_classes = tensor_cpu.shape[1]
            elif tensor_cpu.shape[1] != num_classes:
                raise ValueError(
                    f"Logit tensor for method '{method_name}' has {tensor_cpu.shape[1]} "
                    f"classes, but previous tensors have {num_classes} classes. "
                    "All logit tensors must have the same number of classes."
                )

            logits_cpu[method_name] = tensor_cpu

        # ------------------------------------------------------------------
        # Store validated state.
        # ------------------------------------------------------------------
        self.logits: Dict[str, torch.Tensor] = logits_cpu
        self.labels: torch.Tensor = labels_cpu

        # Sorted method names for deterministic ordering across all methods.
        # Deterministic ordering ensures consistent column ordering in DataFrames
        # and consistent matrix indexing in per_method_accuracy().
        self.method_names: List[str] = sorted(logits_cpu.keys())

        self.num_samples: int = num_samples
        self.num_classes: int = num_classes if num_classes is not None else 0
        self.num_methods: int = len(self.method_names)

        _logger.info(
            "EnsembleEvaluator initialised: %d methods, %d samples, %d classes. "
            "Methods: %s",
            self.num_methods,
            self.num_samples,
            self.num_classes,
            self.method_names,
        )

    # ------------------------------------------------------------------
    # Ensemble prediction methods
    # ------------------------------------------------------------------

    def majority_vote(self) -> torch.Tensor:
        """Computes ensemble predictions via majority vote across all methods.

        For each sample, each method casts a vote for its predicted class
        (argmax of logits). The class receiving the most votes wins. Ties
        are broken by selecting the smallest class index (scipy.stats.mode
        default behavior with method='low').

        This implements the hard-voting ensemble strategy. The paper also
        discusses "average logits" (soft voting) — see average_logits().

        Returns:
            (N,) LongTensor of predicted class indices, one per sample.
            Each prediction is the modal class across all num_methods methods.
        """
        if self.num_samples == 0:
            _logger.warning(
                "majority_vote called with 0 samples. Returning empty tensor."
            )
            return torch.zeros(0, dtype=torch.long)

        # ------------------------------------------------------------------
        # Step 1: Compute per-method predictions via argmax.
        # Stack into (num_methods, N) matrix.
        # ------------------------------------------------------------------
        per_method_preds: List[np.ndarray] = []

        for method_name in self.method_names:
            logit_tensor: torch.Tensor = self.logits[method_name]
            preds: torch.Tensor = logit_tensor.argmax(dim=1)  # (N,)
            per_method_preds.append(preds.numpy())

        # Stack: (num_methods, N)
        predictions_matrix: np.ndarray = np.stack(per_method_preds, axis=0)

        # ------------------------------------------------------------------
        # Step 2: Compute mode across methods (axis=0) for each sample.
        # scipy.stats.mode returns the smallest value in case of ties.
        # ------------------------------------------------------------------
        # scipy >= 1.9.0 changed the keepdims default; use explicit keepdims=False.
        mode_result = stats.mode(predictions_matrix, axis=0, keepdims=False)

        # mode_result.mode has shape (N,) — the modal prediction per sample.
        modal_preds: np.ndarray = mode_result.mode.astype(np.int64)

        # ------------------------------------------------------------------
        # Step 3: Convert back to torch LongTensor.
        # ------------------------------------------------------------------
        ensemble_predictions: torch.Tensor = torch.from_numpy(modal_preds).long()

        _logger.debug(
            "majority_vote: computed ensemble predictions for %d samples "
            "via majority vote across %d methods.",
            self.num_samples,
            self.num_methods,
        )

        return ensemble_predictions

    def average_logits(self) -> torch.Tensor:
        """Computes ensemble predictions by averaging raw logits across methods.

        Averages the pre-softmax logits from all methods, then takes argmax.
        This is the "soft voting" or "logit averaging" strategy.

        Paper config: ensemble.strategy: average_logits
        Paper: "An ensemble prediction is generated based on the average logits
        of all PEFT methods for each test sample." (Figure 4 details, Appendix C)

        Averaging raw logits (before softmax) is equivalent to a geometric mean
        of the predicted probability distributions, which is a well-established
        ensemble technique that typically outperforms hard majority voting.

        Returns:
            (N,) LongTensor of predicted class indices, one per sample.
            Each prediction is the argmax of the mean logits across all methods.
        """
        if self.num_samples == 0:
            _logger.warning(
                "average_logits called with 0 samples. Returning empty tensor."
            )
            return torch.zeros(0, dtype=torch.long)

        # ------------------------------------------------------------------
        # Step 1: Stack all logit tensors into (num_methods, N, C).
        # ------------------------------------------------------------------
        logit_tensors: List[torch.Tensor] = [
            self.logits[method_name] for method_name in self.method_names
        ]
        stacked_logits: torch.Tensor = torch.stack(logit_tensors, dim=0)
        # stacked_logits shape: (num_methods, N, C)

        # ------------------------------------------------------------------
        # Step 2: Average across methods (dim=0) → (N, C).
        # ------------------------------------------------------------------
        mean_logits: torch.Tensor = stacked_logits.mean(dim=0)
        # mean_logits shape: (N, C)

        # ------------------------------------------------------------------
        # Step 3: Argmax to get ensemble predictions → (N,).
        # ------------------------------------------------------------------
        ensemble_predictions: torch.Tensor = mean_logits.argmax(dim=1)
        # ensemble_predictions shape: (N,)

        _logger.debug(
            "average_logits: computed ensemble predictions for %d samples "
            "by averaging logits across %d methods.",
            self.num_samples,
            self.num_methods,
        )

        return ensemble_predictions

    # ------------------------------------------------------------------
    # Accuracy computation methods
    # ------------------------------------------------------------------

    def ensemble_accuracy(self) -> float:
        """Computes the ensemble Top-1 accuracy using average-logits strategy.

        Uses the average_logits() strategy (config.yaml: ensemble.strategy:
        average_logits) to generate ensemble predictions, then computes the
        fraction of correctly predicted samples.

        Paper Figure 4: The ensemble accuracy is the primary metric shown,
        compared against individual method accuracies and the worst-method
        baseline.

        Returns:
            Ensemble Top-1 accuracy as a Python float in [0.0, 1.0].
            Returns 0.0 if num_samples == 0.
        """
        if self.num_samples == 0:
            _logger.warning(
                "ensemble_accuracy called with 0 samples. Returning 0.0."
            )
            return 0.0

        # Use average_logits strategy (config.yaml: ensemble.strategy: average_logits).
        ensemble_preds: torch.Tensor = self.average_logits()

        # Compute accuracy: fraction of correct predictions.
        correct: torch.Tensor = (ensemble_preds == self.labels).float()
        accuracy: float = correct.mean().item()

        _logger.debug(
            "ensemble_accuracy (average_logits): %.4f (%.2f%%)",
            accuracy,
            accuracy * 100,
        )

        return accuracy

    def per_method_accuracy(self) -> Dict[str, float]:
        """Computes Top-1 accuracy for each individual PEFT method.

        For each method, computes argmax of its logits and compares against
        ground-truth labels. Returns a dict with consistent ordering
        (sorted method names) for reproducible downstream processing.

        Used by:
        - gain_over_worst(): to find the minimum accuracy (worst method)
        - compute_all_gains(): to populate per-method columns in the DataFrame
        - main.py: for logging individual method performance

        Returns:
            Dict mapping method name to Top-1 accuracy float in [0.0, 1.0].
            Keys are sorted alphabetically (self.method_names order).
            Example: {'adaptformer': 0.821, 'bitfit': 0.756, 'lora': 0.826, ...}
            Returns empty dict if num_samples == 0.
        """
        if self.num_samples == 0:
            _logger.warning(
                "per_method_accuracy called with 0 samples. Returning empty dict."
            )
            return {}

        method_accs: Dict[str, float] = {}

        for method_name in self.method_names:
            logit_tensor: torch.Tensor = self.logits[method_name]

            # Compute predictions via argmax.
            preds: torch.Tensor = logit_tensor.argmax(dim=1)  # (N,)

            # Compute accuracy.
            correct: torch.Tensor = (preds == self.labels).float()
            accuracy: float = correct.mean().item()

            method_accs[method_name] = accuracy

            _logger.debug(
                "Method '%s' accuracy: %.4f (%.2f%%)",
                method_name,
                accuracy,
                accuracy * 100,
            )

        return method_accs

    def gain_over_worst(self) -> float:
        """Computes the ensemble accuracy gain over the worst single method.

        Computes: ensemble_accuracy() - min(per_method_accuracy().values())

        Paper: "we use the worst PEFT method as the baseline (∼). Each
        represents the relative performance of other PEFT methods compared
        to this baseline. An ensemble prediction (●) is generated based on
        the average logits of all PEFT methods for each test sample."
        (Figure 4 details, Appendix C)

        Config: ensemble.baseline: worst_method

        This metric directly corresponds to the Y-axis values in Figure 4,
        showing how much the ensemble improves over the worst single method
        on each VTAB-1K dataset.

        Returns:
            Accuracy gain as a Python float. Positive values indicate the
            ensemble outperforms the worst method. Returns 0.0 if there are
            no methods or no samples.
        """
        if self.num_samples == 0 or self.num_methods == 0:
            _logger.warning(
                "gain_over_worst called with 0 samples or 0 methods. "
                "Returning 0.0."
            )
            return 0.0

        # Compute ensemble accuracy (average logits strategy).
        ensemble_acc: float = self.ensemble_accuracy()

        # Compute per-method accuracies.
        method_accs: Dict[str, float] = self.per_method_accuracy()

        if not method_accs:
            _logger.warning(
                "per_method_accuracy returned empty dict. Returning 0.0."
            )
            return 0.0

        # Find the worst (minimum) accuracy across all methods.
        worst_acc: float = min(method_accs.values())
        worst_method: str = min(method_accs, key=lambda k: method_accs[k])

        gain: float = ensemble_acc - worst_acc

        _logger.info(
            "Ensemble gain over worst method: %.4f (%.2f%%). "
            "Ensemble acc=%.4f, worst method='%s' acc=%.4f.",
            gain,
            gain * 100,
            ensemble_acc,
            worst_method,
            worst_acc,
        )

        return gain

    # ------------------------------------------------------------------
    # Batch evaluation across all datasets
    # ------------------------------------------------------------------

    def compute_all_gains(
        self,
        all_dataset_logits: Dict[str, Dict[str, torch.Tensor]],
        all_labels: Dict[str, torch.Tensor],
    ) -> pd.DataFrame:
        """Computes ensemble gains across all VTAB-1K datasets for Figure 4.

        Creates a fresh EnsembleEvaluator instance per dataset and computes
        per-method accuracies, ensemble accuracy, and gain over worst method.
        Returns a DataFrame where each row corresponds to one dataset.

        Paper Figure 4: "Ensemble (majority vote) shows consistent gain on
        most datasets thanks to the diverse predictions."
        Note: The paper uses "majority vote" in the figure caption but the
        config specifies "average_logits" as the strategy. We use average_logits
        (soft voting) which is generally superior and matches the config.

        Args:
            all_dataset_logits: Nested dict mapping dataset name to a dict of
                method logits. Structure:
                    {
                        'dtd': {'lora': Tensor(N,C), 'bitfit': Tensor(N,C), ...},
                        'caltech101': {'lora': Tensor(N,C), ...},
                        ...
                    }
                Each inner dict maps method name to (N, C) float tensor of
                raw logits. N may differ across datasets (different test set sizes).
                Missing methods for a dataset result in NaN in the DataFrame.
            all_labels: Dict mapping dataset name to (N,) long tensor of
                ground-truth class indices. Must have the same keys as
                all_dataset_logits.

        Returns:
            pd.DataFrame with one row per dataset and columns:
            - 'dataset': dataset name string
            - One column per method (sorted alphabetically): per-method accuracy
              as float in [0.0, 1.0]. NaN if method not available for dataset.
            - 'ensemble_acc': ensemble accuracy (average_logits strategy)
            - 'gain': ensemble_acc - min(per_method_accs) (gain over worst)

            Example columns for 3 methods:
                ['dataset', 'adaptformer', 'bitfit', 'lora', 'ensemble_acc', 'gain']

            Rows are ordered by the iteration order of all_dataset_logits.

        Note:
            Datasets present in all_dataset_logits but not in all_labels are
            skipped with a warning. Datasets with empty logits dicts are also
            skipped.
        """
        if not all_dataset_logits:
            _logger.warning(
                "compute_all_gains called with empty all_dataset_logits. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        # ------------------------------------------------------------------
        # Step 1: Collect all unique method names across all datasets for
        # consistent column ordering in the output DataFrame.
        # ------------------------------------------------------------------
        all_method_names_set: set = set()
        for dataset_logits in all_dataset_logits.values():
            all_method_names_set.update(dataset_logits.keys())

        all_method_names_sorted: List[str] = sorted(all_method_names_set)

        _logger.info(
            "compute_all_gains: processing %d datasets, %d unique methods. "
            "Datasets: %s",
            len(all_dataset_logits),
            len(all_method_names_sorted),
            list(all_dataset_logits.keys()),
        )

        # ------------------------------------------------------------------
        # Step 2: Process each dataset.
        # ------------------------------------------------------------------
        rows: List[Dict[str, Any]] = []

        for dataset_name, dataset_logits in all_dataset_logits.items():
            # ------------------------------------------------------------------
            # Validate that labels are available for this dataset.
            # ------------------------------------------------------------------
            if dataset_name not in all_labels:
                _logger.warning(
                    "Dataset '%s' found in all_dataset_logits but not in "
                    "all_labels. Skipping.",
                    dataset_name,
                )
                continue

            dataset_labels: torch.Tensor = all_labels[dataset_name]

            # ------------------------------------------------------------------
            # Skip datasets with empty logits.
            # ------------------------------------------------------------------
            if not dataset_logits:
                _logger.warning(
                    "Dataset '%s' has empty logits dict. Skipping.",
                    dataset_name,
                )
                continue

            # ------------------------------------------------------------------
            # Create a fresh EnsembleEvaluator for this dataset.
            # ------------------------------------------------------------------
            try:
                evaluator: EnsembleEvaluator = EnsembleEvaluator(
                    logits=dataset_logits,
                    labels=dataset_labels,
                )
            except ValueError as exc:
                _logger.error(
                    "Failed to create EnsembleEvaluator for dataset '%s': %s. "
                    "Skipping.",
                    dataset_name,
                    exc,
                )
                continue

            # ------------------------------------------------------------------
            # Compute per-method accuracies.
            # ------------------------------------------------------------------
            method_accs: Dict[str, float] = evaluator.per_method_accuracy()

            # ------------------------------------------------------------------
            # Compute ensemble accuracy (average_logits strategy per config).
            # ------------------------------------------------------------------
            ensemble_acc: float = evaluator.ensemble_accuracy()

            # ------------------------------------------------------------------
            # Compute gain over worst method.
            # ------------------------------------------------------------------
            gain: float = evaluator.gain_over_worst()

            # ------------------------------------------------------------------
            # Build the row dict.
            # Use NaN for methods not available for this dataset.
            # ------------------------------------------------------------------
            row: Dict[str, Any] = {"dataset": dataset_name}

            for method_name in all_method_names_sorted:
                row[method_name] = method_accs.get(method_name, float("nan"))

            row["ensemble_acc"] = ensemble_acc
            row["gain"] = gain

            rows.append(row)

            _logger.info(
                "Dataset '%s': ensemble_acc=%.4f (%.2f%%), "
                "gain=%.4f (%.2f%%), num_methods=%d.",
                dataset_name,
                ensemble_acc,
                ensemble_acc * 100,
                gain,
                gain * 100,
                len(method_accs),
            )

        # ------------------------------------------------------------------
        # Step 3: Build DataFrame with consistent column ordering.
        # ------------------------------------------------------------------
        if not rows:
            _logger.warning(
                "compute_all_gains: no valid datasets processed. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        df: pd.DataFrame = pd.DataFrame(rows)

        # ------------------------------------------------------------------
        # Step 4: Enforce consistent column ordering.
        # Order: ['dataset'] + sorted_method_names + ['ensemble_acc', 'gain']
        # Only include columns that actually exist in the DataFrame.
        # ------------------------------------------------------------------
        desired_column_order: List[str] = (
            ["dataset"]
            + [m for m in all_method_names_sorted if m in df.columns]
            + [col for col in ["ensemble_acc", "gain"] if col in df.columns]
        )

        # Filter to only columns present in the DataFrame (handles edge cases).
        final_column_order: List[str] = [
            col for col in desired_column_order if col in df.columns
        ]

        df = df[final_column_order]

        _logger.info(
            "compute_all_gains complete: DataFrame shape=%s, "
            "columns=%s",
            df.shape,
            list(df.columns),
        )

        return df
