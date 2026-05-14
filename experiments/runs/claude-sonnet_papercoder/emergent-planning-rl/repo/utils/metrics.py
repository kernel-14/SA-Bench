## utils/metrics.py
"""Evaluation metrics utilities for the emergent planning interpretability pipeline.

This module provides the core evaluation primitives used throughout the probing
pipeline. The primary challenge addressed here is class imbalance: in Sokoban
episodes, most squares are assigned NEVER for both C_A and C_B concepts, making
accuracy a misleading metric. The paper (Section 4.2) explicitly uses macro F1
to give equal weight to each class regardless of frequency.

This module has zero project-level dependencies and is importable by all other
modules without risk of circular imports.

Constants:
    CLASS_NAMES: Ordered list of concept class names, establishing the integer-to-class
        mapping used across the entire pipeline.
    CLASS_TO_IDX: Inverse mapping from class name to integer index.
    N_CLASSES: Number of concept classes (5).

Example:
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 2, 3, 4, 0, 0, 0])
    >>> y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    >>> MetricsUtils.compute_macro_f1(y_true, y_pred, n_classes=5)
    0.7...
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Shared constants — imported by concept_labeler, probe_models, probe_trainer,
# visualize_plans, and intervention_engine to ensure a consistent mapping.
# ---------------------------------------------------------------------------

#: Ordered class names matching integer label indices 0–4.
#: Index 0 → NEVER (majority class), indices 1–4 → directional classes.
#: This ordering is the single source of truth for the entire pipeline.
CLASS_NAMES: List[str] = ["NEVER", "UP", "DOWN", "LEFT", "RIGHT"]

#: Inverse mapping from class name string to integer label index.
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

#: Total number of concept classes, matching config.yaml probing.n_classes: 5.
N_CLASSES: int = 5

# Validate consistency at import time.
assert len(CLASS_NAMES) == N_CLASSES, (
    f"CLASS_NAMES length {len(CLASS_NAMES)} must equal N_CLASSES {N_CLASSES}"
)
assert set(CLASS_TO_IDX.keys()) == set(CLASS_NAMES), (
    "CLASS_TO_IDX keys must match CLASS_NAMES"
)


class MetricsUtils:
    """Static utility methods for evaluating linear probe performance.

    All methods are pure functions over numpy arrays with no instance state.
    They are designed to handle the class imbalance inherent in Sokoban concept
    labeling, where NEVER dominates the label distribution.

    The class is not meant to be instantiated; use its static methods directly:
        macro_f1 = MetricsUtils.compute_macro_f1(y_true, y_pred, n_classes=5)
    """

    @staticmethod
    def compute_macro_f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_classes: int = N_CLASSES,
    ) -> float:
        """Compute macro-averaged F1 score across all concept classes.

        Macro F1 gives equal weight to each class regardless of frequency,
        which is critical for Sokoban concept probing where NEVER dominates.
        This is the primary evaluation metric used throughout the paper
        (Section 4.2, Figures 4, 6, 9, 22, 35, 38, 39, etc.).

        The ``labels`` parameter forces sklearn to include all ``n_classes``
        classes in the macro average, even if some classes are absent from
        ``y_true`` or ``y_pred`` in a given batch. Without this, sklearn would
        silently exclude absent classes, inflating the reported macro F1.

        Args:
            y_true: Ground-truth integer class labels, shape (N,). Values in
                [0, n_classes). Should be flattened from spatial predictions
                (B, H, W) → (B*H*W,) before calling.
            y_pred: Predicted integer class labels, same shape as y_true.
            n_classes: Number of classes to include in the macro average.
                Defaults to N_CLASSES=5 matching config.yaml probing.n_classes.

        Returns:
            Macro-averaged F1 score as a float in [0.0, 1.0]. Returns 0.0 if
            y_true is empty or all classes have zero support.

        Raises:
            ValueError: If y_true and y_pred have different shapes.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true shape {y_true.shape} must match y_pred shape {y_pred.shape}"
            )

        if len(y_true) == 0:
            return 0.0

        labels = list(range(n_classes))
        return float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
                labels=labels,
            )
        )

    @staticmethod
    def compute_per_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_classes: int = N_CLASSES,
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class precision, recall, and F1 for all concept classes.

        Returns a nested dict matching the structure of Tables 2–5 in the paper
        (Appendix D.2), which reports per-class metrics for 1×1 and 3×3 probes
        across all layers and the baseline.

        Each class is treated as the positive class with all other classes
        forming a single negative class (one-vs-rest evaluation).

        Args:
            y_true: Ground-truth integer class labels, shape (N,). Values in
                [0, n_classes).
            y_pred: Predicted integer class labels, same shape as y_true.
            n_classes: Number of classes. Defaults to N_CLASSES=5.

        Returns:
            Nested dict with structure:
                {
                    'NEVER': {'precision': float, 'recall': float, 'f1': float},
                    'UP':    {'precision': float, 'recall': float, 'f1': float},
                    'DOWN':  {'precision': float, 'recall': float, 'f1': float},
                    'LEFT':  {'precision': float, 'recall': float, 'f1': float},
                    'RIGHT': {'precision': float, 'recall': float, 'f1': float},
                }
            All float values are in [0.0, 1.0]. Classes with zero support
            receive 0.0 for precision, recall, and F1 (zero_division=0).

        Raises:
            ValueError: If y_true and y_pred have different shapes.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true shape {y_true.shape} must match y_pred shape {y_pred.shape}"
            )

        labels = list(range(n_classes))

        if len(y_true) == 0:
            # Return zero metrics for all classes when input is empty.
            return {
                CLASS_NAMES[i]: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                for i in range(n_classes)
            }

        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0,
            labels=labels,
        )

        result: Dict[str, Dict[str, float]] = {}
        for i in range(n_classes):
            class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
            result[class_name] = {
                "precision": float(precisions[i]),
                "recall": float(recalls[i]),
                "f1": float(f1s[i]),
            }

        return result

    @staticmethod
    def compute_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Compute classification accuracy (fraction of correct predictions).

        Used in Appendix D.5 (future action probing falsification experiment)
        where class imbalance is less severe and accuracy is the reported metric.
        Also useful as a sanity check during probe training alongside macro F1.

        Note: Accuracy is NOT the primary metric for concept probing due to
        class imbalance. Use compute_macro_f1 for the main evaluation.

        Args:
            y_true: Ground-truth integer class labels, shape (N,).
            y_pred: Predicted integer class labels, same shape as y_true.

        Returns:
            Accuracy as a float in [0.0, 1.0]. Returns 0.0 if y_true is empty.

        Raises:
            ValueError: If y_true and y_pred have different shapes.
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"y_true shape {y_true.shape} must match y_pred shape {y_pred.shape}"
            )

        if len(y_true) == 0:
            return 0.0

        return float(accuracy_score(y_true, y_pred))

    @staticmethod
    def aggregate_metrics_over_seeds(
        per_seed_metrics: List[float],
    ) -> Dict[str, float]:
        """Compute mean and standard deviation over multiple probe seeds.

        The paper reports mean ± 1 standard deviation across 5 seeds for all
        probe evaluations (Section 4.1, Figure 4, Tables 2–5).

        Args:
            per_seed_metrics: List of scalar metric values (e.g., macro F1),
                one per seed. Typically length 5 matching probing.n_seeds.

        Returns:
            Dict with keys 'mean' and 'std', both as floats.
        """
        if not per_seed_metrics:
            return {"mean": 0.0, "std": 0.0}

        values = np.array(per_seed_metrics, dtype=np.float64)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    @staticmethod
    def aggregate_per_class_metrics_over_seeds(
        per_seed_per_class: List[Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Aggregate per-class metrics (mean ± std) over multiple probe seeds.

        Produces the format used in Tables 2–5 of the paper (Appendix D.2),
        which reports mean ± std for precision, recall, and F1 per class.

        Args:
            per_seed_per_class: List of per-class metric dicts, one per seed.
                Each element has the structure returned by compute_per_class_metrics.

        Returns:
            Nested dict with structure:
                {
                    'NEVER': {
                        'precision': {'mean': float, 'std': float},
                        'recall':    {'mean': float, 'std': float},
                        'f1':        {'mean': float, 'std': float},
                    },
                    'UP': { ... },
                    ...
                }
        """
        if not per_seed_per_class:
            return {
                class_name: {
                    metric: {"mean": 0.0, "std": 0.0}
                    for metric in ("precision", "recall", "f1")
                }
                for class_name in CLASS_NAMES
            }

        result: Dict[str, Dict[str, Dict[str, float]]] = {}

        for class_name in CLASS_NAMES:
            result[class_name] = {}
            for metric_name in ("precision", "recall", "f1"):
                seed_values = [
                    seed_dict[class_name][metric_name]
                    for seed_dict in per_seed_per_class
                    if class_name in seed_dict
                ]
                agg = MetricsUtils.aggregate_metrics_over_seeds(seed_values)
                result[class_name][metric_name] = agg

        return result
