# losses.py
"""
Loss functions for the three experiments in "Conformal Prediction as Bayesian
Quadrature".  All losses are monotonic non‑increasing in the parameter λ and
bounded by B = 1.0.  They implement the `LossFunction` protocol (abstract
base class).

Classes:
    LossFunction     : abstract base class with `__call__(z, lam) -> float`.
    BinomialLoss     : synthetic binomial loss (Section 5.1).
    MiscoverageLoss  : miscoverage loss for heteroskedastic regression (Section 5.2).
    FNRLoss          : false negative rate loss for MS‑COCO multilabel (Section 5.3).

Usage:
    from losses import BinomialLoss, MiscoverageLoss, FNRLoss
    loss_fn = BinomialLoss(K=4)
    val = loss_fn(z_array, lam=0.5)
"""

from __future__ import annotations

import abc
from typing import Any, List, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Abstract loss function protocol
# ---------------------------------------------------------------------------

class LossFunction(abc.ABC):
    """
    Protocol for a loss function `ℓ(z, λ)` that is monotonic non‑increasing in λ
    and takes values in (-∞, B] with B considered 1.0 throughout.
    """

    @abc.abstractmethod
    def __call__(self, z: Any, lam: float) -> float:
        """
        Evaluate the loss for a single calibration/test point.

        Args:
            z: Observation (type varies per experiment).
            lam: Decision threshold (scalar).

        Returns:
            Loss value as a float.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Synthetic binomial loss (Section 5.1)
# ---------------------------------------------------------------------------

class BinomialLoss(LossFunction):
    """
    Loss for the synthetic binomial experiment.

    .. math::
        \\ell(z_i, \\lambda) = \\frac{1}{K} \\sum_{k=1}^{K} \\mathbf{1}\\{V_{ik} > \\lambda\\}

    where ``z_i`` is a 1‑D array of ``K`` Uniform(0,1) random variables.

    Args:
        K: Number of trials per calibration point (default 4).
    """

    def __init__(self, K: int) -> None:
        if K <= 0:
            raise ValueError("K must be a positive integer.")
        self.K = K

    def __call__(self, z: np.ndarray, lam: float) -> float:
        """
        Compute the average exceedance over the K draws.

        Args:
            z: numpy array of shape (K,) containing uniform variates.
            lam: threshold in [0,1].

        Returns:
            Average loss, in [0, 1].
        """
        # Vectorised comparison and average
        exceeds = np.mean(z > lam, dtype=np.float64)
        return float(exceeds)


# ---------------------------------------------------------------------------
# Miscoverage loss for heteroskedastic regression (Section 5.2)
# ---------------------------------------------------------------------------

class MiscoverageLoss(LossFunction):
    """
    Indicator loss for the heteroskedatic experiment.

    .. math::
        \\ell(z_i, \\lambda) = \\mathbf{1}\\{ |Y_i| > \\lambda \\}

    where ``z_i = (X_i, Y_i)``.

    The loss is 1 if the prediction interval ``[-λ, λ]`` fails to cover ``Y_i``,
    otherwise 0.
    """

    def __call__(self, z: Tuple[float, float], lam: float) -> float:
        """
        Evaluate the miscoverage loss.

        Args:
            z: A tuple (X, Y) where X and Y are scalars.
            lam: Half‑width of the prediction interval.

        Returns:
            1.0 if |Y| > λ, else 0.0.
        """
        _, y = z
        return 1.0 if abs(y) > lam else 0.0


# ---------------------------------------------------------------------------
# False negative rate loss for MS‑COCO multilabel (Section 5.3)
# ---------------------------------------------------------------------------

class FNRLoss(LossFunction):
    """
    False negative rate loss for the MS‑COCO multilabel experiment.

    .. math::
        \\ell(z_i, \\lambda) = 1 - \\frac{ | \\mathcal{C}_\\lambda(x_i) \\cap y_i | }{ |y_i| }

    where ``\\mathcal{C}_\\lambda(x) = \\{ j \\mid p_j(x) \\ge 1 - \\lambda \\}`` and
    ``y_i`` is the set of ground‑truth class indices.  If ``|y_i| = 0`` the loss
    is defined as 0.0.

    The input ``z_i`` is a tuple ``(probs, labels)`` where ``probs`` is a 1‑D
    numpy array of sigmoid probabilities (length = number of classes) and
    ``labels`` is a list of ground‑truth class indices (can be empty).
    """

    def __call__(self, z: Tuple[np.ndarray, Union[list, List[int]]],
                 lam: float) -> float:
        """
        Evaluate the false negative rate loss.

        Args:
            z: Tuple of (probs, labels).
                probs: 1‑D numpy array of shape (num_classes,) with probabilities
                       in [0, 1].
                labels: List of integer class indices that are present in the
                        ground truth.  May be empty.
            lam: Decision threshold.

        Returns:
            False negative rate as a float in [0, 1].
        """
        probs, labels = z
        if labels is None or len(labels) == 0:
            # No ground‑truth labels → no false negatives possible
            return 0.0

        # Prediction set: indices where probability >= 1 - lam
        threshold = 1.0 - lam
        pred_set = set(np.where(probs >= threshold)[0].tolist())
        gt_set = set(labels)

        # Intersection size
        intersect = len(pred_set & gt_set)

        # Recall
        recall = intersect / len(gt_set)
        return 1.0 - recall

