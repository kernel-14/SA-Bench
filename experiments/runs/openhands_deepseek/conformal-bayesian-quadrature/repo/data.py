"""Data generation for synthetic experiments and MS-COCO loader.

Section 5.1 - Synthetic Binomial Data
Section 5.2 - Synthetic Heteroskedastic Data
Section 5.3 - MS-COCO Multilabel Classification (False Negative Rate)
"""

import numpy as np
from typing import Optional, Tuple, Callable


def generate_binomial_losses(
    n: int,
    K: int,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate binomial loss values for n calibration points at threshold λ.

    ℓ(z_i, λ) = (1/K) Σ_{k=1}^K 1{V_ik > λ}
    where V_ik ~ Uniform(0, 1). (Eq 34)

    Args:
        n: Number of calibration samples.
        K: Number of binomial trials per sample.
        lam: Threshold λ.
        rng: Random number generator.

    Returns:
        Array of n loss values in [0, 1].
    """
    V = rng.uniform(0, 1, size=(n, K))  # (n, K)
    losses = np.mean(V > lam, axis=1)  # (n,)
    return losses


def generate_binomial_losses_multilambda(
    n: int,
    K: int,
    lambdas: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Generate losses at multiple λ values for the same calibration set.

    Uses the same V_ik across all λ for consistency.

    Args:
        n: Number of calibration samples.
        K: Number of binomial trials per sample.
        lambdas: Array of λ values.
        rng: Random number generator.

    Returns:
        Dict mapping λ to array of n losses.
    """
    V = rng.uniform(0, 1, size=(n, K))
    result = {}
    for lam in lambdas:
        result[lam] = np.mean(V > lam, axis=1)
    return result


def expected_binomial_loss(lam: float) -> float:
    """Expected loss for binomial data: E[ℓ] = 1 - λ."""
    return 1.0 - lam


def generate_heteroskedastic_data(
    n_cal: int,
    n_test: int,
    x_range: Tuple[float, float] = (0.0, 4.0),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate heteroskedastic data (Section 5.2).

    X ~ Uniform[x_range], Y | X ~ N(0, X²).
    Scores are s = |Y|. Prediction interval: [-λ, λ].
    Miscoverage loss: ℓ = 1{|Y| > λ}.

    Args:
        n_cal: Number of calibration points.
        n_test: Number of test points.
        x_range: Range of X values.
        rng: Random number generator.

    Returns:
        (X_cal, Y_cal, X_test, Y_test): Calibration and test data.
    """
    if rng is None:
        rng = np.random.default_rng()

    X_cal = rng.uniform(x_range[0], x_range[1], size=n_cal)
    Y_cal = rng.normal(0, X_cal)  # std = X

    X_test = rng.uniform(x_range[0], x_range[1], size=n_test)
    Y_test = rng.normal(0, X_test)

    return X_cal, Y_cal, X_test, Y_test


def compute_heteroskedastic_scores(Y: np.ndarray) -> np.ndarray:
    """Compute absolute scores for heteroskedastic data: s = |Y|."""
    return np.abs(Y)


def compute_miscoverage_losses_heteroskedastic(
    Y: np.ndarray,
    lambdas: np.ndarray,
) -> dict:
    """Compute miscoverage losses for heteroskedastic data.

    ℓ = 1{|Y| > λ} for each λ.

    Args:
        Y: Array of Y values.
        lambdas: Array of λ values.

    Returns:
        Dict mapping λ to array of binary loss values.
    """
    abs_y = np.abs(Y)
    result = {}
    for lam in lambdas:
        result[lam] = (abs_y > lam).astype(np.float64)
    return result


def compute_miscoverage_risk(Y_test: np.ndarray, lam: float) -> float:
    """Compute true miscoverage risk on test data.

    R(θ, λ) = E[1{|Y| > λ}].

    Args:
        Y_test: Test Y values.
        lam: Threshold λ.

    Returns:
        True risk.
    """
    return float(np.mean(np.abs(Y_test) > lam))


def compute_prediction_interval_length(lam: float) -> float:
    """Length of symmetric prediction interval [-λ, λ]: 2λ."""
    return 2.0 * lam


class SyntheticMSCOCODataset:
    """Simulated MS-COCO multilabel classification data.

    Mirrors the setup in Angelopoulos & Bates (2023, Section 5.1).
    Since actual MS-COCO data requires download, this generates
    synthetic data that mimics the multilabel structure with 80 classes,
    calibrated to match the paper's experimental parameters.

    In a real deployment, replace with actual MS-COCO loading.
    """

    def __init__(
        self,
        n_cal: int = 1000,
        n_test: int = 3952,
        num_classes: int = 80,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        self.num_classes = num_classes

        base_prob = 0.08
        self.cal_labels = rng.binomial(1, base_prob, size=(n_cal, num_classes)).astype(np.float64)

        cal_probs = np.clip(
            self.cal_labels * rng.uniform(0.6, 0.95, size=(n_cal, num_classes))
            + (1 - self.cal_labels) * rng.uniform(0.05, 0.4, size=(n_cal, num_classes)),
            0.01, 0.99,
        )
        self.cal_scores = cal_probs

        self.test_labels = rng.binomial(1, base_prob, size=(n_test, num_classes)).astype(np.float64)

        test_probs = np.clip(
            self.test_labels * rng.uniform(0.6, 0.95, size=(n_test, num_classes))
            + (1 - self.test_labels) * rng.uniform(0.05, 0.4, size=(n_test, num_classes)),
            0.01, 0.99,
        )
        self.test_scores = test_probs

    def compute_fnr_losses(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        lam: float,
    ) -> np.ndarray:
        """Compute false negative rate losses at threshold λ.

        For each example: loss = (#FN) / max(1, #true positives)

        Prediction: ŷ_c = 1{score_c > λ}.
        ℓ_i = (Σ_c y_ic * (1 - ŷ_ic)) / max(1, Σ_c y_ic)

        Args:
            scores: (m, C) array of class scores in [0, 1].
            labels: (m, C) array of ground-truth binary labels.
            lam: Threshold λ.

        Returns:
            Array of m loss values in [0, 1].
        """
        predictions = (scores > lam).astype(np.float64)
        false_negatives = labels * (1 - predictions)
        fn_count = false_negatives.sum(axis=1)
        label_count = np.maximum(1.0, labels.sum(axis=1))
        return fn_count / label_count

    def compute_fnr_losses_multilambda(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        lambdas: np.ndarray,
    ) -> dict:
        """Compute FNR losses for all λ values.

        Args:
            scores: (m, C) array.
            labels: (m, C) array.
            lambdas: Array of λ values.

        Returns:
            Dict mapping λ to array of m losses.
        """
        result = {}
        for lam in lambdas:
            result[lam] = self.compute_fnr_losses(scores, labels, lam)
        return result

    def compute_prediction_set_size(
        self,
        scores: np.ndarray,
        lam: float,
    ) -> float:
        """Average prediction set size at threshold λ.

        |C_λ(x)| = Σ_c 1{score_c > λ}.

        Args:
            scores: (m, C) array.
            lam: Threshold.

        Returns:
            Mean prediction set size.
        """
        predictions = (scores > lam).astype(np.float64)
        return float(predictions.sum(axis=1).mean())


def load_coco_dummy(
    n_cal: int = 1000,
    n_test: int = 3952,
    num_classes: int = 80,
    seed: int = 42,
) -> SyntheticMSCOCODataset:
    """Load dummy MS-COCO dataset for development/testing."""
    rng = np.random.default_rng(seed)
    return SyntheticMSCOCODataset(
        n_cal=n_cal,
        n_test=n_test,
        num_classes=num_classes,
        rng=rng,
    )
