"""
Evaluation metrics for uncertainty quantification.

Metrics from the paper (Appendix D.4):
  - RMSE: root mean squared error of mean predictions
  - Marginal NLL: negative log-likelihood under Gaussian predictive distribution
  - Marginal χ² statistic: average squared error normalized by predicted variance

All metrics are computed as expected values over 250 test input-output pairs.

Definitions:
  RMSE = sqrt(1/n * Σ_i (y_i - ŷ_i)²)
  NLL = -Σ_i log N(y_i; ŷ_i, σ_i²)
      = Σ_i [0.5 * log(2π σ_i²) + (y_i - ŷ_i)² / (2 σ_i²)]
  χ² = 1/n * Σ_i (y_i - ŷ_i)² / σ_i²

where y_i is ground truth, ŷ_i is predicted mean, σ_i is predicted std.
"""

from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np


def compute_rmse(
    mean_pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Root mean squared error.

    Args:
        mean_pred: predicted mean, any shape
        target: ground truth, same shape
    Returns:
        RMSE scalar
    """
    return float(np.sqrt(np.mean((mean_pred - target) ** 2)))


def compute_marginal_nll(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Marginal negative log-likelihood under Gaussian predictive distribution.

    NLL = Σ_i [0.5 * log(2π σ_i²) + (y_i - ŷ_i)² / (2 σ_i²)]
        = 0.5 * Σ_i [log(2π) + log(σ_i²) + (y_i - ŷ_i)² / σ_i²]

    The paper reports the expected NLL over test pairs, so we average over
    all spatial points and test samples.

    Args:
        mean_pred: predicted mean, shape (n_test, ...)
        std_pred: predicted std, shape (n_test, ...)
        target: ground truth, shape (n_test, ...)
        eps: numerical stability for std
    Returns:
        mean NLL per spatial point
    """
    std_pred = np.maximum(std_pred, eps)
    var_pred = std_pred ** 2

    nll = 0.5 * (np.log(2 * np.pi * var_pred) + (target - mean_pred) ** 2 / var_pred)
    return float(np.mean(nll))


def compute_chi2(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Marginal χ² statistic.

    χ² = 1/n * Σ_i (y_i - ŷ_i)² / σ_i²

    A value close to 1 indicates well-calibrated uncertainty.
    Values > 1: overconfident; values < 1: underconfident.

    Args:
        mean_pred: predicted mean
        std_pred: predicted std
        target: ground truth
        eps: numerical stability
    Returns:
        χ² scalar
    """
    std_pred = np.maximum(std_pred, eps)
    chi2 = np.mean((target - mean_pred) ** 2 / std_pred ** 2)
    return float(chi2)


def compute_all_metrics(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        mean_pred: predicted mean, shape (n_test, ...)
        std_pred: predicted std, shape (n_test, ...)
        target: ground truth, shape (n_test, ...)
        eps: numerical stability
    Returns:
        dict with "rmse", "nll", "chi2"
    """
    return {
        "rmse": compute_rmse(mean_pred, target),
        "nll": compute_marginal_nll(mean_pred, std_pred, target, eps),
        "chi2": compute_chi2(mean_pred, std_pred, target, eps),
    }


def evaluate_method_on_dataset(
    predict_fn,
    test_loader,
    n_test: int = 250,
) -> Dict[str, float]:
    """
    Evaluate a UQ method on a test dataset.

    Args:
        predict_fn: callable(a) -> (mean, std) where a is a single test input
        test_loader: iterable of (a, u) test pairs
        n_test: number of test pairs to evaluate
    Returns:
        dict with "rmse", "nll", "chi2"
    """
    all_means = []
    all_stds = []
    all_targets = []
    n_evaluated = 0

    for a_batch, u_batch in test_loader:
        if n_evaluated >= n_test:
            break

        for i in range(a_batch.shape[0]):
            if n_evaluated >= n_test:
                break

            a_i = a_batch[i:i + 1]
            u_i = u_batch[i:i + 1]

            mean_i, std_i = predict_fn(a_i)

            all_means.append(np.array(mean_i))
            all_stds.append(np.array(std_i))
            all_targets.append(np.array(u_i))
            n_evaluated += 1

    mean_pred = np.concatenate(all_means, axis=0)
    std_pred = np.concatenate(all_stds, axis=0)
    target = np.concatenate(all_targets, axis=0)

    return compute_all_metrics(mean_pred, std_pred, target)
