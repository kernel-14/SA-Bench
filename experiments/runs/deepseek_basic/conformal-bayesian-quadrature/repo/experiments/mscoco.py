"""Experiment 5.3: False Negative Rate Control on MS-COCO.

This experiment reproduces the results from Section 5.3 / Table 3 of the paper.

Setup:
  - Multilabel classification on MS-COCO dataset
  - The experimental setup mirrors Angelopoulos & Bates (2023, Section 5.1)
  - Each random split: 1000 calibration examples, 3952 test examples
  - Loss: false negative rate (FNR)
  - Goal: control FNR <= alpha

Decision rules:
  1. CRC (Conformal Risk Control)
  2. RCPS (Risk-controlling Prediction Sets)
  3. Ours (HPD, beta=0.95)

Note: Since we cannot access the actual MS-COCO dataset in this static
environment, this implementation provides the framework for running the
experiment and demonstrates the methodology. The actual data loading
would require downloading MS-COCO and a pre-trained model.

This module implements:
  - FNR loss function for multilabel classification
  - Calibration/evaluation pipeline
  - Comparison of all three decision rules
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bayesian_quadrature import (
    compute_crc_decision_rule,
    compute_rcps_hoeffding_lambda,
    compute_hpd_lambda,
)


def false_negative_rate_loss(y_true: np.ndarray, y_pred_probs: np.ndarray,
                              lambda_val: float) -> np.ndarray:
    """Compute the false negative rate loss for multilabel classification.

    Following Angelopoulos & Bates (2023, Section 5.1):
    For each example, we form a prediction set by thresholding predicted
    probabilities at lambda. The FNR loss is:
        FNR = 1 - |Y_hat ∩ Y_true| / |Y_true|
    with the convention 0/0 = 0.

    Args:
        y_true: Binary matrix of shape (n_samples, n_classes), ground truth labels.
        y_pred_probs: Matrix of shape (n_samples, n_classes), predicted probabilities.
        lambda_val: Threshold for inclusion in prediction set.

    Returns:
        Array of shape (n_samples,) with FNR loss for each sample.
    """
    n_samples = y_true.shape[0]

    # Prediction set: include class j if prob >= lambda_val
    in_set = y_pred_probs >= lambda_val  # shape (n_samples, n_classes)

    # Number of true labels per sample
    n_true = y_true.sum(axis=1)  # shape (n_samples,)

    # Number of correctly identified true labels (intersection)
    n_correct = (in_set & y_true.astype(bool)).sum(axis=1)  # shape (n_samples,)

    # FNR loss: 1 - |intersection| / |Y_true|
    # Handle edge case where n_true == 0
    fnr = np.ones(n_samples)
    nonzero_mask = n_true > 0
    fnr[nonzero_mask] = 1.0 - n_correct[nonzero_mask] / n_true[nonzero_mask]

    return fnr


def run_multilabel_experiment(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float,
    beta: float = 0.95,
    n_lambda: int = 200,
    n_dirichlet_samples: int = 1000,
    seed: int = 42,
    verbose: bool = True,
):
    """Run the full MS-COCO multilabel experiment for a single split.

    Args:
        y_true_cal: Calibration ground truth labels (n_cal, n_classes).
        y_pred_cal: Calibration predicted probabilities (n_cal, n_classes).
        y_true_test: Test ground truth labels (n_test, n_classes).
        y_pred_test: Test predicted probabilities (n_test, n_classes).
        alpha: Target FNR.
        beta: Confidence level for HPD.
        n_lambda: Number of lambda grid points.
        n_dirichlet_samples: Number of Dirichlet MC samples.
        seed: Random seed.
        verbose: Whether to print progress.

    Returns:
        Dictionary with results.
    """
    rng = np.random.default_rng(seed)
    lambda_grid = np.linspace(0, 1, n_lambda)
    B = 1.0  # FNR is bounded in [0, 1]

    # Loss function for calibration data
    def loss_fn(lam):
        return false_negative_rate_loss(y_true_cal, y_pred_cal, lam)

    if verbose:
        print("  Computing CRC lambda...")
    lambda_crc, info_crc = compute_crc_decision_rule(
        loss_fn=loss_fn,
        alpha=alpha,
        B=B,
        lambda_grid=lambda_grid,
    )

    if verbose:
        print("  Computing RCPS lambda...")
    lambda_rcps, info_rcps = compute_rcps_hoeffding_lambda(
        loss_fn=loss_fn,
        alpha=alpha,
        B=B,
        delta=1 - beta,
        lambda_grid=lambda_grid,
    )

    if verbose:
        print("  Computing HPD lambda...")
    lambda_hpd, info_hpd = compute_hpd_lambda(
        loss_fn=loss_fn,
        alpha=alpha,
        B=B,
        beta=beta,
        lambda_grid=lambda_grid,
        n_dirichlet_samples=n_dirichlet_samples,
        rng=rng,
    )

    # Evaluate true risk on test set
    def true_fnr(lam):
        return np.mean(false_negative_rate_loss(y_true_test, y_pred_test, lam))

    risk_crc = true_fnr(lambda_crc)
    risk_rcps = true_fnr(lambda_rcps)
    risk_hpd = true_fnr(lambda_hpd)

    # Compute prediction set sizes on test set
    def pred_set_size(lam):
        return np.mean(np.sum(y_pred_test >= lam, axis=1))

    size_crc = pred_set_size(lambda_crc)
    size_rcps = pred_set_size(lambda_rcps)
    size_hpd = pred_set_size(lambda_hpd)

    results = {
        "lambda_crc": lambda_crc,
        "lambda_rcps": lambda_rcps,
        "lambda_hpd": lambda_hpd,
        "risk_crc": risk_crc,
        "risk_rcps": risk_rcps,
        "risk_hpd": risk_hpd,
        "size_crc": size_crc,
        "size_rcps": size_rcps,
        "size_hpd": size_hpd,
        "risk_exceeds_crc": risk_crc > alpha,
        "risk_exceeds_rcps": risk_rcps > alpha,
        "risk_exceeds_hpd": risk_hpd > alpha,
    }

    return results


def run_synthetic_mscoco_experiment(
    n_trials: int = 10000,
    n_cal: int = 1000,
    n_test: int = 3952,
    n_classes: int = 80,
    alpha: float = 0.1,
    beta: float = 0.95,
    seed: int = 42,
    verbose: bool = True,
):
    """Run a synthetic version of the MS-COCO experiment.

    Since we cannot access the actual MS-COCO data, we simulate the
    experimental setup with synthetic multilabel data that mimics the
    structure of the problem. This demonstrates the methodology.

    The simulated data uses:
      - Random binary labels with sparsity pattern
      - Random prediction scores correlated with true labels

    Args:
        n_trials: Number of random splits.
        n_cal: Number of calibration samples.
        n_test: Number of test samples.
        n_classes: Number of classes (80 for MS-COCO).
        alpha: Target FNR.
        beta: Confidence level.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dictionary of aggregate results.
    """
    rng = np.random.default_rng(seed)

    crc_exceeds = 0
    rcps_exceeds = 0
    hpd_exceeds = 0
    crc_sizes = []
    rcps_sizes = []
    hpd_sizes = []

    for trial in range(n_trials):
        if verbose and (trial + 1) % 500 == 0:
            print(f"  Trial {trial + 1}/{n_trials}")

        # Generate synthetic multilabel data
        # Each class has some base probability of being present
        class_probs = rng.beta(0.5, 5.0, size=n_classes)  # sparse labels
        class_probs = class_probs / class_probs.max() * 0.3  # cap at ~30%

        # Calibration data
        y_true_cal = rng.binomial(1, class_probs, size=(n_cal, n_classes)).astype(float)
        # Predictions: add noise to true labels
        y_pred_cal = y_true_cal * (0.6 + 0.4 * rng.beta(2, 1, size=(n_cal, n_classes))) + \
                      (1 - y_true_cal) * rng.beta(1, 3, size=(n_cal, n_classes)) * 0.3

        # Test data
        y_true_test = rng.binomial(1, class_probs, size=(n_test, n_classes)).astype(float)
        y_pred_test = y_true_test * (0.6 + 0.4 * rng.beta(2, 1, size=(n_test, n_classes))) + \
                       (1 - y_true_test) * rng.beta(1, 3, size=(n_test, n_classes)) * 0.3

        result = run_multilabel_experiment(
            y_true_cal=y_true_cal,
            y_pred_cal=y_pred_cal,
            y_true_test=y_true_test,
            y_pred_test=y_pred_test,
            alpha=alpha,
            beta=beta,
            seed=trial,
            verbose=False,
        )

        if result["risk_exceeds_crc"]:
            crc_exceeds += 1
        if result["risk_exceeds_rcps"]:
            rcps_exceeds += 1
        if result["risk_exceeds_hpd"]:
            hpd_exceeds += 1

        crc_sizes.append(result["size_crc"])
        rcps_sizes.append(result["size_rcps"])
        hpd_sizes.append(result["size_hpd"])

    results = {
        "n_trials": n_trials,
        "crc": {
            "exceed_rate": crc_exceeds / n_trials,
            "mean_size": np.mean(crc_sizes),
        },
        "rcps": {
            "exceed_rate": rcps_exceeds / n_trials,
            "mean_size": np.mean(rcps_sizes),
        },
        "hpd": {
            "exceed_rate": hpd_exceeds / n_trials,
            "mean_size": np.mean(hpd_sizes),
        },
    }

    return results


def print_results(results):
    """Print results in a formatted table (matching Table 3 in the paper)."""
    print("\n" + "=" * 60)
    print("Table 3: MS-COCO Results")
    print("=" * 60)
    print(f"{'Method':<15} {'Relative Freq.':>15} {'Pred. Set Size':>18}")
    print("-" * 48)
    print(f"{'CRC':<15} {results['crc']['exceed_rate']*100:>14.2f}%  "
          f"{results['crc']['mean_size']:>17.2f}")
    print(f"{'RCPS':<15} {results['rcps']['exceed_rate']*100:>14.2f}%  "
          f"{results['rcps']['mean_size']:>17.2f}")
    print(f"{'Ours (b=0.95)':<15} {results['hpd']['exceed_rate']*100:>14.2f}%  "
          f"{results['hpd']['mean_size']:>17.2f}")
    print("-" * 48)
    print()


if __name__ == "__main__":
    print("Running MS-COCO Experiment (Section 5.3)...")
    print("Note: Using synthetic data to demonstrate methodology.")
    print("For actual MS-COCO results, download the dataset and use a pre-trained model.")
    print()
    results = run_synthetic_mscoco_experiment(
        n_trials=10000,
        n_cal=1000,
        n_test=3952,
        n_classes=80,
        alpha=0.1,
        beta=0.95,
        seed=42,
    )
    print_results(results)
