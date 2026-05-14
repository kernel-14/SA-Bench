"""
Experiment 5.3: False Negative Rate on MS-COCO

Reproduces Table 3 from the paper.

Setup:
- MS-COCO dataset for multilabel classification
- Goal: control false negative rate (FNR)
- Each random split: 1000 calibration examples, 3952 test examples
- M = 10,000 random trials
- Target risk: alpha = 0.1 (10% FNR)
- Maximum failure rate: 5% (beta = 0.95)

The experimental setup mirrors Angelopoulos & Bates (2023, Section 5.1).

For multilabel classification:
- lambda in [0, 1] controls the threshold for including a class
- Prediction set: {c : score_c >= 1 - lambda}
- Loss = FNR = fraction of true labels not in prediction set
- Loss is monotonically non-increasing in lambda

Data: Pre-computed softmax scores from a ResNet-101 model on MS-COCO.
Download with: python download_coco_data.py
"""

import numpy as np
import os
from scipy import stats

from methods import (
    conformal_risk_control,
    rcps_hoeffding,
    bayesian_quadrature_decision_rule,
)


def compute_fnr_loss(scores, labels, lam):
    """
    Compute false negative rate loss for multilabel classification.

    Prediction set: include class c if score_c >= 1 - lambda.
    FNR = fraction of true labels not in prediction set.

    Parameters
    ----------
    scores : array of shape (n, C)
        Softmax scores for each class.
    labels : array of shape (n, C)
        Binary labels (1 if class is present, 0 otherwise).
    lam : float
        Threshold parameter in [0, 1].

    Returns
    -------
    losses : array of shape (n,)
        Individual FNR losses in [0, 1].
    """
    threshold = 1.0 - lam
    predictions = (scores >= threshold).astype(float)

    n_true = labels.sum(axis=1)
    n_false_neg = (labels * (1.0 - predictions)).sum(axis=1)

    # FNR = 0 for examples with no true labels
    fnr = np.where(n_true > 0, n_false_neg / n_true, 0.0)
    return fnr


def load_coco_data(data_path="data/coco"):
    """
    Load MS-COCO model predictions and labels.

    Expected files:
      data/coco/scores.npy  — softmax scores, shape (N, C)
      data/coco/labels.npy  — binary labels,  shape (N, C)

    Parameters
    ----------
    data_path : str
        Path to data directory.

    Returns
    -------
    scores : array of shape (N, C)
    labels : array of shape (N, C)
    """
    scores_path = os.path.join(data_path, "scores.npy")
    labels_path = os.path.join(data_path, "labels.npy")

    if not os.path.exists(scores_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"COCO data not found at {data_path}. "
            "Run 'python download_coco_data.py' first."
        )

    scores = np.load(scores_path)
    labels = np.load(labels_path)
    return scores, labels


def run_coco_experiment(
    scores,
    labels,
    n_cal=1000,
    alpha=0.1,
    beta=0.95,
    M=10000,
    B=1.0,
    n_bq_samples=1000,
    lambda_grid=None,
    seed=42,
):
    """
    Run the MS-COCO false negative rate experiment.

    Parameters
    ----------
    scores : array of shape (N, C)
    labels : array of shape (N, C)
    n_cal : int
        Number of calibration examples per trial.
    alpha : float
        Target FNR level.
    beta : float
        Confidence level for BQ method.
    M : int
        Number of random trials.
    B : float
        Upper bound on losses (1 for FNR).
    n_bq_samples : int
        Number of Monte Carlo samples for BQ.
    lambda_grid : array-like, optional
        Grid of lambda values in [0, 1].
    seed : int
        Random seed.

    Returns
    -------
    results : dict
    """
    rng = np.random.default_rng(seed)
    N = len(scores)

    if lambda_grid is None:
        lambda_grid = np.linspace(0, 1, 101)

    lambdas_crc = np.zeros(M)
    lambdas_rcps = np.zeros(M)
    lambdas_bq = np.zeros(M)

    # Arrays for test-set metrics
    risks_crc = np.zeros(M)
    risks_rcps = np.zeros(M)
    risks_bq = np.zeros(M)
    pred_sizes_crc = np.zeros(M)
    pred_sizes_rcps = np.zeros(M)
    pred_sizes_bq = np.zeros(M)

    for trial in range(M):
        if trial % 1000 == 0:
            print(f"  Trial {trial}/{M}")

        # Random split
        idx = rng.permutation(N)
        cal_idx = idx[:n_cal]
        test_idx = idx[n_cal:]

        scores_cal = scores[cal_idx]
        labels_cal = labels[cal_idx]
        scores_test = scores[test_idx]
        labels_test = labels[test_idx]

        # Capture calibration data in closures
        sc, lc = scores_cal, labels_cal

        def losses_fn(lam, _sc=sc, _lc=lc):
            return compute_fnr_loss(_sc, _lc, lam)

        # CRC
        lam_crc = conformal_risk_control(losses_fn, lambda_grid, alpha, B=B)
        lambdas_crc[trial] = lam_crc

        # RCPS
        delta = 1.0 - beta
        lam_rcps = rcps_hoeffding(losses_fn, lambda_grid, alpha, delta=delta, B=B)
        lambdas_rcps[trial] = lam_rcps

        # BQ
        lam_bq = bayesian_quadrature_decision_rule(
            losses_fn, lambda_grid, alpha, beta=beta, B=B,
            n_samples=n_bq_samples, rng=rng
        )
        lambdas_bq[trial] = lam_bq

        # Evaluate on test set
        for lam, risks_arr, sizes_arr in [
            (lam_crc,  risks_crc,  pred_sizes_crc),
            (lam_rcps, risks_rcps, pred_sizes_rcps),
            (lam_bq,   risks_bq,   pred_sizes_bq),
        ]:
            test_losses = compute_fnr_loss(scores_test, labels_test, lam)
            risks_arr[trial] = np.mean(test_losses)

            threshold = 1.0 - lam
            pred_sets = (scores_test >= threshold)
            sizes_arr[trial] = np.mean(pred_sets.sum(axis=1))

    # Compute exceedance
    exceed_crc  = risks_crc  > alpha
    exceed_rcps = risks_rcps > alpha
    exceed_bq   = risks_bq   > alpha

    def clopper_pearson_ci(k, n_trials, confidence=0.95):
        alpha_ci = 1 - confidence
        lower = stats.beta.ppf(alpha_ci / 2, k, n_trials - k + 1) if k > 0 else 0.0
        upper = stats.beta.ppf(1 - alpha_ci / 2, k + 1, n_trials - k) if k < n_trials else 1.0
        return lower, upper

    results = {}
    for name, exceed, pred_sizes in [
        ("CRC",            exceed_crc,  pred_sizes_crc),
        ("RCPS",           exceed_rcps, pred_sizes_rcps),
        ("Ours (beta=0.95)", exceed_bq, pred_sizes_bq),
    ]:
        freq = np.mean(exceed)
        k = int(np.sum(exceed))
        ci_low, ci_high = clopper_pearson_ci(k, M)
        results[name] = {
            "relative_freq": freq,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean_pred_set_size": np.mean(pred_sizes),
        }

    return results


def print_table(results, M=10000):
    """Print Table 3 from the paper."""
    print("\n" + "=" * 65)
    print("Table 3: MS-COCO results")
    print("=" * 65)
    print(f"{'Method':<30} {'Relative Freq.':<20} {'Pred. Set Size'}")
    print("-" * 65)
    for name, res in results.items():
        freq_pct = res["relative_freq"] * 100
        pred_size = res["mean_pred_set_size"]
        print(f"{name:<30} {freq_pct:>6.2f}%{'':<13} {pred_size:.2f}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    os.makedirs("data/coco", exist_ok=True)

    print("Running MS-COCO False Negative Rate Experiment (Section 5.3)...")
    print("Parameters: n_cal=1000, alpha=0.1, beta=0.95, M=10000")
    print()

    try:
        scores, labels = load_coco_data("data/coco")
        print(f"Loaded COCO data: {scores.shape[0]} examples, {scores.shape[1]} classes")

        results = run_coco_experiment(
            scores, labels,
            n_cal=1000, alpha=0.1, beta=0.95, M=10000,
            B=1.0, n_bq_samples=1000, seed=42
        )

        print_table(results, M=10000)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'python download_coco_data.py' to download the required data.")

    print("Done!")
