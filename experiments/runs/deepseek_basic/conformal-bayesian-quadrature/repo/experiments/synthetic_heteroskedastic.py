"""Experiment 5.2: Synthetic Heteroskedastic Data.

This experiment reproduces the results from Section 5.2 / Table 2 of the paper.

Setup:
  - X ~ Uniform[0, 4]
  - Y | X ~ N(0, X^2)
  - n = 200 calibration samples
  - Prediction intervals: [-lambda_hat, lambda_hat]
  - Loss: miscoverage loss (0-1 loss)
  - alpha = 0.1 (i.e., 90% coverage target)
  - beta = 0.95
  - M = 10,000 trials

Decision rules:
  1. Split Conformal Prediction / CRC
  2. RCPS (Hoeffding)
  3. Ours (HPD, beta=0.95)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bayesian_quadrature import (
    compute_split_conformal_lambda,
    compute_crc_decision_rule,
    compute_rcps_hoeffding_lambda,
    compute_hpd_lambda,
)
from scipy.stats import binom


def run_single_trial(
    n: int,
    alpha: float,
    B: float,
    beta: float,
    lambda_grid: np.ndarray,
    n_dirichlet_samples: int,
    rng: np.random.Generator,
):
    """Run a single trial of the synthetic heteroskedastic experiment.

    Args:
        n: Number of calibration samples.
        alpha: Target miscoverage rate.
        B: Maximum loss (1 for 0-1 loss).
        beta: Confidence level for HPD.
        lambda_grid: Grid of lambda values.
        n_dirichlet_samples: Number of Dirichlet MC samples.
        rng: Random generator.

    Returns:
        Dictionary with selected lambdas and prediction interval lengths.
    """
    # Generate calibration data
    X_cal = rng.uniform(0, 4, size=n)
    Y_cal = rng.normal(0, X_cal)  # std = X

    # For a given lambda, the prediction interval is [-lambda, lambda]
    # The miscoverage loss for point i is 1 if |Y_i| > lambda, else 0
    def loss_fn(lam):
        return (np.abs(Y_cal) > lam).astype(float)

    # True expected loss: P(|Y| > lambda) where Y ~ N(0, X^2), X ~ U[0,4]
    # We'll evaluate this empirically with a large test set
    # For the purpose of determining if risk exceeds alpha, we can compute
    # the true risk by numerical integration or large-sample approximation.

    # --- Split Conformal / CRC ---
    # For miscoverage loss with 0-1 loss, CRC reduces to SCP
    # Use scores = |Y_i| (nonconformity scores)
    scores = np.abs(Y_cal)
    lambda_scp = compute_split_conformal_lambda(scores, alpha)

    # Also compute via CRC for consistency
    lambda_crc, info_crc = compute_crc_decision_rule(
        loss_fn=loss_fn,
        alpha=alpha,
        B=B,
        lambda_grid=lambda_grid,
    )

    # --- RCPS ---
    lambda_rcps, info_rcps = compute_rcps_hoeffding_lambda(
        loss_fn=loss_fn,
        alpha=alpha,
        B=B,
        delta=1 - beta,
        lambda_grid=lambda_grid,
    )

    # --- Ours (HPD) ---
    lambda_hpd, info_hpd = compute_hpd_lambda(
        loss_fn=loss_fn,
        alpha=alpha,
        B=B,
        beta=beta,
        lambda_grid=lambda_grid,
        n_dirichlet_samples=n_dirichlet_samples,
        rng=rng,
    )

    # Prediction interval length = 2 * lambda
    return {
        "lambda_scp": lambda_scp,
        "lambda_crc": lambda_crc,
        "lambda_rcps": lambda_rcps,
        "lambda_hpd": lambda_hpd,
        "interval_length_scp": 2 * lambda_scp,
        "interval_length_crc": 2 * lambda_crc,
        "interval_length_rcps": 2 * lambda_rcps,
        "interval_length_hpd": 2 * lambda_hpd,
    }


def compute_true_risk(lambda_val, n_mc=100000, rng=None):
    """Compute true expected miscoverage risk for a given lambda.

    P(|Y| > lambda) where X ~ U[0,4], Y|X ~ N(0, X^2).

    We can compute this analytically or via MC.
    Using MC here for simplicity.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    X_test = rng.uniform(0, 4, size=n_mc)
    Y_test = rng.normal(0, X_test)
    return np.mean(np.abs(Y_test) > lambda_val)


def run_experiment(
    n_trials: int = 10000,
    n: int = 200,
    alpha: float = 0.1,
    B: float = 1.0,
    beta: float = 0.95,
    n_lambda: int = 200,
    n_dirichlet_samples: int = 1000,
    seed: int = 42,
    verbose: bool = True,
):
    """Run the full synthetic heteroskedastic experiment.

    Returns:
        Dictionary of results.
    """
    rng = np.random.default_rng(seed)
    lambda_grid = np.linspace(0, 15, n_lambda)  # wider range for this problem

    # Pre-compute true risk function at grid points
    # We use a shared test set to evaluate true risk
    n_mc_risk = 50000
    X_test_shared = rng.uniform(0, 4, size=n_mc_risk)
    Y_test_shared = rng.normal(0, X_test_shared)

    def true_risk(lam):
        return np.mean(np.abs(Y_test_shared) > lam)

    # Storage
    lambdas_scp = np.zeros(n_trials)
    lambdas_crc = np.zeros(n_trials)
    lambdas_rcps = np.zeros(n_trials)
    lambdas_hpd = np.zeros(n_trials)
    interval_lengths_scp = np.zeros(n_trials)
    interval_lengths_rcps = np.zeros(n_trials)
    interval_lengths_hpd = np.zeros(n_trials)

    # For the risk threshold, we need to find lambda such that true risk = alpha
    # Solve for lambda where P(|Y| > lambda) = 0.1
    # We binary search using our MC estimate
    lo, hi = 0.0, 15.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if true_risk(mid) > alpha:
            lo = mid
        else:
            hi = mid
    lambda_risk_boundary = hi

    for trial in range(n_trials):
        if verbose and (trial + 1) % 1000 == 0:
            print(f"  Trial {trial + 1}/{n_trials}")

        result = run_single_trial(
            n=n,
            alpha=alpha,
            B=B,
            beta=beta,
            lambda_grid=lambda_grid,
            n_dirichlet_samples=n_dirichlet_samples,
            rng=rng,
        )

        lambdas_scp[trial] = result["lambda_scp"]
        lambdas_crc[trial] = result["lambda_crc"]
        lambdas_rcps[trial] = result["lambda_rcps"]
        lambdas_hpd[trial] = result["lambda_hpd"]
        interval_lengths_scp[trial] = result["interval_length_scp"]
        interval_lengths_rcps[trial] = result["interval_length_rcps"]
        interval_lengths_hpd[trial] = result["interval_length_hpd"]

    # A trial exceeds the target risk if chosen lambda < lambda_risk_boundary
    # (because smaller lambda means wider interval -> less coverage -> more risk)
    scp_exceed = np.mean(lambdas_scp < lambda_risk_boundary)
    rcps_exceed = np.mean(lambdas_rcps < lambda_risk_boundary)
    hpd_exceed = np.mean(lambdas_hpd < lambda_risk_boundary)

    # Clopper-Pearson CI
    def clopper_pearson_ci(k, n_total, conf=0.95):
        from scipy.stats import binom
        alpha_ci = 1 - conf
        lower = binom.ppf(alpha_ci / 2, n_total, k / n_total) / n_total if k > 0 else 0.0
        upper = binom.ppf(1 - alpha_ci / 2, n_total, k / n_total) / n_total if k < n_total else 1.0
        return lower, upper

    scp_k = int(np.sum(lambdas_scp < lambda_risk_boundary))
    rcps_k = int(np.sum(lambdas_rcps < lambda_risk_boundary))
    hpd_k = int(np.sum(lambdas_hpd < lambda_risk_boundary))

    scp_ci = clopper_pearson_ci(scp_k, n_trials)
    rcps_ci = clopper_pearson_ci(rcps_k, n_trials)
    hpd_ci = clopper_pearson_ci(hpd_k, n_trials)

    results = {
        "n_trials": n_trials,
        "n": n,
        "alpha": alpha,
        "B": B,
        "beta": beta,
        "lambda_risk_boundary": lambda_risk_boundary,
        "lambdas_scp": lambdas_scp,
        "lambdas_rcps": lambdas_rcps,
        "lambdas_hpd": lambdas_hpd,
        "mean_interval_length_scp": np.mean(interval_lengths_scp),
        "mean_interval_length_rcps": np.mean(interval_lengths_rcps),
        "mean_interval_length_hpd": np.mean(interval_lengths_hpd),
        "scp": {
            "exceed_rate": scp_exceed,
            "ci_lower": scp_ci[0],
            "ci_upper": scp_ci[1],
        },
        "rcps": {
            "exceed_rate": rcps_exceed,
            "ci_lower": rcps_ci[0],
            "ci_upper": rcps_ci[1],
        },
        "hpd": {
            "exceed_rate": hpd_exceed,
            "ci_lower": hpd_ci[0],
            "ci_upper": hpd_ci[1],
        },
    }

    return results


def print_results(results):
    """Print results in a formatted table (matching Table 2 in the paper)."""
    print("\n" + "=" * 85)
    print("Table 2: Synthetic Heteroskedastic Experiment Results")
    print("=" * 85)
    print(f"{'Decision Rule':<35} {'Relative Freq.':>14} {'95% CI':>25} {'Mean Int. Len.':>15}")
    print("-" * 85)
    print(f"{'Split Conformal / CRC':<35} {results['scp']['exceed_rate']*100:>13.2f}%  "
          f"[{results['scp']['ci_lower']*100:.2f}%, {results['scp']['ci_upper']*100:.2f}%]  "
          f"{results['mean_interval_length_scp']:>14.2f}")
    print(f"{'RCPS':<35} {results['rcps']['exceed_rate']*100:>13.2f}%  "
          f"[{results['rcps']['ci_lower']*100:.2f}%, {results['rcps']['ci_upper']*100:.2f}%]  "
          f"{results['mean_interval_length_rcps']:>14.2f}")
    print(f"{'Ours (beta=0.95)':<35} {results['hpd']['exceed_rate']*100:>13.2f}%  "
          f"[{results['hpd']['ci_lower']*100:.2f}%, {results['hpd']['ci_upper']*100:.2f}%]  "
          f"{results['mean_interval_length_hpd']:>14.2f}")
    print("-" * 85)
    print()


if __name__ == "__main__":
    print("Running Synthetic Heteroskedastic Experiment (Section 5.2)...")
    results = run_experiment(
        n_trials=10000,
        n=200,
        alpha=0.1,
        B=1.0,
        beta=0.95,
        n_dirichlet_samples=1000,
        seed=42,
    )
    print_results(results)
