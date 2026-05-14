"""Experiment 5.1: Synthetic Binomial Data.

This experiment reproduces the results from Section 5.1 / Table 1 of the paper.

Setup:
  - Loss distribution: scaled binomial, normalized to B=1
  - Probability of failure: 1 - lambda
  - l(z_i, lambda) = (1/K) * sum_{k=1}^K 1{V_ik > lambda}
  - V_ik ~ Uniform(0, 1) i.i.d.
  - n = 10 calibration samples, K = 4, alpha = 0.4
  - M = 10,000 random trials (data splits)

Since expectation of loss is 1 - lambda, any trial with lambda < 0.6
constitutes risk exceeding alpha = 0.4.

Decision rules compared:
  1. CRC (Conformal Risk Control) - marginal guarantee
  2. RCPS (Risk-controlling Prediction Sets) with Hoeffding UCB
  3. Ours (HPD, beta = 0.95) - the proposed Bayesian quadrature approach
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bayesian_quadrature import (
    compute_L_plus_distribution,
    compute_crc_decision_rule,
    compute_rcps_hoeffding_lambda,
    compute_hpd_lambda,
)
from scipy.stats import binom


def run_single_trial(
    n: int,
    K: int,
    alpha: float,
    B: float,
    beta: float,
    lambda_grid: np.ndarray,
    n_dirichlet_samples: int,
    rng: np.random.Generator,
):
    """Run a single trial of the synthetic binomial experiment.

    Args:
        n: Number of calibration samples.
        K: Number of binomial trials per sample.
        alpha: Target risk threshold.
        B: Maximum loss.
        beta: Confidence level for HPD.
        lambda_grid: Grid of lambda values to search.
        n_dirichlet_samples: Number of Dirichlet MC samples.
        rng: Random generator.

    Returns:
        Dictionary with results for each method.
    """
    # Generate calibration data: n samples, each with K uniform draws
    V = rng.uniform(0, 1, size=(n, K))  # shape (n, K)

    # Loss function: for a given lambda, compute losses for all n samples
    def loss_fn(lam):
        # l_i(lambda) = (1/K) * sum_k 1{V_ik > lambda}
        return np.mean(V > lam, axis=1)

    # True expected loss for any lambda is 1 - lambda
    # Risk exceeds alpha when expected loss > alpha, i.e., 1 - lambda > alpha,
    # so lambda < 1 - alpha. With alpha=0.4, threshold is lambda < 0.6.

    # --- CRC ---
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

    return {
        "lambda_crc": lambda_crc,
        "lambda_rcps": lambda_rcps,
        "lambda_hpd": lambda_hpd,
    }


def run_experiment(
    n_trials: int = 10000,
    n: int = 10,
    K: int = 4,
    alpha: float = 0.4,
    B: float = 1.0,
    beta: float = 0.95,
    n_lambda: int = 200,
    n_dirichlet_samples: int = 1000,
    seed: int = 42,
    verbose: bool = True,
):
    """Run the full synthetic binomial experiment.

    Returns:
        Dictionary of results.
    """
    rng = np.random.default_rng(seed)
    lambda_grid = np.linspace(0, 1, n_lambda)

    # Storage for results
    lambdas_crc = np.zeros(n_trials)
    lambdas_rcps = np.zeros(n_trials)
    lambdas_hpd = np.zeros(n_trials)

    risk_threshold = 1.0 - alpha  # true expected loss = 1 - lambda

    for trial in range(n_trials):
        if verbose and (trial + 1) % 1000 == 0:
            print(f"  Trial {trial + 1}/{n_trials}")

        result = run_single_trial(
            n=n,
            K=K,
            alpha=alpha,
            B=B,
            beta=beta,
            lambda_grid=lambda_grid,
            n_dirichlet_samples=n_dirichlet_samples,
            rng=rng,
        )

        lambdas_crc[trial] = result["lambda_crc"]
        lambdas_rcps[trial] = result["lambda_rcps"]
        lambdas_hpd[trial] = result["lambda_hpd"]

    # Compute relative frequency of exceeding risk threshold
    # Risk exceeds alpha when lambda < 1 - alpha
    crc_exceed = np.mean(lambdas_crc < risk_threshold)
    rcps_exceed = np.mean(lambdas_rcps < risk_threshold)
    hpd_exceed = np.mean(lambdas_hpd < risk_threshold)

    # Compute 95% Clopper-Pearson confidence intervals
    def clopper_pearson_ci(k, n_trials, conf=0.95):
        alpha_ci = 1 - conf
        lower = binom.ppf(alpha_ci / 2, n_trials, k / n_trials) / n_trials if k > 0 else 0.0
        upper = binom.ppf(1 - alpha_ci / 2, n_trials, k / n_trials) / n_trials if k < n_trials else 1.0
        return lower, upper

    crc_k = int(np.sum(lambdas_crc < risk_threshold))
    rcps_k = int(np.sum(lambdas_rcps < risk_threshold))
    hpd_k = int(np.sum(lambdas_hpd < risk_threshold))

    crc_ci = clopper_pearson_ci(crc_k, n_trials)
    rcps_ci = clopper_pearson_ci(rcps_k, n_trials)
    hpd_ci = clopper_pearson_ci(hpd_k, n_trials)

    # Mean risk
    mean_risk_crc = np.mean(1.0 - lambdas_crc)
    mean_risk_hpd = np.mean(1.0 - lambdas_hpd)

    results = {
        "n_trials": n_trials,
        "n": n,
        "K": K,
        "alpha": alpha,
        "B": B,
        "beta": beta,
        "lambdas_crc": lambdas_crc,
        "lambdas_rcps": lambdas_rcps,
        "lambdas_hpd": lambdas_hpd,
        "crc": {
            "exceed_rate": crc_exceed,
            "ci_lower": crc_ci[0],
            "ci_upper": crc_ci[1],
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
        "mean_risk_crc": mean_risk_crc,
        "mean_risk_hpd": mean_risk_hpd,
    }

    return results


def print_results(results):
    """Print results in a formatted table (matching Table 1 in the paper)."""
    print("\n" + "=" * 70)
    print("Table 1: Relative frequency of trials exceeding target risk threshold")
    print("=" * 70)
    print(f"{'Decision Rule':<20} {'Relative Freq.':>15} {'95% CI':>25}")
    print("-" * 60)
    print(f"{'CRC':<20} {results['crc']['exceed_rate']*100:>14.2f}%  "
          f"[{results['crc']['ci_lower']*100:.2f}%, {results['crc']['ci_upper']*100:.2f}%]")
    print(f"{'RCPS':<20} {results['rcps']['exceed_rate']*100:>14.2f}%  "
          f"[{results['rcps']['ci_lower']*100:.2f}%, {results['rcps']['ci_upper']*100:.2f}%]")
    print(f"{'Ours (beta=0.95)':<20} {results['hpd']['exceed_rate']*100:>14.2f}%  "
          f"[{results['hpd']['ci_lower']*100:.2f}%, {results['hpd']['ci_upper']*100:.2f}%]")
    print("-" * 60)
    print(f"\nMean risk (CRC): {results['mean_risk_crc']:.4f}")
    print(f"Mean risk (HPD): {results['mean_risk_hpd']:.4f}")
    print()


if __name__ == "__main__":
    # Run the experiment
    print("Running Synthetic Binomial Experiment (Section 5.1)...")
    results = run_experiment(
        n_trials=10000,
        n=10,
        K=4,
        alpha=0.4,
        B=1.0,
        beta=0.95,
        n_dirichlet_samples=1000,
        seed=42,
    )
    print_results(results)
