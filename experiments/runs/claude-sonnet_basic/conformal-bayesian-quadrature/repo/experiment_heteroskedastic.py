"""
Experiment 5.2: Synthetic Heteroskedastic Data

Reproduces Table 2 from the paper.

Setup:
- X ~ Uniform[0, 4]
- Y | X ~ N(0, X^2)
- Prediction intervals: [-lambda_hat, lambda_hat]
- Loss: miscoverage loss (1 if |Y| > lambda, 0 otherwise)
- Target: alpha = 0.1 (90% coverage)
- n = 200 calibration samples
- M = 10,000 random trials
- Maximum failure rate: 5% (beta = 0.95)

Methods compared:
1. Split Conformal Prediction / CRC (equivalent for miscoverage loss)
2. RCPS (Risk-Controlling Prediction Sets with Hoeffding UCB)
3. Ours (Bayesian Quadrature with beta=0.95)

Note: For miscoverage loss, SCP and CRC are equivalent.
The true risk for a given lambda is:
  R(theta, lambda) = Pr(|Y| > lambda) = E_X[2 * Phi(-lambda/X)]
where X ~ Uniform[0, 4] and Y | X ~ N(0, X^2).
"""

import numpy as np
from scipy import stats, integrate

from methods import (
    split_conformal_prediction,
    rcps_hoeffding,
    bayesian_quadrature_decision_rule,
)


def compute_miscoverage_loss(Y, lam):
    """
    Compute miscoverage loss for prediction interval [-lam, lam].

    Loss = 1 if |Y| > lam, else 0.

    Parameters
    ----------
    Y : array of shape (n,)
        Response values.
    lam : float
        Half-width of prediction interval.

    Returns
    -------
    losses : array of shape (n,)
        Individual miscoverage losses (0 or 1).
    """
    return (np.abs(Y) > lam).astype(float)


def true_risk_heteroskedastic(lam):
    """
    Compute the true expected miscoverage risk analytically.

    R(theta, lambda) = E_X[Pr(|Y| > lam | X)]
                     = (1/4) * integral_0^4 2*Phi(-lam/x) dx

    Uses scipy.integrate.quad for numerical integration.

    Parameters
    ----------
    lam : float
        Half-width of prediction interval.

    Returns
    -------
    risk : float
        True expected miscoverage risk.
    """
    if np.isinf(lam):
        return 0.0
    if lam <= 0:
        return 1.0

    def integrand(x):
        if x <= 0:
            return 1.0  # Pr(|Y| > 0 | X=0) = 1 for continuous Y
        return 2.0 * stats.norm.cdf(-lam / x)

    # Integrate over X ~ Uniform[0, 4], density = 1/4
    result, _ = integrate.quad(integrand, 1e-10, 4.0, limit=200)
    return result / 4.0


def precompute_true_risks(lambda_grid):
    """
    Precompute true risks for all lambda values in the grid.

    Parameters
    ----------
    lambda_grid : array-like
        Grid of lambda values.

    Returns
    -------
    risk_lookup : dict
        Mapping from lambda value to true risk.
    """
    risk_lookup = {}
    for lam in lambda_grid:
        risk_lookup[lam] = true_risk_heteroskedastic(lam)
    return risk_lookup


def run_heteroskedastic_experiment(
    n=200,
    alpha=0.1,
    beta=0.95,
    M=10000,
    B=1.0,
    n_bq_samples=1000,
    lambda_grid=None,
    seed=42,
):
    """
    Run the synthetic heteroskedastic experiment.

    Parameters
    ----------
    n : int
        Number of calibration samples.
    alpha : float
        Target miscoverage level (e.g., 0.1 for 90% coverage).
    beta : float
        Confidence level for BQ method.
    M : int
        Number of random trials.
    B : float
        Upper bound on losses (1 for miscoverage).
    n_bq_samples : int
        Number of Monte Carlo samples for BQ.
    lambda_grid : array-like, optional
        Grid of lambda values.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
    """
    rng = np.random.default_rng(seed)

    if lambda_grid is None:
        # For X ~ Uniform[0,4] and Y|X ~ N(0,X^2), the 90th percentile of |Y|
        # is roughly 4 * 1.28 ≈ 5.1, so a grid up to ~20 is more than enough.
        lambda_grid = np.linspace(0, 20, 201)

    # Precompute true risks for all grid values (much faster than per-trial MC)
    print("  Precomputing true risks for lambda grid...")
    risk_lookup = precompute_true_risks(lambda_grid)
    # Also handle np.inf
    risk_lookup[np.inf] = 0.0

    lambdas_scp  = np.zeros(M)
    lambdas_rcps = np.zeros(M)
    lambdas_bq   = np.zeros(M)

    delta = 1.0 - beta

    for trial in range(M):
        if trial % 1000 == 0:
            print(f"  Trial {trial}/{M}")

        # Generate calibration data
        X_cal = rng.uniform(0, 4, size=n)
        Y_cal = rng.normal(0, X_cal)  # Y | X ~ N(0, X^2)

        Y_trial = Y_cal.copy()

        def losses_fn(lam, _Y=Y_trial):
            return compute_miscoverage_loss(_Y, lam)

        # SCP (equivalent to CRC for miscoverage loss)
        scores = np.abs(Y_cal)
        lam_scp = split_conformal_prediction(scores, alpha)
        lambdas_scp[trial] = lam_scp

        # RCPS
        lam_rcps = rcps_hoeffding(losses_fn, lambda_grid, alpha, delta=delta, B=B)
        lambdas_rcps[trial] = lam_rcps

        # BQ
        lam_bq = bayesian_quadrature_decision_rule(
            losses_fn, lambda_grid, alpha, beta=beta, B=B,
            n_samples=n_bq_samples, rng=rng
        )
        lambdas_bq[trial] = lam_bq

    # Look up true risks from precomputed table.
    # SCP may return values not on the grid (order statistics of |Y|),
    # so we compute those analytically on the fly.
    def get_risk(lam):
        if lam in risk_lookup:
            return risk_lookup[lam]
        return true_risk_heteroskedastic(lam)

    print("  Computing true risks for SCP lambdas...")
    risks_scp  = np.array([get_risk(lam) for lam in lambdas_scp])
    risks_rcps = np.array([risk_lookup.get(lam, get_risk(lam)) for lam in lambdas_rcps])
    risks_bq   = np.array([risk_lookup.get(lam, get_risk(lam)) for lam in lambdas_bq])

    # Compute exceedance
    exceed_scp  = risks_scp  > alpha
    exceed_rcps = risks_rcps > alpha
    exceed_bq   = risks_bq   > alpha

    def clopper_pearson_ci(k, n_trials, confidence=0.95):
        alpha_ci = 1 - confidence
        lower = stats.beta.ppf(alpha_ci / 2, k, n_trials - k + 1) if k > 0 else 0.0
        upper = stats.beta.ppf(1 - alpha_ci / 2, k + 1, n_trials - k) if k < n_trials else 1.0
        return lower, upper

    results = {}
    for name, exceed, lambdas in [
        ("Split Conformal Prediction / CRC", exceed_scp,  lambdas_scp),
        ("RCPS",                             exceed_rcps, lambdas_rcps),
        ("Ours (beta=0.95)",                 exceed_bq,   lambdas_bq),
    ]:
        freq = np.mean(exceed)
        k = int(np.sum(exceed))
        ci_low, ci_high = clopper_pearson_ci(k, M)

        finite_lambdas = lambdas[np.isfinite(lambdas)]
        mean_interval_length = (
            2.0 * np.mean(finite_lambdas) if len(finite_lambdas) > 0 else np.inf
        )

        results[name] = {
            "relative_freq": freq,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean_interval_length": mean_interval_length,
            "lambdas": lambdas,
        }

    return results


def print_table(results, M=10000):
    """Print Table 2 from the paper."""
    print("\n" + "=" * 95)
    print("Table 2: Heteroskedastic experiment — relative frequency exceeding target risk")
    print("=" * 95)
    print(
        f"{'Decision Rule':<42} {'Relative Freq.':<18} {'95% CI':<26} {'Mean PI Length'}"
    )
    print("-" * 95)
    for name, res in results.items():
        freq_pct    = res["relative_freq"] * 100
        ci_low_pct  = res["ci_low"]        * 100
        ci_high_pct = res["ci_high"]       * 100
        mean_len    = res["mean_interval_length"]
        print(
            f"{name:<42} {freq_pct:>6.2f}%{'':<10} "
            f"[{ci_low_pct:.2f}%, {ci_high_pct:.2f}%]{'':<5} {mean_len:.2f}"
        )
    print("=" * 95)
    print("Note: Error bars are 95% Clopper-Pearson confidence intervals.")
    print()


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    print("Running Synthetic Heteroskedastic Experiment (Section 5.2)...")
    print("Parameters: n=200, alpha=0.1, beta=0.95, M=10000")
    print()

    results = run_heteroskedastic_experiment(
        n=200, alpha=0.1, beta=0.95, M=10000,
        B=1.0, n_bq_samples=1000, seed=42
    )

    print_table(results, M=10000)
    print("Done!")
