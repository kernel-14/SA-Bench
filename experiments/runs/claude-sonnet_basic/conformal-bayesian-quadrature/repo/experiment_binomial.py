"""
Experiment 5.1: Synthetic Binomial Data

Reproduces Table 1, Figure 3, and Figure 4 from the paper.

Setup:
- Loss function: ell(z_i, lambda) = (1/K) * sum_{k=1}^K 1{V_ik > lambda}
  where V_ik ~ Uniform(0, 1)
- Parameters: n=10, K=4, alpha=0.4, beta=0.95, M=10,000 trials
- True expected loss: E[ell] = 1 - lambda
- Risk exceeds alpha=0.4 when lambda < 0.6

Methods compared:
1. CRC  — Conformal Risk Control
2. RCPS — Risk-Controlling Prediction Sets (Hoeffding UCB)
3. Ours — Bayesian Quadrature (lambda_hpd^0.95)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from methods import (
    conformal_risk_control,
    rcps_hoeffding,
    bayesian_quadrature_decision_rule,
    sample_L_plus,
)


def compute_binomial_losses(V, lam):
    """
    Compute individual losses for the binomial experiment.

    ell(z_i, lambda) = (1/K) * sum_{k=1}^K 1{V_ik > lambda}

    Parameters
    ----------
    V : array of shape (n, K)
        Uniform random variables.
    lam : float
        Threshold parameter.

    Returns
    -------
    losses : array of shape (n,)
    """
    return np.mean(V > lam, axis=1)


def run_binomial_experiment(
    n=10,
    K=4,
    alpha=0.4,
    beta=0.95,
    M=10000,
    B=1.0,
    n_bq_samples=1000,
    lambda_grid=None,
    seed=42,
):
    """
    Run the synthetic binomial experiment.

    Parameters
    ----------
    n : int
        Number of calibration samples.
    K : int
        Number of Bernoulli trials per sample.
    alpha : float
        Target risk level.
    beta : float
        Confidence level for BQ method.
    M : int
        Number of random trials.
    B : float
        Upper bound on losses.
    n_bq_samples : int
        Number of Monte Carlo samples for BQ.
    lambda_grid : array-like, optional
        Grid of lambda values. Defaults to linspace(0, 1, 101).
    seed : int
        Random seed.

    Returns
    -------
    results : dict
    """
    rng = np.random.default_rng(seed)

    if lambda_grid is None:
        lambda_grid = np.linspace(0, 1, 101)

    lambdas_crc  = np.zeros(M)
    lambdas_rcps = np.zeros(M)
    lambdas_bq   = np.zeros(M)

    delta = 1.0 - beta  # failure probability for RCPS

    for trial in range(M):
        if trial % 1000 == 0:
            print(f"  Trial {trial}/{M}")

        # Generate calibration data: V[i, k] ~ Uniform(0, 1)
        V = rng.uniform(0, 1, size=(n, K))

        # Use default-argument capture to avoid closure issues
        def losses_fn(lam, _V=V):
            return compute_binomial_losses(_V, lam)

        lambdas_crc[trial]  = conformal_risk_control(losses_fn, lambda_grid, alpha, B=B)
        lambdas_rcps[trial] = rcps_hoeffding(losses_fn, lambda_grid, alpha, delta=delta, B=B)
        lambdas_bq[trial]   = bayesian_quadrature_decision_rule(
            losses_fn, lambda_grid, alpha, beta=beta, B=B,
            n_samples=n_bq_samples, rng=rng
        )

    # True expected loss = 1 - lambda; risk > alpha iff lambda < 1 - alpha
    threshold = 1.0 - alpha

    exceed_crc  = lambdas_crc  < threshold
    exceed_rcps = lambdas_rcps < threshold
    exceed_bq   = lambdas_bq   < threshold

    def clopper_pearson_ci(k, n_trials, confidence=0.95):
        alpha_ci = 1 - confidence
        lower = stats.beta.ppf(alpha_ci / 2, k, n_trials - k + 1) if k > 0 else 0.0
        upper = stats.beta.ppf(1 - alpha_ci / 2, k + 1, n_trials - k) if k < n_trials else 1.0
        return lower, upper

    results = {}
    for name, exceed, lambdas in [
        ("CRC",             exceed_crc,  lambdas_crc),
        ("RCPS",            exceed_rcps, lambdas_rcps),
        ("Ours (beta=0.95)", exceed_bq,  lambdas_bq),
    ]:
        freq = np.mean(exceed)
        k = int(np.sum(exceed))
        ci_low, ci_high = clopper_pearson_ci(k, M)
        # Mean risk = mean(1 - lambda)
        mean_risk = np.mean(1.0 - lambdas)
        results[name] = {
            "relative_freq": freq,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "mean_risk": mean_risk,
            "lambdas": lambdas,
        }

    return results


def plot_lambda_histograms(results, alpha=0.4, save_path=None):
    """
    Plot histograms of chosen lambda values (Figure 3).

    Left panel:  CRC
    Right panel: Ours (BQ)
    Red region:  lambda < 1 - alpha (risk exceeds alpha)
    """
    threshold = 1.0 - alpha
    bins = np.linspace(0, 1, 21)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, name, title in [
        (axes[0], "CRC",             r"CRC: $\hat{\lambda}_{\mathrm{crc}}$"),
        (axes[1], "Ours (beta=0.95)", r"Ours: $\hat{\lambda}_{\mathrm{hpd}}^{0.95}$"),
    ]:
        lambdas = results[name]["lambdas"]
        exceed_mask = lambdas < threshold

        ax.hist(lambdas[~exceed_mask], bins=bins, color="steelblue", alpha=0.8,
                label=r"Risk $\leq \alpha$")
        ax.hist(lambdas[exceed_mask],  bins=bins, color="red",       alpha=0.8,
                label=r"Risk $> \alpha$")
        ax.axvline(threshold, color="black", linestyle="--",
                   label=rf"$\lambda = {threshold:.1f}$")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_L_plus_density(
    n=10, K=4, B=1.0, n_samples=100000,
    lambda_values=None, seed=42, save_path=None
):
    """
    Plot the probability density of L+ for different lambda values (Figure 4).

    Uses 100,000 Dirichlet samples as stated in the paper.
    """
    if lambda_values is None:
        lambda_values = [0.7, 0.8, 0.9]

    rng = np.random.default_rng(seed)
    V = rng.uniform(0, 1, size=(n, K))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for lam, color in zip(lambda_values, colors):
        losses = compute_binomial_losses(V, lam)
        L_plus_samples = sample_L_plus(losses, B=B, n_samples=n_samples, rng=rng)

        from scipy.stats import gaussian_kde
        kde = gaussian_kde(L_plus_samples)
        x_range = np.linspace(0, B, 500)
        ax.plot(x_range, kde(x_range), color=color, lw=2,
                label=rf"$\lambda = {lam}$")

    ax.set_xlabel(r"$L^+$")
    ax.set_ylabel("Density")
    ax.set_title(r"Probability density of $L^+$")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def print_table(results, M=10000):
    """Print Table 1 from the paper."""
    print("\n" + "=" * 70)
    print("Table 1: Relative frequency of trials exceeding target risk (binomial)")
    print("=" * 70)
    print(f"{'Decision Rule':<30} {'Relative Freq.':<20} {'95% CI'}")
    print("-" * 70)
    for name, res in results.items():
        freq_pct    = res["relative_freq"] * 100
        ci_low_pct  = res["ci_low"]        * 100
        ci_high_pct = res["ci_high"]       * 100
        print(f"{name:<30} {freq_pct:>6.2f}%{'':<12} [{ci_low_pct:.2f}%, {ci_high_pct:.2f}%]")
    print("=" * 70)
    print("Note: Error bars are 95% Clopper-Pearson confidence intervals.")
    print()


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    print("Running Synthetic Binomial Experiment (Section 5.1)...")
    print("Parameters: n=10, K=4, alpha=0.4, beta=0.95, M=10000")
    print()

    results = run_binomial_experiment(
        n=10, K=4, alpha=0.4, beta=0.95, M=10000,
        B=1.0, n_bq_samples=1000, seed=42
    )

    print_table(results, M=10000)

    print("Mean risks:")
    for name, res in results.items():
        print(f"  {name}: {res['mean_risk']:.4f}")
    print()

    # Figure 3: lambda histograms
    plot_lambda_histograms(
        results, alpha=0.4,
        save_path="results/figure3_binomial_histograms.png"
    )

    # Figure 4: L+ density
    plot_L_plus_density(
        n=10, K=4, B=1.0, n_samples=100000,
        lambda_values=[0.7, 0.8, 0.9], seed=42,
        save_path="results/figure4_L_plus_density.png"
    )

    print("Done! Results saved to results/")
