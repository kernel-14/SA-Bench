"""
Numerical experiments to validate the convergence rate of the randomized midpoint sampler.

Reproduces Figure 2 from the paper:
"Instance-dependent Convergence Theory for Diffusion Models"
by Yuchen Jiao and Gen Li (2025).

The experiment:
- Target distribution: d-dimensional Gaussian with zero mean and diagonal covariance
  - First k diagonal entries uniformly distributed in [0, 10]
  - Remaining d-k entries set to zero (degenerate, approximated by small variance)
- Sampler: K=10 rounds, N=2T/K steps per round
- Metric: KL divergence between Y_K and q_K = X_{tau_{K,0}}
- Theoretical rate: O(log^4(T) / T^3)

Three settings are tested (Figure 2):
(a) d=10, k=10
(b) d=100, k=10
(c) d=500, k=100
"""

import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sampler import compute_alpha_hat_schedule


def compute_kl_gaussian_sampler(sigma_sq, T, K, c0=5.0, c1=10.0, n_trials=200, rng=None):
    """
    Compute the KL divergence between Y_K and q_K for a Gaussian target.

    For a Gaussian target X_0 ~ N(0, diag(sigma_sq)), the sampler output Y_{k,n}
    remains Gaussian at each step (since the score function is linear).
    We track the mean and covariance analytically.

    Args:
        sigma_sq: Diagonal variances of the target, shape [d].
        T: Total number of iterations.
        K: Number of rounds.
        c0, c1: Schedule constants.
        n_trials: Number of Monte Carlo trials for averaging over tau randomness.
        rng: Random number generator.

    Returns:
        Average KL divergence.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    d = len(sigma_sq)
    N = 2 * T // K

    hat_alpha = compute_alpha_hat_schedule(T, c0, c1)

    def get_hat_alpha(idx):
        if idx < 0:
            return 1.0
        if idx >= len(hat_alpha):
            return 0.0
        return float(hat_alpha[idx])

    def get_hat_tau(k, n):
        idx = T - k * N // 2 - n
        return 1.0 - get_hat_alpha(idx)

    def sample_tau(k, n):
        tau_lo = get_hat_tau(k, n)
        tau_hi = get_hat_tau(k, n - 1)
        if tau_hi <= tau_lo:
            return tau_lo
        return float(rng.uniform(tau_lo, tau_hi))

    def sigma_tau_fn(tau):
        # Marginal variance: (1-tau)*sigma_sq + tau
        return (1.0 - tau) * sigma_sq + tau

    kl_values = []

    for _ in range(n_trials):
        # Initialize Y_0 ~ N(0, I_d)
        mu = np.zeros(d)
        cov_diag = np.ones(d)

        for k in range(K):
            # Sample tau values for this round
            taus = [sample_tau(k, n) for n in range(N + 1)]

            tau_k0 = taus[0]
            hat_tau_k0 = get_hat_tau(k, 0)

            # Work in normalized coordinates u = x / sqrt(1-tau)
            sqrt_1_minus_tau_k0 = np.sqrt(max(1.0 - tau_k0, 1e-10))
            mu_u = mu / sqrt_1_minus_tau_k0
            cov_u = cov_diag / (1.0 - tau_k0 + 1e-10)

            # First step: score at tau_k0
            # s(x) = -x / sigma_tau, so contribution to u is:
            # u_new = u - coeff_0 * sqrt(1-tau_k0) * u / sigma_tau_k0
            sig_k0 = sigma_tau_fn(tau_k0)
            coeff_0 = (tau_k0 - hat_tau_k0) / (2.0 * sqrt_1_minus_tau_k0 ** 3)
            factor_0 = 1.0 - coeff_0 * sqrt_1_minus_tau_k0 / sig_k0
            mu_u = factor_0 * mu_u
            cov_u = factor_0 ** 2 * cov_u

            mu_x_n = np.zeros(d)
            cov_x_n = np.zeros(d)

            for n in range(1, N + 1):
                tau_kn = taus[n]
                hat_tau_kn_minus1 = get_hat_tau(k, n - 1)
                hat_tau_kn = get_hat_tau(k, n)
                tau_kn_minus1 = taus[n - 1]

                sqrt_1_minus_tau_kn_minus1 = np.sqrt(max(1.0 - tau_kn_minus1, 1e-10))
                sig_kn_minus1 = sigma_tau_fn(tau_kn_minus1)
                coeff_last = (hat_tau_kn_minus1 - tau_kn) / (2.0 * sqrt_1_minus_tau_kn_minus1 ** 3)
                factor_last = 1.0 - coeff_last * sqrt_1_minus_tau_kn_minus1 / sig_kn_minus1
                mu_u_n = factor_last * mu_u
                cov_u_n = factor_last ** 2 * cov_u

                sqrt_1_minus_tau_kn = np.sqrt(max(1.0 - tau_kn, 1e-10))
                mu_x_n = sqrt_1_minus_tau_kn * mu_u_n
                cov_x_n = (1.0 - tau_kn) * cov_u_n

                if n < N:
                    sig_kn = sigma_tau_fn(tau_kn)
                    coeff_mid = (hat_tau_kn_minus1 - hat_tau_kn) / (2.0 * sqrt_1_minus_tau_kn ** 3)
                    factor_mid = 1.0 - coeff_mid * sqrt_1_minus_tau_kn / sig_kn
                    mu_u = factor_mid * mu_u_n
                    cov_u = factor_mid ** 2 * cov_u_n
                else:
                    mu_u = mu_u_n
                    cov_u = cov_u_n

            mu = mu_x_n
            cov_diag = cov_x_n

            # Noise injection (equation 11)
            tau_k1_0 = get_hat_tau(k + 1, 0)
            tau_kN = get_hat_tau(k, N)
            denom = max(1.0 - tau_kN, 1e-10)
            if tau_k1_0 > tau_kN:
                scale_y = np.sqrt((1.0 - tau_k1_0) / denom)
                scale_z_sq = (tau_k1_0 - tau_kN) / denom
                mu = scale_y * mu
                cov_diag = scale_y ** 2 * cov_diag + scale_z_sq

        # Compute KL divergence between Y_K ~ N(mu, diag(cov_diag))
        # and q_K = X_{tau_{K,0}} ~ N(0, Sigma_{tau_{K,0}})
        tau_K0 = get_hat_tau(K, 0)
        sig_K0 = sigma_tau_fn(tau_K0)

        # KL(N(mu, diag(cov)) || N(0, diag(sig_K0)))
        # = 0.5 * [sum(cov/sig_K0) + sum(mu^2/sig_K0) - d + sum(log(sig_K0/cov))]
        cov_safe = np.maximum(cov_diag, 1e-300)
        kl = 0.5 * (np.sum(cov_diag / sig_K0) + np.sum(mu ** 2 / sig_K0)
                    - d + np.sum(np.log(sig_K0) - np.log(cov_safe)))
        kl_values.append(max(float(kl), 0.0))

    return float(np.mean(kl_values))


def theoretical_rate(T, log_power=4.0):
    """Compute the theoretical convergence rate O(log^4(T) / T^3)."""
    return (np.log(T) ** log_power) / (T ** 3)


def run_experiment(d, k_active, T_values, K=10, n_trials=200, seed=42):
    """
    Run the convergence experiment for a given setting.

    Args:
        d: Data dimension.
        k_active: Number of active (non-zero variance) dimensions.
        T_values: List of T values to test.
        K: Number of rounds.
        n_trials: Number of Monte Carlo trials.
        seed: Random seed.

    Returns:
        Dictionary with T_values and corresponding KL divergences.
    """
    rng = np.random.default_rng(seed)

    # Set up target distribution
    # First k_active entries: uniform in [0, 10]
    # Remaining d - k_active entries: 0 (degenerate, approximated by small variance)
    sigma_sq = np.zeros(d)
    sigma_sq[:k_active] = rng.uniform(0, 10, size=k_active)
    sigma_sq[k_active:] = 1e-6  # Small but non-zero for numerical stability

    kl_values = []
    for T in T_values:
        print(f"  d={d}, k={k_active}, T={T}...", flush=True)
        kl = compute_kl_gaussian_sampler(sigma_sq, T, K, n_trials=n_trials, rng=rng)
        kl_values.append(kl)
        print(f"    KL = {kl:.6e}")

    return {
        "T_values": T_values,
        "kl_values": kl_values,
        "d": d,
        "k_active": k_active,
        "K": K,
    }


def generate_figure2(results, T_values, output_dir):
    """Generate Figure 2 from the paper."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, result in zip(axes, results):
        T_arr = np.array(result["T_values"])
        kl_arr = np.array(result["kl_values"])

        # Empirical results (blue line)
        ax.loglog(T_arr, kl_arr, "b-o", label="Empirical KL", linewidth=2, markersize=6)

        # Theoretical rate: fit the constant (black line)
        theo_rates = np.array([theoretical_rate(T) for T in T_values])
        valid = kl_arr > 0
        if np.sum(valid) > 1:
            log_kl = np.log(kl_arr[valid])
            log_theo = np.log(theo_rates[valid])
            const = np.exp(np.mean(log_kl - log_theo))
            theo_fitted = const * theo_rates
            ax.loglog(T_arr, theo_fitted, "k--",
                      label=r" \cdot \log^4(T)/T^3$", linewidth=2)

        ax.set_xlabel("T (iterations)", fontsize=12)
        ax.set_ylabel("KL Divergence", fontsize=12)
        ax.set_title(result["label"], fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Convergence Rate of Randomized Midpoint Sampler\n(Figure 2 from Jiao & Li, 2025)",
        fontsize=13
    )
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"figure2_convergence_rate.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {path}")
    plt.close()


def main():
    """Run all three experiments from Figure 2."""
    # Experiment settings from the paper (Appendix A)
    settings = [
        {"d": 10, "k_active": 10, "label": "(a) d=10, k=10"},
        {"d": 100, "k_active": 10, "label": "(b) d=100, k=10"},
        {"d": 500, "k_active": 100, "label": "(c) d=500, k=100"},
    ]

    # T values to test (log-spaced)
    T_values = [10, 20, 50, 100, 200, 500, 1000]

    results = []
    for setting in settings:
        print(f"\nRunning experiment: {setting['label']}")
        result = run_experiment(
            d=setting["d"],
            k_active=setting["k_active"],
            T_values=T_values,
            K=10,
            n_trials=100,
        )
        result["label"] = setting["label"]
        results.append(result)

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to figures/experiment_results.json")

    generate_figure2(results, T_values, output_dir)
    return results


if __name__ == "__main__":
    main()
