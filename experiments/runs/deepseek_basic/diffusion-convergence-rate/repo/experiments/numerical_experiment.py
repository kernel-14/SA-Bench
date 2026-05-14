"""Numerical experiment from Appendix A.

Validates the theoretical convergence rate by sampling from a Gaussian
target distribution and measuring KL divergence between sampler output
and the approximate forward process starting point X_{tau_{K,0}}.

Target: d-dimensional Gaussian with zero mean and diagonal covariance.
  - First k diagonal entries ~ Unif[0, 10]
  - Remaining d-k entries = 0

Settings: K = 10, N = 2T/K, varying T.

As described in Appendix A, we compute KL(Y_K || q_K) where q_K is the
distribution of X_{tau_{K,0}} (approximately the starting point of the
forward process), and verify the O(log^4(T)/T^3) rate.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.schedule import build_hat_alphas, sample_bar_alphas
from src.score_functions import GaussianScore


def run_experiment(d, k, T_values, c0=10.0, c1=100.0, num_samples=5000):
    """Run the numerical experiment from Appendix A.

    Args:
        d: dimension
        k: number of non-zero variance entries
        T_values: list of T values to test
        c0, c1: schedule constants
        num_samples: number of Monte Carlo samples for KL estimation

    Returns:
        dict with KL divergence estimates for each T
    """
    rng = np.random.default_rng(42)

    # Build target distribution
    sigma2 = np.zeros(d)
    sigma2[:k] = rng.uniform(0, 10, size=k)  # first k: Unif[0,10]
    # rest stay 0 (degenerate)

    score_fn = GaussianScore(sigma2)

    results = {}

    for T in T_values:
        K = 10
        N = 2 * T // K
        print(f"  T={T}, K={K}, N={N}")

        # Build schedule
        hat_alphas = build_hat_alphas(T, c0, c1)
        bar_alphas = sample_bar_alphas(hat_alphas, rng)

        # Compute tau_{K,0}
        base_idx_final = T - (K - 1) * N // 2 + 1
        tau_K0 = 1.0 - bar_alphas[base_idx_final]

        # For a Gaussian target, Y_K is Gaussian with mean 0
        # and we can compute its covariance analytically by
        # simulating the linear transformation.

        # Since the score is linear for Gaussian (s_t^*(x) = -Sigma_t^{-1} x),
        # the sampler is a linear transformation of Gaussian noise.
        # We can compute the exact distribution of Y_K.

        # Actually for the experiment we'll estimate via Monte Carlo
        kl_estimates = []
        for _ in range(3):  # 3 independent runs
            Y_samples = np.zeros((num_samples, d))
            for s in range(num_samples):
                Y = rng.normal(0, 1, size=(d,))
                # Run one round of the sampler (simplified for K=10)
                # We'll use the exact linear transformation for efficiency
                Y_samples[s] = simulate_sampler_gaussian(
                    Y, hat_alphas, bar_alphas, T, K, sigma2, rng
                )

            # Compute KL divergence
            # q_K = X_{tau_{K,0}} = sqrt(1-tau_{K,0}) * X_0 + sqrt(tau_{K,0}) * Z
            # For Gaussian X_0, q_K is Gaussian with covariance:
            # (1-tau_{K,0}) * Sigma_0 + tau_{K,0} * I_d

            # Empirical covariance of Y_K
            cov_Y = np.cov(Y_samples, rowvar=False)

            # Target covariance
            Sigma_q = (1 - tau_K0) * np.diag(sigma2) + tau_K0 * np.eye(d)

            # KL between two Gaussians N(0, cov_Y) and N(0, Sigma_q)
            sign, logdet_Y = np.linalg.slogdet(cov_Y)
            sign_q, logdet_q = np.linalg.slogdet(Sigma_q)
            inv_Sigma_q = np.linalg.inv(Sigma_q)
            trace_term = np.trace(inv_Sigma_q @ cov_Y)
            kl = 0.5 * (trace_term - d + logdet_q - logdet_Y)
            kl_estimates.append(max(kl, 0.0))

        results[T] = {
            'kl_mean': np.mean(kl_estimates),
            'kl_std': np.std(kl_estimates),
            'tau_K0': tau_K0,
        }
        print(f"    KL = {results[T]['kl_mean']:.6f} +/- {results[T]['kl_std']:.6f}")

    return results


def simulate_sampler_gaussian(Y0, hat_alphas, bar_alphas, T, K, sigma2, rng):
    """Simulate one complete run of the sampler for Gaussian target.

    For Gaussian score s_t^*(x) = -Sigma_t^{-1} x, the sampler is linear
    so we can compute the transformation efficiently.
    """
    d = len(sigma2)
    N = 2 * T // K

    Y = Y0.copy()

    for k in range(K):
        base_idx = T - k * N // 2 + 1

        # Build tau values for this round
        tau_vals = {}
        hat_tau_vals = {}
        for n in range(-1, N + 1):
            bar_idx_n = base_idx - n
            if 0 <= bar_idx_n <= T + 1:
                tau_vals[n] = 1.0 - bar_alphas[bar_idx_n]
            hat_idx_n = base_idx - n - 1
            if 0 <= hat_idx_n <= T + 1:
                hat_tau_vals[n] = 1.0 - hat_alphas[hat_idx_n]

        Y_k0 = Y.copy()
        Y_kn = [Y_k0]
        Y_current = Y_k0.copy()

        for n in range(1, N + 1):
            tau_0 = tau_vals.get(0, 0)
            tau_n = tau_vals.get(n, 0)
            tau_nm1 = tau_vals.get(n - 1, 0)
            hat_tau_nm1 = hat_tau_vals.get(n - 1, 0)
            hat_tau_0 = hat_tau_vals.get(0, 0)
            hat_tau_n = hat_tau_vals.get(n, 0)

            # Score function: s(x) = -Sigma_t^{-1} x
            bar_alpha_0 = 1.0 - tau_0
            bar_alpha_nm1 = 1.0 - tau_nm1

            sigma_t0_diag = bar_alpha_0 * sigma2 + (1 - bar_alpha_0)
            sigma_t_nm1_diag = bar_alpha_nm1 * sigma2 + (1 - bar_alpha_nm1)

            Y_k0_scaled = Y_k0 / np.sqrt(max(1 - tau_0, 1e-15))
            term_init = (-Y_k0 / sigma_t0_diag) * (tau_0 - hat_tau_0) / (2 * max(1 - tau_0, 1e-15) ** 1.5)

            sum_intermediate = np.zeros(d)
            for i in range(1, n):
                tau_i = tau_vals.get(i, 0)
                bar_alpha_i = 1.0 - tau_i
                sigma_ti_diag = bar_alpha_i * sigma2 + (1 - bar_alpha_i)
                si = -Y_kn[i] / sigma_ti_diag
                sum_intermediate += si * (hat_tau_vals.get(i - 1, 0) - hat_tau_vals.get(i, 0)) / (2 * max(1 - tau_i, 1e-15) ** 1.5)

            term_final = (-Y_current / sigma_t_nm1_diag) * (hat_tau_nm1 - tau_n) / (2 * max(1 - tau_nm1, 1e-15) ** 1.5)

            Y_kn_scaled = Y_k0_scaled + term_init + sum_intermediate + term_final
            Y_current = Y_kn_scaled * np.sqrt(max(1 - tau_n, 1e-15))
            Y_kn.append(Y_current.copy())

        # Noise injection
        tau_next_0 = 1.0 - bar_alphas[T - (k + 1) * N // 2 + 1] if k < K - 1 else 0.0
        tau_kN = tau_vals.get(N, 0)

        if k < K - 1:
            scale = np.sqrt((1 - tau_next_0) / max(1 - tau_kN, 1e-15))
            noise_scale = np.sqrt((tau_next_0 - tau_kN) / max(1 - tau_kN, 1e-15))
            Y = scale * Y_current + noise_scale * rng.normal(0, 1, size=(d,))
        else:
            Y = Y_current.copy()

    return Y


def fit_rate(T_values, kl_values):
    """Fit the theoretical rate O(log^4(T)/T^3) to empirical KL values.

    Returns the constant factor C such that KL ~ C * log^4(T)/T^3.
    """
    log_terms = (np.log(T_values)) ** 4
    T_cubed = T_values ** 3
    theoretical = log_terms / T_cubed

    # Linear regression: kl = C * theoretical
    C = np.sum(kl_values * theoretical) / np.sum(theoretical ** 2)

    fitted = C * theoretical
    return C, fitted


def main():
    """Reproduce the numerical experiment from Appendix A, Figure 2."""
    print("=" * 60)
    print("Numerical Experiment: Appendix A")
    print("=" * 60)

    # Experiment configurations from Figure 2:
    configs = [
        {'d': 10, 'k': 10, 'label': '(a) d=10, k=10'},
        {'d': 100, 'k': 10, 'label': '(b) d=100, k=10'},
        {'d': 500, 'k': 100, 'label': '(c) d=500, k=100'},
    ]

    for cfg in configs:
        print(f"\n{'='*40}")
        print(f"Configuration: {cfg['label']}")
        print(f"{'='*40}")

        # T values: choose range appropriate for dimension
        d = cfg['d']
        k = cfg['k']

        if d <= 10:
            T_values = np.array([50, 100, 200, 400, 800])
        elif d <= 100:
            T_values = np.array([100, 200, 400, 800, 1600])
        else:
            T_values = np.array([200, 400, 800, 1600, 3200])

        results = run_experiment(d, k, T_values, num_samples=2000)

        kl_values = np.array([results[T]['kl_mean'] for T in T_values])
        C, fitted = fit_rate(T_values, kl_values)

        print(f"\n  Fitted rate: KL ~ {C:.6f} * log^4(T)/T^3")
        print(f"\n  {'T':>8s}  {'KL':>12s}  {'log^4(T)/T^3':>14s}  {'Fitted':>12s}")
        print(f"  {'-'*8}  {'-'*12}  {'-'*14}  {'-'*12}")
        for i, T in enumerate(T_values):
            log_term = (np.log(T)) ** 4
            T3 = T ** 3
            print(f"  {T:8d}  {kl_values[i]:12.6e}  {log_term/T3:14.6e}  {fitted[i]:12.6e}")

        # Compute approximate convergence rate
        # If KL ~ T^{-alpha}, then log(KL) ~ -alpha * log(T) + const
        # For Figure 2, the paper shows KL ~ log^4(T)/T^3
        # So rate should be approximately T^{-3} ignoring log factors
        log_T = np.log(T_values)
        log_KL = np.log(kl_values + 1e-15)
        slope, intercept = np.polyfit(log_T, log_KL, 1)
        print(f"\n  Empirical rate: KL ~ T^{{{slope:.3f}}} (theoretical: T^{-3})")


if __name__ == '__main__':
    main()
