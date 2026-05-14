"""Randomized midpoint sampler from Section 2.2, Equation (10).

Implements the discretization of the probability flow ODE:
    dY_tau = -1/(2(1-tau)) * (Y_tau + s_tau(Y_tau)) dtau

using the randomized midpoint method with the schedule from Section 2.2.
"""

import numpy as np


def sample_ddpm_randomized_midpoint(
    T, K, hat_alphas, bar_alphas, score_fn, d, rng=None
):
    """Run the randomized midpoint sampler.

    Args:
        T: total iterations
        K: number of rounds
        hat_alphas: deterministic hat_alpha schedule
        bar_alphas: randomized bar_alpha schedule
        score_fn: function (x, bar_alpha) -> estimated score s_t(x)
                  bar_alpha corresponds to bar_alpha_t in the paper
        d: dimension
        rng: numpy random generator

    Returns:
        Y_K: final sample (approximately from target distribution)
        diagnostics: dict with intermediate values for analysis
    """
    if rng is None:
        rng = np.random.default_rng()

    N = 2 * T // K

    # Initialize Y_0 ~ N(0, I_d)
    Y_k = rng.normal(0, 1, size=(d,))

    # For tracking intermediate states
    Y_k_history = [Y_k.copy()]

    for k in range(K):
        # Get the score function index for this round
        # s_{T - kN/2 + 1} corresponds to bar_alpha_{T - kN/2 + 1}
        base_idx = T - k * N // 2 + 1

        # Y_{k,0} = Y_k
        Y_k0 = Y_k.copy()

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

        # Iteratively compute Y_{k,n} for n = 1, ..., N
        Y_kn = [Y_k0]  # Y_kn[0] = Y_{k,0}
        Y_k_current = Y_k0.copy()

        for n in range(1, N + 1):
            # Get relevant tau values
            tau_0 = tau_vals.get(0, 0)
            hat_tau_0 = hat_tau_vals.get(0, hat_tau_vals.get(-1, 0))
            tau_n = tau_vals.get(n, 0)
            tau_nm1 = tau_vals.get(n - 1, 0)
            hat_tau_nm1 = hat_tau_vals.get(n - 1, 0)
            hat_tau_n = hat_tau_vals.get(n, 0)

            # Compute bar_alpha for score function calls
            bar_alpha_0 = 1.0 - tau_0  # for the initial score
            bar_alpha_nm1 = 1.0 - tau_nm1  # for s at step n-1

            # Corresponding t indices
            t_idx_0 = base_idx  # T - kN/2 + 1
            t_idx_nm1 = base_idx - (n - 1)

            # First term: Y_{k,0} / sqrt(1 - tau_{k,0}) is the base
            Y_k0_scaled = Y_k0 / np.sqrt(max(1 - tau_0, 1e-15))

            # Score at initial point
            s0 = score_fn(Y_k0, 1.0 - tau_0)  # s corresponding to bar_alpha_0

            # Initial correction term
            term_init = s0 * (tau_0 - hat_tau_vals.get(0, 0)) / (2 * max(1 - tau_0, 1e-15) ** 1.5)

            # Sum over intermediate steps i = 1, ..., n-1
            sum_intermediate = np.zeros(d)
            for i in range(1, n):
                tau_i = tau_vals.get(i, 0)
                bar_alpha_i = 1.0 - tau_i
                si = score_fn(Y_kn[i], bar_alpha_i)
                sum_intermediate += si * (hat_tau_vals.get(i - 1, 0) - hat_tau_vals.get(i, 0)) / (2 * max(1 - tau_i, 1e-15) ** 1.5)

            # Final term at n-1
            si_final = score_fn(Y_k_current, bar_alpha_nm1)
            term_final = si_final * (hat_tau_nm1 - tau_n) / (2 * max(1 - tau_nm1, 1e-15) ** 1.5)

            Y_kn_scaled = Y_k0_scaled + term_init + sum_intermediate + term_final
            Y_k_current = Y_kn_scaled * np.sqrt(max(1 - tau_n, 1e-15))

            Y_kn.append(Y_k_current.copy())

        # Noise injection step (Equation 11)
        tau_next_0 = 1.0 - bar_alphas[T - (k + 1) * N // 2 + 1] if k < K - 1 else 0.0
        tau_kN = tau_vals.get(N, 0)

        if k < K - 1:
            scale = np.sqrt((1 - tau_next_0) / max(1 - tau_kN, 1e-15))
            noise_scale = np.sqrt((tau_next_0 - tau_kN) / max(1 - tau_kN, 1e-15))
            Z_k = rng.normal(0, 1, size=(d,))
            Y_k = scale * Y_k_current + noise_scale * Z_k
        else:
            Y_k = Y_k_current.copy()

        Y_k_history.append(Y_k.copy())

    diagnostics = {
        'Y_history': Y_k_history,
        'N': N,
    }

    return Y_k, diagnostics


def compute_forward_distribution(X0, bar_alpha):
    """Compute X_t = sqrt(bar_alpha_t) * X_0 + sqrt(1 - bar_alpha_t) * W_t.

    Args:
        X0: sample from target distribution p_data
        bar_alpha: cumulative product of alphas

    Returns:
        X_t: forward process sample
    """
    d = len(X0)
    W = np.random.normal(0, 1, size=(d,))
    return np.sqrt(bar_alpha) * X0 + np.sqrt(1 - bar_alpha) * W
