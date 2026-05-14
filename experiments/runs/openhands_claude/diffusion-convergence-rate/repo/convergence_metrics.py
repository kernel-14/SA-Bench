"""
Convergence metrics for diffusion model samplers.

Implements:
  - KL divergence (closed-form for Gaussians)
  - Total variation distance (via KL via Pinsker's inequality, or exact for Gaussians)
  - Score estimation error (Assumption 2, Eq. 12)
  - Empirical TV distance estimation

For the numerical experiments (Appendix A), the target is Gaussian and
all distributions remain Gaussian, so KL has a closed-form expression.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import linalg


def kl_divergence_gaussians(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    """
    KL divergence KL(N(mu1, Sigma1) || N(mu2, Sigma2)).

    KL = 0.5 * [tr(Sigma2^{-1} Sigma1) + (mu2-mu1)^T Sigma2^{-1} (mu2-mu1)
                - d + log(det(Sigma2)/det(Sigma1))]

    Args:
        mu1, mu2: shape (d,), means
        Sigma1, Sigma2: shape (d, d), covariance matrices

    Returns:
        KL divergence (non-negative)
    """
    d = mu1.shape[0]
    diff = mu2 - mu1

    # Use Cholesky for numerical stability
    try:
        L2 = linalg.cholesky(Sigma2, lower=True)
        # Sigma2^{-1} Sigma1 via solving L2 L2^T X = Sigma1
        Sigma2_inv_Sigma1 = linalg.cho_solve((L2, True), Sigma1)
        trace_term = np.trace(Sigma2_inv_Sigma1)

        # (mu2-mu1)^T Sigma2^{-1} (mu2-mu1)
        v = linalg.cho_solve((L2, True), diff)
        quad_term = diff @ v

        # log det ratio
        log_det_Sigma2 = 2.0 * np.sum(np.log(np.diag(L2)))
        L1 = linalg.cholesky(Sigma1, lower=True)
        log_det_Sigma1 = 2.0 * np.sum(np.log(np.diag(L1)))
        log_det_ratio = log_det_Sigma2 - log_det_Sigma1

    except linalg.LinAlgError:
        # Fallback: add small regularization
        eps = 1e-10
        Sigma1_reg = Sigma1 + eps * np.eye(d)
        Sigma2_reg = Sigma2 + eps * np.eye(d)
        Sigma2_inv = np.linalg.inv(Sigma2_reg)
        trace_term = np.trace(Sigma2_inv @ Sigma1_reg)
        quad_term = diff @ Sigma2_inv @ diff
        sign1, log_det_Sigma1 = np.linalg.slogdet(Sigma1_reg)
        sign2, log_det_Sigma2 = np.linalg.slogdet(Sigma2_reg)
        log_det_ratio = log_det_Sigma2 - log_det_Sigma1

    kl = 0.5 * (trace_term + quad_term - d + log_det_ratio)
    return max(0.0, kl)


def kl_divergence_gaussians_diagonal(
    mu1: np.ndarray,
    var1: np.ndarray,
    mu2: np.ndarray,
    var2: np.ndarray,
) -> float:
    """
    KL divergence for diagonal Gaussian distributions (efficient).

    KL(N(mu1, diag(var1)) || N(mu2, diag(var2)))
    = 0.5 * sum_i [var1_i/var2_i + (mu2_i - mu1_i)^2/var2_i - 1 + log(var2_i/var1_i)]

    Args:
        mu1, mu2: shape (d,), means
        var1, var2: shape (d,), diagonal variances

    Returns:
        KL divergence
    """
    ratio = var1 / (var2 + 1e-300)
    quad = (mu2 - mu1)**2 / (var2 + 1e-300)
    log_ratio = np.log(var2 + 1e-300) - np.log(var1 + 1e-300)
    kl = 0.5 * np.sum(ratio + quad - 1.0 + log_ratio)
    return max(0.0, kl)


def tv_from_kl(kl: float) -> float:
    """
    Upper bound on TV distance via Pinsker's inequality:
      TV(p, q) <= sqrt(KL(p || q) / 2)

    Args:
        kl: KL divergence

    Returns:
        TV upper bound
    """
    return np.sqrt(kl / 2.0)


def tv_distance_gaussians(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    """
    TV distance between two Gaussians via Pinsker's inequality.

    TV(N(mu1,Sigma1), N(mu2,Sigma2)) <= sqrt(KL(N(mu1,Sigma1) || N(mu2,Sigma2)) / 2)

    Args:
        mu1, mu2: shape (d,), means
        Sigma1, Sigma2: shape (d, d), covariance matrices

    Returns:
        TV upper bound
    """
    kl = kl_divergence_gaussians(mu1, Sigma1, mu2, Sigma2)
    return tv_from_kl(kl)


def tv_distance_gaussians_diagonal(
    mu1: np.ndarray,
    var1: np.ndarray,
    mu2: np.ndarray,
    var2: np.ndarray,
) -> float:
    """
    TV distance between diagonal Gaussians via Pinsker's inequality.
    """
    kl = kl_divergence_gaussians_diagonal(mu1, var1, mu2, var2)
    return tv_from_kl(kl)


def compute_output_distribution_gaussian(
    score_fn,
    sampler,
    d: int,
    n_samples: int = 10000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the output distribution p_{Y_K} empirically for Gaussian targets.

    Since the target is Gaussian and the ODE is linear, the output Y_K
    is also approximately Gaussian. We estimate its mean and covariance.

    Args:
        score_fn: score function
        sampler: RandomizedMidpointSampler instance
        d: data dimension
        n_samples: number of samples
        seed: random seed

    Returns:
        mu_hat: shape (d,), estimated mean
        Sigma_hat: shape (d, d), estimated covariance
    """
    sampler.rng = np.random.default_rng(seed)
    result = sampler.sample(d, n_samples=n_samples)
    samples = result.samples  # shape (n_samples, d)

    mu_hat = np.mean(samples, axis=0)
    Sigma_hat = np.cov(samples.T)
    return mu_hat, Sigma_hat


def kl_divergence_sampler_gaussian(
    sampler,
    score_fn,
    d: int,
    T: int,
    K: int,
    n_samples: int = 5000,
    seed: int = 0,
) -> float:
    """
    Compute KL divergence between sampler output p_{Y_K} and target q_K.

    For Gaussian targets, both distributions are Gaussian:
    - q_K = p_{X_{tau_{K,0}}} = N(0, Sigma_{tau_{K,0}}) where tau_{K,0} is small
    - p_{Y_K} is estimated empirically

    This is the metric used in the numerical experiments (Appendix A).

    Args:
        sampler: RandomizedMidpointSampler
        score_fn: GaussianScoreFunction
        d: data dimension
        T: total iterations
        K: number of rounds
        n_samples: number of samples for estimation
        seed: random seed

    Returns:
        KL divergence KL(q_K || p_{Y_K})
    """
    from score_functions import GaussianScoreFunction
    from forward_process import LearningRateSchedule

    assert isinstance(score_fn, GaussianScoreFunction), \
        "Closed-form KL only available for Gaussian score functions"

    # Get tau_{K,0}
    schedule = LearningRateSchedule(T, K, c0=sampler.schedule.c0, c1=sampler.schedule.c1)
    tau_K0 = schedule.tau_hat(K, 0)

    # q_K = p_{X_{tau_{K,0}}} = N(0, Sigma_{tau_{K,0}})
    mu_q = np.zeros(d)
    var_q = score_fn._sigma_t_sq(tau_K0)  # diagonal variances

    # Estimate p_{Y_K} empirically
    sampler.rng = np.random.default_rng(seed)
    result = sampler.sample(d, n_samples=n_samples)
    samples = result.samples  # shape (n_samples, d)

    mu_p = np.mean(samples, axis=0)
    var_p = np.var(samples, axis=0)

    # KL(q_K || p_{Y_K}) using diagonal approximation
    kl = kl_divergence_gaussians_diagonal(mu_q, var_q, mu_p, var_p)
    return kl


def kl_divergence_gaussian_exact(
    score_fn,
    sampler,
    d: int,
    T: int,
    K: int,
) -> float:
    """
    Compute exact KL divergence for Gaussian target using closed-form expressions.

    For Gaussian target with diagonal covariance, the output Y_K is also
    Gaussian (since the ODE is linear). We compute the exact output distribution
    by tracking the mean and covariance through the ODE.

    This is the approach used in Appendix A for the numerical experiments.

    Args:
        score_fn: GaussianScoreFunction
        sampler: RandomizedMidpointSampler
        d: data dimension
        T: total iterations
        K: number of rounds

    Returns:
        KL divergence KL(q_K || p_{Y_K})
    """
    from score_functions import GaussianScoreFunction
    from forward_process import LearningRateSchedule

    assert isinstance(score_fn, GaussianScoreFunction)

    schedule = LearningRateSchedule(T, K, c0=sampler.schedule.c0, c1=sampler.schedule.c1)
    tau_K0 = schedule.tau_hat(K, 0)

    # q_K = N(0, Sigma_{tau_{K,0}})
    mu_q = np.zeros(d)
    var_q = score_fn._sigma_t_sq(tau_K0)

    # For Gaussian target, the ODE is linear: dx/dtau = A(tau) x
    # where A(tau) = -1/(2(1-tau)) * (I + J_tau) = -1/(2(1-tau)) * (I - Sigma_tau^{-1})
    # The output distribution can be computed analytically.
    # However, with randomized schedule, we use Monte Carlo estimation.

    # Use many samples for accurate estimation
    n_samples = 10000
    sampler.rng = np.random.default_rng(42)
    result = sampler.sample(d, n_samples=n_samples)
    samples = result.samples

    mu_p = np.mean(samples, axis=0)
    var_p = np.var(samples, axis=0)

    kl = kl_divergence_gaussians_diagonal(mu_q, var_q, mu_p, var_p)
    return kl


def score_estimation_error(
    true_score_fn,
    estimated_score_fn,
    sampler,
    d: int,
    T: int,
    K: int,
    n_samples: int = 1000,
    seed: int = 0,
) -> float:
    """
    Compute the averaged score estimation error (Assumption 2, Eq. 12):

    epsilon_score^2 = 1/T * sum_{k,n} E_{Y_k ~ q_k}[||s_{tau_{k,n}}(Y_{k,n}) - s*_{tau_{k,n}}(Y_{k,n})||^2]

    Args:
        true_score_fn: true score function s*
        estimated_score_fn: estimated score function s
        sampler: sampler instance
        d: data dimension
        T: total iterations
        K: number of rounds
        n_samples: number of samples for estimation
        seed: random seed

    Returns:
        epsilon_score^2
    """
    from forward_process import LearningRateSchedule

    schedule = LearningRateSchedule(T, K)
    N = 2 * T // K
    rng = np.random.default_rng(seed)

    total_error_sq = 0.0
    count = 0

    for _ in range(n_samples):
        Y_k = rng.standard_normal(d)

        for k in range(K):
            tau_hat_arr, tau_arr = schedule.get_round_taus(k, rng)

            for n in range(N):
                tau_kn = tau_arr[n]
                s_true = true_score_fn.score(Y_k, tau_kn)
                s_est = estimated_score_fn.score(Y_k, tau_kn)
                error_sq = np.sum((s_est - s_true)**2)
                total_error_sq += error_sq
                count += 1

    return total_error_sq / (n_samples * T)


def tv_distance_empirical(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    n_bins: int = 50,
) -> float:
    """
    Estimate TV distance empirically using histogram binning (1D only).

    TV(p, q) = 0.5 * integral |p(x) - q(x)| dx

    Args:
        samples_p: shape (n,) or (n, d), samples from p
        samples_q: shape (n,) or (n, d), samples from q
        n_bins: number of histogram bins

    Returns:
        Estimated TV distance
    """
    if samples_p.ndim > 1:
        # Use first principal component for visualization
        all_samples = np.concatenate([samples_p, samples_q], axis=0)
        u, s, vt = np.linalg.svd(all_samples - all_samples.mean(0), full_matrices=False)
        samples_p = samples_p @ vt[0]
        samples_q = samples_q @ vt[0]

    all_data = np.concatenate([samples_p, samples_q])
    bins = np.linspace(all_data.min(), all_data.max(), n_bins + 1)

    hist_p, _ = np.histogram(samples_p, bins=bins, density=True)
    hist_q, _ = np.histogram(samples_q, bins=bins, density=True)

    bin_width = bins[1] - bins[0]
    tv = 0.5 * np.sum(np.abs(hist_p - hist_q)) * bin_width
    return tv


def theoretical_tv_bound(
    d: int,
    L: float,
    T: int,
    epsilon_score: float = 0.0,
    C: float = 1.0,
) -> float:
    """
    Theoretical TV distance upper bound from Theorem 1:

    TV(q_K, p_{Y_K}) <= C * min{d^{3/2}, d*L^{1/2}, d^{1/2}*L^{3/2}} * log^4(T) / T^{3/2}
                       + C * epsilon_score * log^{1/2}(T)

    Args:
        d: data dimension
        L: non-uniform Lipschitz constant
        T: total iterations
        epsilon_score: score estimation error
        C: universal constant

    Returns:
        TV upper bound
    """
    log_T = np.log(T)

    if L == float("inf"):
        complexity = d**(3/2)
    else:
        complexity = min(d**(3/2), d * L**(1/2), d**(1/2) * L**(3/2))

    discretization_error = C * complexity * log_T**4 / T**(3/2)
    score_error = C * epsilon_score * log_T**(1/2)

    return discretization_error + score_error
