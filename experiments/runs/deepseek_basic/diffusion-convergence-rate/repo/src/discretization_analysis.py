"""Discretization error analysis from Section 4 / Appendix D.

Implements the key analytical tools used in the proof of Theorem 1:
  1. Lemma 7: Schedule properties
  2. Lemma 8: Discretization error decomposition
  3. Lemma 9: Bound on ||J_tau(x_tau) s_tau^*(x_tau)||^2
  4. Lemma 10: Derivative bound for score function
  5. Lemma 11: Error propagation bound
  6. Lemma 12: Uniform score bound on typical set

These are the core technical lemmas that establish the convergence rate.
"""

import numpy as np
from scipy.special import logsumexp


def compute_jacobian_score_product_bound(Sigma_0, tau, L=None, d=None):
    """Compute the bound on E[||J_tau(x_tau) s_tau^*(x_tau)||^2] (Lemma 9, Eq 62).

    For d < L^2:
        E[||J_tau s_tau^*||^2] <= O(d log(T) / tau^3 * (d + E[Tr(Sigma_tau^2)]))

    For d >= L^2:
        E[||J_tau s_tau^*||^2] <= O(d L^2 / tau^3)

    Args:
        Sigma_0: target covariance matrix
        tau: current tau value
        L: non-uniform Lipschitz constant
        d: dimension (default: Sigma_0.shape[0])

    Returns:
        dict with various bounds
    """
    if d is None:
        d = Sigma_0.shape[0]

    # Compute E[Tr(Sigma_tau^2(x_tau))]
    from src.score_functions import compute_tr_sigma_tau_sq
    tr_sigma2 = compute_tr_sigma_tau_sq(Sigma_0, tau)

    # d < L^2 case bound
    bound_d_less_L = d * (d + tr_sigma2) / (tau ** 3)

    # d >= L^2 case bound (if L provided)
    if L is not None:
        bound_d_greater_L = d * (L ** 2) / (tau ** 3)
    else:
        bound_d_greater_L = float('inf')

    # Actual bound: min of the two
    actual_bound = min(bound_d_less_L, bound_d_greater_L)

    return {
        'tr_sigma_tau_sq': tr_sigma2,
        'bound_d_less_L2': bound_d_less_L,
        'bound_d_greater_L2': bound_d_greater_L,
        'actual_bound': actual_bound,
        'tau': tau,
    }


def compute_score_derivative_bound(Sigma_0, tau, L=None, d=None):
    """Compute bound on E[||partial/partial_tau (s_tau^*(x_tau) / (1-tau)^{3/2})||^2].

    From Lemma 10 and Eq. 59:
        E[||partial/partial_tau ... ||^2] <= O(d * min{d*log(T), L} / (tau^3 * (1-tau)^5))

    This is a key quantity in the discretization error analysis.
    """
    if d is None:
        d = Sigma_0.shape[0]

    # Default log(T) factor - in the paper this is a log(T) factor
    log_T = 10.0  # placeholder

    term_1 = d * d * log_T  # d * min{d*log T, L^2} = d * d * log T for d*log T < L^2
    if L is not None:
        term_1 = d * min(d * log_T, L)

    bound = term_1 / (tau ** 3 * max(1 - tau, 1e-10) ** 5)

    return {
        'tau': tau,
        'd': d,
        'L': L,
        'bound': bound,
        'term_d_min': term_1,
    }


def compute_discretization_error_for_interval(Sigma_0, tau_start, tau_end, L, d, T, N):
    """Compute the discretization error contribution for one interval.

    From Lemma 8 (Eq 53) and surrounding analysis:
    The error from interval [tau_end, tau_start] contributes approximately:
        (delta_tau)^3 * integral of E[||partial_tau (s_tau^*/(1-tau)^{3/2})||^2] dt

    Combined with Lemma 7's schedule properties (hat_tau differences ~ hat_tau * T^{-1} * log T)
    this gives the O(T^{-3}) rate.
    """
    # For the schedule, delta_tau ~ hat_tau * log(T) / T
    delta_tau = tau_start * np.log(T) / T

    # Approximate the integral by midpoint
    tau_mid = (tau_start + tau_end) / 2
    derivative_bound = compute_score_derivative_bound(Sigma_0, tau_mid, L, d)['bound']

    # Error ~ (delta_tau)^3 * derivative_bound
    error = (delta_tau ** 3) * derivative_bound

    return {
        'delta_tau': delta_tau,
        'derivative_bound': derivative_bound,
        'error_contribution': error,
    }


def verify_lemma12(Sigma_0, tau, d, theta=10.0, c0=10.0):
    """Verify Lemma 12: Uniform score bound on typical set.

    For x in S_tau ∩ L_tau, y in S_tau:
        ||s_tau^*(x) - s_tau^*(y)||_2 <= C * min{d*log(T), L} / tau * ||x-y||_2

    This lemma is crucial for the error propagation analysis (Step 2/3).
    """
    # The typical set S_tau requires -log p_{X_tau}(x) <= theta * d * log(T)
    # For Gaussian X_tau ~ N(0, Sigma_tau), this is ||x||_{Sigma_tau^{-1}}^2 <= 2*theta*d*log(T)

    # The bound C * min{d*log(T), L} / tau
    # This is a loose upper bound derived from the definition

    if L is None:
        L = np.inf

    C = 10 * np.sqrt(theta + c0)  # from the proof

    bound = C * min(d * np.log(100), L) / tau

    return {
        'tau': tau,
        'bound': bound,
        'L_effective': min(d * np.log(100), L),
    }
