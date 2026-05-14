"""
MDP complexity constants from Table 1 of the paper.

Constants:
  C_m  = max_π ||(I - Φ P^π)^{-1}||_∞        (mixing rate)
  C_p  = max_{π,π'} ||P^{π'} - P^π||_∞ / ||π' - π||_2   (transition diameter)
  C_r  = max_{π,π'} ||r^{π'} - r^π||_∞ / ||π' - π||_2   (reward diameter)
  κ_r  = max_π ||Φ r^π||_∞                    (reward variance)
  L_1^Π = 2(C_r + C_p C_m κ_r + 2(C_m^2 C_p κ_r + C_m C_r))
  L_2^Π = 4(C_p^2 C_m^2 κ_r + C_p C_m C_r
            + (C_p+1)(C_m^2 C_p κ_r + C_m C_r)
            + 4(C_m^3 C_p^2 κ_r + C_m^2 C_p C_r))
  C_PL  = max_{π,s} d^{π*}(s) / d^π(s)

All operator norms are the L_∞ induced norm (max row absolute sum).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mdp import AverageRewardMDP


@dataclass
class MDPComplexity:
    """Container for all MDP complexity constants."""

    C_m: float
    C_p: float
    C_r: float
    kappa_r: float
    L1: float
    L2: float
    C_PL: float

    def __str__(self) -> str:
        return (
            f"MDPComplexity(\n"
            f"  C_m    = {self.C_m:.4f}\n"
            f"  C_p    = {self.C_p:.4f}\n"
            f"  C_r    = {self.C_r:.4f}\n"
            f"  kappa_r= {self.kappa_r:.4f}\n"
            f"  L1     = {self.L1:.4f}\n"
            f"  L2     = {self.L2:.4f}\n"
            f"  C_PL   = {self.C_PL:.4f}\n"
            f")"
        )


def _inf_norm(A: NDArray[np.float64]) -> float:
    """L_∞ induced operator norm: max row sum of absolute values."""
    return float(np.max(np.sum(np.abs(A), axis=1)))


def _sample_policies(
    S: int,
    A: int,
    n_samples: int,
    rng: np.random.Generator,
) -> list[NDArray[np.float64]]:
    """Sample n_samples random policies plus the uniform policy."""
    policies = [np.ones((S, A)) / A]
    for _ in range(n_samples - 1):
        pi = rng.dirichlet(np.ones(A), size=S)
        policies.append(pi)
    return policies


def compute_C_m(
    mdp: AverageRewardMDP,
    policies: list[NDArray[np.float64]],
) -> float:
    """
    C_m = max_π ||(I - Φ P^π)^{-1}||_∞

    Lemma 18 item 4: C_m ≤ 2 C_e |S| / (1 - λ).
    """
    S = mdp.S
    I = np.eye(S)
    C_m = 0.0
    for pi in policies:
        P_pi = mdp.transition_matrix(pi)
        A_mat = I - mdp.Phi @ P_pi
        try:
            M = np.linalg.inv(A_mat)
        except np.linalg.LinAlgError:
            continue
        C_m = max(C_m, _inf_norm(M))
    return C_m


def compute_C_p(
    mdp: AverageRewardMDP,
    policies: list[NDArray[np.float64]],
) -> float:
    """
    C_p = max_{π,π'∈Π} ||P^{π'} - P^π||_∞ / ||π' - π||_2

    Lemma 18 item 5: C_p ≤ sqrt(|A|).
    """
    C_p = 0.0
    n = len(policies)
    for i in range(n):
        for j in range(i + 1, n):
            pi, pi_prime = policies[i], policies[j]
            diff_norm = float(np.linalg.norm(pi_prime - pi))
            if diff_norm < 1e-12:
                continue
            P_pi = mdp.transition_matrix(pi)
            P_pi_prime = mdp.transition_matrix(pi_prime)
            kernel_diff = _inf_norm(P_pi_prime - P_pi)
            C_p = max(C_p, kernel_diff / diff_norm)
    return C_p


def compute_C_r(
    mdp: AverageRewardMDP,
    policies: list[NDArray[np.float64]],
) -> float:
    """
    C_r = max_{π,π'∈Π} ||r^{π'} - r^π||_∞ / ||π' - π||_2

    Lemma 18 item 6: C_r ≤ sqrt(|A|).
    """
    C_r = 0.0
    n = len(policies)
    for i in range(n):
        for j in range(i + 1, n):
            pi, pi_prime = policies[i], policies[j]
            diff_norm = float(np.linalg.norm(pi_prime - pi))
            if diff_norm < 1e-12:
                continue
            r_pi = mdp.reward_vector(pi)
            r_pi_prime = mdp.reward_vector(pi_prime)
            reward_diff = float(np.max(np.abs(r_pi_prime - r_pi)))
            C_r = max(C_r, reward_diff / diff_norm)
    return C_r


def compute_kappa_r(
    mdp: AverageRewardMDP,
    policies: list[NDArray[np.float64]],
) -> float:
    """
    κ_r = max_π ||Φ r^π||_∞

    Lemma 18 item 3: κ_r ≤ 2.
    """
    kappa_r = 0.0
    for pi in policies:
        r_pi = mdp.reward_vector(pi)
        kappa_r = max(kappa_r, float(np.max(np.abs(mdp.Phi @ r_pi))))
    return kappa_r


def compute_C_PL(
    mdp: AverageRewardMDP,
    pi_star: NDArray[np.float64],
    policies: list[NDArray[np.float64]],
) -> float:
    """
    C_PL = max_{π,s} d^{π*}(s) / d^π(s)

    This is the gradient domination constant from Lemma 7.
    """
    d_star = mdp.stationary_distribution(pi_star)
    C_PL = 0.0
    for pi in policies:
        d_pi = mdp.stationary_distribution(pi)
        # Avoid division by zero for states with zero stationary probability
        mask = d_pi > 1e-12
        if mask.any():
            ratio = np.max(d_star[mask] / d_pi[mask])
            C_PL = max(C_PL, float(ratio))
    return C_PL


def compute_smoothness_constants(
    C_m: float, C_p: float, C_r: float, kappa_r: float
) -> tuple[float, float]:
    """
    Compute L_1^Π and L_2^Π from Lemmas 3 and 4.

    L_1^Π = 2(C_r + C_p C_m κ_r + 2(C_m^2 C_p κ_r + C_m C_r))

    L_2^Π = 4(C_p^2 C_m^2 κ_r + C_p C_m C_r
              + (C_p+1)(C_m^2 C_p κ_r + C_m C_r)
              + 4(C_m^3 C_p^2 κ_r + C_m^2 C_p C_r))
    """
    L1 = 2.0 * (
        C_r
        + C_p * C_m * kappa_r
        + 2.0 * (C_m**2 * C_p * kappa_r + C_m * C_r)
    )

    L2 = 4.0 * (
        C_p**2 * C_m**2 * kappa_r
        + C_p * C_m * C_r
        + (C_p + 1.0) * (C_m**2 * C_p * kappa_r + C_m * C_r)
        + 4.0 * (C_m**3 * C_p**2 * kappa_r + C_m**2 * C_p * C_r)
    )

    return L1, L2


def compute_convergence_rate(
    C_PL: float, S: int, L2: float
) -> tuple[float, float]:
    """
    Compute the convergence rate ν from Theorem 1.

    c = 1 / (32 C_PL^2 |S| L_2^Π)
    ν = c * (1 + 4c)^{-3/2}
    """
    if L2 < 1e-15 or C_PL < 1e-15:
        return float("inf"), float("inf")
    c = 1.0 / (32.0 * C_PL**2 * S * L2)
    nu = c * (1.0 + 4.0 * c) ** (-1.5)
    return c, nu


def compute_all_complexity(
    mdp: AverageRewardMDP,
    pi_star: NDArray[np.float64],
    n_samples: int = 200,
    seed: int = 0,
) -> MDPComplexity:
    """
    Compute all MDP complexity constants by sampling policies.

    Parameters
    ----------
    mdp : AverageRewardMDP
    pi_star : (S, A) array
        Optimal policy (used for C_PL).
    n_samples : int
        Number of random policies to sample for the max computations.
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)
    policies = _sample_policies(mdp.S, mdp.A, n_samples, rng)
    # Always include the optimal policy
    policies.append(pi_star)

    C_m = compute_C_m(mdp, policies)
    C_p = compute_C_p(mdp, policies)
    C_r = compute_C_r(mdp, policies)
    kappa_r = compute_kappa_r(mdp, policies)
    L1, L2 = compute_smoothness_constants(C_m, C_p, C_r, kappa_r)
    C_PL = compute_C_PL(mdp, pi_star, policies)

    return MDPComplexity(
        C_m=C_m,
        C_p=C_p,
        C_r=C_r,
        kappa_r=kappa_r,
        L1=L1,
        L2=L2,
        C_PL=C_PL,
    )
