"""
Projected Policy Gradient (PPG) algorithm for average-reward MDPs.

Implements the update rule from Equation 6:
    π_{k+1} = Proj_Π[π_k + η ∂ρ^π/∂π|_{π=π_k}]

and the theoretical convergence bound from Theorem 1:
    ρ* - ρ^{π_k} ≤ 1 / (1/(ρ* - ρ^{π_0}) + ν k)

where ν = c (1 + 4c)^{-3/2}  and  c = 1/(32 C_PL^2 |S| L_2^Π).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mdp import AverageRewardMDP
from utils import project_policy


@dataclass
class PPGResult:
    """Results returned by the PPG algorithm."""

    rewards: list[float] = field(default_factory=list)
    suboptimality: list[float] = field(default_factory=list)
    policies: list[NDArray[np.float64]] = field(default_factory=list)
    theoretical_bound: list[float] = field(default_factory=list)


def projected_policy_gradient(
    mdp: AverageRewardMDP,
    pi_init: NDArray[np.float64],
    eta: float,
    n_iterations: int,
    rho_star: float | None = None,
    nu: float | None = None,
    store_policies: bool = False,
) -> PPGResult:
    """
    Run the Projected Policy Gradient algorithm (Equation 6).

    π_{k+1} = Proj_Π[π_k + η ∂ρ^π/∂π|_{π=π_k}]

    Parameters
    ----------
    mdp : AverageRewardMDP
    pi_init : (S, A) array
        Initial policy.
    eta : float
        Step size.  Should satisfy η < 1/L_2^Π.
    n_iterations : int
        Number of gradient ascent steps.
    rho_star : float, optional
        Optimal average reward.  If provided, suboptimality is tracked.
    nu : float, optional
        Convergence rate ν from Theorem 1.  If provided, the theoretical
        bound is computed alongside the empirical curve.
    store_policies : bool
        Whether to store the full policy at every iteration (memory-heavy).

    Returns
    -------
    PPGResult
    """
    result = PPGResult()
    pi = pi_init.copy()

    rho_0 = mdp.average_reward(pi)
    result.rewards.append(rho_0)

    if rho_star is not None:
        gap_0 = rho_star - rho_0
        result.suboptimality.append(gap_0)

        if nu is not None and gap_0 > 0:
            # Theorem 1: ρ* - ρ^{π_k} ≤ 1 / (1/gap_0 + ν k)
            result.theoretical_bound.append(gap_0)

    if store_policies:
        result.policies.append(pi.copy())

    for k in range(1, n_iterations + 1):
        grad = mdp.policy_gradient(pi)
        pi = project_policy(pi + eta * grad)

        rho_k = mdp.average_reward(pi)
        result.rewards.append(rho_k)

        if rho_star is not None:
            gap_k = rho_star - rho_k
            result.suboptimality.append(gap_k)

            if nu is not None and gap_0 > 0:
                bound_k = 1.0 / (1.0 / gap_0 + nu * k)
                result.theoretical_bound.append(bound_k)

        if store_policies:
            result.policies.append(pi.copy())

    return result


def compute_step_size(L2: float, safety_factor: float = 0.5) -> float:
    """
    Compute a safe step size η = safety_factor / L_2^Π.

    The paper requires η < 1/L_2^Π.  We use a safety factor < 1.
    """
    if L2 < 1e-15:
        return 1.0
    return safety_factor / L2


def theoretical_bound(
    gap_0: float,
    nu: float,
    k: int,
) -> float:
    """
    Theorem 1 convergence bound:
        ρ* - ρ^{π_k} ≤ 1 / (1/gap_0 + ν k)
    """
    if gap_0 <= 0:
        return 0.0
    return 1.0 / (1.0 / gap_0 + nu * k)


def exponential_bound(
    gap_0: float,
    c: float,
    k: int,
) -> float:
    """
    Theorem 1 exponential bound for simple MDPs (1/c < 1):
        ρ* - ρ^{π_k} ≤ c^{-k/2} * gap_0^{1/2^k}
    """
    if gap_0 <= 0 or c <= 0:
        return 0.0
    return (1.0 / c) ** (k / 2.0) * gap_0 ** (1.0 / 2.0**k)
