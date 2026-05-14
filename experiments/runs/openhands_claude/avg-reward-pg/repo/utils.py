"""
Utility functions shared across modules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def project_simplex(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Project a vector v onto the probability simplex {p : p ≥ 0, Σp_i = 1}.

    Uses the O(n log n) algorithm from Duchi et al. (2008).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def project_policy(pi: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Project a (S, A) matrix onto the space of randomised policies Π.

    Each row is projected independently onto the probability simplex.
    This implements Proj_Π from Equation 6 of the paper.
    """
    pi_proj = np.empty_like(pi)
    for s in range(pi.shape[0]):
        pi_proj[s] = project_simplex(pi[s])
    return pi_proj


def uniform_policy(S: int, A: int) -> NDArray[np.float64]:
    """Return the uniform policy π(a|s) = 1/A for all s, a."""
    return np.ones((S, A)) / A


def random_policy(S: int, A: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Return a random policy by sampling Dirichlet(1,...,1) for each state."""
    pi = rng.dirichlet(np.ones(A), size=S)
    return pi
