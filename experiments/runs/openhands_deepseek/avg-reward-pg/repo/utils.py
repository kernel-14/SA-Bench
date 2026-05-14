"""Utility functions for policy gradient experiments."""

import numpy as np
from typing import List, Tuple


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax over last axis."""
    x = x / temperature
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def make_uniform_policy(S: int, A: int) -> np.ndarray:
    """Create uniform random policy pi(s,a) = 1/A for all s,a."""
    return np.ones((S, A)) / A


def make_random_policy(S: int, A: int, seed: int = None) -> np.ndarray:
    """Create random policy with Dirichlet(1,1,...,1) per state."""
    rng = np.random.RandomState(seed)
    pi = rng.rand(S, A)
    return pi / pi.sum(axis=1, keepdims=True)


def compute_diameter_pi(S: int) -> float:
    """Compute theoretical upper bound on diameter of policy class: 2*sqrt(S)."""
    return 2.0 * np.sqrt(S)


def compute_nu(L2_pi: float, C_PL: float, S: int) -> float:
    """Compute convergence rate nu from Theorem 1.

    nu = (1 / (32 * C_PL^2 * |S| * L2^Pi)) * (1 + 4/(32 * C_PL^2 * |S| * L2^Pi))^{-3/2}
    """
    c = 32.0 * C_PL**2 * S * L2_pi
    nu = (1.0 / c) * (1.0 + 4.0 / c) ** (-1.5)
    return nu


def theoretical_bound(k: int, L2_pi: float, C_PL: float, S: int,
                      initial_gap: float) -> float:
    """Theorem 1 bound: 1 / (1/(rho*-rho_0) + nu * k)."""
    nu = compute_nu(L2_pi, C_PL, S)
    return 1.0 / (1.0 / initial_gap + nu * k)


def exponential_bound(k: int, L2_pi: float, C_PL: float, S: int,
                      initial_gap: float) -> float:
    """Exponential bound for simple MDPs (L2^Pi << 1)."""
    c = 32.0 * S * L2_pi * C_PL**2
    if c >= 1.0:
        return float('inf')
    exponent = 2.0 ** k
    return (c ** (k / 2.0)) * (initial_gap ** (1.0 / exponent))
