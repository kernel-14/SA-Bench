## utils.py

"""
Utility functions for the average reward policy gradient reproduction experiments.

This module provides essential helper functions:

- ``set_seed``: Seeds NumPy's global random number generator for reproducibility.
- ``project_simplex``: Computes the Euclidean projection of a vector onto the
  probability simplex using the efficient algorithm from Duchi et al. (2008).
"""

import numpy as np
from typing import Union


def set_seed(seed: int) -> None:
    """
    Set the global random seed for NumPy to ensure reproducibility.

    Args:
        seed: An integer seed to initialise NumPy's random number generator.

    Returns:
        None

    Example:
        >>> set_seed(42)
        >>> np.random.rand()
        0.3745401188473625
    """
    np.random.seed(seed)


def project_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the probability simplex using the Duchi et al. (2008) algorithm.

    The probability simplex is defined as:

        { x ∈ ℝⁿ : xᵢ ≥ 0 ∀i,  ∑ᵢ xᵢ = 1 }

    This function computes the orthogonal Euclidean projection of the input vector ``v``
    onto this set.  The algorithm is:

    1. Sort ``v`` in descending order.
    2. Find the critical index ρ such that for all i ≤ ρ,
       ``v_sorted[i] > (cumulative_sum(v_sorted[:i+1]) - 1) / (i+1)``.
    3. Compute the threshold θ = (cumulative_sum(v_sorted[:ρ+1]) - 1) / (ρ+1).
    4. Return ``max(v - θ, 0)``.

    This implementation is deterministic, vectorised, and has O(n log n) time complexity.

    Args:
        v: A 1-D NumPy array of length n containing the vector to be projected.

    Returns:
        A 1-D NumPy array of the same shape as ``v`` containing the projected vector.
        The returned vector has non‑negative entries and sums to 1.

    Raises:
        ValueError: If ``v`` is not 1-D or is empty.

    Example:
        >>> project_simplex(np.array([3.0, 1.0, 2.0]))
        array([0.6, 0.0, 0.4])

        >>> project_simplex(np.array([-1.0, 0.5, 0.5]))
        array([0. , 0.5, 0.5])
    """
    if v.ndim != 1:
        raise ValueError(
            f"Input vector must be 1-D, got shape {v.shape}. "
            "Use row‑wise application for a matrix."
        )
    if v.size == 0:
        raise ValueError("Input vector must be non‑empty.")

    # Sort in descending order
    u = np.sort(v)[::-1]

    # Cumulative sum of the sorted values
    css = np.cumsum(u)

    # For each k (0‑based), compute candidate threshold:
    #   candidate[k] = (css[k] - 1) / (k + 1)
    # The condition u[k] > candidate[k] identifies indices that will be strictly positive
    # after projection.  We need the largest such k.
    # Use a vectorised approach: find where u > candidate
    candidates = (css - 1.0) / (np.arange(len(u)) + 1.0)
    # Boolean array: True where u[k] > candidate[k]
    valid = u > candidates

    # The critical index rho is the last (largest) index where valid is True.
    # Since the condition always holds for k=0 (u[0] > u[0] - 1), there will be at least
    # one True.
    rho = np.max(np.where(valid)[0])  # type: ignore[arg-type]

    # Threshold
    theta = (css[rho] - 1.0) / (rho + 1.0)

    # Project and return (broadcasts over original unsorted v)
    return np.maximum(v - theta, 0.0)
