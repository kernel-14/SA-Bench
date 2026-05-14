## utils.py
"""Standalone utility functions shared across all modules.

This module provides pure, stateless utility functions for:
- Euclidean projection onto the probability simplex
- Stationary distribution computation via power iteration
- L_infinity operator norm computation
- Random policy sampling via Dirichlet distribution
- Projection matrix construction (Phi = I - 11^T / S)

All functions are pure (no side effects, no global state). They depend only
on numpy and have no imports from other project files, making them safe to
import from any module without circular dependency risk.

References:
    Murthy et al., "Global Convergence of Policy Gradient in Average Reward
    MDPs", Lemma 1 (projection matrix), Appendix C (MDP constructions).
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


def project_simplex(v: ndarray) -> ndarray:
    """Project a 1D vector onto the probability simplex.

    Computes the Euclidean projection of v onto the set:
        Delta^n = {x in R^n : x >= 0, sum(x) = 1}

    Uses the sort-based O(n log n) algorithm. The projected point has the
    form x = max(v - theta, 0) for a scalar Lagrange multiplier theta.

    Algorithm:
        1. Sort v in descending order: u = sort(v)[::-1]
        2. Compute cumulative sums: cssv[j] = u[0] + ... + u[j]
        3. Find rho = largest j in {0,...,n-1} s.t. u[j] > (cssv[j] - 1)/(j+1)
        4. Compute theta = (cssv[rho] - 1) / (rho + 1)
        5. Return max(v - theta, 0)

    Args:
        v: 1D numpy array of shape (n,). Can contain any real values.

    Returns:
        1D numpy array of shape (n,) lying on the probability simplex.
        Satisfies: result >= 0 and sum(result) == 1 (up to floating point).

    Example:
        >>> project_simplex(np.array([0.5, 0.3, 0.2]))
        array([0.5, 0.3, 0.2])  # already on simplex
        >>> project_simplex(np.array([1.5, 0.5, -0.5]))
        array([0.833..., 0.166..., 0.0])
    """
    n: int = v.shape[0]

    # Handle trivial case
    if n == 1:
        return np.ones(1, dtype=float)

    # Step 1: Sort in descending order
    u: ndarray = np.sort(v)[::-1]

    # Step 2: Compute cumulative sums (cssv[j] = u[0] + ... + u[j])
    cssv: ndarray = np.cumsum(u)

    # Step 3: Find rho — the largest 0-indexed j such that
    #   u[j] > (cssv[j] - 1) / (j + 1)
    # Equivalently: u[j] * (j + 1) > cssv[j] - 1
    # We compute the condition for all j and find the last True index.
    indices: ndarray = np.arange(1, n + 1, dtype=float)  # j+1 for j=0,...,n-1
    condition: ndarray = u - (cssv - 1.0) / indices > 0.0

    # rho is the last index where condition is True (0-indexed)
    # np.nonzero returns indices where condition is True; take the last one.
    nonzero_indices: ndarray = np.nonzero(condition)[0]
    rho: int = int(nonzero_indices[-1])

    # Step 4: Compute the Lagrange multiplier theta
    theta: float = (cssv[rho] - 1.0) / float(rho + 1)

    # Step 5: Project
    result: ndarray = np.maximum(v - theta, 0.0)

    return result


def power_iteration(
    P: ndarray,
    tol: float = 1.0e-10,
    max_iter: int = 10000,
) -> ndarray:
    """Compute the stationary distribution of a row-stochastic matrix.

    Finds d such that d @ P = d, sum(d) = 1, d >= 0 by iterating
    d_{t+1} = d_t @ P until convergence in L1 norm.

    The matrix P must be row-stochastic: P[s, s'] >= 0 and
    sum_{s'} P[s, s'] = 1 for all s. Under Assumption 1 of the paper
    (irreducible, aperiodic), convergence is guaranteed.

    Args:
        P: Row-stochastic matrix of shape (S, S). P[s, s'] is the
            probability of transitioning from state s to state s'.
        tol: Convergence tolerance in L1 norm. Default 1e-10 matches
            config.yaml value_functions.power_iter_tol.
        max_iter: Maximum number of iterations. Default 10000 matches
            config.yaml value_functions.power_iter_max_iter.

    Returns:
        Stationary distribution d of shape (S,). Satisfies:
            - d >= 0 (clipped to handle floating point)
            - sum(d) == 1 (normalized after iteration)
            - d @ P ≈ d (stationarity, up to tol)

    Note:
        The normalization and clipping at the end handle floating point
        drift that accumulates over many iterations.
    """
    S: int = P.shape[0]

    # Initialize with uniform distribution
    d: ndarray = np.ones(S, dtype=float) / S

    for _ in range(max_iter):
        # Left-multiply: d_new[s'] = sum_s d[s] * P[s, s']
        d_new: ndarray = d @ P

        # Check L1 convergence
        if np.sum(np.abs(d_new - d)) < tol:
            d = d_new
            break

        d = d_new

    # Post-processing: clip negatives (floating point drift) and renormalize
    d = np.maximum(d, 0.0)
    d_sum: float = float(np.sum(d))
    if d_sum > 0.0:
        d = d / d_sum
    else:
        # Fallback: return uniform if something went catastrophically wrong
        d = np.ones(S, dtype=float) / S

    return d


def operator_norm_inf(A: ndarray) -> float:
    """Compute the L_infinity operator norm of a matrix.

    The L_inf operator norm is defined as:
        ||A||_inf = max_{||v||_inf <= 1} ||Av||_inf
                  = max_i sum_j |A[i, j]|
                  = maximum absolute row-sum

    This is used in complexity.py to compute C_m (the operator norm of
    (I - Phi P^pi)^{-1}) and related constants from Table 1/2 of the paper.

    Args:
        A: Matrix of shape (m, n). Can be any real-valued matrix.

    Returns:
        The L_infinity operator norm as a non-negative float.

    Example:
        >>> operator_norm_inf(np.eye(3))
        1.0
        >>> operator_norm_inf(np.zeros((3, 3)))
        0.0
    """
    return float(np.max(np.sum(np.abs(A), axis=1)))


def sample_dirichlet_policy(
    S: int,
    A: int,
    alpha: float = 1.0,
) -> ndarray:
    """Sample a random stochastic policy using the Dirichlet distribution.

    Generates a policy pi of shape (S, A) where each row pi[s, :] is
    independently drawn from Dirichlet(alpha * ones(A)). With alpha=1.0
    (the default from config.yaml complexity.dirichlet_alpha), this is
    equivalent to sampling uniformly from the probability simplex.

    Used in complexity.py to sample random policies for estimating the
    MDP complexity constants C_m, C_p, C_r, kappa_r via Monte Carlo.

    Args:
        S: Number of states. Must be a positive integer.
        A: Number of actions. Must be a positive integer.
        alpha: Dirichlet concentration parameter. Default 1.0 corresponds
            to uniform sampling over the simplex (from config.yaml).
            Larger values produce policies closer to uniform; smaller
            values produce more peaked policies.

    Returns:
        Policy array of shape (S, A). Each row is a valid probability
        vector: pi[s, :] >= 0 and sum(pi[s, :]) == 1 for all s.

    Note:
        This function uses numpy.random.dirichlet, which is affected by
        the global numpy random seed set in Config.__post_init__. No
        local seed management is needed.
    """
    # numpy.random.dirichlet with size=S generates S independent draws,
    # each of length A, from Dirichlet(alpha * ones(A)).
    alpha_vec: ndarray = alpha * np.ones(A, dtype=float)
    policy: ndarray = np.random.dirichlet(alpha=alpha_vec, size=S)
    return policy


def make_projection_matrix(S: int) -> ndarray:
    """Construct the projection matrix Phi that projects onto 1^perp.

    From Lemma 1 of the paper, the orthogonal projection matrix onto the
    subspace perpendicular to the all-ones vector 1 in R^S is:

        Phi = I - (1 * 1^T) / S

    where I is the S x S identity matrix and 1 is the all-ones vector.

    Properties:
        - Phi @ ones(S) = zeros(S)  (null space contains 1)
        - Phi @ Phi = Phi            (idempotent projection)
        - Phi^T = Phi                (symmetric)
        - Eigenvalues are 0 (multiplicity 1) and 1 (multiplicity S-1)

    This matrix is used in value_functions.py to compute the unique
    projected value function v_phi^pi = (I - Phi P^pi)^{-1} Phi r^pi,
    which removes the additive constant ambiguity in the average reward
    Bellman equation.

    Args:
        S: Dimension of the space (number of states). Must be a positive
            integer.

    Returns:
        Projection matrix of shape (S, S). Satisfies all properties
        listed above up to floating point precision.

    Example:
        >>> Phi = make_projection_matrix(3)
        >>> Phi @ np.ones(3)  # should be near zeros
        array([0., 0., 0.])
        >>> Phi @ Phi  # should equal Phi (idempotent)
        array([[0.667, -0.333, -0.333], ...])
    """
    # I - (1 * 1^T) / S
    # np.ones((S, S)) / S creates the outer product 1*1^T / S directly.
    Phi: ndarray = np.eye(S, dtype=float) - np.ones((S, S), dtype=float) / S
    return Phi
