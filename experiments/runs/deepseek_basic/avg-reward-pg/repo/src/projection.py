"""
Projection operators and the unique value function representation.

Implements Lemma 1 from the paper:
- The orthogonal projection matrix Φ = I - 11^T/|S|
- The unique value function v_φ^π = (I - ΦP^π)^{-1} Φ r^π

Key insight: Since the value function in average reward MDPs is unique only up to an
additive constant, we project onto the subspace orthogonal to the all-ones vector to
obtain a unique representation. This is critical for establishing smoothness.
"""

import numpy as np
from typing import Tuple


def make_projection_matrix(n_states: int) -> np.ndarray:
    """
    Construct the projection matrix Φ = I - 11^T / |S|.
    
    This projects vectors onto the subspace orthogonal to the all-ones vector.
    (Lemma 1, Equation 14)
    
    Args:
        n_states: Number of states |S|
    
    Returns:
        Phi: Projection matrix, shape (n_states, n_states)
    """
    I = np.eye(n_states)
    ones = np.ones((n_states, 1))
    Phi = I - (ones @ ones.T) / n_states
    return Phi


def projected_value_function(P_pi: np.ndarray, r_pi: np.ndarray, 
                              Phi: np.ndarray) -> np.ndarray:
    """
    Compute the unique projected value function v_φ^π.
    
    v_φ^π = (I - Φ P^π)^{-1} Φ r^π
    (Lemma 1, Equation 15)
    
    Args:
        P_pi: Transition matrix under policy π, shape (n_states, n_states)
        r_pi: Expected reward vector under π, shape (n_states,)
        Phi: Projection matrix, shape (n_states, n_states)
    
    Returns:
        v_phi: Projected value function, shape (n_states,)
    """
    n_states = P_pi.shape[0]
    I = np.eye(n_states)
    A = I - Phi @ P_pi
    
    # (I - ΦP^π) is guaranteed to be invertible by Lemma 12
    # The eigenvalues of ΦP^π are all < 1 in absolute value (except one zero eigenvalue)
    v_phi = np.linalg.solve(A, Phi @ r_pi)
    return v_phi


def verify_projection_properties(n_states: int = 5) -> bool:
    """
    Verify key properties of the projection matrix Φ.
    
    Returns:
        True if all properties hold.
    """
    Phi = make_projection_matrix(n_states)
    ones = np.ones(n_states)
    
    # Property 1: Φ 1 = 0
    prop1 = np.allclose(Phi @ ones, np.zeros(n_states))
    
    # Property 2: Φ^2 = Φ (idempotent)
    prop2 = np.allclose(Phi @ Phi, Phi)
    
    # Property 3: ||Φ||_∞ ≤ 2 (Lemma 18, item 1)
    # Operator norm w.r.t. L_∞
    prop3 = np.max(np.abs(Phi).sum(axis=1)) <= 2.0 + 1e-10
    
    return prop1 and prop2 and prop3
