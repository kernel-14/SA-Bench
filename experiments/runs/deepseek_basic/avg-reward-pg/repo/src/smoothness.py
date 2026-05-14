"""
Smoothness analysis of the average reward.

Implements the theoretical proofs from Appendix A:
- Lemma 1: Projection matrix and unique value function
- Lemma 2 (14): Lipschitzness of the projected value function v_φ^π
- Lemma 2 (15): Smoothness of the projected value function v_φ^π
- Lemma 3 (16): Lipschitzness of the average reward ρ^π
- Lemma 4 (17): Smoothness of the average reward ρ^π

These results are the theoretical foundation for the convergence analysis.
They prove that the average reward is smooth without assuming it a priori.
"""

import numpy as np
from typing import Tuple, Callable
from .projection import make_projection_matrix, projected_value_function


def directional_derivative_first_order(
    mdp, pi: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """
    Compute the first-order directional derivative of the projected value function
    in direction u (where u = π' - π).
    
    Following Lemma 14: ∂v_φ^{π_α} / ∂α
    
    Args:
        mdp: AverageRewardMDP instance
        pi: Policy π
        u: Direction vector (π' - π)
    
    Returns:
        dv_dalpha: First derivative of v_φ w.r.t. α
    """
    n_states = mdp.n_states
    Phi = make_projection_matrix(n_states)
    
    P_pi = mdp.get_transition_matrix(pi)
    r_pi = mdp.get_reward_vector(pi)
    
    # M(π) = (I - ΦP^π)^{-1}
    I = np.eye(n_states)
    M = np.linalg.inv(I - Phi @ P_pi)
    
    # P^u(s'|s) = Σ_a u(a|s) P(s'|s,a)
    P_u = np.einsum('sa,san->sn', u, mdp.P)
    # r^u(s) = Σ_a u(a|s) r(s,a)
    r_u = np.einsum('sa,sa->s', u, mdp.r)
    
    # ∂M/∂α = M (∂(ΦP^{π_α})/∂α) M = M Φ P^u M
    # ∂v_φ/∂α = M Φ P^u M Φ r^π + M Φ r^u
    dv_dalpha = M @ Phi @ P_u @ M @ Phi @ r_pi + M @ Phi @ r_u
    
    return dv_dalpha


def directional_derivative_second_order(
    mdp, pi: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """
    Compute the second-order directional derivative of the projected value function
    in direction u.
    
    Following Lemma 15: ∂²v_φ^{π_α} / ∂α²
    
    Args:
        mdp: AverageRewardMDP instance
        pi: Policy π
        u: Direction vector (π' - π)
    
    Returns:
        d2v_dalpha2: Second derivative of v_φ w.r.t. α
    """
    n_states = mdp.n_states
    Phi = make_projection_matrix(n_states)
    
    P_pi = mdp.get_transition_matrix(pi)
    r_pi = mdp.get_reward_vector(pi)
    
    I = np.eye(n_states)
    M = np.linalg.inv(I - Phi @ P_pi)
    
    P_u = np.einsum('sa,san->sn', u, mdp.P)
    r_u = np.einsum('sa,sa->s', u, mdp.r)
    
    # From Lemma 15:
    # ∂²v_φ/∂α² = M Φ P^u M Φ P^u M Φ r^π + M Φ P^u M Φ P^u M Φ r^π
    #            + M Φ P^u M Φ r^u + M Φ P^u M Φ r^u
    term1 = M @ Phi @ P_u @ M @ Phi @ P_u @ M @ Phi @ r_pi
    term2 = M @ Phi @ P_u @ M @ Phi @ P_u @ M @ Phi @ r_pi
    term3 = M @ Phi @ P_u @ M @ Phi @ r_u
    term4 = M @ Phi @ P_u @ M @ Phi @ r_u
    
    d2v_dalpha2 = term1 + term2 + term3 + term4
    
    return d2v_dalpha2


def average_reward_directional_derivative(
    mdp, pi: np.ndarray, u: np.ndarray
) -> float:
    """
    Compute the directional derivative of the average reward ρ^π in direction u.
    
    Using the policy gradient theorem: ⟨∇ρ, u⟩ = Σ_s d^π(s) Σ_a u(s,a) Q^π(s,a)
    
    Args:
        mdp: The MDP
        pi: Policy π
        u: Direction vector
    
    Returns:
        drho_dalpha: Directional derivative (scalar)
    """
    d_pi = mdp.get_stationary_distribution(pi)
    Q_pi = mdp.compute_q_function(pi)
    
    drho_dalpha = np.sum(d_pi[:, np.newaxis] * u * Q_pi)
    return float(drho_dalpha)


def verify_smoothness_constant(
    mdp, 
    pi_list: list,
    policy_pairs: list,
    L2_theoretical: float,
    tol: float = 1.0,
) -> dict:
    """
    Empirically verify that the smoothness constant L_2^Π holds.
    
    Checks that for all policy pairs (π, π'):
    |⟨π' - π, ∇²ρ(π' - π)⟩| ≤ (L_2^Π / 2) ||π' - π||²
    
    Args:
        mdp: The MDP
        pi_list: List of policies to test
        policy_pairs: List of (π, π') pairs
        L2_theoretical: Theoretical smoothness constant
        tol: Tolerance factor (should be ≤ 1 for theoretical bound to hold)
    
    Returns:
        Dictionary with verification results
    """
    max_ratio = 0.0
    violations = 0
    
    for pi, pi_prime in policy_pairs:
        u = pi_prime - pi
        norm_u_sq = np.sum(u ** 2)
        if norm_u_sq < 1e-12:
            continue
        
        # Compute directional second derivative of ρ
        # We compute this via finite differences of the directional derivative
        # or analytically via the second-order expression
        
        # For empirical verification, use finite differences
        eps = 1e-5
        d_pi_plus = mdp.get_stationary_distribution(pi + eps * u)
        d_pi_minus = mdp.get_stationary_distribution(pi - eps * u)
        d_pi = mdp.get_stationary_distribution(pi)
        
        Q_pi_plus = mdp.compute_q_function(pi + eps * u)
        Q_pi_minus = mdp.compute_q_function(pi - eps * u)
        
        # First derivative at pi+eps and pi-eps
        drho_plus = np.sum(d_pi_plus[:, np.newaxis] * u * Q_pi_plus)
        drho_minus = np.sum(d_pi_minus[:, np.newaxis] * u * Q_pi_minus)
        
        # Second derivative approximation
        d2rho = (drho_plus - drho_minus) / (2 * eps)
        
        ratio = abs(d2rho) / (L2_theoretical * norm_u_sq / 2.0)
        max_ratio = max(max_ratio, ratio)
        
        if ratio > tol:
            violations += 1
    
    return {
        'max_ratio': max_ratio,
        'violations': violations,
        'total_pairs': len(policy_pairs),
        'smoothness_holds': max_ratio <= tol,
    }


def verify_lipschitz_constant(
    mdp,
    policy_pairs: list,
    L1_theoretical: float,
    tol: float = 1.0,
) -> dict:
    """
    Empirically verify that the Lipschitz constant L_1^Π holds.
    
    Checks that for all policy pairs (π, π'):
    |⟨π' - π, ∇ρ(π)⟩| ≤ L_1^Π ||π' - π||
    
    Args:
        mdp: The MDP
        policy_pairs: List of (π, π') pairs
        L1_theoretical: Theoretical Lipschitz constant
        tol: Tolerance factor
    
    Returns:
        Dictionary with verification results
    """
    max_ratio = 0.0
    violations = 0
    
    for pi, pi_prime in policy_pairs:
        u = pi_prime - pi
        norm_u = np.sqrt(np.sum(u ** 2))
        if norm_u < 1e-12:
            continue
        
        d_pi = mdp.get_stationary_distribution(pi)
        Q_pi = mdp.compute_q_function(pi)
        
        # ⟨∇ρ(π), u⟩ = Σ_s d^π(s) Σ_a u(s,a) Q^π(s,a)
        directional_deriv = np.sum(d_pi[:, np.newaxis] * u * Q_pi)
        
        ratio = abs(directional_deriv) / (L1_theoretical * norm_u)
        max_ratio = max(max_ratio, ratio)
        
        if ratio > tol:
            violations += 1
    
    return {
        'max_ratio': max_ratio,
        'violations': violations,
        'total_pairs': len(policy_pairs),
        'lipschitz_holds': max_ratio <= tol,
    }
