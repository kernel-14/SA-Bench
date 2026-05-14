"""
MDP Complexity Constants.

Implements the computation of constants from Table 1 and Lemma 18 of the paper:
- C_m: maximum operator norm of (I - ΦP^π)^{-1}
- C_p: diameter of transition kernel
- C_r: diameter of reward function
- κ_r: variance of reward function
- L_1^Π: restricted Lipschitz constant of average reward
- L_2^Π: restricted smoothness constant of average reward
- C_PL: distribution mismatch coefficient

These constants capture the complexity of the underlying MDP and appear in the
convergence bounds of Theorem 1.
"""

import numpy as np
from typing import Tuple, List
from .projection import make_projection_matrix


def compute_operator_norm_inf(A: np.ndarray) -> float:
    """
    Compute the operator norm ||A||_∞ = max_{||v||_∞ ≤ 1} ||A v||_∞.
    
    For a matrix A, this is max_i Σ_j |A_{ij}|.
    
    Args:
        A: Matrix
    
    Returns:
        norm: Operator norm w.r.t. L_∞
    """
    return float(np.max(np.abs(A).sum(axis=1)))


def compute_C_m(P_pi_list: List[np.ndarray], Phi: np.ndarray) -> float:
    """
    Compute C_m = max_π ||(I - ΦP^π)^{-1}||_∞.
    
    (Table 1, Lemma 18 item 4)
    
    Args:
        P_pi_list: List of transition matrices under different policies
        Phi: Projection matrix
    
    Returns:
        C_m: Maximum operator norm
    """
    n_states = Phi.shape[0]
    I = np.eye(n_states)
    max_norm = 0.0
    
    for P_pi in P_pi_list:
        M = np.linalg.inv(I - Phi @ P_pi)
        norm = compute_operator_norm_inf(M)
        max_norm = max(max_norm, norm)
    
    return max_norm


def compute_C_p(P: np.ndarray, policy_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Compute C_p = max_{π,π'∈Π} ||P^{π'} - P^π||_∞ / ||π' - π||_2.
    
    Diameter of transition kernel as a function of the policy class.
    (Table 1, Lemma 18 item 5)
    
    Args:
        P: Transition kernel, shape (n_states, n_actions, n_states)
        policy_pairs: List of (π, π') policy pairs to evaluate
    
    Returns:
        C_p: Diameter constant
    """
    max_ratio = 0.0
    
    for pi, pi_prime in policy_pairs:
        pi_diff = pi_prime - pi
        norm_diff = np.sqrt(np.sum(pi_diff ** 2))
        if norm_diff < 1e-12:
            continue
        
        # P^{π'}(s'|s) = Σ_a π'(a|s) P(s'|s,a)
        P_pi = np.einsum('sa,san->sn', pi, P)
        P_pi_prime = np.einsum('sa,san->sn', pi_prime, P)
        
        P_diff = P_pi_prime - P_pi
        norm_P_diff = compute_operator_norm_inf(P_diff)
        
        ratio = norm_P_diff / norm_diff
        max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def compute_C_r(r: np.ndarray, policy_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Compute C_r = max_{π,π'∈Π} ||r^{π'} - r^π||_∞ / ||π' - π||_2.
    
    Diameter of reward function as a function of the policy class.
    (Table 1, Lemma 18 item 6)
    
    Args:
        r: Reward function, shape (n_states, n_actions)
        policy_pairs: List of (π, π') policy pairs to evaluate
    
    Returns:
        C_r: Diameter constant
    """
    max_ratio = 0.0
    
    for pi, pi_prime in policy_pairs:
        pi_diff = pi_prime - pi
        norm_diff = np.sqrt(np.sum(pi_diff ** 2))
        if norm_diff < 1e-12:
            continue
        
        r_pi = np.einsum('sa,sa->s', pi, r)
        r_pi_prime = np.einsum('sa,sa->s', pi_prime, r)
        
        r_diff_norm = np.max(np.abs(r_pi_prime - r_pi))
        ratio = r_diff_norm / norm_diff
        max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def compute_kappa_r(r: np.ndarray, pi_list: List[np.ndarray], 
                     Phi: np.ndarray) -> float:
    """
    Compute κ_r = max_π ||Φ r^π||_∞.
    
    Captures the variance of the reward function.
    (Table 1, Lemma 18 item 3)
    
    Args:
        r: Reward function, shape (n_states, n_actions)
        pi_list: List of policies to evaluate
        Phi: Projection matrix
    
    Returns:
        kappa_r: Reward variance constant
    """
    max_norm = 0.0
    
    for pi in pi_list:
        r_pi = np.einsum('sa,sa->s', pi, r)
        phi_r_pi = Phi @ r_pi
        norm = np.max(np.abs(phi_r_pi))
        max_norm = max(max_norm, norm)
    
    return max_norm


def compute_L1_Pi(C_r: float, C_p: float, C_m: float, kappa_r: float) -> float:
    """
    Compute the restricted Lipschitz constant L_1^Π of the average reward.
    
    L_1^Π = 2(C_r + C_p C_m κ_r + 2(C_m^2 C_p κ_r + C_m C_r))
    (Lemma 3, Equation at line 204)
    
    Args:
        C_r: Reward diameter constant
        C_p: Transition kernel diameter constant
        C_m: Mixing rate constant
        kappa_r: Reward variance constant
    
    Returns:
        L1: Restricted Lipschitz constant
    """
    return 2.0 * (C_r + C_p * C_m * kappa_r + 2.0 * (C_m**2 * C_p * kappa_r + C_m * C_r))


def compute_L2_Pi(C_r: float, C_p: float, C_m: float, kappa_r: float) -> float:
    """
    Compute the restricted smoothness constant L_2^Π of the average reward.
    
    L_2^Π = 4(C_p^2 C_m^2 κ_r + C_p C_m C_r + (C_p + 1)(C_m^2 C_p κ_r + C_m C_r) 
              + 4(C_m^3 C_p^2 κ_r + C_m^2 C_p C_r))
    (Lemma 4, Equation at line 211)
    
    Args:
        C_r: Reward diameter constant
        C_p: Transition kernel diameter constant
        C_m: Mixing rate constant
        kappa_r: Reward variance constant
    
    Returns:
        L2: Restricted smoothness constant
    """
    term1 = C_p**2 * C_m**2 * kappa_r
    term2 = C_p * C_m * C_r
    term3 = (C_p + 1.0) * (C_m**2 * C_p * kappa_r + C_m * C_r)
    term4 = 4.0 * (C_m**3 * C_p**2 * kappa_r + C_m**2 * C_p * C_r)
    
    return 4.0 * (term1 + term2 + term3 + term4)


def compute_C_PL(d_pi_star: np.ndarray, d_pi: np.ndarray) -> float:
    """
    Compute the distribution mismatch coefficient C_PL.
    
    C_PL = max_{π∈Π} max_s d^{π^*}(s) / d^{π}(s)
    (Lemma 7)
    
    Args:
        d_pi_star: Stationary distribution of optimal policy
        d_pi: Stationary distribution of current policy
    
    Returns:
        C_PL: Distribution mismatch coefficient
    """
    # Avoid division by zero
    d_pi_safe = np.maximum(d_pi, 1e-12)
    ratios = d_pi_star / d_pi_safe
    return float(np.max(ratios))


def compute_all_constants(mdp, pi_list: List[np.ndarray], 
                          pi_star: np.ndarray,
                          policy_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
    """
    Compute all MDP complexity constants.
    
    Args:
        mdp: AverageRewardMDP instance
        pi_list: List of policies to evaluate constants over
        pi_star: Optimal policy
        policy_pairs: List of (π, π') pairs for C_p, C_r computation
    
    Returns:
        Dictionary of all constants
    """
    Phi = make_projection_matrix(mdp.n_states)
    
    # Compute transition matrices for all policies
    P_pi_list = [mdp.get_transition_matrix(pi) for pi in pi_list]
    
    C_m = compute_C_m(P_pi_list, Phi)
    C_p = compute_C_p(mdp.P, policy_pairs)
    C_r = compute_C_r(mdp.r, policy_pairs)
    kappa_r = compute_kappa_r(mdp.r, pi_list, Phi)
    L1 = compute_L1_Pi(C_r, C_p, C_m, kappa_r)
    L2 = compute_L2_Pi(C_r, C_p, C_m, kappa_r)
    
    # C_PL for the initial policy
    d_star = mdp.get_stationary_distribution(pi_star)
    d_init = mdp.get_stationary_distribution(pi_list[0])
    C_PL = compute_C_PL(d_star, d_init)
    
    return {
        'C_m': C_m,
        'C_p': C_p,
        'C_r': C_r,
        'kappa_r': kappa_r,
        'L1_Pi': L1,
        'L2_Pi': L2,
        'C_PL': C_PL,
    }
