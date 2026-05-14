"""
Extension to Discounted Reward MDPs (Section 3.2 and Appendix D).

Implements the analysis showing that the average reward smoothness framework
can be applied to discounted reward MDPs, yielding instance-dependent bounds
that can be tighter than the existing MDP-agnostic bounds.

Key result: The smoothness constant L_2^Π for discounted return can be
expressed in terms of MDP complexity constants, with:
- C_m = ||(I - γP^π)^{-1}|| ≤ 1/(1-γ)
- C_p ≤ γ√|A|
- C_r ≤ √|A|
- κ_r ≤ 1

Leading to iteration complexity O(|S| L_2^Π / ε) compared to the
state-of-the-art O(|S||A|/((1-γ)^5 ε)).
"""

import numpy as np
from typing import Tuple, Optional
from .mdp import AverageRewardMDP
from .constants import compute_L2_Pi, compute_C_PL


class DiscountedRewardMDP:
    """
    Discounted reward MDP for the extension in Section 3.2.
    
    The discounted return is:
    ρ_γ^π = μ^T (1 - γP^π)^{-1} r^π
    
    where γ ∈ [0,1) is the discount factor.
    """
    
    def __init__(self, mdp: AverageRewardMDP, gamma: float, 
                 mu: Optional[np.ndarray] = None):
        """
        Initialize the discounted MDP.
        
        Args:
            mdp: The base average reward MDP
            gamma: Discount factor γ ∈ [0,1)
            mu: Initial state distribution; if None, uniform
        """
        self.mdp = mdp
        self.gamma = gamma
        self.n_states = mdp.n_states
        self.n_actions = mdp.n_actions
        
        if mu is None:
            self.mu = np.ones(self.n_states) / self.n_states
        else:
            self.mu = mu
    
    def get_transition_matrix(self, pi: np.ndarray) -> np.ndarray:
        """Get transition matrix under policy π."""
        return self.mdp.get_transition_matrix(pi)
    
    def get_reward_vector(self, pi: np.ndarray) -> np.ndarray:
        """Get expected reward vector under policy π."""
        return self.mdp.get_reward_vector(pi)
    
    def compute_discounted_return(self, pi: np.ndarray) -> float:
        """
        Compute the discounted return ρ_γ^π.
        
        ρ_γ^π = μ^T (I - γP^π)^{-1} r^π
        
        Args:
            pi: Policy
        
        Returns:
            Discounted return
        """
        P_pi = self.mdp.get_transition_matrix(pi)
        r_pi = self.mdp.get_reward_vector(pi)
        I = np.eye(self.n_states)
        
        v = np.linalg.solve(I - self.gamma * P_pi, r_pi)
        return float(np.dot(self.mu, v))
    
    def compute_value_function(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute the discounted value function v_γ^π.
        
        v_γ^π = (I - γP^π)^{-1} r^π
        
        Args:
            pi: Policy
        
        Returns:
            Value function
        """
        P_pi = self.mdp.get_transition_matrix(pi)
        r_pi = self.mdp.get_reward_vector(pi)
        I = np.eye(self.n_states)
        
        v = np.linalg.solve(I - self.gamma * P_pi, r_pi)
        return v
    
    def compute_discounted_C_m(self, pi_list: list) -> float:
        """
        Compute Č_m = max_π ||(I - γP^π)^{-1}||_∞ for discounted setting.
        
        Note: Č_m ≤ 1/(1-γ) always.
        
        Args:
            pi_list: List of policies
        
        Returns:
            Discounted C_m
        """
        max_norm = 0.0
        I = np.eye(self.n_states)
        
        for pi in pi_list:
            P_pi = self.mdp.get_transition_matrix(pi)
            M = np.linalg.inv(I - self.gamma * P_pi)
            norm = np.max(np.abs(M).sum(axis=1))
            max_norm = max(max_norm, norm)
        
        return max_norm
    
    def compute_discounted_L2(self, C_r: float, C_p: float, 
                               C_m: float, kappa_r: float) -> float:
        """
        Compute the restricted smoothness constant for discounted return.
        
        Same formula as average reward case but with discounted constants.
        L_2^Π = 8(C_m^3 C_p^2 κ_r + C_m^2 C_p C_r)
        
        (Lemma 28, Equation at line 1145)
        
        Args:
            C_r: Reward diameter
            C_p: Transition kernel diameter (already includes γ)
            C_m: Discounted mixing constant
            kappa_r: Reward variance
        
        Returns:
            Discounted smoothness constant
        """
        # Note: In discounted case, the smoothness constant from Lemma 28 is:
        # 8(C_m^3 C_p^2 κ_r + C_m^2 C_p C_r)
        return 8.0 * (C_m**3 * C_p**2 * kappa_r + C_m**2 * C_p * C_r)
    
    def iteration_complexity_bound(self, L2: float, C_PL: float, 
                                     epsilon: float) -> float:
        """
        Compute the iteration complexity to achieve ε-suboptimality.
        
        From Section 3.2: O(|S| L_2^Π / ε) iterations.
        
        Args:
            L2: Smoothness constant
            C_PL: Distribution mismatch coefficient
            epsilon: Target suboptimality
        
        Returns:
            Approximate number of iterations needed
        """
        # From the sublinear bound: a_k ≤ 1/(1/a_0 + νk)
        # To get a_k ≤ ε, we need k ≈ 1/(ν ε)
        # ν = c (1+4c)^{-3/2}, c = 1/(32 C_PL^2 |S| L_2^Π)
        c = 1.0 / (32.0 * C_PL**2 * self.n_states * L2)
        nu = c * (1.0 + 4.0 * c) ** (-1.5)
        
        # Approximate iterations needed
        k_approx = 1.0 / (nu * epsilon)
        return k_approx
    
    def compare_with_state_of_art(self, epsilon: float, 
                                    C_PL: float = 1.0) -> dict:
        """
        Compare our instance-dependent bound with state-of-the-art bounds.
        
        Xiao (2022a): O(|S||A|/((1-γ)^5 ε))
        Ours: O(|S| L_2^Π / ε)
        
        Args:
            epsilon: Target suboptimality
            C_PL: Distribution mismatch coefficient
        
        Returns:
            Comparison dictionary
        """
        # State-of-the-art bound (Xiao 2022a)
        sota_bound = self.n_states * self.n_actions / ((1 - self.gamma)**5 * epsilon)
        
        # Our instance-dependent bound
        # We need L_2^Π which depends on the MDP constants
        # For a worst-case estimate, use the bounds from Lemma 18
        C_m_bound = 1.0 / (1 - self.gamma)
        C_p_bound = self.gamma * np.sqrt(self.n_actions)
        C_r_bound = np.sqrt(self.n_actions)
        kappa_r_bound = 1.0
        
        L2_worst_case = self.compute_discounted_L2(
            C_r_bound, C_p_bound, C_m_bound, kappa_r_bound
        )
        
        our_bound = self.n_states * L2_worst_case / epsilon
        
        return {
            'sota_bound': sota_bound,
            'our_bound': our_bound,
            'improvement_factor': sota_bound / max(our_bound, 1e-12),
            'L2_worst_case': L2_worst_case,
        }


def discounted_policy_gradient_step(
    mdp_disc: DiscountedRewardMDP, pi: np.ndarray, eta: float
) -> Tuple[np.ndarray, dict]:
    """
    Perform one step of projected policy gradient for discounted MDP.
    
    π_{k+1} = Proj_Π[π_k + η ∇ρ_γ^π|_{π=π_k}]
    
    Args:
        mdp_disc: Discounted MDP
        pi: Current policy
        eta: Step size
    
    Returns:
        (pi_next, info)
    """
    from .policy_gradient import project_policy
    
    n_states = mdp_disc.n_states
    n_actions = mdp_disc.n_actions
    gamma = mdp_disc.gamma
    
    P_pi = mdp_disc.mdp.get_transition_matrix(pi)
    r_pi = mdp_disc.mdp.get_reward_vector(pi)
    I = np.eye(n_states)
    
    # Compute state visitation distribution
    # d_γ^π = (1-γ) μ^T (I - γP^π)^{-1}
    d_gamma = (1 - gamma) * mdp_disc.mu @ np.linalg.inv(I - gamma * P_pi)
    
    # Compute Q-function for discounted setting
    v_pi = np.linalg.solve(I - gamma * P_pi, r_pi)
    
    Q_pi = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            next_v = np.dot(mdp_disc.mdp.P[s, a, :], v_pi)
            Q_pi[s, a] = mdp_disc.mdp.r[s, a] + gamma * next_v
    
    # Policy gradient for discounted MDP
    # ∇ρ_γ^π(s,a) = (1/(1-γ)) * d_γ^π(s) * Q^π(s,a) * (1-γ)
    # Simplified: ∇ρ_γ^π(s,a) = d_γ^π(s) * Q^π(s,a)  
    grad = d_gamma[:, np.newaxis] * Q_pi
    
    pi_next = project_policy(pi + eta * grad)
    
    info = {
        'gradient_norm': np.linalg.norm(grad),
        'policy_change': np.linalg.norm(pi_next - pi),
    }
    
    return pi_next, info
