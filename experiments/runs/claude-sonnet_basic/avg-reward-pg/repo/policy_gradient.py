"""
Projected Policy Gradient for Average Reward MDPs.

Implements the projected policy gradient update:
    pi_{k+1} = Proj_Pi [ pi_k + eta * d rho^pi / d pi |_{pi=pi_k} ]

Reference: "Global Convergence of Policy Gradient in Average Reward MDPs"
"""

import numpy as np
from typing import List, Optional, Tuple
from mdp import AverageRewardMDP


def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a vector v onto the probability simplex.
    
    Uses the algorithm from Duchi et al. (2008).
    
    Args:
        v: Input vector of shape (n,)
    Returns:
        w: Projected vector on the simplex
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w


def project_policy(pi: np.ndarray) -> np.ndarray:
    """
    Project a policy matrix onto the space of valid stochastic policies.
    Each row (state) is projected onto the probability simplex.
    
    Args:
        pi: Policy matrix of shape (S, A)
    Returns:
        pi_proj: Projected policy of shape (S, A)
    """
    S, A = pi.shape
    pi_proj = np.zeros_like(pi)
    for s in range(S):
        pi_proj[s] = project_onto_simplex(pi[s])
    return pi_proj


def projected_policy_gradient(
    mdp: AverageRewardMDP,
    eta: float,
    num_iterations: int,
    pi_init: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Run projected policy gradient for average reward MDP.
    
    Update rule:
        pi_{k+1} = Proj_Pi [ pi_k + eta * grad_rho(pi_k) ]
    
    Args:
        mdp: AverageRewardMDP instance
        eta: Step size (learning rate)
        num_iterations: Number of gradient steps
        pi_init: Initial policy (uniform if None)
        seed: Random seed for initialization
    
    Returns:
        rewards: List of average rewards at each iteration
        policies: List of policies at each iteration
    """
    S, A = mdp.S, mdp.A
    
    if pi_init is None:
        rng = np.random.RandomState(seed)
        # Start with uniform policy
        pi = np.ones((S, A)) / A
    else:
        pi = pi_init.copy()
    
    rewards = []
    policies = [pi.copy()]
    
    # Compute initial reward
    rho = mdp.get_average_reward(pi)
    rewards.append(rho)
    
    for k in range(num_iterations):
        # Compute policy gradient
        grad = mdp.get_policy_gradient(pi)
        
        # Gradient ascent step
        pi_new = pi + eta * grad
        
        # Project onto policy space (simplex per state)
        pi_new = project_policy(pi_new)
        
        pi = pi_new
        
        # Compute average reward
        rho = mdp.get_average_reward(pi)
        rewards.append(rho)
        policies.append(pi.copy())
    
    return rewards, policies


def compute_suboptimality_gap(
    mdp: AverageRewardMDP,
    rewards: List[float],
    rho_star: Optional[float] = None,
) -> List[float]:
    """
    Compute suboptimality gap rho* - rho^{pi_k} for each iteration.
    
    Args:
        mdp: AverageRewardMDP instance
        rewards: List of average rewards
        rho_star: Optimal average reward (computed if None)
    
    Returns:
        gaps: List of suboptimality gaps
    """
    if rho_star is None:
        _, rho_star = mdp.get_optimal_policy()
    
    gaps = [rho_star - r for r in rewards]
    return gaps


def theoretical_bound(
    rho_star: float,
    rho_0: float,
    L2: float,
    C_PL: float,
    S: int,
    num_iterations: int,
) -> List[float]:
    """
    Compute the theoretical convergence bound from Theorem 1:
    
    rho* - rho^{pi_k} <= 1 / (1/(rho* - rho^{pi_0}) + nu * k)
    
    where nu = c * (1 + 4c)^{-3/2} and c = 1 / (32 * C_PL^2 * |S| * L2)
    
    Args:
        rho_star: Optimal average reward
        rho_0: Initial average reward
        L2: Smoothness constant
        C_PL: Gradient domination constant
        S: Number of states
        num_iterations: Number of iterations
    
    Returns:
        bounds: List of theoretical upper bounds
    """
    a0 = rho_star - rho_0
    
    # c = 1 / (32 * C_PL^2 * S * L2)
    c = 1.0 / (32.0 * C_PL**2 * S * L2) if L2 > 0 else float('inf')
    
    # nu = c * (1 + 4c)^{-3/2}
    nu = c * (1.0 + 4.0 * c) ** (-1.5)
    
    bounds = []
    for k in range(num_iterations + 1):
        if a0 <= 0:
            bounds.append(0.0)
        else:
            bound = 1.0 / (1.0 / a0 + nu * k)
            bounds.append(bound)
    
    return bounds
