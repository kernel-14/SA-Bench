"""
Policy Gradient for Average Reward MDPs.

Implements the projected policy gradient algorithm and the theoretical analysis
from the paper "Global Convergence of Policy Gradient in Average Reward MDPs".

Key components:
- Average reward policy gradient theorem (Equation 5)
- Projected policy gradient update (Equation 6)
- Convergence bounds from Theorem 1

The policy gradient is computed analytically (not estimated) since this is the
planning setting with exact gradients.
"""

import numpy as np
from typing import Tuple, List, Optional
from .mdp import AverageRewardMDP
from .projection import make_projection_matrix
from .constants import (
    compute_L2_Pi, compute_C_PL,
    compute_operator_norm_inf
)


def compute_policy_gradient(mdp: AverageRewardMDP, pi: np.ndarray) -> np.ndarray:
    """
    Compute the average reward policy gradient ∇_π ρ^π.
    
    Using the average reward policy gradient theorem (Equation 5):
    ∂ρ/∂π(s,a) = d^π(s) Q^π(s,a)
    
    Args:
        mdp: The MDP
        pi: Current policy, shape (n_states, n_actions)
    
    Returns:
        grad: Policy gradient, shape (n_states, n_actions)
    """
    d_pi = mdp.get_stationary_distribution(pi)
    Q_pi = mdp.compute_q_function(pi)
    
    # Gradient: d^π(s) * Q^π(s,a)
    grad = d_pi[:, np.newaxis] * Q_pi
    return grad


def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the probability simplex.
    
    Uses the Euclidean projection algorithm. This is the per-state projection
    in the policy gradient update (Equation 6).
    
    Args:
        v: Vector to project, shape (n_actions,)
    
    Returns:
        Projected vector on simplex
    """
    # Sort in descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, len(v) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        theta = 0.0
    else:
        theta = (cssv[rho[-1]] - 1) / (rho[-1] + 1)
    w = np.maximum(v - theta, 0)
    return w


def project_policy(pi: np.ndarray) -> np.ndarray:
    """
    Project a policy matrix onto the space of randomized policies Π.
    
    Each row (state) is projected onto the probability simplex independently.
    This implements Proj_Π from Equation 6.
    
    Args:
        pi: Policy to project, shape (n_states, n_actions)
    
    Returns:
        Projected policy
    """
    n_states = pi.shape[0]
    pi_proj = np.zeros_like(pi)
    for s in range(n_states):
        pi_proj[s] = project_onto_simplex(pi[s])
    return pi_proj


def projected_policy_gradient_step(mdp: AverageRewardMDP, pi: np.ndarray, 
                                    eta: float) -> Tuple[np.ndarray, dict]:
    """
    Perform one step of projected policy gradient ascent.
    
    π_{k+1} = Proj_Π[π_k + η ∇ρ^π|_{π=π_k}]
    (Equation 6)
    
    Args:
        mdp: The MDP
        pi: Current policy, shape (n_states, n_actions)
        eta: Step size (learning rate)
    
    Returns:
        pi_next: Next policy
        info: Dict with diagnostic information
    """
    grad = compute_policy_gradient(mdp, pi)
    pi_next = project_policy(pi + eta * grad)
    
    info = {
        'gradient': grad,
        'gradient_norm': np.linalg.norm(grad),
        'policy_change': np.linalg.norm(pi_next - pi),
    }
    
    return pi_next, info


def run_projected_policy_gradient(
    mdp: AverageRewardMDP,
    n_iterations: int,
    eta: float,
    pi_0: Optional[np.ndarray] = None,
    seed: int = 42
) -> dict:
    """
    Run the projected policy gradient algorithm.
    
    Args:
        mdp: The MDP
        n_iterations: Number of iterations T
        eta: Step size
        pi_0: Initial policy; if None, uniform random is used
        seed: Random seed
    
    Returns:
        Dictionary with:
            - policies: List of policies at each iteration
            - average_rewards: List of average rewards
            - policy_changes: List of ||π_{k+1} - π_k||
            - gradient_norms: List of gradient norms
            - optimality_gaps: List of ρ^* - ρ^{π_k}
    """
    rng = np.random.RandomState(seed)
    
    if pi_0 is None:
        # Initialize with uniform random policy
        pi_0 = rng.rand(mdp.n_states, mdp.n_actions)
        pi_0 = pi_0 / pi_0.sum(axis=1, keepdims=True)
    
    policies = [pi_0.copy()]
    rewards = [mdp.average_reward(pi_0)]
    policy_changes = []
    gradient_norms = []
    
    pi = pi_0.copy()
    
    for k in range(n_iterations):
        pi_next, info = projected_policy_gradient_step(mdp, pi, eta)
        
        policies.append(pi_next.copy())
        rewards.append(mdp.average_reward(pi_next))
        policy_changes.append(info['policy_change'])
        gradient_norms.append(info['gradient_norm'])
        
        pi = pi_next
    
    # Compute optimality gaps (using max reward across trajectory as proxy for optimum)
    # For small MDPs, we can compute the true optimum via value iteration
    rho_star = max(rewards)  # proxy; for rigorous comparison, use value iteration
    
    optimality_gaps = [rho_star - r for r in rewards]
    
    return {
        'policies': policies,
        'average_rewards': rewards,
        'policy_changes': policy_changes,
        'gradient_norms': gradient_norms,
        'optimality_gaps': optimality_gaps,
    }


def compute_theoretical_bound(
    mdp: AverageRewardMDP,
    pi_0: np.ndarray,
    pi_star: np.ndarray,
    L2: float,
    C_PL: float,
    n_iterations: int,
) -> dict:
    """
    Compute the theoretical convergence bounds from Theorem 1.
    
    Returns both the sublinear and exponential bounds.
    
    Args:
        mdp: The MDP
        pi_0: Initial policy
        pi_star: Optimal policy
        L2: Restricted smoothness constant L_2^Π
        C_PL: Distribution mismatch coefficient
        n_iterations: Number of iterations
    
    Returns:
        Dictionary with theoretical bounds
    """
    n_states = mdp.n_states
    rho_0 = mdp.average_reward(pi_0)
    rho_star = mdp.average_reward(pi_star)
    suboptimality_0 = rho_star - rho_0
    
    # Constants from Theorem 1
    c = 1.0 / (32.0 * C_PL**2 * n_states * L2)
    
    # Sublinear convergence parameter ν
    nu = c * (1.0 + 4.0 * c) ** (-1.5)
    
    # Sublinear bound: a_k ≤ 1 / (1/a_0 + ν k)
    # where a_k = ρ^* - ρ^{π_k}
    bounds_sublinear = []
    for k in range(n_iterations + 1):
        if suboptimality_0 > 0:
            bound = 1.0 / (1.0 / suboptimality_0 + nu * k)
        else:
            bound = 0.0
        bounds_sublinear.append(bound)
    
    # Exponential bound for simple MDPs (when 1/c = 32|S|L2 C_PL^2 < 1)
    inv_c = 32.0 * n_states * L2 * C_PL**2
    is_simple = inv_c < 1.0
    bounds_exponential = []
    if is_simple:
        for k in range(n_iterations + 1):
            bound = (inv_c) ** (k / 2.0) * suboptimality_0 ** (1.0 / (2 ** k))
            bounds_exponential.append(bound)
    
    return {
        'bounds_sublinear': bounds_sublinear,
        'bounds_exponential': bounds_exponential if is_simple else None,
        'is_simple': is_simple,
        'c': c,
        'nu': nu,
        'inv_c': inv_c,
        'suboptimality_0': suboptimality_0,
    }
