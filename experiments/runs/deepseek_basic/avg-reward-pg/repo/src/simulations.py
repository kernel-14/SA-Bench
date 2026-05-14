"""
Simulations from Section 4 of the paper.

Reproduces the three experiments:
1. Convergence with different action and state space sizes (Figure 1a)
2. Convergence with different reward functions (Figure 1b)
3. Convergence with different transition kernels (Figure 2)

See Appendix C for detailed simulation setup.
"""

import numpy as np
from typing import Dict, List, Tuple
import time

from .mdp import AverageRewardMDP
from .policy_gradient import run_projected_policy_gradient
from .constants import compute_all_constants


def make_transition_kernel_symmetric(n_states: int, n_actions: int) -> np.ndarray:
    """
    Create the transition kernel from Appendix C.1.
    
    P(·|s,·) = (1_{S×A} + 1/S) / 2
    
    So P(i|s,i) = (1 + 1/S)/2 and P(i|s,j) = 1/(2S) for i ≠ j.
    This is a symmetric, ergodic transition kernel.
    """
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            # Action index a maps to some next state (deterministic part)
            target = a % n_states
            for s_prime in range(n_states):
                if s_prime == target:
                    P[s, a, s_prime] = (1.0 + 1.0 / n_states) / 2.0
                else:
                    P[s, a, s_prime] = 1.0 / (2.0 * n_states)
    return P


def make_reward_max_variance(n_states: int, n_actions: int) -> np.ndarray:
    """
    Create max-variance reward function from Appendix C.1.
    
    Half the actions get reward 1, half get -1, for every state.
    """
    r = np.zeros((n_states, n_actions))
    half_actions = n_actions // 2
    for s in range(n_states):
        r[s, :half_actions] = 1.0
        r[s, half_actions:] = -1.0
    return r


def make_uniform_transition_kernel(n_states: int, n_actions: int) -> np.ndarray:
    """
    Create uniform transition kernel from Appendix C.3.
    
    P(s'|s,a) = 1/S for all s, a, s'
    """
    P = np.ones((n_states, n_actions, n_states)) / n_states
    return P


def make_nonuniform_transition_kernel(n_states: int, n_actions: int) -> np.ndarray:
    """
    Create non-uniform stochastic transition kernel from Appendix C.3.
    
    P(i|s,i) = 1/(2S) + 1/2 and P(i|s,j) = 1/(2S) for i ≠ j.
    """
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            target = a % n_states
            for s_prime in range(n_states):
                if s_prime == target:
                    P[s, a, s_prime] = 1.0 / (2.0 * n_states) + 0.5
                else:
                    P[s, a, s_prime] = 1.0 / (2.0 * n_states)
    return P


def make_deterministic_transition_kernel(n_states: int, n_actions: int, 
                                          seed: int = 0) -> np.ndarray:
    """
    Create deterministic transition kernel from Appendix C.3.
    
    P(·|s,·) is an S×A matrix that is a random permutation of the identity matrix.
    Each action leads to a different deterministic next state.
    """
    rng = np.random.RandomState(seed)
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        # Random permutation of states for the actions
        perm = rng.permutation(n_states)
        for a in range(n_actions):
            target = perm[a % n_states]
            P[s, a, target] = 1.0
    return P


def make_reward_variance(n_states: int, n_actions: int, 
                          variance_level: str, seed: int = 0) -> np.ndarray:
    """
    Create reward function with specified variance level, from Appendix C.2.
    
    All states have r(s,a) = 0 except for state s_0 where:
    - No variance: all actions get reward 1
    - Low variance: 1/8 of actions get -1, rest 1
    - High variance: 1/4 of actions get -1, rest 1
    - Max variance: 1/2 of actions get -1, rest 1
    
    Args:
        n_states: Number of states
        n_actions: Number of actions
        variance_level: One of 'none', 'low', 'high', 'max'
        seed: Random seed for which actions get -1
    
    Returns:
        r: Reward matrix
    """
    rng = np.random.RandomState(seed)
    r = np.zeros((n_states, n_actions))
    
    s0 = 0  # The designated state
    
    if variance_level == 'none':
        r[s0, :] = 1.0
    elif variance_level == 'low':
        frac_neg = 1.0 / 8.0
        n_neg = max(1, int(n_actions * frac_neg))
        neg_indices = rng.choice(n_actions, size=n_neg, replace=False)
        r[s0, :] = 1.0
        r[s0, neg_indices] = -1.0
    elif variance_level == 'high':
        frac_neg = 1.0 / 4.0
        n_neg = max(1, int(n_actions * frac_neg))
        neg_indices = rng.choice(n_actions, size=n_neg, replace=False)
        r[s0, :] = 1.0
        r[s0, neg_indices] = -1.0
    elif variance_level == 'max':
        frac_neg = 1.0 / 2.0
        n_neg = max(1, int(n_actions * frac_neg))
        neg_indices = rng.choice(n_actions, size=n_neg, replace=False)
        r[s0, :] = 1.0
        r[s0, neg_indices] = -1.0
    
    return r


def experiment_1_state_action_size(
    sizes: List[Tuple[int, int]] = [(3, 3), (9, 9), (81, 81)],
    n_iterations: int = 500,
    eta: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Experiment 1: Convergence with different state and action space sizes.
    
    Reproduces Figure 1(a) from the paper.
    Uses the symmetric transition kernel and max-variance reward.
    
    Args:
        sizes: List of (|S|, |A|) pairs
        n_iterations: Number of PG iterations
        eta: Step size
        seed: Random seed
    
    Returns:
        Dictionary mapping (S, A) -> results
    """
    results = {}
    
    for n_states, n_actions in sizes:
        P = make_transition_kernel_symmetric(n_states, n_actions)
        r = make_reward_max_variance(n_states, n_actions)
        mdp = AverageRewardMDP(n_states, n_actions, P, r)
        
        result = run_projected_policy_gradient(
            mdp, n_iterations=n_iterations, eta=eta, seed=seed
        )
        results[(n_states, n_actions)] = {
            'mdp': mdp,
            'rewards': result['average_rewards'],
            'gaps': result['optimality_gaps'],
        }
        print(f"  (|S|,|A|) = ({n_states},{n_actions}): "
              f"init reward = {result['average_rewards'][0]:.4f}, "
              f"final reward = {result['average_rewards'][-1]:.4f}")
    
    return results


def experiment_2_reward_variance(
    n_states: int = 16,
    n_actions: int = 16,
    n_iterations: int = 1000,
    eta: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Experiment 2: Convergence with different reward variances.
    
    Reproduces Figure 1(b) from the paper.
    Fixed (|S|,|A|) = (16,16), same random transition kernel,
    varying reward variance.
    
    Args:
        n_states, n_actions: MDP size
        n_iterations: Number of PG iterations
        eta: Step size
        seed: Random seed
    
    Returns:
        Dictionary mapping variance level -> results
    """
    # Generate a fixed random transition kernel
    rng = np.random.RandomState(seed)
    P = rng.rand(n_states, n_actions, n_states)
    # Ensure aperiodicity
    for s in range(n_states):
        for a in range(n_actions):
            P[s, a, s] += 0.5
    P = P / P.sum(axis=2, keepdims=True)
    
    variance_levels = ['none', 'low', 'high', 'max']
    results = {}
    
    for var_level in variance_levels:
        r = make_reward_variance(n_states, n_actions, var_level, seed=seed + 1)
        mdp = AverageRewardMDP(n_states, n_actions, P, r)
        
        result = run_projected_policy_gradient(
            mdp, n_iterations=n_iterations, eta=eta, seed=seed
        )
        results[var_level] = {
            'mdp': mdp,
            'rewards': result['average_rewards'],
            'gaps': result['optimality_gaps'],
        }
        print(f"  Variance={var_level}: "
              f"init reward = {result['average_rewards'][0]:.4f}, "
              f"final reward = {result['average_rewards'][-1]:.4f}")
    
    return results


def experiment_3_transition_kernel(
    n_states: int = 16,
    n_actions: int = 16,
    n_iterations: int = 1000,
    eta: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Experiment 3: Convergence with different transition kernels.
    
    Reproduces Figure 2 from the paper.
    Fixed (|S|,|A|) = (16,16), same reward function (high variance),
    varying transition kernel types.
    
    Args:
        n_states, n_actions: MDP size
        n_iterations: Number of PG iterations
        eta: Step size
        seed: Random seed
    
    Returns:
        Dictionary mapping kernel type -> results
    """
    # High variance reward
    r = make_reward_variance(n_states, n_actions, 'high', seed=seed)
    
    kernel_types = {
        'uniform': make_uniform_transition_kernel(n_states, n_actions),
        'non-uniform': make_nonuniform_transition_kernel(n_states, n_actions),
        'deterministic': make_deterministic_transition_kernel(n_states, n_actions, seed=seed),
    }
    
    results = {}
    
    for ktype, P in kernel_types.items():
        mdp = AverageRewardMDP(n_states, n_actions, P, r)
        
        result = run_projected_policy_gradient(
            mdp, n_iterations=n_iterations, eta=eta, seed=seed
        )
        results[ktype] = {
            'mdp': mdp,
            'rewards': result['average_rewards'],
            'gaps': result['optimality_gaps'],
        }
        print(f"  Kernel={ktype}: "
              f"init reward = {result['average_rewards'][0]:.4f}, "
              f"final reward = {result['average_rewards'][-1]:.4f}")
    
    return results


def run_all_simulations(seed: int = 42) -> Dict:
    """
    Run all three simulation experiments.
    
    Returns:
        Dictionary with all experiment results
    """
    print("=" * 60)
    print("Experiment 1: Different state/action space sizes (Figure 1a)")
    print("=" * 60)
    exp1 = experiment_1_state_action_size(seed=seed)
    
    print("\n" + "=" * 60)
    print("Experiment 2: Different reward variances (Figure 1b)")
    print("=" * 60)
    exp2 = experiment_2_reward_variance(seed=seed)
    
    print("\n" + "=" * 60)
    print("Experiment 3: Different transition kernels (Figure 2)")
    print("=" * 60)
    exp3 = experiment_3_transition_kernel(seed=seed)
    
    return {
        'experiment_1': exp1,
        'experiment_2': exp2,
        'experiment_3': exp3,
    }
