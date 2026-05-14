"""
MDP construction functions for the experiments in:
"Global Convergence of Policy Gradient in Average Reward MDPs"

Three experiments:
1. Figure 1(a): Different state/action space sizes (S,A) in {(3,3), (9,9), (81,81)}
2. Figure 1(b): Different reward variances with fixed (S,A) = (16,16)
3. Figure 2: Different transition kernels with fixed (S,A) = (16,16)
"""

import numpy as np
from mdp import AverageRewardMDP


def make_mdp_varying_size(S: int, A: int, seed: int = 42) -> AverageRewardMDP:
    """
    Construct MDP for Experiment 1 (Figure 1a): varying state/action space sizes.
    
    Transition kernel (from Appendix C.1):
        P(i|s, i) = (1 + 1/S) / 2
        P(i|s, j) = 1/(2S) for i != j
    
    Reward: half the actions get reward 1, rest get -1, for every state.
    
    Args:
        S: Number of states
        A: Number of actions (must equal S for this experiment)
        seed: Random seed
    Returns:
        mdp: AverageRewardMDP instance
    """
    # Transition kernel: P(s'|s, a)
    # P(i|s, i) = (1 + 1/S) / 2, P(i|s, j) = 1/(2S) for i != j
    # Action a leads to state a % S with higher probability
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            for sp in range(S):
                if sp == a % S:
                    P[s, a, sp] = (1.0 + 1.0 / S) / 2.0
                else:
                    P[s, a, sp] = 1.0 / (2.0 * S)
    
    # Verify normalization (should already be normalized)
    # Sum over s': (1+1/S)/2 + (S-1)/(2S) = (S+1)/(2S) + (S-1)/(2S) = 1
    
    # Reward: half actions get +1, half get -1 (maximal variance)
    R = np.zeros((S, A))
    half_A = A // 2
    for s in range(S):
        R[s, :half_A] = 1.0
        R[s, half_A:] = -1.0
    
    return AverageRewardMDP(S, A, P, R)


def _make_random_transition_kernel(S: int, A: int, seed: int = 42) -> np.ndarray:
    """
    Generate a random transition kernel using Dirichlet distribution.
    
    Args:
        S: Number of states
        A: Number of actions
        seed: Random seed
    Returns:
        P: Transition kernel, shape (S, A, S)
    """
    rng = np.random.RandomState(seed)
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            P[s, a] = rng.dirichlet(np.ones(S))
    return P


def make_mdp_varying_reward_variance(
    S: int, A: int, variance_type: str, seed: int = 42
) -> AverageRewardMDP:
    """
    Construct MDP for Experiment 2 (Figure 1b): varying reward variance.
    
    Fixed (S, A) = (16, 16). Same randomly generated transition kernel.
    Reward: r(s, a) = 0 for all s, a except s_0.
    
    Variance types (from Appendix C.2):
    - 'no_variance': all (s_0, a) get reward 1
    - 'low_variance': 1/8 of actions for s_0 get -1, rest get 1
    - 'high_variance': 1/4 of actions for s_0 get -1, rest get 1
    - 'max_variance': 1/2 of actions for s_0 get -1, rest get 1
    
    Args:
        S: Number of states
        A: Number of actions
        variance_type: One of 'no_variance', 'low_variance', 'high_variance', 'max_variance'
        seed: Random seed for transition kernel (same for all variance types)
    Returns:
        mdp: AverageRewardMDP instance
    """
    # Random transition kernel (same for all variance types, same seed)
    P = _make_random_transition_kernel(S, A, seed=seed)
    
    # Reward function
    R = np.zeros((S, A))
    s0 = 0  # Special state s_0
    
    if variance_type == 'no_variance':
        # All (s_0, a) get reward 1
        R[s0, :] = 1.0
    elif variance_type == 'low_variance':
        # 1/8 of actions for s_0 get -1, rest get 1
        n_neg = max(1, A // 8)
        R[s0, :] = 1.0
        R[s0, :n_neg] = -1.0
    elif variance_type == 'high_variance':
        # 1/4 of actions for s_0 get -1, rest get 1
        n_neg = max(1, A // 4)
        R[s0, :] = 1.0
        R[s0, :n_neg] = -1.0
    elif variance_type == 'max_variance':
        # 1/2 of actions for s_0 get -1, rest get 1
        n_neg = A // 2
        R[s0, :] = 1.0
        R[s0, :n_neg] = -1.0
    else:
        raise ValueError(f"Unknown variance_type: {variance_type}")
    
    return AverageRewardMDP(S, A, P, R)


def make_mdp_varying_transition(
    S: int, A: int, kernel_type: str, seed: int = 42
) -> AverageRewardMDP:
    """
    Construct MDP for Experiment 3 (Figure 2): varying transition kernels.
    
    Fixed (S, A) = (16, 16). Same reward function (high variance).
    
    Kernel types (from Appendix C.3):
    - 'uniform': P(s'|s, a) = 1/S for all s, a, s'
    - 'non_uniform': P(i|s, i) = 1/(2S) + 1/2, P(i|s, j) = 1/(2S) for i != j
    - 'deterministic': random permutation of identity matrix per action
    
    Args:
        S: Number of states
        A: Number of actions
        kernel_type: One of 'uniform', 'non_uniform', 'deterministic'
        seed: Random seed
    Returns:
        mdp: AverageRewardMDP instance
    """
    rng = np.random.RandomState(seed)
    
    # Reward function: high variance (1/4 of actions for s_0 get -1)
    R = np.zeros((S, A))
    s0 = 0
    n_neg = max(1, A // 4)
    R[s0, :] = 1.0
    R[s0, :n_neg] = -1.0
    
    P = np.zeros((S, A, S))
    
    if kernel_type == 'uniform':
        # P(s'|s, a) = 1/S for all s, a, s'
        P[:, :, :] = 1.0 / S
    
    elif kernel_type == 'non_uniform':
        # P(i|s, i) = 1/(2S) + 1/2, P(i|s, j) = 1/(2S) for i != j
        # This is the same as the kernel in Experiment 1a
        for s in range(S):
            for a in range(A):
                for sp in range(S):
                    if sp == a % S:
                        P[s, a, sp] = 1.0 / (2.0 * S) + 0.5
                    else:
                        P[s, a, sp] = 1.0 / (2.0 * S)
        # Already normalized: sum = 1/(2S) + 1/2 + (S-1)/(2S) = 1
    
    elif kernel_type == 'deterministic':
        # Random permutation of identity matrix for each action
        # P(s'|s, a) = 1 if s' = perm_a[s], else 0
        # This makes the MDP deterministic but non-trivial
        # "every state leads to a different one"
        for a in range(A):
            perm = rng.permutation(S)
            for s in range(S):
                P[s, a, perm[s]] = 1.0
    
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    
    return AverageRewardMDP(S, A, P, R)
