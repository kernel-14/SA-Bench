
import numpy as np
from typing import Tuple
from model import MDP

def generate_mdp_s4_1(S: int, A: int) -> MDP:
    """
    Generates an MDP as described in Appendix C.1 for Section 4.1 simulations.
    Transition kernel P(s'|s,a) and reward R(s,a) are constructed for varying S, A.

    Args:
        S (int): Number of states.
        A (int): Number of actions.

    Returns:
        MDP: An MDP instance.
    """
    # Transition kernel P(s'|s,a)
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            for s_prime in range(S):
                if s_prime == s:
                    P[s, a, s_prime] = 0.5 * (1 + 1/S)
                else:
                    P[s, a, s_prime] = 0.5 / S
            # Ensure probabilities sum to 1 (due to floating point precision issues)
            P[s, a, :] /= np.sum(P[s, a, :])


    # Reward function R(s,a)
    R = np.zeros((S, A))
    for s in range(S):
        # Rewards of half the actions to 1 and the rest to -1
        # To make it deterministic for reproducibility, let's say first A/2 actions get 1, rest get -1
        half_A = A // 2
        R[s, :half_A] = 1
        R[s, half_A:] = -1

    return MDP(S=S, A=A, P=P, R=R)


def generate_mdp_s4_2(S: int, A: int, reward_variance_type: str, seed: int = 42) -> MDP:
    """
    Generates an MDP as described in Appendix C.2 for Section 4.2 simulations.
    Fixed S, A. Randomly generated transition kernel. Reward function varies by variance.

    Args:
        S (int): Number of states.
        A (int): Number of actions.
        reward_variance_type (str): "no_variance", "low_variance", "high_variance", "max_variance".
        seed (int): Random seed for reproducibility.

    Returns:
        MDP: An MDP instance.
    """
    np.random.seed(seed)

    # Randomly generated transition kernel (constant across different reward functions)
    P = np.random.rand(S, A, S)
    P = P / np.sum(P, axis=2, keepdims=True)

    # Reward function R(s,a)
    R = np.zeros((S, A))
    s0 = 0 # One state denoted by s0, let's pick state 0

    if reward_variance_type == "no_variance":
        R[s0, :] = 1
    elif reward_variance_type == "low_variance":
        # 1/8 of actions for s0 get -1, rest 1
        num_neg_rewards = A // 8
        neg_indices = np.random.choice(A, num_neg_rewards, replace=False)
        R[s0, neg_indices] = -1
        R[s0, [i for i in range(A) if i not in neg_indices]] = 1
    elif reward_variance_type == "high_variance":
        # 1/4 of actions for s0 get -1, rest 1
        num_neg_rewards = A // 4
        neg_indices = np.random.choice(A, num_neg_rewards, replace=False)
        R[s0, neg_indices] = -1
        R[s0, [i for i in range(A) if i not in neg_indices]] = 1
    elif reward_variance_type == "max_variance":
        # 1/2 of actions for s0 get -1, rest 1
        num_neg_rewards = A // 2
        neg_indices = np.random.choice(A, num_neg_rewards, replace=False)
        R[s0, neg_indices] = -1
        R[s0, [i for i in range(A) if i not in neg_indices]] = 1
    else:
        raise ValueError(f"Unknown reward_variance_type: {reward_variance_type}")

    return MDP(S=S, A=A, P=P, R=R)


def generate_mdp_s4_3(S: int, A: int, transition_kernel_type: str, reward_variance_type: str = "high_variance", seed: int = 42) -> MDP:
    """
    Generates an MDP as described in Appendix C.3 for Section 4.3 simulations.
    Fixed S, A. Fixed high variance reward function. Transition kernel varies by type.

    Args:
        S (int): Number of states.
        A (int): Number of actions.
        transition_kernel_type (str): "uniform", "non_uniform", "deterministic".
        reward_variance_type (str): "no_variance", "low_variance", "high_variance", "max_variance".
                                    (Defaulting to "high_variance" as per paper)
        seed (int): Random seed for reproducibility.

    Returns:
        MDP: An MDP instance.
    """
    np.random.seed(seed)

    # Reward function R(s,a) - fixed to high variance as per paper
    R = np.zeros((S, A))
    s0 = 0 # One state denoted by s0, let's pick state 0
    num_neg_rewards = A // 4
    neg_indices = np.random.choice(A, num_neg_rewards, replace=False)
    R[s0, neg_indices] = -1
    R[s0, [i for i in range(A) if i not in neg_indices]] = 1

    # Transition kernel P(s'|s,a)
    P = np.zeros((S, A, S))
    if transition_kernel_type == "uniform":
        for s in range(S):
            for a in range(A):
                P[s, a, :] = 1 / S
    elif transition_kernel_type == "non_uniform":
        for s in range(S):
            for a in range(A):
                for s_prime in range(S):
                    if s_prime == s: # P(i|s,i) = 1/(2S) + 0.5
                        P[s, a, s_prime] = 0.5 / S + 0.5 # Paper has P(i|s,i) = 1/(2S) + 0.5 for i != j in description, but formula P(i|s,i) = 1/(2S) + 0.5 in C.1. Let's use 0.5/S + 0.5
                    else: # P(i|s,j) = 1/(2S) for i != j
                        P[s, a, s_prime] = 0.5 / S
            P[s, a, :] /= np.sum(P[s, a, :]) # Normalize to sum to 1
    elif transition_kernel_type == "deterministic":
        for s in range(S):
            for a in range(A):
                # Assign a random permutation of the identity matrix.
                # This means for each (s,a) pair, it transitions deterministically to one s'.
                perm = np.random.permutation(S)
                P[s, a, perm[0]] = 1.0 # Only one state s' has prob 1.0, others 0.0
    else:
        raise ValueError(f"Unknown transition_kernel_type: {transition_kernel_type}")

    return MDP(S=S, A=A, P=P, R=R)
