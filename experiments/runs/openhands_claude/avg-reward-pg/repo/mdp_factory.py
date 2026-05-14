"""
MDP construction functions for the three simulation experiments.

Appendix C of the paper describes three experiments:

  Experiment 1 (Section C.1): Varying state/action space sizes
    (|S|, |A|) ∈ {(3,3), (9,9), (81,81)}
    Transition kernel: P(i|s,i) = (1 + 1/S)/2, P(i|s,j) = 1/(2S) for i≠j
    Reward: half actions → +1, half → -1 (maximal variance)

  Experiment 2 (Section C.2): Varying reward variance
    |S| = |A| = 16, fixed random transition kernel
    r(s,a) = 0 except for state s_0
    Four reward variance levels: none, low, high, max

  Experiment 3 (Section C.3): Varying transition kernels
    |S| = |A| = 16, fixed high-variance reward
    Three transition kernels: uniform, non-uniform, deterministic
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mdp import AverageRewardMDP


# ---------------------------------------------------------------------------
# Experiment 1: Varying (S, A) sizes
# ---------------------------------------------------------------------------

def make_mdp_varying_size(S: int, A: int) -> AverageRewardMDP:
    """
    Construct the MDP for Experiment 1 (Appendix C.1).

    Transition kernel (Equation in C.1):
        P(i | s, i) = (1 + 1/S) / 2
        P(i | s, j) = 1 / (2S)  for i ≠ j

    Reward: for every state, half the actions get reward +1, half get -1.
    This corresponds to maximal reward variance.
    """
    # Build transition kernel P: (S, A, S)
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            for sp in range(S):
                if sp == a % S:
                    # The "self" transition for action a maps to state a % S
                    P[s, a, sp] = (1.0 + 1.0 / S) / 2.0
                else:
                    P[s, a, sp] = 1.0 / (2.0 * S)

    # Normalise rows to ensure valid probability distributions
    for s in range(S):
        for a in range(A):
            P[s, a] /= P[s, a].sum()

    # Reward: half actions → +1, half → -1 (maximal variance)
    r = np.ones((S, A))
    half = A // 2
    r[:, :half] = 1.0
    r[:, half:] = -1.0

    return AverageRewardMDP(P, r)


# ---------------------------------------------------------------------------
# Experiment 2: Varying reward variance
# ---------------------------------------------------------------------------

def make_random_transition_kernel(
    S: int, A: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """
    Sample a random transition kernel P: (S, A, S) with rows summing to 1.
    """
    P = rng.dirichlet(np.ones(S), size=(S, A))
    return P


def make_reward_no_variance(S: int, A: int, s0: int = 0) -> NDArray[np.float64]:
    """
    No variance: r(s_0, a) = 1 for all a; r(s, a) = 0 otherwise.
    """
    r = np.zeros((S, A))
    r[s0, :] = 1.0
    return r


def make_reward_low_variance(S: int, A: int, s0: int = 0) -> NDArray[np.float64]:
    """
    Low variance: 1/8 of actions for s_0 get reward -1, rest get +1.
    """
    r = np.zeros((S, A))
    r[s0, :] = 1.0
    n_neg = max(1, A // 8)
    r[s0, :n_neg] = -1.0
    return r


def make_reward_high_variance(S: int, A: int, s0: int = 0) -> NDArray[np.float64]:
    """
    High variance: 1/4 of actions for s_0 get reward -1, rest get +1.
    """
    r = np.zeros((S, A))
    r[s0, :] = 1.0
    n_neg = max(1, A // 4)
    r[s0, :n_neg] = -1.0
    return r


def make_reward_max_variance(S: int, A: int, s0: int = 0) -> NDArray[np.float64]:
    """
    Max variance: 1/2 of actions for s_0 get reward -1, rest get +1.
    """
    r = np.zeros((S, A))
    r[s0, :] = 1.0
    n_neg = max(1, A // 2)
    r[s0, :n_neg] = -1.0
    return r


def make_mdps_varying_reward(
    S: int = 16, A: int = 16, seed: int = 42
) -> dict[str, AverageRewardMDP]:
    """
    Construct the four MDPs for Experiment 2 (Appendix C.2).

    All MDPs share the same randomly generated transition kernel.
    """
    rng = np.random.default_rng(seed)
    P = make_random_transition_kernel(S, A, rng)

    s0 = 0
    return {
        "no_variance": AverageRewardMDP(P.copy(), make_reward_no_variance(S, A, s0)),
        "low_variance": AverageRewardMDP(P.copy(), make_reward_low_variance(S, A, s0)),
        "high_variance": AverageRewardMDP(P.copy(), make_reward_high_variance(S, A, s0)),
        "max_variance": AverageRewardMDP(P.copy(), make_reward_max_variance(S, A, s0)),
    }


# ---------------------------------------------------------------------------
# Experiment 3: Varying transition kernels
# ---------------------------------------------------------------------------

def make_transition_uniform(S: int, A: int) -> NDArray[np.float64]:
    """
    Uniform transition: P(s'|s,a) = 1/S for all s, a, s'.
    """
    P = np.ones((S, A, S)) / S
    return P


def make_transition_nonuniform(S: int, A: int) -> NDArray[np.float64]:
    """
    Non-uniform transition (Appendix C.3):
        P(i | s, i) = 1/(2S) + 1/2
        P(i | s, j) = 1/(2S)  for i ≠ j
    """
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            for sp in range(S):
                if sp == a % S:
                    P[s, a, sp] = 1.0 / (2.0 * S) + 0.5
                else:
                    P[s, a, sp] = 1.0 / (2.0 * S)
    # Normalise
    for s in range(S):
        for a in range(A):
            P[s, a] /= P[s, a].sum()
    return P


def make_transition_deterministic(
    S: int, A: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """
    Deterministic transition (Appendix C.3):
    P(·|s,·) is a random permutation of the identity matrix.
    Each action deterministically maps to a different next state.
    """
    P = np.zeros((S, A, S))
    for s in range(S):
        # Random permutation of states for each state s
        perm = rng.permutation(S)
        for a in range(A):
            sp = perm[a % S]
            P[s, a, sp] = 1.0
    return P


def make_mdps_varying_kernel(
    S: int = 16, A: int = 16, seed: int = 42
) -> dict[str, AverageRewardMDP]:
    """
    Construct the three MDPs for Experiment 3 (Appendix C.3).

    All MDPs share the same high-variance reward function.
    """
    rng = np.random.default_rng(seed)
    r = make_reward_high_variance(S, A, s0=0)

    P_uniform = make_transition_uniform(S, A)
    P_nonuniform = make_transition_nonuniform(S, A)
    P_deterministic = make_transition_deterministic(S, A, rng)

    return {
        "uniform": AverageRewardMDP(P_uniform, r.copy()),
        "non_uniform": AverageRewardMDP(P_nonuniform, r.copy()),
        "deterministic": AverageRewardMDP(P_deterministic, r.copy()),
    }
