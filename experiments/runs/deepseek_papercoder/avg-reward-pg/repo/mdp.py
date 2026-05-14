## mdp.py

"""
Markov Decision Process (MDP) definitions for the average reward policy gradient
experiments.

This module provides the ``MDP`` class together with static factory methods that
construct the specific MDPs used in the paper:

* ``build_exp1`` – Experiment 1: varying state‑action size (Appendix C.1)
* ``build_exp2`` – Experiment 2: varying reward variance (Appendix C.2)
* ``build_exp3`` – Experiment 3: varying transition kernel (Appendix C.3)

All randomness is controlled through explicit ``seed`` arguments to guarantee exact
reproducibility.  The design follows the interface described in the project design
document.
"""

import numpy as np
from typing import Union


class MDP:
    """
    Tabular Markov Decision Process with finite state and action spaces.

    Attributes
    ----------
    S : int
        Number of states.
    A : int
        Number of actions.
    P : numpy.ndarray
        Transition probabilities, shape ``(S, A, S)``.
        ``P[s, a, s_prime]`` is the probability of moving to state ``s_prime``
        when action ``a`` is taken in state ``s``.
    R : numpy.ndarray
        Immediate rewards, shape ``(S, A)``.
        ``R[s, a]`` is the reward obtained for executing action ``a`` in state ``s``.
    """

    def __init__(self, S: int, A: int, P: np.ndarray, R: np.ndarray) -> None:
        """
        Initialise an MDP.

        Parameters
        ----------
        S : int
            Number of states.
        A : int
            Number of actions.
        P : numpy.ndarray
            Transition probabilities, shape ``(S, A, S)``.  Must satisfy
            ``∑_s' P[s,a,s'] = 1`` for every ``(s,a)``.
        R : numpy.ndarray
            Immediate rewards, shape ``(S, A)``.

        Raises
        ------
        ValueError
            If ``P`` or ``R`` do not have the correct shapes or if ``P`` rows
            do not sum to 1 (within numerical tolerance).
        """
        if not isinstance(S, int) or S <= 0:
            raise ValueError(f"Number of states must be a positive integer, got {S}")
        if not isinstance(A, int) or A <= 0:
            raise ValueError(f"Number of actions must be a positive integer, got {A}")
        if P.shape != (S, A, S):
            raise ValueError(
                f"Transition tensor P must have shape ({S}, {A}, {S}), got {P.shape}"
            )
        if R.shape != (S, A):
            raise ValueError(
                f"Reward matrix R must have shape ({S}, {A}), got {R.shape}"
            )
        # Check row sums (allow small numerical errors)
        row_sums = P.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(
                "Transition probabilities must sum to 1 for every state‑action pair. "
                f"Max deviation: {np.max(np.abs(row_sums - 1.0))}"
            )

        self.S = S
        self.A = A
        self.P = P.copy()
        self.R = R.copy()

    def get_transition_matrix(self, policy: np.ndarray) -> np.ndarray:
        """
        Compute the state‑to‑state transition matrix induced by a stationary policy.

        Parameters
        ----------
        policy : numpy.ndarray
            Policy representation, shape ``(S, A)``.  ``policy[s, a]`` is the
            probability of choosing action ``a`` in state ``s``.  Rows must sum to 1.

        Returns
        -------
        numpy.ndarray
            Transition matrix ``P^π`` of shape ``(S, S)`` defined as::

                P^π(s' | s) = Σ_a policy[s, a] * P[s, a, s'].

        Raises
        ------
        ValueError
            If ``policy`` shape does not match ``(S, A)``.
        """
        if policy.shape != (self.S, self.A):
            raise ValueError(
                f"Policy must have shape ({self.S}, {self.A}), got {policy.shape}"
            )
        # Use einsum for clarity: (S, A) x (S, A, S) -> (S, S)
        # We are summing over the action dimension.
        return np.einsum("sa,sak->sk", policy, self.P)

    def get_reward_vector(self, policy: np.ndarray) -> np.ndarray:
        """
        Compute the state‑wise expected immediate reward under the given policy.

        Parameters
        ----------
        policy : numpy.ndarray
            Policy matrix, shape ``(S, A)``.  Rows sum to 1.

        Returns
        -------
        numpy.ndarray
            Vector ``r^π`` of shape ``(S,)`` where::

                r^π(s) = Σ_a policy[s, a] * R[s, a].

        Raises
        ------
        ValueError
            If ``policy`` shape does not match ``(S, A)``.
        """
        if policy.shape != (self.S, self.A):
            raise ValueError(
                f"Policy must have shape ({self.S}, {self.A}), got {policy.shape}"
            )
        return np.sum(policy * self.R, axis=1)

    # --------------------------------------------------------------------------
    # Static factory methods for the three experiments
    # --------------------------------------------------------------------------

    @staticmethod
    def build_exp1(S: int, A: int, seed: int) -> "MDP":
        """
        Construct an MDP for Experiment 1 (varying state‑action size).

        The transition kernel uses a simple structure (Appendix C.1)::

            P(i | s, i) = (1 + 1/S) / 2
            P(j | s, i) = 1 / (2S)   for j ≠ i

        This is valid only when ``S == A``, because the action index is used as
        the preferred next state.  The reward function assigns +1 to half of the
        actions and -1 to the other half, independently for each state.

        Parameters
        ----------
        S : int
            Number of states.
        A : int
            Number of actions.  Must equal ``S``.
        seed : int
            Seed for the random number generator used to assign rewards.

        Returns
        -------
        MDP
            The constructed MDP instance.

        Raises
        ------
        ValueError
            If ``S`` != ``A``.
        """
        if S != A:
            raise ValueError(
                f"For Experiment 1, S must equal A, but got S={S}, A={A}"
            )

        # ---------- Transition kernel ----------
        # P_base: shape (S, A, S) filled with base value 1/(2S)
        P = np.full((S, A, S), 1.0 / (2 * S))
        # Add the extra 0.5 on the diagonal (action index equals next state index)
        # We can use advanced indexing: P[s, a, a] += 0.5 for all s.
        # Range for s and a: 0..S-1
        s_idx = np.arange(S)[:, None]          # shape (S, 1)
        a_idx = np.arange(A)[None, :]          # shape (1, A)
        P[s_idx, a_idx, a_idx] += 0.5

        # ---------- Reward function ----------
        rng = np.random.RandomState(seed)
        R = np.zeros((S, A))
        half = A // 2
        for s in range(S):
            # Random permutation of action indices
            perm = rng.permutation(A)
            # First half get +1, remaining get -1
            R[s, perm[:half]] = 1.0
            R[s, perm[half:]] = -1.0

        return MDP(S, A, P, R)

    @staticmethod
    def build_exp2(S: int, A: int, reward_variant: str, seed: int) -> "MDP":
        """
        Construct an MDP for Experiment 2 (varying reward variance).

        The transition kernel is generated exactly once using the provided ``seed``
        and is **identical** across all ``reward_variant`` calls with the same seed.
        This is achieved by separating the random state for the transition kernel
        from the one used for rewards.

        The reward configuration follows Appendix C.2: for one special state
        ``s0 = 0`` (other states have reward 0)::

            * "no_var"   : all actions get +1
            * "low_var"  : 1/8 of actions get -1, rest +1
            * "high_var" : 1/4 of actions get -1, rest +1
            * "max_var"  : 1/2 of actions get -1, rest +1

        The subset of negative actions is chosen randomly using a derived seed
        (based on ``reward_variant`` and the original ``seed``) to ensure different
        variants produce different reward patterns while keeping the transition
        kernel fixed.

        Parameters
        ----------
        S : int
            Number of states.
        A : int
            Number of actions.
        reward_variant : str
            One of ``"no_var"``, ``"low_var"``, ``"high_var"``, ``"max_var"``.
        seed : int
            Seed used for generating the transition kernel.  The reward pattern
            seed is derived from this value and the variant string.

        Returns
        -------
        MDP
            The constructed MDP instance.

        Raises
        ------
        ValueError
            If ``reward_variant`` is not recognised.
        """
        variants = {
            "no_var": 0.0,
            "low_var": 1 / 8,
            "high_var": 1 / 4,
            "max_var": 1 / 2,
        }
        if reward_variant not in variants:
            raise ValueError(
                f"Unknown reward_variant '{reward_variant}'. "
                f"Expected one of {list(variants.keys())}"
            )

        # ---- Transition kernel: same for all variants ----
        rng_P = np.random.RandomState(seed)
        P = np.zeros((S, A, S))
        for s in range(S):
            for a in range(A):
                P[s, a, :] = rng_P.dirichlet(np.ones(S))  # uniform over simplex

        # ---- Reward function ----
        R = np.zeros((S, A))                # all states start with 0 reward
        s0 = 0                               # special state

        # Derive a deterministic seed for the reward assignment that depends on
        # both the original seed and the variant string.
        # This keeps the transition kernel identical across variants.
        extra_seed = hash(reward_variant) % 10000
        rng_R = np.random.RandomState(seed + extra_seed)

        frac_neg = variants[reward_variant]
        num_neg = int(frac_neg * A + 0.5)    # round to nearest integer
        num_neg = max(0, min(num_neg, A))   # clamp to valid range

        # Randomly select which actions get -1 (the rest get +1)
        actions = np.arange(A)
        rng_R.shuffle(actions)
        neg_actions = actions[:num_neg]
        pos_actions = actions[num_neg:]

        R[s0, neg_actions] = -1.0
        R[s0, pos_actions] = 1.0

        return MDP(S, A, P, R)

    @staticmethod
    def build_exp3(S: int, A: int, kernel_type: str, seed: int) -> "MDP":
        """
        Construct an MDP for Experiment 3 (varying transition kernel).

        The reward function is fixed to the *high variance* variant (1/4 of
        actions at state 0 have reward -1, the rest +1).  Three different
        transition kernels are supported (Appendix C.3)::

            * ``"uniform"`` : ``P(s'|s,a) = 1/S``
            * ``"non_uniform"`` : the same structure as ``build_exp1``
            * ``"deterministic"`` : for each state ``s``, a random permutation
              of states is generated; action ``a`` leads deterministically
              to the state at index ``a`` in that permutation.

        Parameters
        ----------
        S : int
            Number of states.
        A : int
            Number of actions.
        kernel_type : str
            One of ``"uniform"``, ``"non_uniform"``, ``"deterministic"``.
        seed : int
            Seed used for generating the transition kernel and the reward assignment.

        Returns
        -------
        MDP
            The constructed MDP instance.

        Raises
        ------
        ValueError
            If ``kernel_type`` is unknown.
        """
        valid_kernels = {"uniform", "non_uniform", "deterministic"}
        if kernel_type not in valid_kernels:
            raise ValueError(
                f"Unknown kernel_type '{kernel_type}'. "
                f"Expected one of {valid_kernels}"
            )

        # ---- Common high‑variance reward (same as experiment2 'high_var') ----
        rng_R = np.random.RandomState(seed + 1000)  # decouple from kernel generation
        R = np.zeros((S, A))
        s0 = 0
        frac_neg = 1 / 4
        num_neg = int(frac_neg * A + 0.5)
        num_neg = max(0, min(num_neg, A))
        actions = np.arange(A)
        rng_R.shuffle(actions)
        R[s0, actions[:num_neg]] = -1.0
        R[s0, actions[num_neg:]] = 1.0

        # ---- Transition kernel ----
        rng_P = np.random.RandomState(seed)
        P = np.zeros((S, A, S))

        if kernel_type == "uniform":
            P[:] = 1.0 / S
        elif kernel_type == "non_uniform":
            # Same formula as build_exp1, but does not require S == A.
            # It uses action index as preferred next state; if A != S,
            # out‑of‑range access would occur, but the experiment uses S == A.
            if S != A:
                # Provide a fallback for safety, but still warn.
                # According to the paper, Experiment 3 uses S == A == 16, so we
                # can safely assume equality. If not, we can fall back to a
                # simpler method or raise an error. We'll raise an error here
                # to avoid unexpected behaviour.
                raise ValueError(
                    "For 'non_uniform' kernel, S must equal A to match the "
                    "construction from Appendix C.1. Got S={S}, A={A}."
                )
            P.fill(1.0 / (2 * S))
            # Add 0.5 along the diagonal (action index matches next state)
            s_idx = np.arange(S)[:, None]
            a_idx = np.arange(A)[None, :]
            P[s_idx, a_idx, a_idx] += 0.5
        else:  # deterministic
            for s in range(S):
                perm = rng_P.permutation(S)  # random permutation of states
                for a in range(A):
                    P[s, a, perm[a]] = 1.0

        return MDP(S, A, P, R)
