## mdp.py
"""MDP data container for finite average-reward Markov Decision Processes.

This module defines the MDP class, which is the foundational data structure
for the entire codebase. It stores the MDP tuple (S, A, P, r) and provides
the two core linear-algebraic operations that every downstream module depends
on: computing the policy-induced transition matrix P^pi and the policy-induced
reward vector r^pi.

Mathematical definitions follow Section 2.1 of:
    Murthy et al., "Global Convergence of Policy Gradient in Average Reward
    MDPs", ICLR 2024.

Equation (1): P^pi[s, s'] = sum_a pi[s, a] * P[s, a, s']
Equation (2): r^pi[s]     = sum_a pi[s, a] * r[s, a]
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray


class MDP:
    """Finite average-reward Markov Decision Process.

    Stores the MDP tuple (S, A, P, r) and provides efficient computation
    of the policy-induced transition matrix and reward vector. All downstream
    computations (stationary distributions, value functions, gradients) are
    built on top of these two operations.

    Attributes:
        S: Number of states. Positive integer.
        A: Number of actions. Positive integer.
        P: Transition kernel of shape (S, A, S). P[s, a, s'] is the
            probability of transitioning to state s' from state s under
            action a. Each slice P[s, a, :] must be a valid probability
            vector (non-negative, sums to 1).
        r: Reward function of shape (S, A). r[s, a] is the scalar reward
            for taking action a in state s. In the paper's experiments,
            rewards take values in {-1, +1}.
    """

    def __init__(
        self,
        S: int,
        A: int,
        P: ndarray,
        r: ndarray,
    ) -> None:
        """Initialize the MDP with its defining components.

        Args:
            S: Number of states. Must be a positive integer.
            A: Number of actions. Must be a positive integer.
            P: Transition kernel of shape (S, A, S). P[s, a, s'] is the
                probability of transitioning to state s' from (s, a).
                Each row P[s, a, :] must be a valid probability vector.
            r: Reward function of shape (S, A). r[s, a] is the reward
                for taking action a in state s.

        Note:
            Validation is deferred to validate() to keep construction
            lightweight. This matters for the (81, 81) MDP where
            construction is called during experiments.
        """
        self.S: int = S
        self.A: int = A
        # Store as float64 for numerical precision in linear system solves
        self.P: ndarray = np.asarray(P, dtype=np.float64)
        self.r: ndarray = np.asarray(r, dtype=np.float64)

    def get_policy_transition(self, pi: ndarray) -> ndarray:
        """Compute the policy-induced transition matrix P^pi.

        Implements Equation (1) from Section 2.1 of the paper:
            P^pi[s, s'] = sum_a pi[s, a] * P[s, a, s']

        The result P^pi is a row-stochastic matrix of shape (S, S), where
        P^pi[s, s'] is the one-step probability of moving from state s to
        state s' under policy pi.

        This is the fundamental operation used by:
        - ValueFunctions.stationary_distribution (power iteration on P^pi)
        - ValueFunctions.projected_value_function (linear system with P^pi)

        Args:
            pi: Policy array of shape (S, A). Each row pi[s, :] must be a
                valid probability vector: pi[s, :] >= 0, sum(pi[s, :]) == 1.

        Returns:
            Policy-induced transition matrix of shape (S, S). Row-stochastic:
            each row sums to 1 (up to floating point precision). Entry [s, b]
            equals sum_a pi[s, a] * P[s, a, b].

        Note:
            The einsum uses distinct indices 's' (current state), 'a' (action,
            contracted), and 'b' (next state) to avoid index collision. The
            output shape is (S, S) where the axes correspond to (current state,
            next state).

        Example:
            For a 2-state, 2-action MDP with uniform policy pi = [[0.5, 0.5],
            [0.5, 0.5]], the result is the average of P[:, 0, :] and P[:, 1, :]
            over the action axis.
        """
        # einsum: 'sa,sab->sb'
        #   s = current state (shared, appears in output)
        #   a = action (contracted/summed over)
        #   b = next state (appears in output)
        # Result shape: (S, S) where [s, b] = sum_a pi[s,a] * P[s,a,b]
        P_pi: ndarray = np.einsum('sa,sab->sb', pi, self.P)
        return P_pi

    def get_policy_reward(self, pi: ndarray) -> ndarray:
        """Compute the policy-induced expected reward vector r^pi.

        Implements Equation (2) from Section 2.1 of the paper:
            r^pi[s] = sum_a pi[s, a] * r[s, a]

        The result r^pi is a vector of shape (S,) where r^pi[s] is the
        expected single-step reward at state s under policy pi.

        This is used by:
        - ValueFunctions.average_reward: rho = d^pi @ r^pi
        - ValueFunctions.projected_value_function: b = Phi @ r^pi

        Args:
            pi: Policy array of shape (S, A). Each row pi[s, :] must be a
                valid probability vector: pi[s, :] >= 0, sum(pi[s, :]) == 1.

        Returns:
            Policy-induced reward vector of shape (S,). Entry [s] equals
            sum_a pi[s, a] * r[s, a]. Since r[s, a] in {-1, +1} and pi[s, :]
            is a probability vector, each entry lies in [-1, 1].

        Example:
            For a uniform policy pi[s, :] = 1/A for all s, the result is
            the average reward across all actions at each state.
        """
        # einsum: 'sa,sa->s'
        #   s = state (shared, appears in output)
        #   a = action (contracted/summed over)
        # Equivalent to np.sum(pi * self.r, axis=1) but more explicit.
        r_pi: ndarray = np.einsum('sa,sa->s', pi, self.r)
        return r_pi

    def validate(self) -> bool:
        """Validate the MDP data for correctness.

        Performs the following checks:
        1. Shape consistency: P.shape == (S, A, S), r.shape == (S, A)
        2. Positive dimensions: S > 0, A > 0
        3. Stochasticity of P: each P[s, a, :] sums to 1 (within tolerance)
        4. Non-negativity of P: P[s, a, s'] >= 0 (within tolerance)
        5. Finiteness of r: no NaN or Inf values in reward matrix

        Called by Experiments after building each MDP to catch construction
        errors early, before running potentially expensive PPG iterations.

        Returns:
            True if all checks pass.

        Raises:
            ValueError: If any check fails, with a descriptive error message
                indicating which check failed and the observed values.
        """
        # Check 1: Positive dimensions
        if self.S <= 0:
            raise ValueError(
                f"Number of states S must be positive, got S={self.S}."
            )
        if self.A <= 0:
            raise ValueError(
                f"Number of actions A must be positive, got A={self.A}."
            )

        # Check 2: Shape of transition kernel P
        expected_P_shape: tuple = (self.S, self.A, self.S)
        if self.P.shape != expected_P_shape:
            raise ValueError(
                f"Transition kernel P has wrong shape. "
                f"Expected {expected_P_shape}, got {self.P.shape}."
            )

        # Check 3: Shape of reward function r
        expected_r_shape: tuple = (self.S, self.A)
        if self.r.shape != expected_r_shape:
            raise ValueError(
                f"Reward function r has wrong shape. "
                f"Expected {expected_r_shape}, got {self.r.shape}."
            )

        # Check 4: Non-negativity of P (allow small floating point errors)
        min_P_val: float = float(np.min(self.P))
        if min_P_val < -1e-8:
            raise ValueError(
                f"Transition kernel P has negative entries. "
                f"Minimum value: {min_P_val:.6e}. "
                "P must be non-negative (it represents probabilities)."
            )

        # Check 5: Stochasticity of P — each P[s, a, :] must sum to 1
        row_sums: ndarray = self.P.sum(axis=2)  # shape (S, A)
        expected_sums: ndarray = np.ones((self.S, self.A), dtype=np.float64)
        if not np.allclose(row_sums, expected_sums, atol=1e-6):
            max_deviation: float = float(np.max(np.abs(row_sums - expected_sums)))
            raise ValueError(
                f"Transition kernel P is not row-stochastic. "
                f"Maximum deviation from 1.0: {max_deviation:.6e}. "
                "Each P[s, a, :] must sum to 1."
            )

        # Check 6: Finiteness of reward function r
        if not np.all(np.isfinite(self.r)):
            n_nonfinite: int = int(np.sum(~np.isfinite(self.r)))
            raise ValueError(
                f"Reward function r contains {n_nonfinite} non-finite "
                "values (NaN or Inf). All rewards must be finite."
            )

        return True

    def __repr__(self) -> str:
        """Return a concise string representation of the MDP.

        Returns:
            String of the form 'MDP(S=<S>, A=<A>)' for easy identification
            in logs and debug output.
        """
        return f"MDP(S={self.S}, A={self.A})"
