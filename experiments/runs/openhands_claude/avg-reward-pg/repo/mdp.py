"""
Average Reward MDP implementation.

Implements the tabular MDP model from:
  "Global Convergence of Policy Gradient in Average Reward MDPs"
  Kumar et al., ICLR 2025

Key objects:
  - Transition kernel P: (S, A, S) array
  - Reward function r: (S, A) array
  - Policy π: (S, A) array (each row is a probability distribution)
  - Projected value function v_φ^π = (I - ΦP^π)^{-1} Φr^π  (Lemma 1)
  - Average reward ρ^π = Σ_s d^π(s) r^π(s)
  - Q-function Q^π(s,a) = r(s,a) - ρ^π + Σ_{s'} P(s'|s,a) v^π(s')
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class AverageRewardMDP:
    """
    Tabular average-reward MDP with finite state and action spaces.

    Assumption 1 (ergodicity): every policy induces an irreducible, aperiodic
    Markov chain, so a unique stationary distribution exists for every π ∈ Π.
    """

    def __init__(
        self,
        P: NDArray[np.float64],
        r: NDArray[np.float64],
    ) -> None:
        """
        Parameters
        ----------
        P : (S, A, S) array
            Transition kernel.  P[s, a, s'] = Prob(s' | s, a).
        r : (S, A) array
            Single-step reward function.
        """
        assert P.ndim == 3 and P.shape[0] == P.shape[2], "P must be (S, A, S)"
        assert r.shape == P.shape[:2], "r must be (S, A)"

        self.P = P.astype(np.float64)
        self.r = r.astype(np.float64)
        self.S, self.A = r.shape

        # Projection matrix Φ = I - 11^T / |S|  (Lemma 1)
        self.Phi: NDArray[np.float64] = (
            np.eye(self.S) - np.ones((self.S, self.S)) / self.S
        )

    # ------------------------------------------------------------------
    # Policy-induced quantities
    # ------------------------------------------------------------------

    def transition_matrix(self, pi: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute P^π ∈ R^{S×S}.

        P^π(s' | s) = Σ_a π(a|s) P(s'|s,a)
        """
        # pi: (S, A), P: (S, A, S')  →  P_pi: (S, S')
        # Use distinct index names: s=current state, a=action, t=next state
        return np.einsum("sa,sat->st", pi, self.P)

    def reward_vector(self, pi: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute r^π ∈ R^S.

        r^π(s) = Σ_a π(a|s) r(s,a)
        """
        return np.einsum("sa,sa->s", pi, self.r)

    def stationary_distribution(self, pi: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the stationary distribution d^π of the Markov chain P^π.

        Solves d^π P^π = d^π, Σ_s d^π(s) = 1 via the left eigenvector
        corresponding to eigenvalue 1.
        """
        P_pi = self.transition_matrix(pi)
        # Left eigenvector: solve d (P^π - I) = 0 with Σ d_s = 1
        # Equivalent to right eigenvector of (P^π)^T
        eigvals, eigvecs = np.linalg.eig(P_pi.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigvals - 1.0))
        d = np.real(eigvecs[:, idx])
        d = np.abs(d)
        d /= d.sum()
        return d

    def average_reward(self, pi: NDArray[np.float64]) -> float:
        """
        ρ^π = Σ_s d^π(s) r^π(s)
        """
        d = self.stationary_distribution(pi)
        r_pi = self.reward_vector(pi)
        return float(np.dot(d, r_pi))

    # ------------------------------------------------------------------
    # Projected value function  (Lemma 1, Equation 15)
    # ------------------------------------------------------------------

    def projected_value_function(self, pi: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the unique projected value function v_φ^π.

        v_φ^π = (I - Φ P^π)^{-1} Φ r^π

        This is the unique solution to the fixed-point equation
            v_φ^π = Φ(r^π + P^π v_φ^π)
        subject to the constraint v_φ^{π T} 1 = 0.
        """
        P_pi = self.transition_matrix(pi)
        r_pi = self.reward_vector(pi)

        A_mat = np.eye(self.S) - self.Phi @ P_pi  # (I - Φ P^π)
        b_vec = self.Phi @ r_pi                    # Φ r^π

        v_phi = np.linalg.solve(A_mat, b_vec)
        return v_phi

    # ------------------------------------------------------------------
    # Q-function  (Equation 4)
    # ------------------------------------------------------------------

    def q_function(self, pi: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute Q^π ∈ R^{S×A}.

        Q^π(s,a) = r(s,a) - ρ^π + Σ_{s'} P(s'|s,a) v^π(s')

        Uses the projected value function v_φ^π as the representative v^π.
        """
        rho = self.average_reward(pi)
        v = self.projected_value_function(pi)
        # P: (S, A, S'), v: (S',)  →  (P v)[s,a] = Σ_{s'} P(s'|s,a) v(s')
        # Use distinct index: t = next state
        Pv = np.einsum("sat,t->sa", self.P, v)
        Q = self.r - rho + Pv
        return Q

    # ------------------------------------------------------------------
    # Policy gradient  (Equation 5)
    # ------------------------------------------------------------------

    def policy_gradient(self, pi: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute ∂ρ^π/∂π ∈ R^{S×A}.

        From the average-reward policy gradient theorem:
            ∂ρ/∂π(a|s) = d^π(s) Q^π(s,a)
        """
        d = self.stationary_distribution(pi)
        Q = self.q_function(pi)
        # grad[s, a] = d[s] * Q[s, a]
        grad = d[:, np.newaxis] * Q
        return grad

    # ------------------------------------------------------------------
    # Optimal policy / value
    # ------------------------------------------------------------------

    def optimal_policy(self) -> tuple[NDArray[np.float64], float]:
        """
        Compute the optimal deterministic policy and optimal average reward
        via policy iteration.

        Returns
        -------
        pi_star : (S, A) array
            Optimal policy (deterministic, one-hot rows).
        rho_star : float
            Optimal average reward.
        """
        # Initialise with uniform policy
        pi = np.ones((self.S, self.A)) / self.A

        for _ in range(10_000):
            Q = self.q_function(pi)
            # Greedy improvement: deterministic policy
            pi_new = np.zeros((self.S, self.A))
            best_actions = np.argmax(Q, axis=1)
            pi_new[np.arange(self.S), best_actions] = 1.0

            if np.allclose(pi_new, pi, atol=1e-10):
                break
            pi = pi_new

        rho_star = self.average_reward(pi)
        return pi, rho_star
