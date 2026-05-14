import numpy as np
from typing import Tuple, Optional


class TabularMDP:
    """Tabular average-reward MDP with finite state and action spaces.

    The MDP is defined by:
    - S: number of states
    - A: number of actions
    - P: transition kernel P(s'|s,a) of shape (S, A, S)
    - r: reward function r(s,a) of shape (S, A)
    """

    def __init__(self, S: int, A: int, P: np.ndarray, r: np.ndarray):
        assert P.shape == (S, A, S), f"P shape {P.shape} != ({S}, {A}, {S})"
        assert r.shape == (S, A), f"r shape {r.shape} != ({S}, {A})"
        assert np.allclose(P.sum(axis=2), 1.0), "P must be row-stochastic"
        self.S = S
        self.A = A
        self.P = P
        self.r = r

    def transition_matrix(self, pi: np.ndarray) -> np.ndarray:
        """Compute P^pi(s'|s) = sum_a pi(a|s) P(s'|s,a).
        pi: (S, A) policy matrix.
        """
        return np.einsum('ij,ijk->ik', pi, self.P)

    def reward_vector(self, pi: np.ndarray) -> np.ndarray:
        """Compute r^pi(s) = sum_a pi(a|s) r(s,a).
        pi: (S, A) policy matrix.
        """
        return np.einsum('ij,ij->i', pi, self.r)

    def average_reward(self, pi: np.ndarray) -> float:
        """Compute rho^pi = sum_s d^pi(s) r^pi(s)."""
        d = self.stationary_distribution(pi)
        r_pi = self.reward_vector(pi)
        return float(np.dot(d, r_pi))

    def stationary_distribution(self, pi: np.ndarray) -> np.ndarray:
        """Compute stationary distribution d^pi of P^pi."""
        P_pi = self.transition_matrix(pi)
        eigvals, eigvecs = np.linalg.eig(P_pi.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        d = np.abs(eigvecs[:, idx].real)
        d = d / d.sum()
        return d

    def policy_gradient(self, pi: np.ndarray) -> np.ndarray:
        """Compute policy gradient: ∂ρ/∂π(s,a) = d^pi(s) * Q^pi(s,a).
        Returns array of shape (S, A).
        """
        d = self.stationary_distribution(pi)
        Q = self.compute_Q(pi)
        return d[:, np.newaxis] * Q

    def compute_v_and_rho(self, pi: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve Bellman equation: rho*1 + v = r_pi + P_pi * v.
        Returns (v, rho) where v is unique with sum_s d^pi(s)*v(s)=0.
        """
        P_pi = self.transition_matrix(pi)
        r_pi = self.reward_vector(pi)
        S = self.S
        A_mat = np.eye(S) - P_pi
        ones = np.ones((S, 1))
        # Solve augmented system:
        # [I - P_pi, 1] [v; rho] = r_pi, with constraint d^T v = 0
        A_aug = np.zeros((S + 1, S + 1))
        A_aug[:S, :S] = A_mat
        A_aug[:S, S] = 1.0
        A_aug[S, :S] = self.stationary_distribution(pi)
        A_aug[S, S] = 0.0
        b_aug = np.zeros(S + 1)
        b_aug[:S] = r_pi
        sol = np.linalg.solve(A_aug, b_aug)
        v = sol[:S]
        rho = sol[S]
        return v, rho

    def compute_Q(self, pi: np.ndarray) -> np.ndarray:
        """Compute Q^pi(s,a) = r(s,a) + sum_{s'} P(s'|s,a) v^pi(s') - rho^pi.
        Uses the unique v that satisfies sum_s d^pi(s)*v(s)=0.
        """
        v, rho = self.compute_v_and_rho(pi)
        S, A = self.S, self.A
        Q = np.zeros((S, A))
        for s in range(S):
            for a in range(A):
                Q[s, a] = self.r[s, a] + np.dot(self.P[s, a, :], v) - rho
        return Q

    def projection_matrix(self) -> np.ndarray:
        """Compute Phi = I - 11^T/S."""
        S = self.S
        return np.eye(S) - np.ones((S, S)) / S

    def projected_value_function(self, pi: np.ndarray) -> np.ndarray:
        """Compute v_phi^pi = (I - Phi * P^pi)^{-1} * Phi * r^pi."""
        S = self.S
        Phi = self.projection_matrix()
        P_pi = self.transition_matrix(pi)
        r_pi = self.reward_vector(pi)
        M = np.eye(S) - Phi @ P_pi
        v_phi = np.linalg.solve(M, Phi @ r_pi)
        return v_phi


def generate_random_mdp(
    S: int,
    A: int,
    seed: int = 42,
    deterministic: bool = False,
    uniform_transition: bool = False,
    reward_variance: str = "max",
) -> TabularMDP:
    """Generate an MDP according to paper specifications."""
    rng = np.random.RandomState(seed)

    if uniform_transition:
        P = np.ones((S, A, S)) / S
    elif deterministic:
        P = np.zeros((S, A, S))
        for s in range(S):
            perm = rng.permutation(S)
            for a in range(A):
                next_s = perm[a % S]
                P[s, a, next_s] = 1.0
    else:
        P = np.zeros((S, A, S))
        for s in range(S):
            for a in range(A):
                probs = rng.dirichlet(np.ones(S))
                P[s, a, :] = probs

    r = np.zeros((S, A))
    if reward_variance == "none":
        r[:, :] = 1.0
    elif reward_variance == "low":
        n_neg = max(1, A // 8)
        r[:, :] = 1.0
        r[:, :n_neg] = -1.0
    elif reward_variance == "high":
        n_neg = max(1, A // 4)
        r[:, :] = 1.0
        r[:, :n_neg] = -1.0
    elif reward_variance == "max":
        n_neg = max(1, A // 2)
        r[:, :] = 1.0
        r[:, :n_neg] = -1.0
    else:
        raise ValueError(f"Unknown reward_variance: {reward_variance}")

    return TabularMDP(S, A, P, r)
