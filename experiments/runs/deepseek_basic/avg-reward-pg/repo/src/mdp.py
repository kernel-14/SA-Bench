"""
MDP (Markov Decision Process) for Average Reward setting.

Implements the core MDP model as described in Section 2.1 of the paper:
"Global Convergence of Policy Gradient in Average Reward MDPs"

Key components:
- Finite state space S, finite action space A
- Probability transition kernel P
- Single-step reward function r(s,a)
- Randomized policy class Π
"""

import numpy as np
from typing import Tuple, Optional


class AverageRewardMDP:
    """
    Infinite horizon average reward MDP with finite state and action spaces.
    
    Attributes:
        n_states (int): |S|, number of states
        n_actions (int): |A|, number of actions
        P (np.ndarray): Transition kernel, shape (n_states, n_actions, n_states)
                       P[s, a, s'] = probability of moving s -> s' under action a
        r (np.ndarray): Reward function, shape (n_states, n_actions)
                       r[s, a] = single-step reward for taking action a in state s
    """
    
    def __init__(self, n_states: int, n_actions: int, P: np.ndarray, r: np.ndarray):
        """
        Initialize the MDP.
        
        Args:
            n_states: Number of states |S|
            n_actions: Number of actions |A|
            P: Transition kernel, shape (n_states, n_actions, n_states)
            r: Reward function, shape (n_states, n_actions)
        """
        assert P.shape == (n_states, n_actions, n_states), f"P shape {P.shape} != ({n_states}, {n_actions}, {n_states})"
        assert r.shape == (n_states, n_actions), f"r shape {r.shape} != ({n_states}, {n_actions})"
        # Validate that transition probabilities sum to 1
        assert np.allclose(P.sum(axis=2), 1.0), "Transition probabilities must sum to 1"
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.P = P
        self.r = r
    
    def get_transition_matrix(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute the transition matrix P^π for a given policy π.
        
        P^π(s'|s) = Σ_a π(a|s) P(s'|s,a)
        
        Args:
            pi: Policy, shape (n_states, n_actions), pi[s,a] = π(a|s)
        
        Returns:
            P_pi: Transition matrix under policy π, shape (n_states, n_states)
        """
        assert pi.shape == (self.n_states, self.n_actions)
        assert np.allclose(pi.sum(axis=1), 1.0), "Policy must be a valid probability distribution"
        
        P_pi = np.einsum('sa,san->sn', pi, self.P)
        return P_pi
    
    def get_reward_vector(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute the expected reward vector r^π for a given policy π.
        
        r^π(s) = Σ_a π(a|s) r(s,a)
        
        Args:
            pi: Policy, shape (n_states, n_actions)
        
        Returns:
            r_pi: Expected reward vector, shape (n_states,)
        """
        r_pi = np.einsum('sa,sa->s', pi, self.r)
        return r_pi
    
    def get_stationary_distribution(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute the stationary distribution d^π of the Markov chain induced by π.
        
        Solves d^π P^π = d^π, Σ_s d^π(s) = 1.
        Under Assumption 1 (irreducible, aperiodic), this is unique.
        
        Args:
            pi: Policy, shape (n_states, n_actions)
        
        Returns:
            d_pi: Stationary distribution, shape (n_states,)
        """
        P_pi = self.get_transition_matrix(pi)
        
        # Solve for stationary distribution: d P = d
        # Equivalent to finding left eigenvector for eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(P_pi.T)
        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        d_pi = np.real(eigenvectors[:, idx])
        # Normalize to sum to 1 and ensure non-negative
        d_pi = np.abs(d_pi)
        d_pi = d_pi / d_pi.sum()
        return d_pi
    
    def average_reward(self, pi: np.ndarray) -> float:
        """
        Compute the average reward ρ^π for policy π.
        
        ρ^π = Σ_s d^π(s) r^π(s)
        
        Args:
            pi: Policy, shape (n_states, n_actions)
        
        Returns:
            rho: Average reward
        """
        d_pi = self.get_stationary_distribution(pi)
        r_pi = self.get_reward_vector(pi)
        return float(np.dot(d_pi, r_pi))
    
    def compute_value_function(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute the basic differential value function v_0^π (unique up to additive constant).
        
        Uses the constraint Σ_s d^π(s) v^π(s) = 0 for uniqueness.
        Solves: ρ^π 1 + v^π = r^π + P^π v^π
        
        Args:
            pi: Policy, shape (n_states, n_actions)
        
        Returns:
            v_pi: Value function, shape (n_states,)
        """
        P_pi = self.get_transition_matrix(pi)
        r_pi = self.get_reward_vector(pi)
        rho_pi = self.average_reward(pi)
        d_pi = self.get_stationary_distribution(pi)
        
        # Solve the system: (I - P^π) v = r^π - ρ^π 1
        # with constraint: d^π · v = 0
        I = np.eye(self.n_states)
        A = I - P_pi
        b = r_pi - rho_pi * np.ones(self.n_states)
        
        # Augment with constraint
        A_aug = np.vstack([A, d_pi.reshape(1, -1)])
        b_aug = np.concatenate([b, [0.0]])
        
        # Solve least squares
        v_pi, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
        return v_pi
    
    def compute_q_function(self, pi: np.ndarray) -> np.ndarray:
        """
        Compute the state-action value function Q^π(s,a).
        
        Q^π(s,a) = r(s,a) + Σ_{s'} P(s'|s,a) Σ_{a'} π(a'|s') Q^π(s',a') - ρ^π
        
        Args:
            pi: Policy, shape (n_states, n_actions)
        
        Returns:
            Q_pi: Q-function, shape (n_states, n_actions)
        """
        P_pi = self.get_transition_matrix(pi)
        r_pi = self.get_reward_vector(pi)
        rho_pi = self.average_reward(pi)
        v_pi = self.compute_value_function(pi)
        
        # Bellman equation for Q:
        # Q^π(s,a) = r(s,a) + Σ_{s'} P(s'|s,a) v^π(s') - ρ^π
        # Using v^π(s) = Σ_a π(a|s) Q^π(s,a), we can compute directly:
        
        Q_pi = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                expected_next_v = np.dot(self.P[s, a, :], v_pi)
                Q_pi[s, a] = self.r[s, a] + expected_next_v - rho_pi
        
        return Q_pi


def make_random_mdp(n_states: int, n_actions: int, seed: int = 42) -> AverageRewardMDP:
    """
    Create a random MDP with the specified size.
    
    The transition kernel is generated to ensure irreducibility and aperiodicity
    (Assumption 1 of the paper).
    
    Args:
        n_states: Number of states
        n_actions: Number of actions
        seed: Random seed
    
    Returns:
        mdp: An AverageRewardMDP instance
    """
    rng = np.random.RandomState(seed)
    
    # Generate transition kernel ensuring it's irreducible and aperiodic
    # Start with uniform random, then add self-loops to ensure aperiodicity
    P = rng.rand(n_states, n_actions, n_states)
    # Add self-transition probability to ensure aperiodicity
    for s in range(n_states):
        for a in range(n_actions):
            P[s, a, s] += 0.5
    # Normalize
    P = P / P.sum(axis=2, keepdims=True)
    
    # Generate rewards
    r = rng.randn(n_states, n_actions)
    
    return AverageRewardMDP(n_states, n_actions, P, r)
