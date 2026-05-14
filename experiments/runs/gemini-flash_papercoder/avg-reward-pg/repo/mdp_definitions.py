import numpy as np
from typing import Optional


class MDP:
    """
    Represents a Markov Decision Process (MDP) with finite state and action spaces.

    The MDP is defined by its number of states (S), number of actions (A),
    transition probability kernel (P), and reward function (R).
    It also stores the optimal average reward (rho*) once computed.
    """

    def __init__(self, s_count: int, a_count: int, transitions: np.ndarray, rewards: np.ndarray):
        """
        Initializes an MDP instance.

        Args:
            s_count (int): The number of states in the MDP. Corresponds to |S|.
            a_count (int): The number of actions available in each state. Corresponds to |A|.
            transitions (np.ndarray): The transition probability kernel.
                                      Shape (S, A, S), where transitions[s, a, s_prime] is P(s_prime | s, a).
            rewards (np.ndarray): The reward function.
                                  Shape (S, A), where rewards[s, a] is r(s, a).
        """
        if not isinstance(s_count, int) or s_count <= 0:
            raise ValueError("Number of states (s_count) must be a positive integer.")
        if not isinstance(a_count, int) or a_count <= 0:
            raise ValueError("Number of actions (a_count) must be a positive integer.")
        if not isinstance(transitions, np.ndarray) or transitions.shape != (s_count, a_count, s_count):
            raise ValueError(f"Transitions must be a numpy array of shape ({s_count}, {a_count}, {s_count}).")
        if not isinstance(rewards, np.ndarray) or rewards.shape != (s_count, a_count):
            raise ValueError(f"Rewards must be a numpy array of shape ({s_count}, {a_count}).")
        if not np.allclose(transitions.sum(axis=2), 1.0):
             # For some constructions, sum might be slightly off. Allow small tolerance.
            if not np.all(np.isclose(transitions.sum(axis=2), 1.0, atol=1e-8)):
                raise ValueError("Transition probabilities P[s, a, :] must sum to 1 for all s, a.")


        self.S: int = s_count
        self.A: int = a_count
        self.P: np.ndarray = transitions
        self.R: np.ndarray = rewards
        self.optimal_avg_reward: Optional[float] = None # Will be set by MDPSolver

    def set_optimal_avg_reward(self, reward: float) -> None:
        """
        Sets the pre-computed optimal average reward (ρ*) for this MDP instance.

        Args:
            reward (float): The calculated optimal average reward.
        """
        if not isinstance(reward, (int, float)):
            raise ValueError("Optimal average reward must be a numerical value.")
        self.optimal_avg_reward = float(reward)

    def get_S(self) -> int:
        """Returns the number of states."""
        return self.S

    def get_A(self) -> int:
        """Returns the number of actions."""
        return self.A

    def get_P(self) -> np.ndarray:
        """Returns the transition probability kernel."""
        return self.P

    def get_R(self) -> np.ndarray:
        """Returns the reward function."""
        return self.R

    def generate_policy_P(self, policy: np.ndarray) -> np.ndarray:
        """
        Computes the policy-dependent transition kernel P^π.

        Args:
            policy (np.ndarray): A 2D NumPy array of shape (S, A), where policy[s, a]
                                 represents π(a|s), the probability of taking action a in state s.

        Returns:
            np.ndarray: A 2D NumPy array of shape (S, S), where policy_P[s, s_prime]
                        is the transition probability P^π(s_prime|s).
        """
        if not isinstance(policy, np.ndarray) or policy.shape != (self.S, self.A):
            raise ValueError(f"Policy must be a numpy array of shape ({self.S}, {self.A}).")
        if not np.allclose(policy.sum(axis=1), 1.0):
            raise ValueError("Policy probabilities policy[s, :] must sum to 1 for all s.")

        # P^π(s'|s) = sum_a π(a|s) P(s'|s,a)
        # Using np.einsum for efficient computation:
        # 'sa' -> policy (state, action)
        # 'sas' -> self.P (state, action, next_state)
        # 'ss' -> result (state, next_state) summing over 'a'
        policy_p = np.einsum('sa,sas->ss', policy, self.P)
        return policy_p

    def generate_policy_R(self, policy: np.ndarray) -> np.ndarray:
        """
        Computes the policy-dependent reward vector r^π.

        Args:
            policy (np.ndarray): A 2D NumPy array of shape (S, A), representing π(a|s).

        Returns:
            np.ndarray: A 1D NumPy array of shape (S), where policy_R_vec[s]
                        is the expected reward r^π(s).
        """
        if not isinstance(policy, np.ndarray) or policy.shape != (self.S, self.A):
            raise ValueError(f"Policy must be a numpy array of shape ({self.S}, {self.A}).")
        if not np.allclose(policy.sum(axis=1), 1.0):
            raise ValueError("Policy probabilities policy[s, :] must sum to 1 for all s.")

        # r^π(s) = sum_a π(a|s) r(s,a)
        # Using np.einsum for efficient computation:
        # 'sa' -> policy (state, action)
        # 'sa' -> self.R (state, action)
        # 's' -> result (state) summing over 'a'
        policy_r_vec = np.einsum('sa,sa->s', policy, self.R)
        return policy_r_vec

    def generate_uniform_random_policy(self) -> np.ndarray:
        """
        Creates and returns a uniform random policy where each action in every state
        has an equal probability (1/A).

        Returns:
            np.ndarray: A 2D NumPy array of shape (S, A) representing the uniform random policy.
        """
        if self.A == 0:
            raise ValueError("Cannot generate a uniform random policy for an MDP with 0 actions.")
        uniform_policy = np.ones((self.S, self.A)) / float(self.A)
        return uniform_policy

