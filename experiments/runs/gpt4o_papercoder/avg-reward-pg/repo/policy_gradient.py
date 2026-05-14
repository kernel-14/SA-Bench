"""
policy_gradient.py

This module defines the PolicyGradient class responsible for implementing the policy gradient computation,
projection operation onto the simplex space, and updating policies based on the gradient.
All operations adhere strictly to the methodology described in "Global Convergence of Policy Gradient in Average Reward MDPs".
"""

import numpy as np
from typing import Callable

# Import constants and projection functions
from utils.constants import DEFAULT_LEARNING_RATE
from utils.projections import Projections


class PolicyGradient:
    """
    PolicyGradient Class:
    Handles gradient computation, projection operations, and policy updates for average reward Markov Decision Processes (MDPs).
    """

    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE, projection_fn: Callable = Projections.project_to_simplex) -> None:
        """
        Initializes the policy gradient object with specified learning rate and projection function.

        Args:
            learning_rate (float): Step size for policy updates.
            projection_fn (Callable): Function to project policies onto a valid probability simplex.
        """
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be a positive value. Received: {learning_rate}")
        if not callable(projection_fn):
            raise ValueError("Projection function must be callable.")

        self.learning_rate = learning_rate
        self.projection_fn = projection_fn

    def compute_policy_gradient(
        self, policy: np.ndarray, transition_kernel: np.ndarray, rewards: np.ndarray
    ) -> np.ndarray:
        """
        Computes the policy gradient based on the average reward policy gradient theorem.

        Args:
            policy (np.ndarray): Current policy matrix (\( |\mathcal{S}| \times |\mathcal{A}| \)).
            transition_kernel (np.ndarray): Transition probabilities (\( |\mathcal{S}| \times |\mathcal{A}| \times |\mathcal{S}| \)).
            rewards (np.ndarray): Reward matrix (\( |\mathcal{S}| \times |\mathcal{A}| \)).

        Returns:
            np.ndarray: Policy gradient (\( |\mathcal{S}| \times |\mathcal{A}| \)).
        """
        num_states, num_actions = policy.shape

        # Step 1: Compute policy-induced transition probabilities \( \mathbb{P}^\pi \)
        policy_transition = np.zeros((num_states, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                policy_transition[s] += policy[s, a] * transition_kernel[s, a]

        # Step 2: Compute stationary distribution \( d^\pi \)
        eigenvalues, eigenvectors = np.linalg.eig(policy_transition.T)
        stationary_distribution = eigenvectors[:, np.isclose(eigenvalues, 1)].flatten().real
        stationary_distribution /= stationary_distribution.sum()  # Normalize to ensure it sums to 1
        
        # Step 3: Compute per-state policy reward \( r^\pi(s) \)
        reward_per_state = np.sum(policy * rewards, axis=1)

        # Step 4: Solve for the relative value function \( v^\pi \) using the Bellman equation
        v_pi = np.zeros(num_states)
        bellman_residual_tolerance = 1e-6  # Convergence tolerance for iterative Bellman solver
        max_iterations = 1000
        
        for _ in range(max_iterations):
            new_v_pi = reward_per_state + policy_transition @ v_pi - np.mean(reward_per_state)
            if np.linalg.norm(new_v_pi - v_pi, ord=2) < bellman_residual_tolerance:
                v_pi = new_v_pi
                break
            v_pi = new_v_pi

        # Step 5: Compute action-value function \( Q^\pi(s, a) \)
        q_pi = np.zeros((num_states, num_actions))
        for s in range(num_states):
            for a in range(num_actions):
                q_pi[s, a] = rewards[s, a] + transition_kernel[s, a] @ v_pi - np.mean(reward_per_state)

        # Step 6: Compute the gradient \( \frac{\partial \rho^\pi}{\partial \pi} \)
        policy_gradient = np.zeros_like(policy)
        for s in range(num_states):
            for a in range(num_actions):
                policy_gradient[s, a] = stationary_distribution[s] * q_pi[s, a]

        return policy_gradient

    def apply_projection(self, policy: np.ndarray) -> np.ndarray:
        """
        Applies the projection operation to ensure the policy remains valid (probability simplex per state).

        Args:
            policy (np.ndarray): Policy matrix to be projected (\( |\mathcal{S}| \times |\mathcal{A}| \)).

        Returns:
            np.ndarray: Projected policy matrix.
        """
        if not isinstance(policy, np.ndarray) or policy.ndim != 2:
            raise ValueError(f"Expected a 2D numpy array for policy, got {type(policy)} with shape {policy.shape}.")

        return self.projection_fn(policy)

    def update_policy(
        self, policy: np.ndarray, gradient: np.ndarray, learning_rate: float = None
    ) -> np.ndarray:
        """
        Updates the policy using the computed gradient and projection operation.

        Args:
            policy (np.ndarray): Current policy matrix (\( |\mathcal{S}| \times |\mathcal{A}| \)).
            gradient (np.ndarray): Computed policy gradient (\( |\mathcal{S}| \times |\mathcal{A}| \)).
            learning_rate (float): Step size for the update. Defaults to the learning rate initialized in the class.

        Returns:
            np.ndarray: Updated policy matrix (\( |\mathcal{S}| \times |\mathcal{A}| \)).
        """
        if learning_rate is None:
            learning_rate = self.learning_rate

        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive for policy updates. Received: {learning_rate}")
        if policy.shape != gradient.shape:
            raise ValueError(f"Policy shape {policy.shape} and gradient shape {gradient.shape} must match.")

        # Update policy using gradient ascent
        updated_policy = policy + learning_rate * gradient

        # Apply projection to ensure the policy remains valid
        projected_policy = self.apply_projection(updated_policy)

        return projected_policy


# Example usage
if __name__ == "__main__":
    # Define example MDP parameters
    num_states = 5
    num_actions = 3
    policy = np.random.rand(num_states, num_actions)
    policy = policy / policy.sum(axis=1, keepdims=True)  # Normalize to make initial policy valid
    transition_kernel = np.random.dirichlet(np.ones(num_states), size=(num_states, num_actions))
    rewards = np.random.rand(num_states, num_actions)

    # Initialize PolicyGradient
    pg = PolicyGradient()

    # Compute policy gradient
    gradient = pg.compute_policy_gradient(policy, transition_kernel, rewards)
    print("Policy Gradient:", gradient)

    # Update policy
    updated_policy = pg.update_policy(policy, gradient)
    print("Updated Policy:", updated_policy)
