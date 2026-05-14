"""
trainer.py

This module defines the Trainer class, which oversees the training process for the projected policy gradient algorithm
in average-reward Markov Decision Processes (MDPs). It adheres to the methodology described in the paper and the
design specified in the initial API.
"""

import numpy as np
from typing import Dict

from utils.constants import get_hyperparameters
from policy_gradient import PolicyGradient
from mdp_generator import MDPGenerator


class Trainer:
    """
    Trainer Class:
    Orchestrates the training loop for the projected policy gradient algorithm in average-reward MDPs.
    """

    def __init__(self, policy_gradient: PolicyGradient, mdp: Dict, learning_rate: float = None, num_iterations: int = None):
        """
        Initializes the class with the given policy gradient object, MDP environment, learning rate, and number of iterations.

        Args:
            policy_gradient (PolicyGradient): Instance of the PolicyGradient class to compute gradients and perform policy updates.
            mdp (Dict): MDP environment dictionary that includes state and action spaces, transition kernel, and reward matrix.
            learning_rate (float): Learning rate for gradient updates. Defaults to the value in config.yaml.
            num_iterations (int): Number of iterations for training. Defaults to the value in config.yaml.

        Raises:
            ValueError: If the MDP is incomplete or invalid.
        """
        if not mdp or not isinstance(mdp, dict):
            raise ValueError("Provided MDP must be a valid dictionary containing transition kernel and rewards.")
        if not all(key in mdp for key in ["state_space", "action_space", "transition_kernel", "rewards"]):
            raise ValueError("MDP dictionary is missing required keys: 'state_space', 'action_space', 'transition_kernel', or 'rewards'.")

        self.policy_gradient = policy_gradient
        self.mdp = mdp
        self.learning_rate = learning_rate or get_hyperparameters()["learning_rate"]
        self.num_iterations = num_iterations or get_hyperparameters()["num_iterations"]

        # Metrics storage
        self.metrics = {"iteration_rewards": [], "suboptimality_gaps": [], "policies": []}

        # Initialize policy (uniform)
        self.policy = self._initialize_uniform_policy(len(mdp["state_space"]), len(mdp["action_space"]))

    def _initialize_uniform_policy(self, num_states: int, num_actions: int) -> np.ndarray:
        """
        Initializes a uniform policy across all states and actions.

        Args:
            num_states (int): Number of states in the MDP.
            num_actions (int): Number of actions in the MDP.

        Returns:
            np.ndarray: Uniform policy with shape (num_states, num_actions).
        """
        return np.ones((num_states, num_actions)) / num_actions

    def train(self) -> Dict:
        """
        Executes the training loop for the projected policy gradient algorithm.

        Returns:
            dict: Final metrics including rewards, suboptimality gaps, and optimized policy.
        """
        transition_kernel = self.mdp["transition_kernel"]
        rewards = self.mdp["rewards"]

        # Training loop
        for iteration in range(self.num_iterations):
            # Step 1: Compute policy gradient
            gradient = self.policy_gradient.compute_policy_gradient(self.policy, transition_kernel, rewards)

            # Step 2: Update policy
            self.policy = self.policy_gradient.update_policy(self.policy, gradient, self.learning_rate)

            # Step 3: Evaluate policy and log metrics
            self._log_metrics(iteration)

        return {
            "rewards": self.metrics["iteration_rewards"],
            "suboptimality_gaps": self.metrics["suboptimality_gaps"],
            "final_policy": self.policy,
        }

    def update_policy(self, policy: np.ndarray, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Updates the policy using gradient ascent and ensures the policy remains valid via projection.

        Args:
            policy (np.ndarray): Current policy matrix.
            gradient (np.ndarray): Computed policy gradient matrix.
            learning_rate (float): Learning rate for policy updates.

        Returns:
            np.ndarray: Updated and projected policy matrix.
        """
        return self.policy_gradient.update_policy(policy, gradient, learning_rate)

    def _log_metrics(self, iteration: int) -> None:
        """
        Logs metrics such as average rewards and suboptimality gaps for the current iteration.

        Args:
            iteration (int): Current iteration of the training loop.
        """
        # Compute average reward for the current policy
        transition_kernel = self.mdp["transition_kernel"]
        rewards = self.mdp["rewards"]
        avg_reward = np.sum(self.policy * rewards) / len(self.policy)

        # Normally, optimal reward would be provided; using avg_reward as a placeholder for this example
        optimal_reward = avg_reward  # Placeholder. Replace with actual optimal calculation if available.
        suboptimality_gap = optimal_reward - avg_reward

        self.metrics["iteration_rewards"].append(avg_reward)
        self.metrics["suboptimality_gaps"].append(suboptimality_gap)
        self.metrics["policies"].append(self.policy.copy())  # Store policies for debugging/inspection

        print(f"Iteration {iteration}: Avg. Reward = {avg_reward:.4f}, Suboptimality Gap = {suboptimality_gap:.4f}")


# Example usage
if __name__ == "__main__":
    # Example setup based on config.yaml and generated MDP
    config = {"state_space_sizes": [3, 9, 81], "action_space_sizes": [3, 9, 81], "reward_variance_levels": ["low_variance"]}
    mdp_generator = MDPGenerator(config)
    mdp = mdp_generator.generate_mdp(state_size=9, action_size=9, reward_variance="low_variance")

    # Initialize policy gradient object
    pg = PolicyGradient()

    # Create trainer instance and start training
    trainer = Trainer(policy_gradient=pg, mdp=mdp)
    results = trainer.train()

    print("Final Training Results:", results)
    print("Final Policy:", results["final_policy"])
