"""
mdp_generator.py

This module defines the MDPGenerator class. It generates simulated MDP environments,
including state and action spaces, transition kernels, and reward functions, based on
configurations provided in config.yaml and serves as a foundation for policy gradient experiments.
"""

import numpy as np
from typing import Dict, List

from utils.constants import STATE_SPACE_SIZES, ACTION_SPACE_SIZES, REWARD_VARIANCE_LEVELS, get_reward_variance_config


class MDPGenerator:
    """
    MDPGenerator Class:
    Generates Markov Decision Process (MDP) environments required for policy gradient experiments.
    """

    def __init__(self, config: Dict):
        """
        Initializes the generator with configurations from a dictionary.

        Args:
            config (dict): Configuration dictionary, typically loaded from config.yaml.
        """
        self.state_space_sizes = config.get("state_space_sizes", STATE_SPACE_SIZES)
        self.action_space_sizes = config.get("action_space_sizes", ACTION_SPACE_SIZES)
        self.reward_variance_levels = config.get("reward_variance_levels", REWARD_VARIANCE_LEVELS)

        # Ensure valid configuration
        if not self.state_space_sizes or not self.action_space_sizes or not self.reward_variance_levels:
            raise ValueError("Configuration for state and action space sizes and reward variance levels must be specified.")

    def generate_mdp(self, state_size: int, action_size: int, reward_variance: str) -> Dict:
        """
        Generates a complete MDP given the state size, action size, and reward variance level.

        Args:
            state_size (int): Number of states in the MDP.
            action_size (int): Number of actions in the MDP.
            reward_variance (str): One of the reward variance levels: "no_variance", "low_variance", "high_variance", "max_variance".

        Returns:
            dict: A complete MDP environment with the following structure:
                - state_space: List of states.
                - action_space: List of actions.
                - transition_kernel: Transition probabilities (3D array).
                - rewards: Reward matrix (2D array).
        """
        if state_size not in self.state_space_sizes:
            raise ValueError(f"Invalid state size {state_size}. Supported sizes: {self.state_space_sizes}.")
        if action_size not in self.action_space_sizes:
            raise ValueError(f"Invalid action size {action_size}. Supported sizes: {self.action_space_sizes}.")
        if reward_variance not in self.reward_variance_levels:
            raise ValueError(f"Invalid reward variance {reward_variance}. Supported levels: {self.reward_variance_levels}.")

        # Generate state space and action space
        state_space = list(range(state_size))
        action_space = list(range(action_size))

        # Generate transition kernel
        transition_kernel = self.get_random_transition_kernel(state_size, action_size)

        # Generate reward function
        rewards = self.get_reward_function(state_size, action_size, reward_variance)

        return {
            "state_space": state_space,
            "action_space": action_space,
            "transition_kernel": transition_kernel,
            "rewards": rewards,
        }

    def get_random_transition_kernel(self, state_size: int, action_size: int) -> np.ndarray:
        """
        Generates a random transition kernel for the MDP.

        Args:
            state_size (int): Number of states.
            action_size (int): Number of actions.

        Returns:
            np.ndarray: Transition kernel with shape [state_size x action_size x state_size].
        """
        # Create a transition kernel of shape [state_size x action_size x state_size]
        transition_kernel = np.zeros((state_size, action_size, state_size))

        for s in range(state_size):
            for a in range(action_size):
                # Sample from a Dirichlet distribution to ensure the row sums to 1
                transition_kernel[s, a] = np.random.dirichlet(np.ones(state_size))

        return transition_kernel

    def get_reward_function(self, state_size: int, action_size: int, variance: str) -> np.ndarray:
        """
        Generates a reward function based on reward variance levels.

        Args:
            state_size (int): Number of states in the MDP.
            action_size (int): Number of actions in the MDP.
            variance (str): Reward variance level: "no_variance", "low_variance", "high_variance", "max_variance".

        Returns:
            np.ndarray: Reward matrix with shape [state_size x action_size].
        """
        # Validate variance level
        reward_config = get_reward_variance_config(variance)

        # Create a reward matrix initialized to zeros
        rewards = np.zeros((state_size, action_size))

        # Apply variance-based reward assignments
        reward_assignments = int(action_size * reward_config[1])  # Negative rewards
        for s in range(state_size):
            negative_indices = np.random.choice(action_size, size=reward_assignments, replace=False)
            rewards[s, negative_indices] = -1
            rewards[s, :] += 1  # Default positive reward set to 1

        return rewards


# Example usage
if __name__ == "__main__":
    # Use hardcoded config for demonstration; typically this comes from config.yaml
    config = {
        "state_space_sizes": [3, 9, 81],
        "action_space_sizes": [3, 9, 81],
        "reward_variance_levels": ["no_variance", "low_variance", "high_variance", "max_variance"],
    }

    generator = MDPGenerator(config)
    mdp = generator.generate_mdp(state_size=9, action_size=9, reward_variance="high_variance")

    print("Generated MDP:")
    print("State Space:", mdp["state_space"])
    print("Action Space:", mdp["action_space"])
    print("Transition Kernel Shape:", mdp["transition_kernel"].shape)
    print("Rewards Shape:", mdp["rewards"].shape)
