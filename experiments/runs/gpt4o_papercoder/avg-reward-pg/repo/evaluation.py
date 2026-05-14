"""
evaluation.py

This module defines the Evaluation class which computes metrics such as average reward, suboptimality gaps, 
and visualizes convergence trends for the policy gradient experiments. It strictly adheres to the methodology 
described in the paper and the experimental design provided.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Import constants from utils/constants.py
from utils.constants import STATE_SPACE_SIZES, get_hyperparameters

class Evaluation:
    """
    Evaluation Class:
    Provides methods for computing metrics like average reward, suboptimality gaps, and visualizing convergence trends.
    """

    def __init__(self, mdp: Dict) -> None:
        """
        Initializes the Evaluation class with the provided MDP environment.

        Args:
            mdp (Dict): MDP dictionary containing the transition kernel, reward function, state space, and action space.
        
        Raises:
            ValueError: If the MDP dictionary is invalid or missing required keys.
        """
        if not mdp or not isinstance(mdp, dict):
            raise ValueError("Provided MDP must be a valid dictionary containing transition kernel and rewards.")
        if not all(key in mdp for key in ["state_space", "action_space", "transition_kernel", "rewards"]):
            raise ValueError("MDP dictionary is missing required keys: 'state_space', 'action_space', 'transition_kernel', or 'rewards'.")

        self.transition_kernel = mdp["transition_kernel"]
        self.rewards = mdp["rewards"]
        self.state_space = mdp["state_space"]
        self.action_space = mdp["action_space"]

    def evaluate(self, policy: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluates the given policy in terms of average reward and stationary distribution.

        Args:
            policy (np.ndarray): Current policy matrix (\( |\mathcal{S}| \times |\mathcal{A}| \)).

        Returns:
            Dict[str, np.ndarray]: Contains `average_reward` and `stationary_distribution`.
        """
        num_states, num_actions = policy.shape

        # Step 1: Compute policy-induced transition probabilities \( \mathbb{P}^\pi \)
        policy_transition = np.zeros((num_states, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                policy_transition[s] += policy[s, a] * self.transition_kernel[s, a]

        # Step 2: Compute stationary distribution \( d^\pi \)
        eigenvalues, eigenvectors = np.linalg.eig(policy_transition.T)
        stationary_distribution = eigenvectors[:, np.isclose(eigenvalues, 1)].flatten().real
        stationary_distribution /= stationary_distribution.sum()  # Normalize to ensure it sums to 1

        # Check for edge cases (e.g., singular matrices)
        if np.any(np.isnan(stationary_distribution)):
            raise ValueError("Stationary distribution calculation resulted in NaNs.")

        # Step 3: Calculate average reward \( \rho^\pi \)
        reward_per_state = np.sum(policy * self.rewards, axis=1)
        average_reward = np.sum(stationary_distribution * reward_per_state)

        return {
            "average_reward": average_reward,
            "stationary_distribution": stationary_distribution,
        }

    def compute_suboptimality_gap(self, optimal_reward: float, current_reward: float) -> float:
        """
        Computes the suboptimality gap (\( \rho^* - \rho^{\pi_k} \)).

        Args:
            optimal_reward (float): Global optimal average reward.
            current_reward (float): Average reward for the current policy.

        Returns:
            float: Suboptimality gap.
        """
        if optimal_reward < current_reward:
            raise ValueError("Optimal reward must be greater than or equal to the current reward.")
        return optimal_reward - current_reward

    def plot_convergence(self, metrics: Dict[str, List[float]]) -> None:
        """
        Plots the convergence trends for the policy gradient experiment.

        Args:
            metrics (Dict[str, List[float]]): Contains keys like `average_rewards` and `suboptimality_gaps`
                with corresponding values as lists of floats over iterations.
        
        Raises:
            ValueError: If metrics are improperly formatted or missing required keys.
        """
        if not metrics or not isinstance(metrics, dict):
            raise ValueError("Metrics must be a valid dictionary.")
        if "average_rewards" not in metrics or "suboptimality_gaps" not in metrics:
            raise ValueError("Metrics dictionary must contain keys 'average_rewards' and 'suboptimality_gaps'.")

        average_rewards = metrics["average_rewards"]
        suboptimality_gaps = metrics["suboptimality_gaps"]

        # Validate metric sizes
        if not average_rewards or not suboptimality_gaps:
            raise ValueError("Metrics for average rewards or suboptimality gaps cannot be empty.")

        # Plot convergence trends
        plt.figure(figsize=(12, 6))

        # Subplot 1: Average Rewards over iterations
        plt.subplot(1, 2, 1)
        plt.plot(range(len(average_rewards)), average_rewards, label="Average Reward", color="blue")
        plt.title("Convergence of Average Rewards")
        plt.xlabel("Iterations")
        plt.ylabel("Average Reward")
        plt.grid(True)
        plt.legend()

        # Subplot 2: Suboptimality Gap over iterations
        plt.subplot(1, 2, 2)
        plt.plot(range(len(suboptimality_gaps)), suboptimality_gaps, label="Suboptimality Gap", color="red")
        plt.title("Suboptimality Gap Over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Suboptimality Gap")
        plt.yscale("log")  # Log scale to validate theoretical convergence
        plt.grid(True)
        plt.legend()

        # Save and display the plot
        plt.tight_layout()
        save_path = "convergence_plot.png"
        plt.savefig(save_path)
        print(f"Convergence plot saved to {save_path}")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Dummy MDP setup for testing
    state_space = list(range(5))
    action_space = list(range(3))
    transition_kernel = np.random.dirichlet(np.ones(len(state_space)), size=(len(state_space), len(action_space)))
    rewards = np.random.rand(len(state_space), len(action_space))

    # Instantiate evaluation with dummy MDP
    mdp = {
        "state_space": state_space,
        "action_space": action_space,
        "transition_kernel": transition_kernel,
        "rewards": rewards,
    }
    evaluator = Evaluation(mdp)

    # Dummy policies for evaluation
    policy = np.ones((len(state_space), len(action_space))) / len(action_space)
    evaluated_metrics = evaluator.evaluate(policy)
    print("Evaluated Metrics:", evaluated_metrics)

    # Dummy convergence metrics
    metrics = {
        "average_rewards": [1.0, 1.2, 1.3, 1.35],
        "suboptimality_gaps": [0.5, 0.4, 0.3, 0.2],
    }
    evaluator.plot_convergence(metrics)
