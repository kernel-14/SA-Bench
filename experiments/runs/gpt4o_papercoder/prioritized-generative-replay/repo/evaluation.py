## evaluation.py

import os
import json
from typing import Dict, Any
import numpy as np
import torch
from utils import Utils
from data_loader import DataLoader
from rl_agent import RLAgent
import matplotlib.pyplot as plt


class Evaluation:
    """
    Handles the evaluation of the RL agent’s performance in test environments.
    Computes metrics (e.g., average return, dormant ratio) and generates visualizations.
    """

    def __init__(self, env_name: str, config: dict) -> None:
        """
        Initializes the evaluation module with the environment, agent, and output configurations.

        Args:
            env_name (str): Name of the evaluation environment.
            config (dict): Configuration dictionary parsed from `config.yaml`.
        """
        self.env_name = env_name
        self.config = config

        # Parse settings from config
        self.eval_episodes = self.config["evaluation"]["eval_episodes"]
        self.num_seeds = self.config["evaluation"]["num_seeds"]
        self.metrics_to_compute = self.config["evaluation"]["metrics"]
        self.visualizations_dir = self.config["output"]["visualizations_dir"]
        self.logs_dir = self.config["output"]["logs_dir"]

        # Initialize DataLoader for evaluation environment
        self.data_loader = DataLoader(env_name, config)
        self.env, self.env_config = self.data_loader.setup_environment()

        # Create output directories
        Utils.setup_directories([self.visualizations_dir, self.logs_dir])

    def evaluate(self, agent: RLAgent) -> Dict[str, Any]:
        """
        Evaluates the policy of a trained RL agent.

        Args:
            agent (RLAgent): Trained RLAgent instance.

        Returns:
            Dict[str, Any]: Dictionary of computed evaluation metrics.
        """
        # Seed setup for reproducibility
        Utils.set_seeds(42)

        # Tracking metrics
        average_returns = []
        dormant_ratios = []

        for seed in range(self.num_seeds):
            # Set seed
            Utils.set_seeds(seed)

            # Seed-specific metrics
            seed_returns = []
            neuron_activations = []

            for episode in range(self.eval_episodes):
                # Reset environment
                observation = self.data_loader.reset_environment()
                done = False
                episode_return = 0

                while not done:
                    # Get action from the trained policy
                    observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(agent.device)
                    with torch.no_grad():
                        action = agent.actor(observation_tensor).cpu().numpy()
                    
                    # Step environment
                    next_observation, reward, done, _ = self.env.step(action)

                    # Accumulate reward
                    episode_return += reward

                    # Collect neuron activations for DR computation
                    activations = agent.actor(observation_tensor)
                    neuron_activations.append(activations)

                    # Update observation
                    observation = next_observation

                seed_returns.append(episode_return)

            # Compute metrics
            average_seed_return = np.mean(seed_returns)
            dormant_ratio = self._compute_dormant_ratio(neuron_activations)

            average_returns.append(average_seed_return)
            dormant_ratios.append(dormant_ratio)

        # Metrics aggregation
        metrics = {
            "average_return": np.mean(average_returns),
            "average_return_std": np.std(average_returns),
            "dormant_ratio": np.mean(dormant_ratios),
            "dormant_ratio_std": np.std(dormant_ratios),
        }

        # Save metrics to file
        metrics_path = os.path.join(self.logs_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        return metrics

    @staticmethod
    def _compute_dormant_ratio(activations: list, threshold: float = 0.01) -> float:
        """
        Computes the dormant ratio, which measures the fraction of inactive neurons.

        Args:
            activations (list): List of activation tensors over evaluation steps.
            threshold (float): Threshold below which neurons are considered dormant.

        Returns:
            float: Dormant ratio.
        """
        total_neurons = 0
        dormant_neurons = 0

        for activation in activations:
            total_neurons += activation.numel()
            dormant_neurons += (activation.abs() < threshold).sum().item()

        return dormant_neurons / total_neurons

    def visualize_results(self, metrics: Dict[str, Any]) -> None:
        """
        Visualizes the evaluation metrics into plots saved in the specified directory.

        Args:
            metrics (Dict[str, Any]): Dictionary of computed evaluation metrics.
        """
        # Visualization for average return
        if "average_return" in metrics:
            plt.figure(figsize=(10, 6))
            plt.title("Average Return Across Evaluation Episodes")
            plt.xlabel("Seeds")
            plt.ylabel("Average Return")
            plt.errorbar(
                range(len(metrics["average_return"])),
                metrics["average_return"],
                yerr=metrics["average_return_std"],
                fmt="o",
                label="Average Return (mean ± std)",
            )
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(self.visualizations_dir, "average_return.png"))
            plt.close()

        # Visualization for dormant ratio
        if "dormant_ratio" in metrics:
            plt.figure(figsize=(10, 6))
            plt.title("Dormant Ratio Across Evaluation Seeds")
            plt.xlabel("Seeds")
            plt.ylabel("Dormant Ratio")
            plt.errorbar(
                range(len(metrics["dormant_ratio"])),
                metrics["dormant_ratio"],
                yerr=metrics["dormant_ratio_std"],
                fmt="o",
                label="Dormant Ratio (mean ± std)",
            )
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(self.visualizations_dir, "dormant_ratio.png"))
            plt.close()
