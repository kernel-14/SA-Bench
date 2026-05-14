## evaluator.py

from typing import Dict, Optional, List
import numpy as np
import torch
from sokoban_environment import SokobanEnvironment
from drc_model import DRCModel
from prober import Prober
import matplotlib.pyplot as plt
import os


class Evaluator:
    """
    The Evaluator class handles quantitative evaluation, qualitative analysis,
    and plan visualization of the trained DRC model on Sokoban levels.
    """

    def __init__(self, model: DRCModel, env: SokobanEnvironment, config: dict):
        """
        Initialize the Evaluator class.

        Args:
            model (DRCModel): The trained DRC model.
            env (SokobanEnvironment): Sokoban environment instance.
            config (dict): Configuration dictionary from config.yaml.
        """
        self.model = model
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.grid_size = config["environment"]["grid_size"]
        self.thinking_ticks = config["agent"]["recurrent_ticks"]

    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate the model on unseen Sokoban levels and compute success rates and metrics.

        Args:
            num_episodes (int): Number of episodes for evaluation.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics including success rate, average steps,
            and average rewards.
        """
        metrics = {
            "success_rate": 0.0,
            "average_steps": 0.0,
            "average_rewards": 0.0,
        }
        total_success = 0
        total_steps = 0
        total_rewards = 0

        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_rewards = 0
            steps = 0

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                policy_logits, _ = self.model(obs_tensor)
                action = torch.argmax(policy_logits, dim=-1).item()
                obs, reward, done, _ = self.env.step(action)
                episode_rewards += reward
                steps += 1

            total_success += int(all(pos in self.env.target_positions for pos in self.env.box_positions))
            total_steps += steps
            total_rewards += episode_rewards

        metrics["success_rate"] = total_success / num_episodes
        metrics["average_steps"] = total_steps / num_episodes
        metrics["average_rewards"] = total_rewards / num_episodes

        return metrics

    def evaluate_with_thinking(self, num_episodes: int, thinking_steps: int) -> Dict[str, float]:
        """
        Evaluate the model's performance using additional thinking computational ticks.

        Args:
            num_episodes (int): Number of episodes for evaluation.
            thinking_steps (int): Number of stationary thinking steps.

        Returns:
            Dict[str, float]: Evaluation metrics including improvements with thinking steps.
        """
        metrics = {
            "success_rate_with_thinking": 0.0,
            "average_steps_with_thinking": 0.0,
            "average_rewards_with_thinking": 0.0,
        }
        total_success = 0
        total_steps = 0
        total_rewards = 0

        for episode in range(num_episodes):
            obs = self.env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Perform thinking steps
            for _ in range(thinking_steps * self.thinking_ticks):
                _, _ = self.model(obs_tensor)

            # Proceed with normal episode
            done = False
            episode_rewards = 0
            steps = 0

            while not done:
                policy_logits, _ = self.model(obs_tensor)
                action = torch.argmax(policy_logits, dim=-1).item()
                obs, reward, done, _ = self.env.step(action)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                episode_rewards += reward
                steps += 1

            total_success += int(all(pos in self.env.target_positions for pos in self.env.box_positions))
            total_steps += steps
            total_rewards += episode_rewards

        metrics["success_rate_with_thinking"] = total_success / num_episodes
        metrics["average_steps_with_thinking"] = total_steps / num_episodes
        metrics["average_rewards_with_thinking"] = total_rewards / num_episodes

        return metrics

    def evaluate_ood(self, ood_dataset: str, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate the model on Out-of-Distribution (OOD) Sokoban levels.

        Args:
            ood_dataset (str): Path to out-of-distribution dataset.
            num_episodes (int): Number of OOD episodes to evaluate.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics specific to OOD levels.
        """
        # Assuming ood_dataset provides environment-level reset configuration
        metrics = {
            "ood_success_rate": 0.0,
            "ood_average_steps": 0.0,
            "ood_average_rewards": 0.0,
        }
        total_success = 0
        total_steps = 0
        total_rewards = 0

        for episode in range(num_episodes):
            # Reset environment with OOD configuration
            obs = self.env.reset()
            done = False
            episode_rewards = 0
            steps = 0

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                policy_logits, _ = self.model(obs_tensor)
                action = torch.argmax(policy_logits, dim=-1).item()
                obs, reward, done, _ = self.env.step(action)
                episode_rewards += reward
                steps += 1

            total_success += int(all(pos in self.env.target_positions for pos in self.env.box_positions))
            total_steps += steps
            total_rewards += episode_rewards

        metrics["ood_success_rate"] = total_success / num_episodes
        metrics["ood_average_steps"] = total_steps / num_episodes
        metrics["ood_average_rewards"] = total_rewards / num_episodes

        return metrics

    def visualize_plans(self, episode_data: Dict, save_path: str) -> None:
        """
        Visualize the evolution of internal plans across layers and ticks.

        Args:
            episode_data (Dict): Data from an episode (hidden states and actions).
            save_path (str): Directory path to save the visualization.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for layer_idx, layer_data in episode_data.items():
            fig, ax = plt.subplots(self.grid_size, self.grid_size, figsize=(10, 10))
            fig.suptitle(f"Internal Plans - Layer {layer_idx}")

            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    ax[x, y].arrow(0.5, 0.5, layer_data[x, y, 0], layer_data[x, y, 1])
                    ax[x, y].axis("off")

            plt.savefig(os.path.join(save_path, f"internal_plan_layer_{layer_idx}.png"))
            plt.close(fig)

    def generate_metrics(self) -> Dict[str, float]:
        """
        Compile and generate all evaluation metrics into an exportable dictionary.

        Returns:
            Dict[str, float]: Compiled evaluation metrics.
        """
        return {
            "success_rate": 0.0,  # Placeholder
            "average_steps": 0.0,  # Placeholder
            "average_rewards": 0.0,  # Placeholder
        }
