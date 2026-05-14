"""
evaluation.py
Handles evaluation of the MR.Q algorithm on test benchmarks, including computation of
normalized metrics (TD3-normalized, Human-normalized), learning curves, and result logging.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from utils import Utils


class Evaluation:
    """
    Class to handle MR.Q model evaluation across multiple environments.
    """

    def __init__(self, model: torch.nn.Module, eval_envs: List[Any], metrics: List[str], config: Dict[str, Any]):
        """
        Initialize the Evaluation class.

        Args:
            model (torch.nn.Module): Trained MR.Q model.
            eval_envs (List[Any]): List of evaluation environments to test the model on.
            metrics (List[str]): Metrics to compute (e.g., 'cumulative_rewards', 'mean', 'median', 'IQM').
            config (Dict[str, Any]): Configuration dictionary parsed from the config.yaml file.
        """
        self.model = model
        self.eval_envs = eval_envs
        self.metrics = metrics
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_episodes = config["evaluation"].get("num_episodes", 10)
        self.log_dir = config["logging"]["log_dir"]
        self.results = {}
        self.logger = Utils.setup_logger(log_dir=self.log_dir)

    def evaluate_policy(self, num_episodes: int = None) -> Dict[str, Any]:
        """
        Evaluate the given model across all environments for multiple episodes.

        Args:
            num_episodes (int, optional): Number of episodes for evaluation. Defaults to config value.

        Returns:
            Dict[str, Any]: Results containing cumulative rewards, normalized metrics, and per-environment stats.
        """
        num_episodes = num_episodes or self.num_episodes
        print(f"Evaluating policy for {num_episodes} episodes across {len(self.eval_envs)} environments...")
        results = {}

        for env in self.eval_envs:
            env_name = env.spec.id if hasattr(env, "spec") else str(env)
            print(f"Evaluating on environment: {env_name}")
            episode_rewards = []

            for episode in range(num_episodes):
                obs = env.reset()
                cumulative_reward = 0
                done = False

                while not done:
                    obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action = self.model.policy_head(self.model.forward_state(obs_tensor)).cpu().numpy()
                    action = action.argmax(-1) if self.model.discrete_action_space else action
                    next_obs, reward, done, _ = env.step(action)
                    cumulative_reward += reward
                    obs = next_obs

                episode_rewards.append(cumulative_reward)

            # Aggregate statistics for the current environment
            results[env_name] = self._compute_metrics(episode_rewards)

        self.results = results
        return results

    def _compute_metrics(self, rewards: List[float]) -> Dict[str, Any]:
        """
        Compute metrics for a given set of episode rewards.

        Args:
            rewards (List[float]): List of rewards obtained across episodes for a single environment.

        Returns:
            Dict[str, Any]: Dictionary containing metrics like mean, median, and IQM.
        """
        rewards = np.array(rewards)
        mean = rewards.mean()
        median = np.median(rewards)
        rewards_sorted = np.sort(rewards)
        iqm = rewards_sorted[len(rewards_sorted) // 4 : len(rewards_sorted) * 3 // 4].mean()

        metrics = {
            "mean": mean,
            "median": median,
            "iqm": iqm,
            "raw_rewards": rewards.tolist(),
        }
        return metrics

    def compute_normalized_metrics(self, raw_results: Dict[str, Any], baseline_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Normalize evaluation results based on task-specific baselines.

        Args:
            raw_results (Dict[str, Any]): Raw evaluation results.
            baseline_scores (Dict[str, Dict[str, float]]): Dictionary of task baselines for normalization.

        Returns:
            Dict[str, Any]: Normalized results.
        """
        normalized_results = {}
        for env_name, metrics in raw_results.items():
            if env_name not in baseline_scores:
                print(f"Baseline scores not found for environment {env_name}. Skipping normalization.")
                continue

            baseline = baseline_scores[env_name]
            random_score, reference_score = baseline["random"], baseline["reference"]

            normalized_rewards = [
                (r - random_score) / (reference_score - random_score)
                for r in metrics["raw_rewards"]
            ]
            mean = np.mean(normalized_rewards)
            median = np.median(normalized_rewards)
            iqm = np.sort(normalized_rewards)[len(normalized_rewards) // 4 : len(normalized_rewards) * 3 // 4].mean()

            normalized_results[env_name] = {
                "normalized_mean": mean,
                "normalized_median": median,
                "normalized_iqm": iqm,
                "normalized_rewards": normalized_rewards,
            }

        return normalized_results

    def visualize_learning_curves(self, results: Dict[str, Any], log_dir: str) -> None:
        """
        Visualize learning curves for evaluation results.

        Args:
            results (Dict[str, Any]): Evaluation results, including raw rewards.
            log_dir (str): Directory to save the visualizations.
        """
        os.makedirs(log_dir, exist_ok=True)

        for env_name, metrics in results.items():
            rewards = metrics.get("raw_rewards", [])
            steps = list(range(1, len(rewards) + 1))
            save_path = os.path.join(log_dir, f"{env_name}_learning_curve.png")

            Utils.plot_learning_curve(
                rewards=rewards, steps=steps, save_path=save_path, label=f"Cumulative Rewards ({env_name})"
            )

    def log_results(self, results: Dict[str, Any]) -> None:
        """
        Log evaluation results to TensorBoard and save to a CSV.

        Args:
            results (Dict[str, Any]): Evaluation results to log.
        """
        # Write raw results to a CSV file
        csv_path = os.path.join(self.log_dir, "evaluation_results.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w") as f:
            f.write("Environment,Mean,Median,IQM\n")
            for env_name, metrics in results.items():
                f.write(
                    f"{env_name},{metrics['mean']},{metrics['median']},{metrics['iqm']}\n"
                )

        print(f"Evaluation results saved to: {csv_path}")

        # Log results to TensorBoard
        for env_name, metrics in results.items():
            self.logger.add_scalar(f"{env_name}/Mean", metrics["mean"], 0)
            self.logger.add_scalar(f"{env_name}/Median", metrics["median"], 0)
            self.logger.add_scalar(f"{env_name}/IQM", metrics["iqm"], 0)
