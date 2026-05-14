# evaluation.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List
from rwm_model import RWMModel
from policy_optimizer import PolicyOptimizer


class Evaluation:
    """
    Evaluation class responsible for assessing the predictive accuracy of the Robotic World Model (RWM)
    and the performance of trained policies in simulated environments and hardware deployment.
    """

    def __init__(self, model: RWMModel, trained_policy: object, config: dict):
        """
        Initialize the Evaluation instance.

        Args:
            model (RWMModel): Trained Robotic World Model.
            trained_policy (object): Trained policy from PolicyOptimizer.
            config (dict): Configuration settings from `config.yaml`.
        """
        self.model = model
        self.trained_policy = trained_policy
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Hyperparameters
        self.history_horizon = config["training"]["history_horizon"]
        self.forecast_horizon = config["training"]["forecast_horizon"]
        self.noise_levels = config["evaluation"]["noise_levels"]  # Noise levels for robustness tests

        # Visualization output paths
        self.visualization_dir = "visualizations/"
        os.makedirs(self.visualization_dir, exist_ok=True)

    def evaluate_prediction(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluates multi-step autoregressive prediction accuracy and robustness under noisy conditions.

        Args:
            test_loader (DataLoader): DataLoader containing test trajectories (observations, actions, ground truth).

        Returns:
            Dict[str, Any]: Includes mean prediction errors and robustness metrics.
        """
        self.model.eval()

        metrics = {
            "mean_prediction_error": [],
            "noise_robustness": {},
        }

        # Evaluate prediction error without noise
        metrics["mean_prediction_error"] = self._compute_prediction_error(test_loader)

        # Evaluate prediction robustness under noise
        for noise_level in self.noise_levels:
            metrics["noise_robustness"][noise_level] = self._compute_prediction_error(
                test_loader, noise_level=noise_level
            )

        # Visualize robustness results
        self._visualize_robustness(metrics["noise_robustness"])
        return metrics

    def _compute_prediction_error(self, test_loader: DataLoader, noise_level: float = 0.0) -> float:
        """
        Compute mean prediction error for the RWM across the test dataset, optionally with added noise.

        Args:
            test_loader (DataLoader): DataLoader with test data.
            noise_level (float): Standard deviation of Gaussian noise applied to observations and actions.

        Returns:
            float: Mean prediction error across the forecasting horizon for the test set.
        """
        total_error = 0.0
        num_samples = 0

        with torch.no_grad():
            for history, actions, targets_obs, targets_priv in test_loader:
                # Add Gaussian noise if specified
                if noise_level > 0:
                    history += torch.normal(0, noise_level, size=history.shape).to(self.device)
                    actions += torch.normal(0, noise_level, size=actions.shape).to(self.device)

                # Move tensors to device
                history = history.to(self.device)
                actions = actions.to(self.device)
                targets_obs = targets_obs.to(self.device)

                # Predict using RWM
                predictions, _ = self.model(history, actions)

                # Compute multi-step prediction error
                pred_mean, _ = torch.chunk(predictions, 2, dim=-1)
                error = torch.norm(pred_mean - targets_obs, p=2, dim=-1)  # L2 loss per step
                mean_error = error.mean().item()  # Average for batch
                total_error += mean_error * history.size(0)  # Weighted by batch size
                num_samples += history.size(0)

        return total_error / num_samples

    def evaluate_policy(self, buffer: List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]) -> Dict[str, Any]:
        """
        Evaluates the trained policy's performance in imagination rollouts and zero-shot hardware transfer.

        Args:
            buffer (List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]): Replay buffer from policy training.

        Returns:
            Dict[str, Any]: Includes tracking rewards, stability metrics, and hardware deployment results.
        """
        self.trained_policy.eval()
        imagination_steps = self.config["policy_optimizer"]["imagination_steps"]

        policy_metrics = {"mean_reward": [], "stability_metrics": {}, "hardware_results": {}}

        # Simulated Policy Evaluation
        total_rewards = []
        for i in range(imagination_steps):
            # Sample buffer (state, action, reward, next_state)
            initial_state = buffer[np.random.choice(len(buffer))][0].to(self.device)
            current_state = initial_state
            trajectory_rewards = []

            for step in range(imagination_steps):
                action = self.trained_policy(current_state).detach()
                predicted_obs, _ = self.model(
                    history=current_state.unsqueeze(0), actions=action.unsqueeze(0)
                )
                next_state = predicted_obs.squeeze(0)
                reward = self._compute_task_reward(next_state)
                trajectory_rewards.append(reward)
                current_state = next_state

            total_rewards.append(np.mean(trajectory_rewards))

        policy_metrics["mean_reward"] = np.mean(total_rewards)

        # Additional stability metrics (e.g., noise injection or disturbances)
        policy_metrics["stability_metrics"] = self._evaluate_policy_stability(buffer)

        # Hardware deployment (placeholder for experiment setup)
        policy_metrics["hardware_results"] = self._evaluate_hardware_transfer()
        return policy_metrics

    def _compute_task_reward(self, obs: torch.Tensor) -> float:
        """
        Compute task-specific reward. For example, velocity tracking accuracy.

        Args:
            obs (torch.Tensor): Observation.

        Returns:
            float: Computed reward based on task dynamics.
        """
        velocity_command = obs[..., 9:12]
        actual_velocity = obs[..., 0:3]
        reward = torch.exp(-torch.norm(velocity_command - actual_velocity, p=2)**2)
        return reward.item()

    def _evaluate_policy_stability(
        self, buffer: List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Evaluate policy stability under noise and domain shifts.

        Args:
            buffer (List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]): Replay buffer from policy training.

        Returns:
            Dict[str, float]: Stability metrics such as noise robustness and domain shift adaptation.
        """
        stability_metrics = {}

        for noise_level in self.noise_levels:
            rewards = []
            for i in range(len(buffer)):
                initial_state = buffer[np.random.choice(len(buffer))][0].to(self.device)
                initial_state += torch.normal(0, noise_level, size=initial_state.shape).to(self.device)
                action = self.trained_policy(initial_state).detach()
                predicted_obs, _ = self.model(
                    history=initial_state.unsqueeze(0), actions=action.unsqueeze(0)
                )
                rewards.append(self._compute_task_reward(predicted_obs.squeeze(0)))

            stability_metrics[f"noise_level_{noise_level}"] = np.mean(rewards)

        return stability_metrics

    def _evaluate_hardware_transfer(self) -> Dict[str, Any]:
        """
        Evaluate trained policy transfer on hardware robots (zero-shot deployment).

        Returns:
            Dict[str, Any]: Results from hardware transfer tests.
        """
        hardware_results = {"tracking_reward": 0.0, "stability": 0.0, "observations": {}}
        print("Hardware deployment evaluation placeholder. Implement based on real-world experiments.")
        return hardware_results

    def visualize_rollouts(self, test_loader: DataLoader) -> None:
        """
        Generate and save visualizations for autoregressive rollouts (predictions vs. ground truth).

        Args:
            test_loader (DataLoader): DataLoader with test data.
        """
        self.model.eval()

        for batch_idx, (history, actions, targets_obs, _) in enumerate(test_loader):
            history = history.to(self.device)
            actions = actions.to(self.device)
            targets_obs = targets_obs.to(self.device)

            with torch.no_grad():
                predictions, _ = self.model(history, actions)
                pred_mean, _ = torch.chunk(predictions, 2, dim=-1)

            # Plot rollouts (example visualization for first test case)
            if batch_idx == 0:
                plt.figure(figsize=(12, 6))
                for i in range(pred_mean.shape[-1]):  # Iterate over features
                    plt.plot(
                        range(self.forecast_horizon),
                        targets_obs[0, :, i].cpu().numpy(),
                        label=f"Feature {i} (Ground Truth)"
                    )
                    plt.plot(
                        range(self.forecast_horizon),
                        pred_mean[0, :, i].cpu().numpy(),
                        linestyle="--",
                        label=f"Feature {i} (Prediction)"
                    )
                plt.legend()
                plt.title(f"Autoregressive Prediction Rollout (Batch {batch_idx})")
                plt.xlabel("Forecast Horizon")
                plt.ylabel("Observation")
                plt.savefig(os.path.join(self.visualization_dir, f"rollout_batch_{batch_idx}.png"))
                print(f"Saved rollout visualization: rollout_batch_{batch_idx}.png")
                break
