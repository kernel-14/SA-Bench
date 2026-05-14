"""
utils.py
Utilities for MR.Q implementation, including reward binning, model management, logging utilities,
reward normalization, visualization, and configuration file parsing.
"""

import os
import yaml
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, List, Tuple


# Shared Constants
CONFIG_PATH = "config.yaml"
DEFAULT_REWARD_BINS = 65
DEFAULT_REWARD_RANGE = (-10, 10)
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
PERCENTILE_25, PERCENTILE_50, PERCENTILE_75 = 25, 50, 75


class Utils:
    """Utility functions for the MR.Q project."""

    @staticmethod
    def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
        """Load the configuration file (YAML format)."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

    @staticmethod
    def setup_logger(log_dir: str = "./logs") -> SummaryWriter:
        """Set up TensorBoard logger for metrics."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return SummaryWriter(log_dir)

    @staticmethod
    def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, save_path: str, epoch: int) -> None:
        """Save model and optimizer states."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }, save_path)
        print(f"Model saved at: {save_path}")

    @staticmethod
    def load_model(save_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
        """Load model and optimizer states."""
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Checkpoint file not found: {save_path}")

        checkpoint = torch.load(save_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        print(f"Model successfully loaded from: {save_path}")
        return epoch

    @staticmethod
    def preprocess_reward(
        reward: float, bins: int = DEFAULT_REWARD_BINS, reward_range: Tuple[float, float] = DEFAULT_REWARD_RANGE
    ) -> np.ndarray:
        """
        Convert a reward into a two-hot encoded representation based on non-uniform interval binning.
        """
        low, high = reward_range
        reward = np.clip(reward, low, high)
        bin_edges = np.linspace(low, high, bins)

        # Find two neighboring bins and calculate their weights
        closest_bin = np.digitize(reward, bin_edges) - 1
        closest_bin = np.clip(closest_bin, 0, len(bin_edges) - 2)

        bin_low = bin_edges[closest_bin]
        bin_high = bin_edges[closest_bin + 1]
        weight_high = (reward - bin_low) / (bin_high - bin_low)
        weight_low = 1 - weight_high

        # Encode two-hot encoding as categorical representation
        encoding = np.zeros(bins)
        encoding[closest_bin] = weight_low
        encoding[closest_bin + 1] = weight_high
        return encoding

    @staticmethod
    def normalize_rewards(rewards: List[float], baseline: Tuple[float, float]) -> List[float]:
        """
        Normalize rewards based on a baseline (e.g., human or TD3 scores).
        Args:
            rewards: List of rewards across episodes.
            baseline: A tuple (min_score, max_score) for normalization.
        Returns:
            List of normalized rewards.
        """
        min_score, max_score = baseline
        normalized = [(r - min_score) / (max_score - min_score) for r in rewards]
        return normalized

    @staticmethod
    def plot_learning_curve(rewards: List[float], steps: List[int], save_path: str, label: str = "Cumulative Reward"):
        """
        Plot learning curve of cumulative reward over interactions.
        Args:
            rewards: List of cumulative rewards per evaluation.
            steps: Corresponding evaluation steps.
            save_path: Path to save the plot image.
            label: Label for the y-axis (default: "Cumulative Reward").
        """
        plt.figure(figsize=(8, 5))
        plt.plot(steps, rewards, label=label, color="blue", linewidth=2)
        plt.fill_between(
            steps,
            np.percentile(rewards, PERCENTILE_25),
            np.percentile(rewards, PERCENTILE_75),
            color="blue",
            alpha=0.3,
            label="Confidence Interval (25-75%)",
        )
        plt.xlabel("Steps")
        plt.ylabel(label)
        plt.title(f"{label} Over Training Steps")
        plt.legend()
        plt.grid()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Learning curve saved at: {save_path}")
