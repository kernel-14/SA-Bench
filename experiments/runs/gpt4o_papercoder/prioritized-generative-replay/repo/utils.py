"""
utils.py

Utility module providing functionalities for logging, managing configuration files, setting random seeds, handling checkpoints, calculating metrics, and visualizing experimental results.
"""

import os
import yaml
import logging
from typing import List, Dict, Any
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class Utils:
    """Utility class for shared functionalities."""

    @staticmethod
    def setup_logging(log_dir: str = "./logs", log_file: str = "experiment.log") -> logging.Logger:
        """
        Sets up logging with both console and file handlers.

        Args:
            log_dir (str): Directory to save log files.
            log_file (str): Name of the log file.

        Returns:
            logging.Logger: Configured logger instance.
        """
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        # Configure logger
        logger = logging.getLogger("PGR_Logger")
        logger.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    @staticmethod
    def parse_config(config_path: str = "config.yaml") -> Dict[str, Any]:
        """
        Parses the configuration file.

        Args:
            config_path (str): Path to the configuration YAML file.

        Returns:
            Dict[str, Any]: Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            yaml.YAMLError: If the YAML content is invalid.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        try:
            with open(config_path, "r") as config_file:
                config = yaml.safe_load(config_file)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing the YAML config file: {e}")

    @staticmethod
    def set_seeds(seed: int = 42) -> None:
        """
        Sets random seeds for reproducibility.

        Args:
            seed (int): Seed value for random number generation.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def setup_directories(directories: List[str]) -> None:
        """
        Ensures existence of given directories.

        Args:
            directories (List[str]): List of directory paths to create if they do not exist.
        """
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def save_checkpoint(model: nn.Module, path: str) -> None:
        """
        Saves model checkpoint.

        Args:
            model (nn.Module): PyTorch model to save.
            path (str): Path to save the checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
        """
        Loads model checkpoint.

        Args:
            model (nn.Module): PyTorch model to load weights into.
            path (str): Path to checkpoint file.

        Returns:
            nn.Module: Model with loaded weights.
        """
        model.load_state_dict(torch.load(path))
        return model

    @staticmethod
    def compute_mse(generated: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]) -> float:
        """
        Computes mean squared error (MSE) between generated and ground truth transitions.

        Args:
            generated (List[Dict[str, Any]]): List of generated transitions.
            ground_truth (List[Dict[str, Any]]): List of ground truth transitions.

        Returns:
            float: Mean squared error value.
        """
        mse = 0.0
        for gen, true in zip(generated, ground_truth):
            mse += np.mean((np.array(gen) - np.array(true)) ** 2)
        return mse / len(generated)

    @staticmethod
    def compute_average_return(rewards: List[float]) -> float:
        """
        Computes average episodic return.

        Args:
            rewards (List[float]): List of episodic rewards.

        Returns:
            float: Average return value.
        """
        return np.mean(rewards)

    @staticmethod
    def compute_dormant_ratio(activations: torch.Tensor, threshold: float = 0.01) -> float:
        """
        Computes dormant ratio, the fraction of inactive neurons below a given activation threshold.

        Args:
            activations (torch.Tensor): Tensor representing policy network activations.
            threshold (float): Threshold below which activations are considered dormant.

        Returns:
            float: Dormant ratio.
        """
        total_neurons = activations.numel()
        dormant_neurons = (activations < threshold).sum().item()
        return dormant_neurons / total_neurons

    @staticmethod
    def plot_metrics(metrics: Dict[str, Any], output_dir: str = "./visualizations") -> None:
        """
        Plots and saves metrics visualizations.

        Args:
            metrics (Dict[str, Any]): Dictionary containing metrics to plot.
            output_dir (str): Directory to save the plots.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Example plot: average return over time
        if "average_return" in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics["average_return"], label="Average Return", marker='o', linestyle='-')
            plt.xlabel("Environment Steps")
            plt.ylabel("Average Return")
            plt.title("Average Return vs Environment Steps")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_dir, "average_return.png"))
            plt.close()

        # Example plot: dormant ratio trends
        if "dormant_ratio" in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics["dormant_ratio"], label="Dormant Ratio", marker='o', linestyle='-')
            plt.xlabel("Environment Steps")
            plt.ylabel("Dormant Ratio")
            plt.title("Dormant Ratio Over Time")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_dir, "dormant_ratio.png"))
            plt.close()

        # Additional plots can be defined similarly.

    @staticmethod
    def handle_error(module: str, exception: Exception, logger: logging.Logger = None) -> None:
        """
        Handles exceptions and logs them.

        Args:
            module (str): Name of the module where the error occurred.
            exception (Exception): Exception instance.
            logger (logging.Logger): Logger instance to log the error (optional).
        """
        error_message = f"Error in module {module}: {exception}"
        if logger:
            logger.error(error_message)
        print(error_message)  # Always print for visibility

