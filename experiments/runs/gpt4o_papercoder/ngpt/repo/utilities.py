## utilities.py
import os
import yaml
import torch
from torch import nn
from torch.optim import Optimizer
from typing import Dict, Tuple

class Utilities:
    """A utility class providing helper methods for configuration management, checkpoint handling, and logging."""

    @staticmethod
    def get_config(file_path: str) -> Dict:
        """
        Loads the configuration file (config.yaml) and returns it as a dictionary.

        Args:
            file_path (str): Path to the configuration YAML file.

        Returns:
            dict: Parsed configuration.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Validation and setting default values if necessary
        required_keys = ["training", "model", "dataset", "hardware", "evaluation"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Example of enforcing defaults
        config["training"].setdefault("learning_rate", 2.0e-3)
        config["training"].setdefault("epochs", 100000)
        config["training"].setdefault("weight_decay", 0.0)
        config["model"].setdefault("d_model", 1024)  # Default to 0.5B model specs
        config["model"].setdefault("n_layers", 24)
        config["hardware"].setdefault("gpus", 64)
        return config

    @staticmethod
    def save_checkpoint(
        model: nn.Module, 
        optimizer: Optimizer, 
        scaling_factors: dict, 
        save_path: str
    ) -> None:
        """
        Save the model, optimizer state, and scaling parameters to a checkpoint.

        Args:
            model (nn.Module): The model to save.
            optimizer (Optimizer): The optimizer state to save.
            scaling_factors (dict): Dictionary of trainable scaling parameters.
            save_path (str): Path to save the checkpoint.
        """
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaling_factors": scaling_factors
        }
        try:
            torch.save(checkpoint, save_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint: {e}")

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Tuple[nn.Module, Optimizer, dict]:
        """
        Load the model, optimizer state, and scaling factors from a checkpoint.

        Args:
            checkpoint_path (str): Path to the saved checkpoint.

        Returns:
            Tuple[nn.Module, Optimizer, dict]: Restored model, optimizer, and scaling factors.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if "model_state_dict" not in checkpoint or \
           "optimizer_state_dict" not in checkpoint or \
           "scaling_factors" not in checkpoint:
            raise ValueError("Checkpoint is missing required keys: model, optimizer, or scaling factors.")

        return checkpoint["model_state_dict"], checkpoint["optimizer_state_dict"], checkpoint["scaling_factors"]

    @staticmethod
    def log_metrics(metrics: dict, step: int, log_file: str) -> None:
        """
        Logs training or evaluation metrics to a file.

        Args:
            metrics (dict): Dictionary of metrics to log (e.g., loss, perplexity).
            step (int): The current training or evaluation step.
            log_file (str): Path to the log file.
        """
        log_entry = f"Step: {step} | "
        log_entry += " | ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
        
        print(log_entry)  # Optionally print to console
        
        if not os.path.isdir(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, "a") as file:
            file.write(log_entry + "\n")

    @staticmethod
    def tensor_normalization_check(tensor: torch.Tensor, eps: float = 1e-6) -> bool:
        """
        Validates whether a tensor is normalized to unit norm.

        Args:
            tensor (Tensor): The input tensor to check.
            eps (float): Numerical tolerance for normalization validation.

        Returns:
            bool: True if all norms are within acceptable bounds, otherwise False.
        """
        norms = torch.linalg.norm(tensor, dim=-1)
        if torch.all((norms >= 1 - eps) & (norms <= 1 + eps)):
            return True
        raise ValueError(f"Tensor normalization check failed: Norms outside acceptable range.")

    @staticmethod
    def create_directory(path: str) -> None:
        """
        Ensures a directory exists, creating it if necessary.

        Args:
            path (str): The directory path to create.
        """
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
