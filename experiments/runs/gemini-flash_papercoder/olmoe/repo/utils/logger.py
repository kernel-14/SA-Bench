"""
This module implements a Logger class for experiment tracking using Weights & Biases (wandb).
It provides functionalities to initialize a wandb run, log configuration, record metrics,
and monitor model parameters and gradients.
"""

import dataclasses
import wandb
import torch.nn as nn
from typing import Dict, Any

# Assuming Config is available at the root level or via a relative import path.
# If config.py is in the same directory as utils, it would be 'from .config import Config'.
# Given the provided structure, it's likely a direct import from the project root.
from config import Config


class Logger:
    """
    Manages experiment logging and tracking using Weights & Biases (wandb).

    This class initializes a wandb run, logs the experiment configuration,
    records training and evaluation metrics, and can monitor the model's
    parameters and gradients.
    """

    def __init__(self, project_name: str, run_name: str, config: Config):
        """
        Initializes a new Weights & Biases run and logs the experiment configuration.

        Args:
            project_name: The name of the wandb project.
            run_name: A unique name for the current experiment run.
            config: The global configuration object containing all experiment settings.
        """
        # Initialize wandb run
        wandb.init(
            project=project_name,
            name=run_name,
            config=dataclasses.asdict(config)  # Convert dataclass to dict for wandb.config
        )
        self._config = config  # Store config for internal use, e.g., log_interval

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Records a set of metrics at a specific point in the experiment.

        Args:
            metrics: A dictionary where keys are metric names (strings) and values
                     are their corresponding scalar numerical values.
            step: The global step count or epoch number at which these metrics
                  were recorded.
        """
        wandb.log(metrics, step=step)

    def watch_model(self, model: nn.Module) -> None:
        """
        Enables wandb to automatically track the parameters and gradients of the
        provided PyTorch neural network model.

        Args:
            model: The PyTorch model instance whose parameters and gradients
                   are to be monitored.
        """
        # Ensure that log_interval is a positive integer to avoid issues with wandb.watch
        log_freq = getattr(self._config.training, 'log_interval', 100)
        if not isinstance(log_freq, int) or log_freq <= 0:
            print(f"Warning: Invalid log_interval ({log_freq}). Defaulting to 100 for wandb.watch.")
            log_freq = 100

        wandb.watch(
            model,
            log='all',  # Log both gradients and parameters
            log_freq=log_freq
        )

