import logging
import os
import sys
import json
import yaml
from typing import Dict, Any, Optional


class Logger:
    """
    Manages logging for the experiment, directing messages to both the console
    and a file, and providing utilities to save configuration and results.
    """

    def __init__(self, log_dir: str, filename: str = 'experiment.log') -> None:
        """
        Initializes the Logger, setting up console and file handlers.

        Args:
            log_dir (str): The directory where log files and results will be saved.
            filename (str): The name of the main log file. Defaults to 'experiment.log'.
        """
        self.log_dir = log_dir
        self.filename = filename

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Get a logger instance
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers to prevent duplicate output
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Define a formatter for log messages
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # File handler
        file_path = os.path.join(self.log_dir, self.filename)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.info(f"Logger initialized. Log messages will be saved to: {file_path}")

    def info(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message (str): The message string to log.
        """
        self.logger.info(message)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None,
                    epoch: Optional[int] = None, prefix: str = '') -> None:
        """
        Logs a dictionary of metrics, optionally including step, epoch, and a prefix.

        Args:
            metrics (Dict[str, Any]): A dictionary of metric names and their values.
            step (Optional[int]): The current training step number. Defaults to None.
            epoch (Optional[int]): The current epoch number. Defaults to None.
            prefix (str): A prefix string (e.g., 'Train', 'Validation', 'Test'). Defaults to ''.
        """
        log_parts = []
        if epoch is not None:
            log_parts.append(f"[Epoch {epoch:03d}]")
        if step is not None:
            log_parts.append(f"[Step {step:05d}]")
        if prefix:
            log_parts.append(f"[{prefix}]")

        metric_parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metric_parts.append(f"{key}: {value:.4f}")
            else:
                metric_parts.append(f"{key}: {value}")

        message = " ".join(log_parts + metric_parts)
        self.info(message)

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Saves the full configuration dictionary to a YAML file in the log directory.

        Args:
            config (Dict[str, Any]): The configuration dictionary to save.
        """
        config_path = os.path.join(self.log_dir, 'config.yaml')
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.info(f"Configuration saved to: {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")

    def save_results_to_json(self, results: Dict[str, Any], filename: str) -> None:
        """
        Saves a dictionary of aggregated results to a JSON file in the log directory.

        Args:
            results (Dict[str, Any]): The results dictionary to save.
            filename (str): The name of the JSON file (e.g., 'final_results.json').
        """
        results_path = os.path.join(self.log_dir, filename)
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            self.info(f"Results saved to: {results_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results to {results_path}: {e}")

