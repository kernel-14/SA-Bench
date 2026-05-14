# utils/logger.py
"""Centralized logging utility for the MoE-POT training pipeline.

Provides a Logger class that writes to both Python's standard logging
module (file + stdout) and optionally Weights & Biases (wandb).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional


class Logger:
    """Centralized logger for MoE-POT experiments.

    Handles two output channels:
      - Python logging (file at DEBUG level, stdout at INFO level)
      - Weights & Biases (optional, guarded by try/except)

    Attributes:
        log_dir: Directory where log files are written.
        use_wandb: Whether wandb logging is active.
        _logger: Python logging.Logger instance.
        _wandb_run: wandb run object or None.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_wandb: bool = False,
    ) -> None:
        """Initializes the Logger.

        Creates the log directory, sets up file and stdout handlers,
        and optionally initializes a wandb run. In distributed training,
        only rank 0 writes logs and initializes wandb.

        Args:
            log_dir: Directory path where log files will be written.
            experiment_name: Name used for the logger, log file, and
                wandb project.
            use_wandb: Whether to enable Weights & Biases logging.
                Defaults to False.
        """
        self.log_dir: str = log_dir
        self.use_wandb: bool = use_wandb
        self._wandb_run: Optional[Any] = None

        # Create log directory (safe on restart with exist_ok=True).
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Determine distributed rank for DDP safety.
        local_rank: int = int(os.environ.get("LOCAL_RANK", "0"))
        is_main_process: bool = local_rank == 0

        # Build Python logger. Using experiment_name as the logger name
        # ensures a unique logger per experiment in the global registry.
        self._logger: logging.Logger = logging.getLogger(experiment_name)
        self._logger.setLevel(logging.DEBUG)

        # Guard against duplicate handlers when __init__ is called
        # multiple times (e.g., in DDP where each rank may initialize).
        if not self._logger.handlers:
            formatter = logging.Formatter(
                fmt="[%(asctime)s][%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Stdout handler: INFO and above for clean terminal output.
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)

            # File handler: DEBUG and above for full detail.
            # Non-zero ranks write to rank-specific files to avoid
            # concurrent write conflicts.
            if is_main_process:
                log_filename = f"{experiment_name}.log"
            else:
                log_filename = f"{experiment_name}_rank{local_rank}.log"

            log_filepath = os.path.join(log_dir, log_filename)
            file_handler = logging.FileHandler(log_filepath, mode="a")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        # Initialize wandb only on the main process.
        if use_wandb and is_main_process:
            try:
                import wandb  # pylint: disable=import-outside-toplevel

                self._wandb_run = wandb.init(
                    project=experiment_name,
                    dir=log_dir,
                )
                self._logger.info("Weights & Biases initialized successfully.")
            except ImportError:
                self._wandb_run = None
                self._logger.warning(
                    "wandb is not installed. Disabling wandb logging. "
                    "Install with: pip install wandb"
                )
            except Exception as exc:  # pylint: disable=broad-except
                self._wandb_run = None
                self._logger.warning(
                    "Failed to initialize wandb: %s. Disabling wandb logging.",
                    exc,
                )
        elif use_wandb and not is_main_process:
            # Non-zero ranks never use wandb to avoid duplicate runs.
            self._wandb_run = None

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Logs a dictionary of metrics at a given training step.

        Writes a formatted string to the Python logger (file + stdout)
        and, if enabled, sends the metrics dict to wandb.

        Args:
            metrics: Dictionary mapping metric names to values.
                Example: {"train_loss": 0.042, "val_l2re": 0.031}.
            step: Current training step or epoch number.
        """
        # Format metrics into a human-readable string.
        parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}: {value:.6f}")
            else:
                parts.append(f"{key}: {value}")
        formatted_string = " | ".join(parts)

        self._logger.info("Step %d | %s", step, formatted_string)

        # Send to wandb if active.
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.warning(
                    "Failed to log metrics to wandb at step %d: %s", step, exc
                )

    def log_config(self, config: Any) -> None:
        """Logs the experiment configuration.

        Serializes the config to JSON, writes it to a file in log_dir,
        logs a summary line, and syncs to wandb if active.

        Args:
            config: Config object with a to_dict() method that returns
                a serializable dictionary.
        """
        # Serialize config to a dictionary.
        config_dict: dict = config.to_dict()

        # Write config to a JSON file for permanent record.
        config_filepath = os.path.join(self.log_dir, "config.json")
        try:
            with open(config_filepath, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, default=str)
            self._logger.debug("Config written to %s", config_filepath)
        except OSError as exc:
            self._logger.warning("Failed to write config file: %s", exc)

        # Log a summary line to the log file.
        try:
            config_summary = json.dumps(config_dict, default=str)
            self._logger.info("Config: %s", config_summary)
        except (TypeError, ValueError) as exc:
            self._logger.warning("Failed to serialize config for logging: %s", exc)

        # Sync config to wandb dashboard if active.
        if self._wandb_run is not None:
            try:
                self._wandb_run.config.update(config_dict)
            except Exception as exc:  # pylint: disable=broad-except
                self._logger.warning(
                    "Failed to sync config to wandb: %s", exc
                )

    def info(self, message: str) -> None:
        """Logs an informational message.

        Thin wrapper around Python logger.info() so callers do not need
        to import the logging module directly.

        Args:
            message: Human-readable status message to log.
        """
        self._logger.info(message)

    def debug(self, message: str) -> None:
        """Logs a debug-level message (file only, not stdout).

        Args:
            message: Debug message to log.
        """
        self._logger.debug(message)

    def warning(self, message: str) -> None:
        """Logs a warning message.

        Args:
            message: Warning message to log.
        """
        self._logger.warning(message)

    def error(self, message: str) -> None:
        """Logs an error message.

        Args:
            message: Error message to log.
        """
        self._logger.error(message)
