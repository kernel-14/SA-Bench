## utils/logger.py
"""Logging utility for the PEFT Visual Recognition reproduction study.

This module provides the Logger class, which handles structured logging to
console, file, TensorBoard, and CSV. It has zero dependencies on other project
files and serves as the foundational utility imported by all other modules.

Typical usage:
    logger = Logger(log_dir="./outputs", experiment_name="vtab_lora_dtd")
    logger.log_config(config)
    logger.log_metrics({"train/loss": 0.5, "val/accuracy": 72.1}, step=10)
    logger.save_results_csv({"method": "lora", "accuracy": 72.1}, "results.csv")
    logger.close()
"""

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union


class Logger:
    """Structured logger for PEFT experiments.

    Writes to console, a rotating log file, TensorBoard, and CSV files.
    Uses named Python loggers to avoid handler collisions during grid search.

    Attributes:
        log_dir: Base directory for all log artifacts.
        experiment_name: Unique name for this experiment run.
        full_log_path: Resolved path = os.path.join(log_dir, experiment_name).
        metrics_csv_path: Path to the per-step metrics CSV file.
        writer: TensorBoard SummaryWriter, or None if unavailable.
    """

    def __init__(self, log_dir: str, experiment_name: str) -> None:
        """Initialises all logging infrastructure.

        Creates the experiment directory, sets up a named Python logger with
        both console and file handlers, and optionally initialises a TensorBoard
        SummaryWriter.

        Args:
            log_dir: Base output directory (e.g. "./outputs").
            experiment_name: Unique identifier for this run
                (e.g. "vtab_lora_dtd_lr0.001").
        """
        self.log_dir: str = log_dir
        self.experiment_name: str = experiment_name
        self.full_log_path: str = os.path.join(log_dir, experiment_name)

        # Create experiment directory tree.
        os.makedirs(self.full_log_path, exist_ok=True)

        # ------------------------------------------------------------------
        # Python standard logging setup
        # ------------------------------------------------------------------
        self._logger: logging.Logger = logging.getLogger(experiment_name)
        self._logger.setLevel(logging.INFO)

        # Guard against duplicate handlers when Logger is re-instantiated
        # inside the same Python process (e.g. during hyperparameter search).
        if not self._logger.handlers:
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Console handler.
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

            # File handler — append mode so restarts don't overwrite history.
            log_file_path: str = os.path.join(self.full_log_path, "experiment.log")
            file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

        # ------------------------------------------------------------------
        # TensorBoard SummaryWriter (optional dependency)
        # ------------------------------------------------------------------
        self.writer: Optional[Any] = None  # type: ignore[assignment]
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            tb_dir: str = os.path.join(self.full_log_path, "tensorboard")
            self.writer = SummaryWriter(log_dir=tb_dir)
        except ImportError:
            self._logger.warning(
                "TensorBoard not available. Scalar metrics will only be written to CSV."
            )

        # ------------------------------------------------------------------
        # CSV metrics file (created lazily on first write)
        # ------------------------------------------------------------------
        self.metrics_csv_path: str = os.path.join(self.full_log_path, "metrics.csv")

        self._logger.info(
            "Logger initialised. Artifacts will be written to: %s",
            self.full_log_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Records scalar metrics to TensorBoard and the metrics CSV file.

        Handles both Python floats and PyTorch tensors as metric values.

        Args:
            metrics: Flat mapping of metric name to scalar value.
                Example: {"train/loss": 0.42, "val/top1_acc": 71.3}.
            step: Global training step or epoch number used as the x-axis
                in TensorBoard and the "step" column in the CSV.
        """
        # Normalise tensor values to Python floats.
        clean_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            try:
                # Works for torch.Tensor, numpy scalars, and plain numbers.
                clean_metrics[key] = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                self._logger.warning(
                    "Metric '%s' has non-numeric value %r; skipping.", key, value
                )

        # Write to TensorBoard.
        if self.writer is not None:
            for tag, scalar_value in clean_metrics.items():
                self.writer.add_scalar(tag, scalar_value, global_step=step)

        # Write to CSV (append mode; write header on first row).
        file_exists: bool = (
            os.path.isfile(self.metrics_csv_path)
            and os.path.getsize(self.metrics_csv_path) > 0
        )
        row: Dict[str, Any] = {"step": step, **clean_metrics}
        fieldnames: List[str] = list(row.keys())

        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=fieldnames, extrasaction="ignore"
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def log_config(self, config: Any) -> None:
        """Serialises the experiment configuration to a JSON file.

        Supports Config dataclasses (with a ``to_dict`` method),
        omegaconf DictConfig objects, and plain Python dicts.

        Args:
            config: Experiment configuration object or dict.
        """
        config_dict: Dict[str, Any]

        if hasattr(config, "to_dict") and callable(config.to_dict):
            # Our Config dataclass.
            config_dict = config.to_dict()
        else:
            # Try omegaconf first, then fall back to vars() / dict().
            try:
                from omegaconf import OmegaConf  # type: ignore

                config_dict = OmegaConf.to_container(config, resolve=True)  # type: ignore[assignment]
            except (ImportError, Exception):
                if isinstance(config, dict):
                    config_dict = config
                else:
                    try:
                        config_dict = vars(config)
                    except TypeError:
                        config_dict = {"config_repr": str(config)}

        config_path: str = os.path.join(self.full_log_path, "config.json")
        with open(config_path, "w", encoding="utf-8") as json_file:
            json.dump(config_dict, json_file, indent=2, default=str)

        self._logger.info("Config saved to %s", config_path)

    def info(self, msg: str) -> None:
        """Writes an informational message to the console and log file.

        Args:
            msg: Human-readable message string.
        """
        self._logger.info(msg)

    def save_results_csv(
        self,
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
        path: str,
    ) -> None:
        """Saves structured experiment results to a CSV file.

        Supports both a single result row (dict) and multiple rows (list of
        dicts). If the target file already exists, rows are appended without
        re-writing the header, enabling incremental saving during long runs.

        Args:
            results: A single result dict or a list of result dicts.
                Example single row:
                    {"method": "lora", "dataset": "dtd", "accuracy": 72.1}
                Example multiple rows:
                    [{"method": "lora", ...}, {"method": "bitfit", ...}]
            path: Destination CSV file path. Relative paths are resolved
                relative to ``self.full_log_path``; absolute paths are used
                as-is.
        """
        # Normalise to a list of dicts.
        rows: List[Dict[str, Any]]
        if isinstance(results, dict):
            rows = [results]
        else:
            rows = list(results)

        if not rows:
            self._logger.warning("save_results_csv called with empty results; nothing written.")
            return

        # Resolve path.
        resolved_path: str = (
            path if os.path.isabs(path) else os.path.join(self.full_log_path, path)
        )

        # Ensure parent directory exists.
        parent_dir: str = os.path.dirname(resolved_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Determine fieldnames from the first row.
        fieldnames: List[str] = list(rows[0].keys())

        # Decide whether to write the header.
        file_exists: bool = (
            os.path.isfile(resolved_path) and os.path.getsize(resolved_path) > 0
        )
        write_mode: str = "a" if file_exists else "w"

        with open(resolved_path, write_mode, newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

        self._logger.info(
            "Results (%d row(s)) saved to %s", len(rows), resolved_path
        )

    def close(self) -> None:
        """Flushes and closes all logging resources.

        Should be called at the end of ``main.py`` to ensure TensorBoard
        data is fully written and file handles are released cleanly.
        """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)

        # Use root logger for this final message since our named logger's
        # handlers have just been removed.
        logging.info(
            "Logger '%s' closed. Artifacts at: %s",
            self.experiment_name,
            self.full_log_path,
        )
