## utils/logging_utils.py
"""Logging utilities for LoRA-SB reproduction experiments.

This module provides logging infrastructure used by ExperimentRunner (main.py)
and Trainer (training/trainer.py). It has no dependencies on other project
modules — only Python standard library is used.

Typical usage:
    from utils.logging_utils import get_logger, log_metrics, save_json

    logger = get_logger("experiment", log_file="outputs/experiment.log")
    log_metrics(logger, {"loss": 2.345, "lr": 1e-4}, step=100, prefix="train")
    save_json({"accuracy": 63.38, "seeds": [42, 43, 44]}, "outputs/results.json")
"""

import json
import logging
import os
import sys
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_serializable(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects to native Python types.

    Handles torch.Tensor, numpy arrays, numpy scalar types, dicts, and lists.
    All other types are passed through unchanged (assumed to be natively
    JSON-serializable: str, int, float, bool, None).

    Args:
        obj: Any Python object that may appear in an experiment results dict.

    Returns:
        A JSON-serializable version of ``obj``.
    """
    # Lazy imports to avoid hard dependency at module load time.
    # These packages are always present in the project environment but we
    # avoid importing them at the top level to keep this module lightweight.
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            return obj.tolist()
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # numpy scalar types (np.float32, np.int64, etc.)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]

    # Native Python types (str, int, float, bool, None) pass through unchanged.
    return obj


def _format_value(value: Any) -> str:
    """Format a single metric value for human-readable log output.

    Floats with absolute value < 0.01 are formatted in scientific notation
    (appropriate for learning rates like 1e-4). All other floats use fixed-
    point notation with 4 decimal places (appropriate for loss ~2.3 and
    accuracy ~0.85). Integers are formatted without decimals.

    Args:
        value: The metric value to format.

    Returns:
        A formatted string representation of the value.
    """
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0.0:
            return f"{value:.4e}"
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    # Fallback for other types (e.g., str labels)
    return str(value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create and configure a named Python logger.

    Returns a logger with a console handler (always) and an optional file
    handler. Calling this function multiple times with the same ``name``
    returns the same cached logger instance without adding duplicate handlers.

    The log format is::

        [2024-01-15 10:23:45,123] [experiment] [INFO] Training started.

    This timestamp granularity is important for tracking initialization
    overhead (Table 6 in the paper: 2–4 minutes vs 3–5 hours training time).

    Args:
        name: Logger name, e.g. ``"experiment"``, ``"trainer"``,
            ``"evaluator"``. Used as the ``%(name)s`` field in log records.
        log_file: Optional path to a log file. If provided, the parent
            directory is created automatically. Log records are appended
            (mode ``'a'``) so that resuming a run does not overwrite prior
            logs. If ``None``, only console output is produced.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    # Guard against adding duplicate handlers when called multiple times
    # with the same name (Python's logging module caches loggers globally).
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Consistent format across all handlers: timestamp, logger name, level, message.
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — always present for interactive monitoring.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — only when a log_file path is provided.
    if log_file is not None:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent log records from propagating to the root logger, which would
    # cause duplicate output if the root logger also has handlers configured.
    logger.propagate = False

    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    step: int,
    prefix: str = "",
) -> None:
    """Format and log a dictionary of metric values at a given training step.

    Produces a single INFO-level log line in the format::

        [train] Step 100 | loss=2.3451 | lr=9.8000e-05

    This function is called by:
    - ``Trainer._train_epoch`` every ``config.log_every_n_steps`` steps (100).
    - ``Evaluator`` after each evaluation pass.
    - ``ExperimentRunner`` during initialization to log overhead timing.

    The step-level loss values logged here are the source data for the
    training loss curves in Figure 2 of the paper (LoRA-SB vs LoRA-XS on
    Mistral-7B and Gemma-2 9B).

    Args:
        logger: A configured logger instance from ``get_logger()``.
        metrics: Dictionary mapping metric names to their values. Values may
            be Python floats, ints, or any type handled by ``_format_value``.
            Example: ``{"loss": 2.3451, "lr": 9.8e-05, "grad_norm": 1.02}``.
        step: The current training step or epoch number. Used as the ``Step N``
            field in the log line.
        prefix: Optional prefix string (e.g., ``"train"``, ``"eval"``,
            ``"init"``) that distinguishes log lines from different phases.
            If non-empty, it is prepended in brackets: ``[train]``.
            Defaults to ``""`` (no prefix).
    """
    # Build the key=value pairs string.
    pairs = " | ".join(
        f"{key}={_format_value(value)}" for key, value in metrics.items()
    )

    # Assemble the full log line.
    if prefix:
        message = f"[{prefix}] Step {step} | {pairs}"
    else:
        message = f"Step {step} | {pairs}"

    logger.info(message)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Persist experiment results to a JSON file.

    Handles non-serializable types (torch.Tensor, numpy arrays, numpy scalars)
    by recursively converting them to native Python types before serialization.
    The output is indented for human readability.

    Called by ``ExperimentRunner._save_results`` after all seeds complete.
    The saved JSON contains per-seed metrics, mean, and standard deviation
    across seeds [42, 43, 44], mirroring the reporting format of Tables 1–3
    in the paper.

    Args:
        data: Dictionary of experiment results. May contain nested dicts,
            lists, torch.Tensor objects, numpy arrays, and native Python types.
        path: Destination file path. The parent directory is created
            automatically if it does not exist. Relative paths are resolved
            against the current working directory.

    Example:
        >>> save_json(
        ...     {"gsm8k": 63.38, "math": 17.44, "seeds": [42, 43, 44]},
        ...     "outputs/mistral_math/lora_sb/rank96/results.json"
        ... )
    """
    # Resolve to absolute path so os.path.dirname works correctly for
    # relative paths like "outputs/results.json".
    abs_path = os.path.abspath(path)
    parent_dir = os.path.dirname(abs_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Convert all non-serializable objects before writing.
    serializable_data = _make_serializable(data)

    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2)
