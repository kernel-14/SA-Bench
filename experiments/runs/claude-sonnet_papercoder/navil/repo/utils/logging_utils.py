## utils/logging_utils.py
"""Logging utilities for the NaViL training and evaluation pipeline.

This module provides three components consumed by trainer.py, evaluator.py,
and main.py:

1. ``setup_logger`` — creates a named Python logger with file + console output.
2. ``log_metrics`` — formats and emits a structured metrics line at a given step.
3. ``AverageMeter`` — tracks running averages of scalar metrics across steps.

Design constraints:
- No internal project dependencies (importable before any model code).
- No third-party dependencies (logging, os, sys, typing only).
- Append-mode file logging for safe checkpoint resumption.
- Named loggers with duplicate-handler guard for distributed training safety.

Config alignment:
- ``log_every_steps: 100`` drives AverageMeter.reset() cadence in trainer.py.
- ``log_dir: "logs/navil_2b"`` is the parent directory; callers append filenames.
- Only rank-0 processes should pass a non-None log_file; others pass None.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional, Union


# ---------------------------------------------------------------------------
# Module-level format string shared by all handlers created in this module.
# Format: [timestamp][logger_name][level] message
# ---------------------------------------------------------------------------
_LOG_FORMAT: str = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Union[int, str] = logging.INFO,
) -> logging.Logger:
    """Create and configure a named logger with file and/or console output.

    Uses named loggers (not the root logger) to prevent log pollution across
    modules. Guards against duplicate handler registration so this function
    is safe to call multiple times with the same ``name``.

    Args:
        name:     Logger name, e.g. ``"navil.trainer"`` or
                  ``"navil.evaluator"``. Hierarchical dot-separated names
                  are supported by Python's logging framework.
        log_file: Absolute or relative path to the log file. Parent
                  directories are created automatically. If ``None`` or
                  empty string, only the console handler is attached.
                  In distributed training, non-zero ranks should pass
                  ``None`` to avoid concurrent file writes.
        level:    Logging level as an integer (e.g. ``logging.INFO``) or
                  a string (e.g. ``"INFO"``, ``"DEBUG"``). Defaults to
                  ``logging.INFO``.

    Returns:
        A configured ``logging.Logger`` instance. If the logger was already
        configured (handlers already attached), the existing instance is
        returned unchanged to prevent duplicate log lines.

    Example::

        logger = setup_logger(
            "navil.trainer",
            log_file="logs/navil_2b/trainer.log",
            level=logging.INFO,
        )
        logger.info("Training started.")
    """
    # ------------------------------------------------------------------ #
    # Resolve level to an integer if a string was passed.                 #
    # logging.getLevelName("INFO") returns 20; getLevelName(20) returns   #
    # "INFO". We always want an integer for setLevel().                   #
    # ------------------------------------------------------------------ #
    resolved_level: int
    if isinstance(level, str):
        resolved_level = logging.getLevelName(level.upper())
        if not isinstance(resolved_level, int):
            # Unrecognised string — fall back to INFO.
            resolved_level = logging.INFO
    else:
        resolved_level = int(level)

    logger: logging.Logger = logging.getLogger(name)

    # Guard: if handlers are already attached, return the existing logger.
    # This prevents duplicate log lines when setup_logger is called more
    # than once (e.g., on import in multiple submodules or after a reload).
    if logger.handlers:
        return logger

    logger.setLevel(resolved_level)

    # Prevent log records from propagating to the root logger, which may
    # have its own handlers and would produce duplicate output.
    logger.propagate = False

    # ------------------------------------------------------------------ #
    # Shared formatter                                                     #
    # ------------------------------------------------------------------ #
    formatter: logging.Formatter = logging.Formatter(
        fmt=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
    )

    # ------------------------------------------------------------------ #
    # Console handler — always attached, writes to stdout.                #
    # Using stdout (not stderr) keeps training logs clean and separable   #
    # from Python error tracebacks.                                       #
    # ------------------------------------------------------------------ #
    console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------ #
    # File handler — attached only when log_file is provided.             #
    # Append mode ('a') ensures logs persist across checkpoint restarts.  #
    # ------------------------------------------------------------------ #
    if log_file:
        # Create parent directories if they do not exist.
        log_dir: str = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler: logging.FileHandler = logging.FileHandler(
            log_file,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_metrics(
    logger: logging.Logger,
    metrics_dict: Dict[str, Any],
    step: int,
    prefix: str = "train",
) -> None:
    """Format and emit a structured metrics log line at a given training step.

    Produces a single INFO-level log line of the form::

        [prefix] step=<step> | key1=val1 key2=val2 ...

    Float values are formatted to 4 decimal places for readability.
    Integer values are formatted as plain integers. All other types use
    ``repr()``.

    Args:
        logger:       A configured ``logging.Logger`` instance, typically
                      obtained from ``setup_logger``.
        metrics_dict: Mapping of metric names to scalar values. Keys are
                      strings; values may be ``float``, ``int``, or any
                      type with a meaningful ``repr``.
                      Example: ``{"loss": 2.3456, "lr": 5e-5, "grad_norm": 0.87}``
        step:         Current global training step (0-indexed integer).
        prefix:       Label prepended to the log line to identify the
                      training stage or phase. Defaults to ``"train"``.
                      Typical values: ``"train/s1_1"``, ``"train/s1_2"``,
                      ``"train/s2"``, ``"val"``, ``"eval"``.

    Returns:
        None. This is a pure side-effect function.

    Example::

        log_metrics(
            logger,
            {"loss": 2.3456, "lr": 5e-5, "grad_norm": 0.87},
            step=100,
            prefix="train/s1_1",
        )
        # Emits:
        # [2024-01-01 12:00:00][navil.trainer][INFO]
        #   [train/s1_1] step=100 | loss=2.3456 lr=0.0001 grad_norm=0.8700
    """
    # Build the key=value pairs string.
    parts: list = []
    key: str
    value: Any
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            formatted_value: str = f"{value:.4f}"
        elif isinstance(value, int):
            formatted_value = str(value)
        else:
            formatted_value = repr(value)
        parts.append(f"{key}={formatted_value}")

    metrics_str: str = " ".join(parts)
    log_line: str = f"[{prefix}] step={step} | {metrics_str}"

    logger.info(log_line)


class AverageMeter:
    """Tracks the running average of a scalar metric across multiple updates.

    Designed for use in training loops where loss (or other scalars) should
    be averaged over a window of steps before logging. The standard usage
    pattern is:

    1. Create one ``AverageMeter`` per metric at the start of training.
    2. Call ``update(val)`` after each training step.
    3. Every ``log_every_steps`` steps, log ``meter.avg`` then call
       ``meter.reset()`` to start a fresh window.

    The ``n`` parameter in ``update`` supports weighted averaging: if the
    loss is already averaged over a batch of size ``n``, passing ``n``
    gives a properly weighted global average. For NaViL's NTP loss
    (``reduction='mean'``), ``n=1`` is appropriate since the loss is
    already batch-averaged.

    Attributes:
        val:   Most recent value passed to ``update()``. Zero after ``reset()``.
        sum:   Cumulative weighted sum of all values since last ``reset()``.
        count: Total weight accumulated since last ``reset()``.
        avg:   Running weighted average (``sum / count``). Zero if ``count == 0``.

    Example::

        loss_meter = AverageMeter()
        for step, batch in enumerate(dataloader):
            loss = model(batch)
            loss_meter.update(loss.item())
            if step % 100 == 0 and step > 0:
                log_metrics(logger, {"loss": loss_meter.avg}, step, "train")
                loss_meter.reset()
    """

    def __init__(self) -> None:
        """Initialise all state to zero by delegating to ``reset()``."""
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: float = 0.0
        self.avg: float = 0.0
        # Call reset() to ensure a single source of truth for initialisation.
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated state to zero.

        Call this at the start of each logging window (every
        ``log_every_steps`` steps) to compute a fresh window average
        rather than a cumulative average over the entire training run.

        Returns:
            None.
        """
        self.val = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0

    def update(self, val: float, n: float = 1.0) -> None:
        """Record a new scalar value and update the running average.

        Args:
            val: The scalar value to record. Typically ``loss.item()``
                 from a training step.
            n:   Weight for this update. Defaults to ``1.0``.
                 Set to batch size when ``val`` is a sum (not mean) over
                 the batch. For NaViL's mean-reduced NTP loss, use the
                 default ``n=1.0``.

        Returns:
            None.

        Note:
            If ``count`` is zero after the update (which cannot happen
            with ``n > 0``), ``avg`` is set to ``0.0`` to avoid
            division by zero. This guard handles the edge case where
            ``n=0`` is passed.
        """
        self.val = float(val)
        self.sum += float(val) * float(n)
        self.count += float(n)

        if self.count > 0.0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0.0
