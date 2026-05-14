# utils/logging.py

"""
Centralised logging utility for the NaViL reproduction project.

Provides:

- ``setup_logging`` : Configures root logger with human‑readable,
  timestamped formatting, optional file output, and distributed‑training
  awareness (only the global rank 0 process emits messages).
- ``get_logger`` : Returns a Python logger for a given module name,
  inheriting the root configuration set by ``setup_logging``.

This module should be imported by all other parts of the codebase.
``setup_logging`` is called **once** at the very beginning of ``main.py``,
before any other component is instantiated.

Example usage (in ``main.py``)::

    import logging
    from utils.logging import setup_logging

    setup_logging(level=logging.INFO, log_file="train.log")

Later, in any other module::

    from utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Starting training stage…")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Custom formatter that includes milliseconds in timestamps
# ---------------------------------------------------------------------------

class _MillisecondFormatter(logging.Formatter):
    """
    Extends the standard ``logging.Formatter`` to append milliseconds to the
    timestamp (``asctime``) field.  This yields timestamps like::

        2025-07-17 14:45:03.123

    which is helpful for fine‑grained debugging of long training runs.
    """

    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        """Override to add milliseconds after the seconds."""
        from datetime import datetime

        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            s = ct.strftime("%Y-%m-%d %H:%M:%S")
        return f"{s}.{int(record.msecs):03d}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    logger_name: str = "navil",
) -> logging.Logger:
    """
    Set up the root logger with a consistent format, optional file output,
    and awareness of distributed training (suppresses logging on non‑rank‑0
    processes).

    Subsequent calls are idempotent in the sense that they replace any
    existing root handlers; only the last invocation matters.

    Args:
        level:
            Severity threshold for log messages (e.g., ``logging.INFO``).
            Defaults to ``logging.INFO``.
        log_file:
            If provided, all log messages are written to this file in
            addition to the console.  The file is overwritten each time
            ``setup_logging`` is called.
        logger_name:
            Name of the top‑level logger.  The root logger is configured,
            so this argument is only used for the docstring; all child
            loggers inherit the configuration.  Defaults to ``"navil"``.

    Returns:
        The root logger with the new configuration applied.
    """
    logger = logging.getLogger()  # root logger

    # ------------------------------------------------------------------
    # Distributed training: suppress logs on all processes except rank 0.
    # ------------------------------------------------------------------
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                # Completely disable logging below this level for non‑master
                # processes.  Messages at CRITICAL+1 are never emitted, so
                # this effectively silences EVERYTHING.
                logging.disable(logging.CRITICAL + 1)
                return logger
    except ImportError:
        # torch.distributed is not available; treat as single‑process.
        pass

    # Re‑enable logging (in case it was previously disabled, e.g., on rank≠0)
    logging.disable(logging.NOTSET)

    # ------------------------------------------------------------------
    # Remove existing handlers to achieve idempotent behaviour.
    # ------------------------------------------------------------------
    logger.handlers.clear()

    # ------------------------------------------------------------------
    # Build the message format:
    #   "2025-07-17 14:45:03.123 | INFO     | module:line | message"
    # ------------------------------------------------------------------
    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = _MillisecondFormatter(fmt, datefmt)

    # ------------------------------------------------------------------
    # Console handler (stdout)
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Optional file handler
    # ------------------------------------------------------------------
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set the overall logging level.
    logger.setLevel(level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given *name*, inheriting the configuration set
    by :func:`setup_logging`.

    This function is a thin wrapper around ``logging.getLogger`` and is
    provided purely for stylistic consistency within the project.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(name)

