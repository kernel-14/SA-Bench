## utils/logger.py
"""Centralized logging utility for the adversarial leaderboard manipulation project.

This module provides a single public function, get_logger(), that returns a
configured logging.Logger writing to both the console and a persistent log file.
It has zero internal project dependencies and must never import from any other
project module.

Usage in any other module:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting experiment for model %s", model_name)
"""

from __future__ import annotations

import logging
import os
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level constants mirroring config.yaml logging section.
# These are fixed conventions, not runtime variables, so hardcoding them here
# is intentional — utils/logger.py cannot import config.py.
# ---------------------------------------------------------------------------

# Path from config.yaml: logging.log_file
LOG_FILE: str = "outputs/experiment.log"

# Format from config.yaml: logging.format
LOG_FORMAT: str = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

# Level from config.yaml: logging.level
LOG_LEVEL: int = logging.INFO


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger configured with console and file handlers.

    The logger writes to both sys.stderr (console) and the file at
    LOG_FILE ("outputs/experiment.log") in append mode with UTF-8 encoding.
    Calling this function multiple times with the same name is safe — handlers
    are only attached once, preventing duplicate log lines.

    Args:
        name: Logger name, typically __name__ from the calling module.
            If None or empty, defaults to "root" to avoid accidentally
            configuring the global root logger.

    Returns:
        A logging.Logger instance with INFO-level console and file handlers
        attached, and propagation disabled to prevent interference with
        third-party library loggers.

    Raises:
        OSError: If the log file directory cannot be created or the log file
            cannot be opened for writing (e.g., permission denied). This is
            a configuration error that should surface immediately.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Collecting responses for model %s", "gpt-4o")
        [2024-01-01 12:00:00,000] INFO my_module: Collecting responses for model gpt-4o
    """
    # Guard against empty or None name to avoid touching the root logger.
    effective_name: str = name if name else "root"

    logger: logging.Logger = logging.getLogger(effective_name)

    # Set the logger's own level. This must be <= handler levels for messages
    # to reach the handlers at all.
    logger.setLevel(LOG_LEVEL)

    # Prevent messages from propagating to the root logger, which may have its
    # own handlers attached by third-party libraries (openai, anthropic, etc.)
    # that would produce duplicate or noisy output.
    logger.propagate = False

    # Only attach handlers if none exist yet. This is the critical guard that
    # prevents duplicate log lines when get_logger() is called multiple times
    # for the same name (e.g., during testing or module re-imports).
    if not logger.handlers:
        formatter: logging.Formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # --- Console handler ---
        # Writes to sys.stderr so that log output does not mix with stdout
        # data (e.g., if any module ever prints structured output to stdout).
        console_handler: logging.StreamHandler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # --- File handler ---
        # Ensure the parent directory of LOG_FILE exists before opening.
        # This mirrors the directory creation in Config.__post_init__ but is
        # necessary here because logger.py has no dependency on config.py.
        log_dir: str = os.path.dirname(LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Append mode ('a') accumulates logs across multiple runs.
        # UTF-8 encoding is required because model responses may contain
        # non-ASCII characters (Chinese, Persian, Arabic, etc.).
        file_handler: logging.FileHandler = logging.FileHandler(
            filename=LOG_FILE,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
