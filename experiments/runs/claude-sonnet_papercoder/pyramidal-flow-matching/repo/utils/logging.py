## utils/logging.py
"""Logging utilities for Pyramidal Flow Matching.

Provides a consistent logging interface across all modules, with support
for TensorBoard and optional W&B metric logging. Must be implemented first
as all other modules import it.

Usage:
    from utils.logging import get_logger, log_metrics, configure_logging

    configure_logging(level="INFO", use_wandb=False)
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import os
import sys
from typing import Any, Dict, Optional, Union


## ---------------------------------------------------------------------------
## Module-level state (set by configure_logging, read by all functions)
## ---------------------------------------------------------------------------
_LOG_LEVEL: int = logging.INFO
_USE_COLOR: bool = True
_USE_WANDB: bool = False
_WANDB_PROJECT: str = "pyramidal-flow-matching"
_WANDB_INITIALIZED: bool = False


## ---------------------------------------------------------------------------
## ANSI color codes for terminal output
## ---------------------------------------------------------------------------
_ANSI_RESET: str = "\033[0m"
_ANSI_COLORS: Dict[int, str] = {
    logging.DEBUG: "\033[36m",     # Cyan
    logging.INFO: "\033[32m",      # Green
    logging.WARNING: "\033[33m",   # Yellow
    logging.ERROR: "\033[31m",     # Red
    logging.CRITICAL: "\033[35m",  # Magenta
}


class _ColorFormatter(logging.Formatter):
    """Custom log formatter that adds ANSI color codes to log level names.

    Only applies colors when the output stream supports ANSI escape codes
    (i.e., when writing to a real terminal, not a file or pipe).
    """

    _FMT: str = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    _DATE_FMT: str = "%Y-%m-%d %H:%M:%S"

    def __init__(self, use_color: bool = True) -> None:
        """Initializes the formatter.

        Args:
            use_color: Whether to apply ANSI color codes to level names.
        """
        super().__init__(fmt=self._FMT, datefmt=self._DATE_FMT)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record, optionally with color.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string, with ANSI colors if enabled.
        """
        if self._use_color:
            color = _ANSI_COLORS.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{_ANSI_RESET}"
        return super().format(record)


## ---------------------------------------------------------------------------
## Private helpers
## ---------------------------------------------------------------------------

def _get_rank() -> int:
    """Returns the current process rank for distributed training.

    Checks environment variables set by both ``torchrun`` and ``accelerate``
    launch utilities. Defaults to rank 0 (main process) if not set.

    Returns:
        Integer rank of the current process.
    """
    # torchrun sets LOCAL_RANK and RANK; accelerate also sets these
    rank_str: str = os.environ.get(
        "RANK", os.environ.get("LOCAL_RANK", "0")
    )
    try:
        return int(rank_str)
    except ValueError:
        return 0


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    """Recursively flattens a nested dict using '/' as the separator.

    Non-numeric leaf values are silently skipped.

    Args:
        d: The (possibly nested) dictionary to flatten.
        prefix: Key prefix accumulated during recursion.

    Returns:
        Flat dictionary mapping string keys to float values.

    Example:
        >>> _flatten_dict({'train': {'loss': 0.5, 'lr': 1e-4}})
        {'train/loss': 0.5, 'train/lr': 0.0001}
    """
    result: Dict[str, float] = {}
    for key, value in d.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_dict(value, prefix=full_key))
        else:
            # Attempt to convert to float; skip non-numeric values
            try:
                # Handle torch.Tensor by calling .item()
                if hasattr(value, "item"):
                    result[full_key] = float(value.item())
                else:
                    result[full_key] = float(value)
            except (TypeError, ValueError):
                pass  # Skip non-numeric values silently
    return result


def _supports_color(stream: Any) -> bool:
    """Checks whether the given stream supports ANSI color codes.

    Args:
        stream: A file-like object (e.g., sys.stdout).

    Returns:
        True if the stream is a real terminal that supports color.
    """
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False
    # On Windows, color support requires additional setup
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True


## ---------------------------------------------------------------------------
## Public API
## ---------------------------------------------------------------------------

def configure_logging(
    level: str = "INFO",
    use_color: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "pyramidal-flow-matching",
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Configures the global logging state for the entire project.

    Must be called once from ``main.py`` after loading the config file.
    Safe to call multiple times (idempotent).

    Args:
        level: Log level string, one of "DEBUG", "INFO", "WARNING", "ERROR",
            "CRITICAL". Defaults to "INFO" as per configs/default.yaml.
        use_color: Whether to use ANSI color codes in terminal output.
            Automatically disabled if stdout is not a real terminal.
        use_wandb: Whether to initialize W&B logging. Defaults to False
            as per configs/default.yaml (logging.use_wandb: false).
        wandb_project: W&B project name. Defaults to
            "pyramidal-flow-matching" as per configs/default.yaml.
        wandb_run_name: Optional W&B run name. If None, W&B auto-generates.
        wandb_config: Optional dict of hyperparameters to log to W&B.
    """
    global _LOG_LEVEL, _USE_COLOR, _USE_WANDB, _WANDB_PROJECT
    global _WANDB_INITIALIZED

    # Parse log level string to int
    numeric_level: int = getattr(logging, level.upper(), logging.INFO)
    _LOG_LEVEL = numeric_level

    # Only enable color if the terminal actually supports it
    _USE_COLOR = use_color and _supports_color(sys.stdout)

    _USE_WANDB = use_wandb
    _WANDB_PROJECT = wandb_project

    # Set root logger level so child loggers inherit it
    logging.getLogger().setLevel(_LOG_LEVEL)

    # Initialize W&B only on rank 0 and only if requested
    if use_wandb and not _WANDB_INITIALIZED and _get_rank() == 0:
        try:
            import wandb  # type: ignore[import]
            if wandb.run is None:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=wandb_config or {},
                    resume="allow",
                )
            _WANDB_INITIALIZED = True
        except ImportError:
            # W&B not installed; degrade gracefully
            _USE_WANDB = False
            _temp_logger = logging.getLogger(__name__)
            _temp_logger.warning(
                "wandb not installed. W&B logging disabled. "
                "Install with: pip install wandb"
            )
        except Exception as exc:
            _USE_WANDB = False
            _temp_logger = logging.getLogger(__name__)
            _temp_logger.warning("Failed to initialize W&B: %s", exc)


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger for the given module name.

    Handles handler deduplication (safe to call multiple times for the same
    name). On non-rank-0 processes in distributed training, INFO and DEBUG
    messages are suppressed to avoid 128x log spam.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Model initialized with %d parameters", num_params)
    """
    logger: logging.Logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(_LOG_LEVEL)
    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    # Create StreamHandler writing to stdout
    handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)

    # On non-rank-0 processes, suppress INFO/DEBUG to avoid log spam
    # from 128 GPUs printing the same messages simultaneously
    rank: int = _get_rank()
    if rank != 0:
        # Only show WARNING and above on worker ranks
        handler.setLevel(logging.WARNING)
    else:
        handler.setLevel(_LOG_LEVEL)

    # Attach color-aware formatter
    formatter: _ColorFormatter = _ColorFormatter(use_color=_USE_COLOR)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_metrics(
    metrics: Dict[str, Any],
    step: int,
    writer: Optional[Any] = None,
) -> None:
    """Logs scalar metrics to TensorBoard and optionally W&B.

    Handles nested metric dicts by flattening with '/' separator.
    Handles torch.Tensor values by calling ``.item()``.
    Silently skips non-numeric values.
    Safe to call with ``writer=None`` (skips TensorBoard, still logs W&B).

    Args:
        metrics: Dictionary of metric name -> value. May be nested.
            Example: ``{'train': {'loss': 0.42, 'lr': 1e-4}}``
        step: Global training step for the x-axis in TensorBoard/W&B.
        writer: A ``torch.utils.tensorboard.SummaryWriter`` instance, or
            ``None`` to skip TensorBoard logging.

    Example:
        >>> log_metrics({'train/loss': 0.42, 'train/lr': 1e-4}, step=100,
        ...             writer=summary_writer)
    """
    # Only log metrics on rank 0 to avoid duplicate entries
    if _get_rank() != 0:
        return

    # Flatten nested dicts and convert all values to float
    flat_metrics: Dict[str, float] = _flatten_dict(metrics)

    if not flat_metrics:
        return

    # TensorBoard logging
    if writer is not None:
        for key, value in flat_metrics.items():
            try:
                writer.add_scalar(key, value, global_step=step)
            except Exception as exc:
                _fallback_logger = get_logger(__name__)
                _fallback_logger.warning(
                    "Failed to log metric '%s' to TensorBoard: %s", key, exc
                )

    # W&B logging (optional)
    if _USE_WANDB and _WANDB_INITIALIZED:
        try:
            import wandb  # type: ignore[import]
            if wandb.run is not None:
                wandb.log(flat_metrics, step=step)
        except ImportError:
            pass  # W&B not installed; already warned in configure_logging
        except Exception as exc:
            _fallback_logger = get_logger(__name__)
            _fallback_logger.warning(
                "Failed to log metrics to W&B at step %d: %s", step, exc
            )


def build_summary_writer(
    log_dir: str,
    comment: str = "",
    flush_secs: int = 120,
) -> Optional[Any]:
    """Creates and returns a TensorBoard SummaryWriter for rank-0 process.

    Returns ``None`` on non-rank-0 processes to avoid duplicate TensorBoard
    event files from all 128 GPUs.

    Args:
        log_dir: Directory where TensorBoard event files will be written.
            Corresponds to ``paths.log_dir`` in configs/default.yaml.
        comment: Optional suffix appended to the log directory name.
        flush_secs: How often (in seconds) to flush pending events to disk.
            Defaults to 120 seconds.

    Returns:
        A ``torch.utils.tensorboard.SummaryWriter`` on rank 0, or ``None``
        on all other ranks.

    Example:
        >>> writer = build_summary_writer(config.paths.log_dir)
        >>> log_metrics({'loss': 0.5}, step=0, writer=writer)
    """
    # Only rank 0 creates a writer
    if _get_rank() != 0:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(
            log_dir=log_dir,
            comment=comment,
            flush_secs=flush_secs,
        )
        logger: logging.Logger = get_logger(__name__)
        logger.info("TensorBoard SummaryWriter initialized at: %s", log_dir)
        return writer
    except ImportError:
        logger = get_logger(__name__)
        logger.warning(
            "torch.utils.tensorboard not available. "
            "TensorBoard logging disabled. "
            "Install with: pip install tensorboard"
        )
        return None
    except Exception as exc:
        logger = get_logger(__name__)
        logger.error(
            "Failed to create TensorBoard SummaryWriter at '%s': %s",
            log_dir,
            exc,
        )
        return None


def log_hyperparameters(
    writer: Optional[Any],
    hparam_dict: Dict[str, Any],
    metric_dict: Optional[Dict[str, float]] = None,
) -> None:
    """Logs hyperparameters to TensorBoard's HParams dashboard.

    Args:
        writer: A ``torch.utils.tensorboard.SummaryWriter``, or ``None``.
        hparam_dict: Dictionary of hyperparameter name -> value.
            Values must be scalar (int, float, str, bool).
        metric_dict: Optional dictionary of final metric name -> value
            to associate with these hyperparameters.

    Example:
        >>> log_hyperparameters(writer,
        ...     {'lr': 1e-4, 'batch_size': 768, 'num_stages': 3},
        ...     {'final_fid': 12.3})
    """
    if writer is None or _get_rank() != 0:
        return

    # Sanitize hparam values: TensorBoard only accepts scalar types
    sanitized: Dict[str, Union[int, float, str, bool]] = {}
    for key, value in hparam_dict.items():
        if isinstance(value, (int, float, str, bool)):
            sanitized[key] = value
        elif hasattr(value, "item"):
            sanitized[key] = value.item()
        else:
            sanitized[key] = str(value)

    metric_dict = metric_dict or {}

    try:
        writer.add_hparams(sanitized, metric_dict)
    except Exception as exc:
        logger: logging.Logger = get_logger(__name__)
        logger.warning("Failed to log hyperparameters to TensorBoard: %s", exc)
