## utils/logging_utils.py
"""Logging utilities for OLMoE: Weights & Biases experiment tracking and console logging.

Provides two complementary logging mechanisms for the OLMoE training system:

  1. WandbLogger — structured experiment tracking via Weights & Biases.
     Reproduces the paper's training curves and metrics visible at:
     https://wandb.ai/ai2-llm/olmoe/reports/Plot-OLMoE-1B-7B--Vmlldzo4OTcyMjU3

  2. get_logger() — standard Python logging for operational console output.
     Used by all modules for startup messages, checkpoint saves, errors, etc.

Both mechanisms are safe to call from any rank in a distributed FSDP setup.
All output is silently suppressed on non-rank-0 processes via the _enabled flag
(WandbLogger) and Rank0Filter (console logger).

Configuration values used (from config.yaml):
  pretraining.wandb_project: "olmoe"
  pretraining.run_name: "olmoe-1b-7b"
  pretraining.log_every_steps: 1
  pretraining.eval_every_steps: 1000

Metrics logged per training step (from training/trainer.py):
  train/total_loss      — L = L_CE + α·L_LB + β·L_RZ (Equation 2)
  train/ce_loss         — cross-entropy component
  train/lb_loss         — load balancing loss unscaled (Section 4.1.6)
  train/router_z_loss   — router z-loss unscaled (Section 4.1.7)
  train/grad_norm       — gradient norm before clipping (Section 4.2.3, Figure 16)
  train/lr              — current learning rate from LRScheduler
  train/tokens_per_sec  — training throughput (Section 4.1.1)

Metrics logged every eval_every_steps=1000 steps:
  eval/hellaswag        — HellaSwag CF 0-shot char (Figure 3, Table 11)
  eval/mmlu_var         — MMLU Var CF 0-5 shot char (Figure 3, Table 11)
  eval/arc_challenge    — ARC-C CF 0-shot char (Table 11)
  eval/piqa             — PIQA CF 0-shot char (Table 11)
  eval/perplexity_c4    — C4 validation perplexity (Figure 24)
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Internal dependency: is_main_process() from utils/distributed.py.
# This is the ONLY internal import in this file to avoid circular imports.
# utils/distributed.py has no imports from utils/logging_utils.py.
#
# is_main_process() contract:
#   - If torch.distributed is initialized: returns rank == 0
#   - If not initialized (single process): returns True
# This ensures logging works correctly in both distributed and single-GPU modes.
# ---------------------------------------------------------------------------
from utils.distributed import DistributedUtils

# ---------------------------------------------------------------------------
# Optional wandb import.
# wandb may not be installed in all environments (e.g., minimal inference setups).
# We handle ImportError gracefully: WandbLogger sets _enabled=False and
# training continues without experiment tracking.
# ---------------------------------------------------------------------------
try:
    import wandb
    WANDB_AVAILABLE: bool = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]


class Rank0Filter(logging.Filter):
    """Logging filter that suppresses records on non-rank-0 processes.

    Added to StreamHandler instances created by get_logger() to prevent
    duplicate console output in distributed training. Without this filter,
    all 256 processes would print the same log messages, flooding the terminal.

    The filter checks is_main_process() on every record. This is safe because:
      - is_main_process() is a lightweight check (rank comparison or env var read)
      - The filter is only applied to console output, not to wandb logging
      - is_main_process() returns True when torch.distributed is not initialized,
        so single-process scripts (analysis, evaluation) work correctly

    Example:
        >>> handler = logging.StreamHandler(sys.stdout)
        >>> handler.addFilter(Rank0Filter())
        >>> # Only rank-0 process will produce output
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True (allow) only on rank-0 process.

        Args:
            record: The log record to filter. Not inspected — the decision
                    is based solely on the current process rank.

        Returns:
            True if this is the main (rank-0) process, False otherwise.
            Returning False suppresses the log record on non-rank-0 processes.
        """
        return DistributedUtils.is_main_process()


class WandbLogger:
    """Weights & Biases experiment tracker for OLMoE training.

    Wraps the wandb Python client with rank-0 gating, graceful error handling,
    and a clean interface for the training loop. Only rank-0 initializes and
    writes to wandb — all other ranks have _enabled=False and all methods
    return immediately.

    The wandb run captures all hyperparameters from config_dict, enabling
    full reproducibility of the paper's experiments. The paper's public logs
    are at: https://wandb.ai/ai2-llm/olmoe

    Attributes:
        project: Weights & Biases project name (config.yaml: pretraining.wandb_project).
        run_name: Experiment run name (config.yaml: pretraining.run_name).
        _enabled: Whether wandb logging is active. False on non-rank-0 processes,
                  or if wandb is unavailable/initialization failed.
        _run: The active wandb.Run object, or None if not enabled.

    Example:
        >>> logger = WandbLogger(
        ...     project="olmoe",
        ...     run_name="olmoe-1b-7b",
        ...     config_dict={"hidden_dim": 2048, "num_experts": 64, ...}
        ... )
        >>> logger.log({"train/loss": 2.34, "train/lr": 4e-4}, step=1000)
        >>> logger.finish()
    """

    def __init__(
        self,
        project: str = "olmoe",
        run_name: str = "olmoe-1b-7b",
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize WandbLogger.

        Initializes wandb only on rank-0. All other ranks set _enabled=False
        and return immediately without any wandb interaction.

        The config_dict should contain the full merged configuration (model +
        training hyperparameters) so the wandb run captures all settings for
        reproducibility. Typical contents from config.yaml:
            - model: hidden_dim=2048, num_experts=64, top_k=8, lb_loss_weight=0.01, ...
            - training: learning_rate=4e-4, adam_eps=1e-8, warmup_steps=2500, ...

        Args:
            project: Weights & Biases project name.
                     Default: "olmoe" (config.yaml: pretraining.wandb_project).
            run_name: Experiment run name displayed in the wandb UI.
                      Default: "olmoe-1b-7b" (config.yaml: pretraining.run_name).
            config_dict: Dictionary of hyperparameters to log to wandb.
                         If None, an empty dict is used. Should contain all
                         model and training configuration for reproducibility.
                         Values must be JSON-serializable (no tensors).
        """
        self.project: str = project
        self.run_name: str = run_name
        self._enabled: bool = False
        self._run: Optional[Any] = None  # wandb.Run or None

        # -----------------------------------------------------------------------
        # Rank-0 gate: only the main process initializes wandb.
        # All other ranks return immediately with _enabled=False.
        # This prevents 256 wandb runs from being created during pretraining.
        # -----------------------------------------------------------------------
        if not DistributedUtils.is_main_process():
            return

        # -----------------------------------------------------------------------
        # Check wandb availability.
        # wandb may not be installed in minimal environments.
        # -----------------------------------------------------------------------
        if not WANDB_AVAILABLE:
            _get_internal_logger().warning(
                "wandb is not installed. Experiment tracking disabled. "
                "Install with: pip install wandb"
            )
            return

        # -----------------------------------------------------------------------
        # Initialize wandb run.
        # Wrap in try/except to handle:
        #   - Missing API key (wandb.errors.UsageError)
        #   - Network connectivity issues
        #   - Any other wandb initialization errors
        # Training must never fail due to logging issues.
        # -----------------------------------------------------------------------
        effective_config: Dict[str, Any] = config_dict if config_dict is not None else {}

        try:
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=effective_config,
                # resume="allow" enables resuming a run from a checkpoint.
                # The run_id would need to be stored in the checkpoint for
                # true resume support; for now we create a new run on each start.
                resume="allow",
            )
            self._enabled = True
            _get_internal_logger().info(
                f"WandbLogger initialized: project='{project}', "
                f"run_name='{run_name}', "
                f"run_id='{self._run.id if self._run else 'unknown'}'"
            )
        except Exception as exc:
            # Catch all exceptions from wandb.init to prevent training crashes.
            # Common causes: missing API key, network issues, invalid project name.
            _get_internal_logger().warning(
                f"Failed to initialize wandb (project='{project}', "
                f"run_name='{run_name}'): {type(exc).__name__}: {exc}. "
                f"Experiment tracking disabled. Training will continue without wandb."
            )
            self._enabled = False
            self._run = None

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Log scalar metrics to wandb at the given training step.

        Called every training step (config.yaml: pretraining.log_every_steps: 1)
        with training metrics, and every eval_every_steps=1000 steps with
        additional evaluation metrics.

        All metric values must be Python scalars (int or float), not tensors.
        The caller (training/trainer.py) is responsible for calling .item()
        on any tensor values before passing them here.

        Args:
            metrics: Dictionary mapping metric names to scalar values.
                     Keys should use "/" for namespacing (e.g., "train/loss",
                     "eval/hellaswag") to organize metrics in the wandb UI.
                     Common keys logged during pretraining:
                       "train/total_loss"    — L = L_CE + α·L_LB + β·L_RZ
                       "train/ce_loss"       — cross-entropy component
                       "train/lb_loss"       — load balancing loss (unscaled)
                       "train/router_z_loss" — router z-loss (unscaled)
                       "train/grad_norm"     — gradient norm before clipping
                       "train/lr"            — current learning rate
                       "train/tokens_per_sec" — training throughput
            step: Global training step (0-indexed). Used as the x-axis in
                  wandb plots. Corresponds to the number of optimizer steps
                  completed, not the number of tokens processed.

        Returns:
            None. Silently no-ops if _enabled is False.
        """
        if not self._enabled:
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as exc:
            # Log warning but don't crash training.
            _get_internal_logger().warning(
                f"wandb.log failed at step {step}: "
                f"{type(exc).__name__}: {exc}"
            )

    def log_table(self, name: str, data: List[Dict[str, Any]]) -> None:
        """Log a table of results to wandb.

        Used for logging evaluation results in tabular form, such as:
          - OLMES evaluation results (Table 4 in the paper)
          - Adaptation evaluation results (Table 5 in the paper)
          - Router saturation analysis results (Figure 20)
          - Domain specialization results (Figure 22)

        The table columns are inferred from the keys of the first dict in data.
        All dicts in data should have the same keys.

        Args:
            name: Table name as it will appear in the wandb UI.
                  Examples: "eval/olmes_results", "analysis/domain_specialization".
            data: List of row dictionaries. Each dict maps column names to values.
                  Values can be strings, numbers, or other wandb-compatible types.
                  Example:
                    [
                        {"task": "hellaswag", "accuracy": 80.0, "shots": 5},
                        {"task": "mmlu", "accuracy": 54.1, "shots": 5},
                    ]
                  Empty list is handled gracefully (no-op).

        Returns:
            None. Silently no-ops if _enabled is False or data is empty.
        """
        if not self._enabled:
            return

        # Guard against empty data — wandb.Table requires at least one row
        # to infer column names.
        if not data:
            _get_internal_logger().debug(
                f"log_table called with empty data for '{name}'. Skipping."
            )
            return

        try:
            # Infer column names from the keys of the first row.
            # All rows are assumed to have the same keys.
            columns: List[str] = list(data[0].keys())

            # Build the wandb Table by populating rows from the data list.
            table: Any = wandb.Table(columns=columns)
            for row_dict in data:
                # Extract values in the same order as columns.
                row_values: List[Any] = [row_dict.get(col) for col in columns]
                table.add_data(*row_values)

            # Log the table to wandb. No step parameter — tables are logged
            # at the current wandb step (most recent step from log() calls).
            wandb.log({name: table})

            _get_internal_logger().debug(
                f"Logged wandb table '{name}' with "
                f"{len(data)} rows, {len(columns)} columns."
            )
        except Exception as exc:
            _get_internal_logger().warning(
                f"wandb table logging failed for '{name}': "
                f"{type(exc).__name__}: {exc}"
            )

    def finish(self) -> None:
        """Finalize and close the wandb run.

        Properly closes the wandb run, uploading any remaining buffered data
        and marking the run as finished in the wandb UI. Should be called at
        the end of training in main.py, ideally in a finally block to ensure
        it runs even if training fails.

        Example:
            >>> try:
            ...     trainer.train()
            ... finally:
            ...     wandb_logger.finish()

        Returns:
            None. Silently no-ops if _enabled is False.
        """
        if not self._enabled:
            return

        try:
            wandb.finish()
            _get_internal_logger().info(
                f"WandbLogger finished: project='{self.project}', "
                f"run_name='{self.run_name}'"
            )
        except Exception as exc:
            _get_internal_logger().warning(
                f"wandb.finish() failed: {type(exc).__name__}: {exc}"
            )
        finally:
            # Mark as disabled after finish to prevent further logging attempts.
            self._enabled = False
            self._run = None

    @property
    def enabled(self) -> bool:
        """Whether wandb logging is currently active.

        Returns:
            True if wandb was successfully initialized and is active.
            False on non-rank-0 processes, if wandb is unavailable, or
            after finish() has been called.
        """
        return self._enabled

    def __repr__(self) -> str:
        """Return string representation of the WandbLogger.

        Returns:
            Human-readable string showing project, run_name, and enabled state.
        """
        return (
            f"WandbLogger("
            f"project='{self.project}', "
            f"run_name='{self.run_name}', "
            f"enabled={self._enabled}"
            f")"
        )


def get_logger(name: str = "olmoe") -> logging.Logger:
    """Get or create a named logger with rank-0 filtering and stdout output.

    Returns a standard Python logger configured for OLMoE training:
      - Output to stdout (not stderr) for clean log capture
      - Timestamp + name + level + message format
      - INFO level by default
      - Rank-0 filter: suppresses output on non-rank-0 processes in
        distributed training to prevent duplicate console output

    The logger is idempotent: calling get_logger("olmoe.trainer") multiple
    times returns the same logger instance without adding duplicate handlers.
    This is guaranteed by Python's logging module which maintains a global
    registry of named loggers.

    Naming convention used across the codebase:
        "olmoe"              — root OLMoE logger (fallback)
        "olmoe.trainer"      — training/trainer.py
        "olmoe.evaluator"    — evaluation/evaluator.py
        "olmoe.checkpoint"   — utils/checkpoint.py
        "olmoe.data"         — data/dataset_loader.py
        "olmoe.analysis"     — analysis/*.py modules
        "olmoe.sft"          — adaptation/sft_trainer.py
        "olmoe.dpo"          — adaptation/dpo_trainer.py

    Args:
        name: Logger name. Use hierarchical dot notation for module-specific
              loggers (e.g., "olmoe.trainer"). The root "olmoe" logger is
              the parent of all "olmoe.*" loggers.
              Default: "olmoe" (root OLMoE logger).

    Returns:
        Configured logging.Logger instance. Safe to use immediately after
        this call, even before torch.distributed is initialized.

    Example:
        >>> logger = get_logger("olmoe.trainer")
        >>> logger.info("Starting training for 1,223,958 steps")
        2024-09-15 10:23:45,123 - olmoe.trainer - INFO - Starting training for 1,223,958 steps
        >>> logger.warning("Gradient norm spike detected: 12.3")
        2024-09-15 10:23:46,456 - olmoe.trainer - WARNING - Gradient norm spike detected: 12.3
    """
    logger: logging.Logger = logging.getLogger(name)

    # -----------------------------------------------------------------------
    # Idempotency guard: only configure the logger if it has no handlers yet.
    # logging.getLogger() returns the same instance for the same name, so
    # calling get_logger("olmoe.trainer") multiple times would add duplicate
    # handlers without this check.
    # -----------------------------------------------------------------------
    if not logger.handlers:
        # Set the default log level to INFO.
        # Individual modules can override this via logger.setLevel(logging.DEBUG)
        # for more verbose output during debugging.
        logger.setLevel(logging.INFO)

        # -----------------------------------------------------------------------
        # Create a StreamHandler writing to stdout.
        # stdout is preferred over stderr for training logs because:
        #   - Training logs are informational, not errors
        #   - stdout is typically captured by job schedulers (SLURM, etc.)
        #   - Easier to pipe to files: python train.py > training.log
        # -----------------------------------------------------------------------
        handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)  # Handler accepts all levels; logger filters

        # -----------------------------------------------------------------------
        # Configure log format.
        # Format: "2024-09-15 10:23:45,123 - olmoe.trainer - INFO - message"
        # Components:
        #   %(asctime)s    — timestamp with milliseconds
        #   %(name)s       — logger name (e.g., "olmoe.trainer")
        #   %(levelname)s  — log level (INFO, WARNING, ERROR, etc.)
        #   %(message)s    — the actual log message
        # -----------------------------------------------------------------------
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        # -----------------------------------------------------------------------
        # Add Rank0Filter to suppress output on non-rank-0 processes.
        # The filter is added to the HANDLER (not the logger) so that:
        #   1. The logger object itself is always functional (can be used for
        #      in-memory log capture, testing, etc.)
        #   2. Only the console output is suppressed on non-rank-0 processes
        #   3. The filter is applied per-record, allowing dynamic rank changes
        #      (though in practice rank is fixed for the lifetime of a process)
        # -----------------------------------------------------------------------
        rank0_filter: Rank0Filter = Rank0Filter()
        handler.addFilter(rank0_filter)

        # Add the configured handler to the logger.
        logger.addHandler(handler)

        # -----------------------------------------------------------------------
        # Prevent log records from propagating to the root logger.
        # Without this, records would be handled twice: once by our handler
        # and once by the root logger's default handler (if configured).
        # Setting propagate=False ensures clean, non-duplicate output.
        # -----------------------------------------------------------------------
        logger.propagate = False

    return logger


def _get_internal_logger() -> logging.Logger:
    """Get the internal logger for utils/logging_utils.py itself.

    Used within this module to log warnings about wandb initialization failures,
    table logging errors, etc. Separate from the public get_logger() to avoid
    any potential recursion or initialization ordering issues.

    Returns:
        A logging.Logger named "olmoe.logging_utils" configured for rank-0 output.
    """
    return get_logger("olmoe.logging_utils")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure the root OLMoE logger with optional file output.

    Sets up the "olmoe" root logger with the specified log level and
    optionally adds a file handler for persistent log storage. This is
    a convenience function for main.py to call once at startup.

    The file handler (if configured) writes to all ranks' logs to separate
    files named "rank_{rank}.log" to avoid file write conflicts. Only the
    rank-0 file is typically useful for debugging, but having all ranks'
    logs can help diagnose distributed training issues.

    Args:
        log_level: Logging level string. One of: "DEBUG", "INFO", "WARNING",
                   "ERROR", "CRITICAL". Default: "INFO".
                   Use "DEBUG" for verbose output during development.
        log_file: Optional path to a log file. If provided, logs are written
                  to "{log_file}.rank_{rank}" for each process. If None,
                  only console output is used. Default: None.

    Example:
        >>> # In main.py, called once at startup:
        >>> setup_logging(log_level="INFO", log_file="outputs/training.log")
        >>> logger = get_logger("olmoe.main")
        >>> logger.info("OLMoE training started")
    """
    # Map string level to logging constant.
    level_map: Dict[str, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    numeric_level: int = level_map.get(log_level.upper(), logging.INFO)

    # Configure the root "olmoe" logger.
    root_olmoe_logger: logging.Logger = get_logger("olmoe")
    root_olmoe_logger.setLevel(numeric_level)

    # Update all existing handlers to use the new level.
    for handler in root_olmoe_logger.handlers:
        handler.setLevel(logging.DEBUG)  # Handler accepts all; logger filters by level

    # -----------------------------------------------------------------------
    # Optional file handler for persistent log storage.
    # Each rank writes to a separate file to avoid concurrent write conflicts.
    # -----------------------------------------------------------------------
    if log_file is not None:
        rank: int = DistributedUtils.get_rank()
        rank_log_file: str = f"{log_file}.rank_{rank}"

        try:
            # Ensure the directory exists.
            log_dir: str = os.path.dirname(rank_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler: logging.FileHandler = logging.FileHandler(
                rank_log_file,
                mode="a",  # Append mode: don't overwrite existing logs on resume
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)  # File captures all levels

            # Use the same format as the console handler.
            file_formatter: logging.Formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)

            # File handler does NOT have Rank0Filter — each rank writes its own file.
            root_olmoe_logger.addHandler(file_handler)

            # Log the file handler setup (only on rank 0 via console).
            _get_internal_logger().info(
                f"File logging enabled: {rank_log_file} "
                f"(rank={rank}, level={log_level})"
            )
        except OSError as exc:
            _get_internal_logger().warning(
                f"Failed to create log file '{rank_log_file}': "
                f"{type(exc).__name__}: {exc}. "
                f"Continuing with console logging only."
            )

    _get_internal_logger().info(
        f"Logging configured: level={log_level}, "
        f"file={'enabled' if log_file else 'disabled'}, "
        f"rank={DistributedUtils.get_rank()}, "
        f"world_size={DistributedUtils.get_world_size()}"
    )
