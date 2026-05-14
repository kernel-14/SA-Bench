## utils/logging_utils.py
"""Logging utilities for SCoRe: Self-Correction via Reinforcement Learning.

This module implements LoggingUtils, the unified experiment tracking class
for the SCoRe training pipeline. It provides dual-channel logging:
    1. Python standard logging (console + file) — always active.
    2. Weights & Biases (wandb) — active when available and on rank 0.

All wandb calls are wrapped in try/except blocks so logging failures never
interrupt training. In distributed training (DeepSpeed), wandb is only
initialized on rank 0 to prevent duplicate runs.

The log_trajectories() method supports qualitative analysis of self-correction
behavior by logging correction type counts (i→c, c→i, i→i, c→c) in real time,
directly corresponding to the paper's Δ^{i→c} and Δ^{c→i} metrics.

Config fields used (from config.yaml, flattened into Config dataclass):
    config.wandb_project  → experiment.wandb_project
    config.run_name       → experiment.run_name
    config.output_dir     → experiment.output_dir
    config.log_level      → experiment.log_level (e.g., "INFO")

Typical usage:
    from config import Config
    from utils.logging_utils import LoggingUtils

    logger_util = LoggingUtils(config)
    logger_util.log_metrics({'loss': 0.42, 'mean_reward_t2': 0.61}, step=100)
    logger_util.log_trajectories(trajectories, step=100)
    logger_util.finish()
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from config import Config

if TYPE_CHECKING:
    # Guard against circular imports:
    # rollout_buffer.py → model_wrapper.py → config.py
    # We only need Trajectory for type annotations, not at runtime.
    from training.rollout_buffer import Trajectory

# ---------------------------------------------------------------------------
# Optional wandb import — guarded so the module can be imported even if
# wandb is not installed (tests, linting, CI without full deps).
# ---------------------------------------------------------------------------
try:
    import wandb as _wandb_module

    _WANDB_AVAILABLE: bool = True
except ImportError:
    _WANDB_MODULE = None
    _WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Maximum number of trajectories to log per log_trajectories() call.
# Prevents excessive wandb storage for large batches.
_MAX_TRAJECTORIES_TO_LOG: int = 5

# Maximum character length for problem text in wandb Table rows.
_MAX_PROBLEM_CHARS: int = 500

# Maximum character length for response text in wandb Table rows.
_MAX_RESPONSE_CHARS: int = 1000

# Threshold for classifying binary rewards as correct (consistent with Metrics).
_CORRECT_THRESHOLD: float = 0.5

# Log format string for console and file handlers.
_LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

# Date format for log timestamps.
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Name of the root logger for the SCoRe project.
_LOGGER_NAME: str = "score"

# Name of the training log file written to config.output_dir.
_LOG_FILENAME: str = "training.log"


class LoggingUtils:
    """Unified experiment tracking for the SCoRe training pipeline.

    Provides dual-channel logging:
        - Python standard logging (console + file): always active on all ranks.
        - Weights & Biases: active only on rank 0 and when wandb is available.

    All wandb calls are wrapped in try/except blocks — logging failures
    never interrupt training. This is critical in distributed training
    environments where wandb may only be initialized on rank 0.

    Attributes:
        _config: The global Config instance. Stored for access in methods.
        _wandb_run: The active wandb run object, or None if wandb is
            disabled (not installed, initialization failed, or non-rank-0).
        _logger: The named Python logger instance ("score").
        _log_dir: Resolved output directory for file logging and checkpoints.
        _is_rank_zero: Whether this process is rank 0 in distributed training.
            Determined from the LOCAL_RANK environment variable.
    """

    def __init__(self, config: Config) -> None:
        """Initialize LoggingUtils.

        Sets up Python logging (console + file) and initializes the wandb
        run. In distributed training, wandb is only initialized on rank 0.
        All wandb initialization failures are caught and logged as warnings —
        training continues without wandb if initialization fails.

        Args:
            config: The global Config instance. Reads:
                config.wandb_project (str): W&B project name.
                config.run_name (str): Human-readable run name.
                config.output_dir (str): Directory for log files and checkpoints.
                config.log_level (str): Python logging level (e.g., "INFO").
        """
        self._config: Config = config
        self._wandb_run: Optional[Any] = None  # wandb.sdk.wandb_run.Run | None
        self._log_dir: str = config.output_dir

        # ------------------------------------------------------------------
        # Step 1: Determine distributed training rank.
        # In DeepSpeed / torch.distributed, LOCAL_RANK is set by the launcher.
        # Default to 0 (rank 0) if not in a distributed context.
        # ------------------------------------------------------------------
        local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
        self._is_rank_zero: bool = local_rank == 0

        # ------------------------------------------------------------------
        # Step 2: Create output directory for log files.
        # ------------------------------------------------------------------
        os.makedirs(self._log_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Step 3: Set up Python logging (active on ALL ranks for debugging).
        # ------------------------------------------------------------------
        self._logger: logging.Logger = self._setup_python_logging(config)

        # ------------------------------------------------------------------
        # Step 4: Initialize wandb (rank 0 only).
        # ------------------------------------------------------------------
        if self._is_rank_zero:
            self._init_wandb(config)
        else:
            self._logger.debug(
                "LoggingUtils: rank %d (non-zero). "
                "Skipping wandb initialization to prevent duplicate runs.",
                local_rank,
            )

        # ------------------------------------------------------------------
        # Step 5: Log initialization summary.
        # ------------------------------------------------------------------
        self._logger.info(
            "LoggingUtils initialized. "
            "Output dir: '%s', W&B project: '%s', run_name: '%s', "
            "rank: %d, wandb_active: %s.",
            self._log_dir,
            config.wandb_project,
            config.run_name,
            local_rank,
            self._wandb_run is not None,
        )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log a dict of metrics to both wandb and the Python logger.

        Logs all key-value pairs in metrics to wandb (if active) and formats
        a human-readable summary line for the Python logger. Non-float values
        are formatted with str() rather than the float format spec.

        This method is called at every training step by SCoReStage1Trainer,
        SCoReStage2Trainer, and REINFORCETrainer, and once per evaluation
        by Evaluator. The step parameter ensures consistent x-axis alignment
        across all logged metrics in wandb.

        Args:
            metrics: Dict mapping metric names to values. Expected keys vary
                by caller:
                    Stage I training: 'loss', 'mean_reward_t2', 'mean_kl_t1',
                        'mean_kl_t2', 'delta_t1_t2', 'fraction_answer_changed'.
                    Stage II training: 'loss', 'mean_reward_t1', 'mean_reward_t2',
                        'mean_shaped_reward_t2', 'delta_t1_t2', 'mean_kl_t1',
                        'mean_kl_t2'.
                    Evaluation: 'accuracy_t1', 'accuracy_t2', 'delta_t1_t2',
                        'i2c_rate', 'c2i_rate'.
                All keys are logged without filtering — callers control content.
            step: Global training step counter. Used as the x-axis value in
                wandb plots and as a prefix in the Python log message.

        Returns:
            None. This method is a no-op (silent) if both wandb and the
            Python logger are unavailable — never raises exceptions.
        """
        if not metrics:
            return

        # ------------------------------------------------------------------
        # Log to wandb (rank 0 only, wrapped in try/except)
        # ------------------------------------------------------------------
        if self._wandb_run is not None and _WANDB_AVAILABLE:
            try:
                _wandb_module.log(metrics, step=step)
            except Exception as exc:
                self._logger.warning(
                    "W&B logging failed at step %d: %s. "
                    "Continuing without W&B for this step.",
                    step,
                    exc,
                )

        # ------------------------------------------------------------------
        # Format and emit a human-readable summary via Python logger.
        # Float values use 4 decimal places; non-float values use str().
        # ------------------------------------------------------------------
        metric_parts: List[str] = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metric_parts.append(f"{key}: {value:.4f}")
            else:
                metric_parts.append(f"{key}: {value!s}")

        summary_line: str = f"Step {step} | " + " | ".join(metric_parts)
        self._logger.info(summary_line)

    def log_trajectories(
        self,
        trajectories: List["Trajectory"],
        step: int,
    ) -> None:
        """Log a sample of trajectories for qualitative self-correction analysis.

        Supports the qualitative analysis described in Appendix D/E of the
        paper by logging examples of i→c, c→i, i→i, and c→c transitions.
        Also logs aggregate correction type counts as scalar metrics, providing
        a real-time view of behavior collapse during training.

        If wandb is not active, falls back to logging a text summary via
        the Python logger at DEBUG level.

        The correction type classification:
            i→c: reward_t1=0, reward_t2=1 — successful self-correction
            c→i: reward_t1=1, reward_t2=0 — harmful change (behavior collapse)
            i→i: reward_t1=0, reward_t2=0 — failed correction
            c→c: reward_t1=1, reward_t2=1 — maintained correctness

        Args:
            trajectories: List of Trajectory objects from
                RolloutBuffer.sample_trajectories(). Each trajectory must
                have non-None turn1_response, turn2_response, reward_t1,
                reward_t2 fields. The shaped_reward_t2 field may be 0.0
                if compute_shaped_rewards() has not yet been called (Stage I).
            step: Global training step counter for wandb x-axis alignment.

        Returns:
            None. Never raises exceptions — all failures are caught and
            logged as warnings.
        """
        if not trajectories:
            self._logger.debug(
                "log_trajectories: Empty trajectories list at step %d. "
                "Skipping.",
                step,
            )
            return

        # ------------------------------------------------------------------
        # Compute correction type counts for ALL trajectories (not just the
        # logged subset). These aggregate counts are the primary signal for
        # monitoring behavior collapse in real time.
        # ------------------------------------------------------------------
        count_i2c: int = 0
        count_c2i: int = 0
        count_ii: int = 0
        count_cc: int = 0

        for traj in trajectories:
            r1_correct: bool = float(traj.reward_t1) >= _CORRECT_THRESHOLD
            r2_correct: bool = float(traj.reward_t2) >= _CORRECT_THRESHOLD

            if not r1_correct and r2_correct:
                count_i2c += 1
            elif r1_correct and not r2_correct:
                count_c2i += 1
            elif not r1_correct and not r2_correct:
                count_ii += 1
            else:
                count_cc += 1

        n: int = len(trajectories)

        # ------------------------------------------------------------------
        # Fallback: Python logger debug output when wandb is not active.
        # ------------------------------------------------------------------
        if self._wandb_run is None or not _WANDB_AVAILABLE:
            self._logger.debug(
                "log_trajectories (step=%d, n=%d): "
                "i→c=%d (%.1f%%), c→i=%d (%.1f%%), "
                "i→i=%d (%.1f%%), c→c=%d (%.1f%%).",
                step,
                n,
                count_i2c,
                100.0 * count_i2c / n,
                count_c2i,
                100.0 * count_c2i / n,
                count_ii,
                100.0 * count_ii / n,
                count_cc,
                100.0 * count_cc / n,
            )
            return

        # ------------------------------------------------------------------
        # Log aggregate correction type counts as scalar metrics to wandb.
        # These directly correspond to the paper's Δ^{i→c} and Δ^{c→i}
        # metrics, providing real-time monitoring of self-correction behavior.
        # ------------------------------------------------------------------
        correction_count_metrics: Dict[str, Any] = {
            "correction_type/i2c_count": count_i2c,
            "correction_type/c2i_count": count_c2i,
            "correction_type/ii_count": count_ii,
            "correction_type/cc_count": count_cc,
            "correction_type/i2c_rate": count_i2c / n if n > 0 else 0.0,
            "correction_type/c2i_rate": count_c2i / n if n > 0 else 0.0,
        }

        try:
            _wandb_module.log(correction_count_metrics, step=step)
        except Exception as exc:
            self._logger.warning(
                "W&B correction count logging failed at step %d: %s.",
                step,
                exc,
            )

        # ------------------------------------------------------------------
        # Build a wandb.Table with a sample of trajectories for qualitative
        # analysis. Select the first min(5, n) trajectories.
        # ------------------------------------------------------------------
        num_to_log: int = min(_MAX_TRAJECTORIES_TO_LOG, n)
        sampled_trajectories: List["Trajectory"] = trajectories[:num_to_log]

        try:
            table = _wandb_module.Table(
                columns=[
                    "step",
                    "problem",
                    "turn1_response",
                    "turn2_response",
                    "reward_t1",
                    "reward_t2",
                    "shaped_reward_t2",
                    "correction_type",
                ]
            )

            for traj in sampled_trajectories:
                # Determine correction type label
                r1_correct_traj: bool = (
                    float(traj.reward_t1) >= _CORRECT_THRESHOLD
                )
                r2_correct_traj: bool = (
                    float(traj.reward_t2) >= _CORRECT_THRESHOLD
                )

                if not r1_correct_traj and r2_correct_traj:
                    correction_type: str = "i→c"
                elif r1_correct_traj and not r2_correct_traj:
                    correction_type = "c→i"
                elif not r1_correct_traj and not r2_correct_traj:
                    correction_type = "i→i"
                else:
                    correction_type = "c→c"

                # Safely access shaped_reward_t2 — may be 0.0 in Stage I
                # before compute_shaped_rewards() is called.
                shaped_r2: float = 0.0
                try:
                    shaped_r2_raw = traj.shaped_reward_t2
                    if shaped_r2_raw is not None:
                        shaped_r2 = float(shaped_r2_raw)
                except (AttributeError, TypeError, ValueError):
                    shaped_r2 = 0.0

                # Truncate text fields to prevent wandb storage limits
                problem_text: str = str(traj.problem or "")[:_MAX_PROBLEM_CHARS]
                t1_response: str = str(traj.turn1_response or "")[
                    :_MAX_RESPONSE_CHARS
                ]
                t2_response: str = str(traj.turn2_response or "")[
                    :_MAX_RESPONSE_CHARS
                ]

                table.add_data(
                    step,
                    problem_text,
                    t1_response,
                    t2_response,
                    float(traj.reward_t1),
                    float(traj.reward_t2),
                    shaped_r2,
                    correction_type,
                )

            _wandb_module.log({"trajectories": table}, step=step)

            self._logger.debug(
                "log_trajectories: Logged %d/%d trajectories to W&B at "
                "step %d. Correction types — i→c: %d, c→i: %d, "
                "i→i: %d, c→c: %d.",
                num_to_log,
                n,
                step,
                count_i2c,
                count_c2i,
                count_ii,
                count_cc,
            )

        except Exception as exc:
            self._logger.warning(
                "W&B trajectory table logging failed at step %d: %s. "
                "Continuing without trajectory logging for this step.",
                step,
                exc,
            )

    def finish(self) -> None:
        """Close the wandb run and flush all logging handlers.

        Properly finalizes the experiment run by:
            1. Calling wandb.finish() to sync remaining data and close the run.
            2. Logging a completion message.
            3. Flushing all Python logging handlers.

        This method is idempotent — calling it multiple times is safe.
        After the first call, _wandb_run is set to None to prevent
        double-finish errors.

        Returns:
            None. Never raises exceptions.
        """
        # ------------------------------------------------------------------
        # Step 1: Finalize wandb run (rank 0 only, idempotent guard).
        # ------------------------------------------------------------------
        if self._wandb_run is not None and _WANDB_AVAILABLE:
            try:
                _wandb_module.finish()
                self._logger.info(
                    "LoggingUtils.finish(): W&B run '%s' finished successfully.",
                    self._config.run_name,
                )
            except Exception as exc:
                self._logger.warning(
                    "LoggingUtils.finish(): W&B finish() raised an exception: %s. "
                    "The run may not have been properly closed.",
                    exc,
                )
            finally:
                # Set to None to prevent double-finish on subsequent calls
                self._wandb_run = None
        else:
            self._logger.info(
                "LoggingUtils.finish(): W&B was not active. "
                "No W&B run to close."
            )

        # ------------------------------------------------------------------
        # Step 2: Log final completion message.
        # ------------------------------------------------------------------
        self._logger.info(
            "Training complete. LoggingUtils.finish() called. "
            "All metrics have been logged."
        )

        # ------------------------------------------------------------------
        # Step 3: Flush all Python logging handlers to ensure all buffered
        # log records are written to disk before the process exits.
        # ------------------------------------------------------------------
        for handler in self._logger.handlers:
            try:
                handler.flush()
            except Exception as exc:
                # Cannot log this failure since the logger itself may be
                # in a bad state — print to stderr as last resort.
                import sys
                print(
                    f"LoggingUtils.finish(): Failed to flush handler "
                    f"{handler}: {exc}",
                    file=sys.stderr,
                )

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _setup_python_logging(self, config: Config) -> logging.Logger:
        """Configure and return the named Python logger for the SCoRe project.

        Creates a logger named "score" with:
            - A StreamHandler for console output.
            - A FileHandler writing to config.output_dir/training.log.

        Avoids adding duplicate handlers if __init__ is called multiple times
        (e.g., in unit tests) by checking logger.handlers before adding.

        Args:
            config: The global Config instance. Reads config.log_level and
                config.output_dir.

        Returns:
            The configured logging.Logger instance.
        """
        logger: logging.Logger = logging.getLogger(_LOGGER_NAME)

        # Resolve the numeric log level from the string (e.g., "INFO" → 20)
        numeric_level: int = getattr(logging, config.log_level.upper(), logging.INFO)
        logger.setLevel(numeric_level)

        # Create the shared formatter
        formatter: logging.Formatter = logging.Formatter(
            fmt=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
        )

        # ------------------------------------------------------------------
        # Add StreamHandler (console) if not already present.
        # Check by handler type to avoid duplicates across multiple __init__
        # calls (common in unit tests).
        # ------------------------------------------------------------------
        has_stream_handler: bool = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )
        if not has_stream_handler:
            stream_handler: logging.StreamHandler = logging.StreamHandler()
            stream_handler.setLevel(numeric_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        # ------------------------------------------------------------------
        # Add FileHandler (training.log) if not already present.
        # Check by handler type and filename to avoid duplicates.
        # ------------------------------------------------------------------
        log_file_path: str = os.path.join(self._log_dir, _LOG_FILENAME)
        has_file_handler: bool = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == os.path.abspath(log_file_path)
            for h in logger.handlers
        )
        if not has_file_handler:
            try:
                file_handler: logging.FileHandler = logging.FileHandler(
                    log_file_path,
                    mode="a",  # Append mode — preserves logs across restarts
                    encoding="utf-8",
                )
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except OSError as exc:
                # Cannot write to log file — log to console only
                logger.warning(
                    "_setup_python_logging: Could not create file handler "
                    "at '%s': %s. Logging to console only.",
                    log_file_path,
                    exc,
                )

        # Prevent log records from propagating to the root logger to avoid
        # duplicate output when the root logger also has handlers configured.
        logger.propagate = False

        return logger

    def _init_wandb(self, config: Config) -> None:
        """Initialize the wandb run on rank 0.

        Calls wandb.init() with the project name, run name, and full config
        dict (via config.to_dict()). The config dict serializes all
        hyperparameters (α=10, β₁=0.01, β₂=0.1/0.25, learning rates, etc.)
        so they appear in the wandb run config panel for reproducibility.

        All failures are caught and logged as warnings — training continues
        without wandb if initialization fails (e.g., no API key, offline mode,
        network unavailable).

        Args:
            config: The global Config instance. Reads config.wandb_project,
                config.run_name, and calls config.to_dict() for the full
                hyperparameter dict.

        Side effects:
            Sets self._wandb_run to the initialized run object on success,
            or leaves it as None on failure.
        """
        if not _WANDB_AVAILABLE:
            self._logger.warning(
                "_init_wandb: wandb is not installed. "
                "W&B experiment tracking will be disabled. "
                "Install wandb==0.17.5: pip install wandb==0.17.5"
            )
            return

        try:
            # Serialize all hyperparameters for reproducibility tracking.
            # config.to_dict() returns a flat dict with all Config fields,
            # including α, β₁, β₂, learning rates, batch sizes, etc.
            config_dict: Dict[str, Any] = config.to_dict()

            run = _wandb_module.init(
                project=config.wandb_project,
                name=config.run_name,
                config=config_dict,
                # resume="allow" allows resuming interrupted runs by run_name
                resume="allow",
                # dir: store wandb files in the output directory
                dir=self._log_dir,
            )
            self._wandb_run = run

            self._logger.info(
                "_init_wandb: W&B run initialized. "
                "Project: '%s', Run: '%s', Run ID: '%s'.",
                config.wandb_project,
                config.run_name,
                run.id if run is not None else "unknown",
            )

        except Exception as exc:
            self._logger.warning(
                "_init_wandb: W&B initialization failed: %s. "
                "Continuing without W&B experiment tracking. "
                "To enable W&B, ensure you are logged in (wandb login) "
                "and have network access.",
                exc,
            )
            self._wandb_run = None

    @staticmethod
    def _classify_correction_type(
        reward_t1: float, reward_t2: float
    ) -> str:
        """Classify a trajectory's correction type based on binary rewards.

        Used internally by log_trajectories() to populate the
        'correction_type' column in the wandb Table.

        Args:
            reward_t1: Binary reward for the first attempt (0.0 or 1.0).
            reward_t2: Binary reward for the second attempt (0.0 or 1.0).

        Returns:
            One of: "i→c", "c→i", "i→i", "c→c".
        """
        r1_correct: bool = reward_t1 >= _CORRECT_THRESHOLD
        r2_correct: bool = reward_t2 >= _CORRECT_THRESHOLD

        if not r1_correct and r2_correct:
            return "i→c"
        elif r1_correct and not r2_correct:
            return "c→i"
        elif not r1_correct and not r2_correct:
            return "i→i"
        else:
            return "c→c"
