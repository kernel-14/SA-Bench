## utils/logger.py
"""Logging utility for SC-FNO experiments.

Provides a unified Logger class that writes to stdout, a log file, and
optionally TensorBoard. All training metrics, evaluation results, and
experiment outputs flow through this module.

No project-internal imports — depends only on standard library and the
optional tensorboard package.
"""

import json
import logging
import math
import os
from typing import Any, Optional


class Logger:
    """Centralizes all output from the SC-FNO training and evaluation pipeline.

    Writes to three sinks simultaneously:
      1. Python logging (stdout + file at run_dir/run.log)
      2. TensorBoard SummaryWriter (if tensorboard is installed)
      3. JSON files via save_results()

    Attributes:
        log_dir: Base output directory (e.g., "outputs/logs").
        run_name: Identifier for this specific run (e.g., "pde1_sc_fno").
        run_dir: Full path os.path.join(log_dir, run_name), created on init.
        writer: TensorBoard SummaryWriter, or None if unavailable.
        _logger: Standard Python logger writing to stdout and run.log.

    Example:
        >>> logger = Logger("outputs/logs", "pde1_sc_fno")
        >>> logger.log_scalar("train/loss_u", 0.042, step=10)
        >>> logger.log_dict({"u": {"r2": 0.983, "rel_l2": 0.017}}, step=0)
        >>> logger.save_results({"r2": 0.983}, "forward_metrics.json")
        >>> logger.close()
    """

    def __init__(self, log_dir: str = "outputs/logs", run_name: str = "default_run") -> None:
        """Initializes the Logger, creating the run directory and all sinks.

        Args:
            log_dir: Base directory for all log output. Created if absent.
                     Sourced from config.yaml key 'log_dir'.
            run_name: Unique identifier for this run, typically assembled by
                      main.py as "{equation}_{variant}_{timestamp}".
        """
        self.log_dir: str = log_dir
        self.run_name: str = run_name
        self.run_dir: str = os.path.join(log_dir, run_name)
        self.writer: Optional[Any] = None  # SummaryWriter or None

        # Create the run directory (safe if it already exists).
        os.makedirs(self.run_dir, exist_ok=True)

        # Set up the Python logger.
        self._logger: logging.Logger = self._setup_python_logger()

        # Attempt to initialize TensorBoard.
        self._setup_tensorboard()

        self._logger.info("Logger initialized. Run: %s, Dir: %s", run_name, self.run_dir)

    # ------------------------------------------------------------------
    # Private setup helpers
    # ------------------------------------------------------------------

    def _setup_python_logger(self) -> logging.Logger:
        """Creates and configures the Python logging.Logger instance.

        Adds a StreamHandler (stdout) and a FileHandler (run.log) with a
        consistent timestamp + level + message format. Guards against
        duplicate handlers when Logger is instantiated multiple times with
        the same run_name (e.g., during data-scaling experiments).

        Returns:
            Configured logging.Logger instance.
        """
        logger = logging.getLogger(self.run_name)
        logger.setLevel(logging.INFO)

        # Avoid accumulating duplicate handlers across multiple instantiations.
        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Stdout handler.
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File handler — writes to run_dir/run.log.
        log_file_path = os.path.join(self.run_dir, "run.log")
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Prevent log records from propagating to the root logger, which
        # would cause duplicate output if the root logger has handlers.
        logger.propagate = False

        return logger

    def _setup_tensorboard(self) -> None:
        """Attempts to initialize a TensorBoard SummaryWriter.

        Gracefully degrades to None if tensorboard is not installed.
        Logs a warning in that case so the user is aware.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self.writer = SummaryWriter(log_dir=self.run_dir)
        except ImportError:
            self.writer = None
            self._logger.warning(
                "TensorBoard not available; scalar logging to stdout/file only. "
                "Install tensorboard to enable TensorBoard logging."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Logs a single scalar value to TensorBoard and the Python logger.

        Handles non-finite values (NaN, Inf) with an additional warning so
        that training instabilities are immediately visible in the log file.

        Args:
            tag: Metric identifier, e.g. "train/loss_total", "val/loss",
                 "eval/r2_u". Forward slashes create TensorBoard groups.
            value: Scalar metric value. Converted to float before logging.
            step: Global step (epoch number or iteration count).
        """
        float_value: float = float(value)

        # Warn on non-finite values — common early in SC-FNO training due to
        # second-order gradients before the model has stabilized.
        if math.isnan(float_value) or math.isinf(float_value):
            self._logger.warning(
                "WARNING: non-finite value for '%s' at step %d: %s",
                tag,
                step,
                float_value,
            )

        # Write to TensorBoard if available.
        if self.writer is not None:
            self.writer.add_scalar(tag, float_value, global_step=step)

        # Always write to the Python logger.
        self._logger.info("[step %d] %s: %.6f", step, tag, float_value)

    def log_dict(self, metrics: dict, step: int) -> None:
        """Logs all scalar values in a (possibly nested) metrics dictionary.

        Recursively flattens nested dicts using '/' as the separator, which
        maps naturally to TensorBoard's tag hierarchy. Non-numeric leaf values
        are silently skipped.

        Args:
            metrics: Flat or nested dict of metric values. Example:
                     {"u": {"r2": 0.983, "rel_l2": 0.017},
                      "du_dalpha": {"r2": 0.925, "rel_l2": 0.075}}
                     Nested keys become "u/r2", "u/rel_l2", etc.
            step: Global step passed to each log_scalar call.
        """
        self._log_dict_recursive(metrics, prefix="", step=step)

    def _log_dict_recursive(self, obj: Any, prefix: str, step: int) -> None:
        """Recursively traverses a nested dict and logs scalar leaf values.

        Args:
            obj: Current node in the metrics tree (dict, numeric, or other).
            prefix: Accumulated tag prefix with '/' separators.
            step: Global step for all scalar calls in this traversal.
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                # Build the hierarchical tag, e.g. "sensitivity/alpha/r2".
                child_prefix = f"{prefix}/{key}" if prefix else str(key)
                self._log_dict_recursive(value, child_prefix, step)
        elif isinstance(obj, (int, float)):
            self.log_scalar(prefix, float(obj), step)
        else:
            # Attempt to convert tensor scalars or numpy scalars.
            try:
                scalar_value = float(obj)
                self.log_scalar(prefix, scalar_value, step)
            except (TypeError, ValueError):
                # Non-numeric leaf (e.g., string metadata) — skip silently.
                pass

    def save_results(self, results: dict, filename: str) -> None:
        """Serializes a results dictionary to a JSON file in run_dir.

        Converts torch.Tensor scalars and numpy arrays to Python native types
        before serialization so that json.dump never raises TypeError.

        Args:
            results: Arbitrarily nested dict of experiment results. May
                     contain torch.Tensor scalars, numpy scalars/arrays,
                     Python floats, ints, strings, lists, and nested dicts.
            filename: Output filename, e.g. "forward_metrics.json". The file
                      is written to self.run_dir/filename.
        """
        path = os.path.join(self.run_dir, filename)

        # Sanitize the entire results tree for JSON serialization.
        serializable_results = self._to_serializable(results)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)

        self._logger.info("Results saved to %s", path)

    def _to_serializable(self, obj: Any) -> Any:
        """Recursively converts an object tree to JSON-serializable types.

        Handles:
          - dict: recurse over values
          - list / tuple: recurse over elements, return as list
          - torch.Tensor: .item() for scalars, .tolist() for arrays
          - numpy scalar / ndarray: float() or .tolist()
          - int, float, str, bool, None: pass through unchanged

        Args:
            obj: Any Python object that may appear in a results dict.

        Returns:
            A JSON-serializable version of obj.
        """
        if isinstance(obj, dict):
            return {str(k): self._to_serializable(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(item) for item in obj]

        # Handle torch.Tensor without importing torch at module level
        # (avoids a hard dependency — Logger should work even without torch).
        type_name = type(obj).__name__
        module_name = type(obj).__module__

        if module_name == "torch" and type_name == "Tensor":
            # Scalar tensor → Python float; multi-element tensor → nested list.
            if obj.numel() == 1:  # type: ignore[union-attr]
                return float(obj.item())  # type: ignore[union-attr]
            return obj.tolist()  # type: ignore[union-attr]

        # Handle numpy scalars and arrays.
        if module_name == "numpy" or (hasattr(obj, "__module__") and
                                       getattr(obj, "__module__", "").startswith("numpy")):
            if hasattr(obj, "tolist"):
                return obj.tolist()  # type: ignore[union-attr]
            try:
                return float(obj)
            except (TypeError, ValueError):
                return str(obj)

        # Python native types pass through unchanged.
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj

        # Fallback: attempt float conversion, then str.
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)

    def close(self) -> None:
        """Flushes and closes the TensorBoard writer and file handlers.

        Should be called by main.py at the end of each run to ensure all
        pending TensorBoard writes are flushed to disk.
        """
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self._logger.info("TensorBoard writer closed.")

        # Close and remove all file handlers to release the log file handle.
        handlers_to_remove = [
            h for h in self._logger.handlers if isinstance(h, logging.FileHandler)
        ]
        for handler in handlers_to_remove:
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)

    def __del__(self) -> None:
        """Ensures the TensorBoard writer is closed on garbage collection.

        This is a safety net — callers should prefer explicit close() calls.
        """
        try:
            if self.writer is not None:
                self.writer.close()
        except Exception:  # pylint: disable=broad-except
            # Suppress all exceptions in __del__ — the interpreter may be
            # shutting down and some objects may already be finalized.
            pass
