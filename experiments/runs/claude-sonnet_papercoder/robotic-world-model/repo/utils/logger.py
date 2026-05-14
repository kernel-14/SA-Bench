## utils/logger.py
"""Unified logging module for the Robotic World Model (RWM) project.

Provides a single Logger class that writes metrics simultaneously to both
wandb and TensorBoard backends. Consumed by RWMTrainer, MBPOPPOTrainer,
and Benchmark — every component that emits metrics during training or
evaluation.

Design follows the Logger interface specified in the project design document:
    - log(metrics, step): write scalar metrics to both backends
    - log_model_summary(model): print architecture summary and parameter count
    - save_config(config): persist hydra config as JSON
    - close(): flush and finalize both backends

Usage:
    logger = Logger(run_name="rwm_anymal_seed0", config=cfg, use_wandb=False)
    logger.log({"train/loss": 0.42}, step=100)
    logger.close()
"""

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn

from utils.common import count_parameters

# ---------------------------------------------------------------------------
# Optional dependency guards — graceful fallback if packages are missing
# ---------------------------------------------------------------------------

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    WANDB_AVAILABLE = False
    warnings.warn(
        "wandb is not installed. Wandb logging will be disabled. "
        "Install with: pip install wandb==0.17.0",
        UserWarning,
        stacklevel=1,
    )

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]
    TENSORBOARD_AVAILABLE = False
    warnings.warn(
        "TensorBoard SummaryWriter is not available. TensorBoard logging "
        "will be disabled. Install with: pip install tensorboard==2.17.0",
        UserWarning,
        stacklevel=1,
    )

try:
    from omegaconf import DictConfig, OmegaConf

    OMEGACONF_AVAILABLE = True
except ImportError:
    DictConfig = None  # type: ignore[assignment,misc]
    OmegaConf = None  # type: ignore[assignment]
    OMEGACONF_AVAILABLE = False


def _to_plain_dict(config: Any) -> Dict[str, Any]:
    """Convert a config object to a plain Python dict for serialization.

    Handles OmegaConf DictConfig objects (from hydra) as well as plain
    Python dicts. The resolved=True flag expands interpolations in hydra
    configs (e.g., ${robot} references).

    Args:
        config: Configuration object. May be an OmegaConf DictConfig,
            a plain Python dict, or any other type (returned as-is wrapped
            in a dict).

    Returns:
        A plain Python dict suitable for JSON serialization and wandb.init.
    """
    if OMEGACONF_AVAILABLE and DictConfig is not None and isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True, throw_on_missing=False)  # type: ignore[return-value]
    if isinstance(config, dict):
        return config
    # Fallback: wrap in a dict with a single "config" key
    return {"config": str(config)}


class Logger:
    """Unified logger writing to both wandb and TensorBoard simultaneously.

    Provides a single interface for all metric logging in the RWM project.
    Each Logger instance is independent, allowing multiple seeds to log to
    separate directories (e.g., logs/seed_0/, logs/seed_1/).

    The Logger is NOT a singleton — instantiate one per training run.

    Attributes:
        run_name: Human-readable identifier for this run (e.g.,
            "rwm_anymal_seed0"). Used as the wandb run name and as part
            of the log directory path.
        use_wandb: Whether wandb logging is active. Set to False if wandb
            is not installed or if use_wandb=false in config.yaml.
        log_dir: Directory where TensorBoard event files and config.json
            are written. Corresponds to config.yaml field log_dir: "logs".
        tb_writer: TensorBoard SummaryWriter instance, or None if
            TensorBoard is unavailable.
        wandb_run: Active wandb run object, or None if wandb is disabled.
    """

    def __init__(
        self,
        run_name: str,
        config: Any,
        use_wandb: bool = False,
        log_dir: str = "logs",
    ) -> None:
        """Initialize Logger with both TensorBoard and optional wandb backends.

        Creates the log directory, initializes TensorBoard SummaryWriter,
        optionally initializes wandb, and immediately persists the config
        as config.json for experiment traceability.

        Args:
            run_name: Identifier for this run. Used as wandb run name and
                included in log output. Example: "rwm_anymal_seed0".
            config: Experiment configuration. Accepts OmegaConf DictConfig
                (from hydra) or plain Python dict. Converted to plain dict
                internally for JSON serialization and wandb.
            use_wandb: Whether to enable wandb logging. Corresponds to
                config.yaml field use_wandb: false. Forced to False if
                wandb is not installed. Default: False.
            log_dir: Directory for TensorBoard event files and config.json.
                Corresponds to config.yaml field log_dir: "logs".
                Created automatically if it does not exist. Default: "logs".
        """
        self.run_name: str = run_name
        self.log_dir: str = log_dir

        # Resolve use_wandb: respect the config.yaml default (false) and
        # force False if wandb package is not installed.
        if use_wandb and not WANDB_AVAILABLE:
            warnings.warn(
                f"use_wandb=True was requested for run '{run_name}' but "
                "wandb is not installed. Disabling wandb logging.",
                UserWarning,
                stacklevel=2,
            )
            self.use_wandb: bool = False
        else:
            self.use_wandb = use_wandb

        # Track which models have been registered with wandb.watch to
        # prevent duplicate calls if log_model_summary is called multiple times.
        self._watched_models: Set[int] = set()

        # Create log directory (parents=True handles nested paths like
        # logs/rwm_anymal/seed_0/).
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard SummaryWriter.
        if TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            self.tb_writer: Optional[Any] = SummaryWriter(log_dir=log_dir)
        else:
            self.tb_writer = None

        # Convert config to plain dict once — used for both wandb.init and
        # save_config to avoid repeated conversion.
        config_dict: Dict[str, Any] = _to_plain_dict(config)

        # Initialize wandb if requested and available.
        self.wandb_run: Optional[Any] = None
        if self.use_wandb:
            try:
                self.wandb_run = wandb.init(  # type: ignore[union-attr]
                    project="robotic-world-model",
                    name=run_name,
                    config=config_dict,
                    dir=log_dir,
                    # resume="allow" lets wandb handle interrupted runs
                    resume="allow",
                )
            except Exception as exc:
                warnings.warn(
                    f"wandb.init failed for run '{run_name}': {exc}. "
                    "Continuing without wandb logging.",
                    UserWarning,
                    stacklevel=2,
                )
                self.use_wandb = False
                self.wandb_run = None

        # Persist config immediately so it is available even if training
        # crashes before close() is called.
        self.save_config(config_dict)

        print(
            f"[Logger] Initialized run '{run_name}'. "
            f"TensorBoard: {'enabled' if self.tb_writer is not None else 'disabled'}, "
            f"wandb: {'enabled' if self.wandb_run is not None else 'disabled'}. "
            f"Log dir: {log_dir}"
        )

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Write scalar metrics to both TensorBoard and wandb.

        Converts all metric values to Python floats before writing.
        Skips non-finite values (NaN, Inf) with a warning to prevent
        corrupting run history — especially important during early MBPO-PPO
        training when the world model may produce unstable predictions.

        Metric naming convention (forward-slash namespacing for clean
        grouping in dashboards):
            - World model training: "train/loss", "train/obs_loss",
              "train/priv_loss", "train/learning_rate"
            - Policy training: "policy/mean_reward", "policy/model_error",
              "policy/ppo_loss", "policy/value_loss", "policy/entropy",
              "policy/approx_kl", "policy/clip_fraction"
            - Evaluation: "eval/mean_reward", "eval/std_reward",
              "eval/model_error"
            - Benchmark: "benchmark/relative_error"

        Args:
            metrics: Dictionary mapping metric name (str) to scalar value.
                Values may be Python float/int, numpy scalars, or 0-dim
                PyTorch tensors. All are converted to Python float internally.
            step: Global training step or iteration number. Used as the
                x-axis in both TensorBoard and wandb plots.
        """
        if not metrics:
            return

        # Accumulate valid float metrics for a single wandb.log call
        # (more efficient than one call per metric).
        wandb_metrics: Dict[str, float] = {}

        for key, value in metrics.items():
            # Convert to Python float — handles torch.Tensor, numpy scalars,
            # and Python numeric types uniformly.
            try:
                float_value: float = float(value)
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    f"[Logger] Could not convert metric '{key}' value "
                    f"'{value}' to float: {exc}. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Skip non-finite values to protect run history integrity.
            if not math.isfinite(float_value):
                warnings.warn(
                    f"[Logger] Non-finite value detected for metric '{key}': "
                    f"{float_value} at step {step}. Skipping this metric. "
                    "This may indicate numerical instability in the world "
                    "model or policy — check learning rates and gradient norms.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Write to TensorBoard.
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(key, float_value, global_step=step)

            wandb_metrics[key] = float_value

        # Single wandb.log call for all metrics at this step.
        if self.wandb_run is not None and wandb_metrics:
            try:
                wandb.log(wandb_metrics, step=step)  # type: ignore[union-attr]
            except Exception as exc:
                warnings.warn(
                    f"[Logger] wandb.log failed at step {step}: {exc}. "
                    "Continuing without wandb for this step.",
                    UserWarning,
                    stacklevel=2,
                )

    def log_model_summary(self, model: nn.Module) -> None:
        """Print model architecture summary and log parameter count.

        Prints a human-readable summary of the model to stdout, including
        total trainable parameter count and per-layer structure. Also logs
        the parameter count as a scalar metric at step 0 for tracking in
        dashboards.

        If wandb is enabled, registers the model for gradient tracking via
        wandb.watch. This is useful for diagnosing training instability in
        the GRU and MLP heads of RWM. Each model is only registered once
        (tracked by object id) to prevent duplicate wandb.watch calls.

        Args:
            model: PyTorch module to summarize. Typically the GRUWorldModel,
                PolicyNetwork, or ValueNetwork.
        """
        total_params: int = count_parameters(model)
        model_class_name: str = model.__class__.__name__

        # Build a compact summary string from named modules.
        summary_lines = [
            f"\n{'='*60}",
            f"  Model Summary: {model_class_name}",
            f"  Run: {self.run_name}",
            f"{'='*60}",
        ]

        for name, module in model.named_modules():
            # Skip the top-level module itself (empty name) to avoid
            # redundancy — it is already shown in the header.
            if name == "":
                continue
            # Indent nested modules for readability.
            depth = name.count(".")
            indent = "  " + "  " * depth
            module_str = repr(module).split("\n")[0]  # First line only
            summary_lines.append(f"{indent}{name}: {module_str}")

        summary_lines.extend([
            f"{'='*60}",
            f"  Total trainable parameters: {total_params:,}",
            f"{'='*60}\n",
        ])

        print("\n".join(summary_lines))

        # Log parameter count as a scalar metric for dashboard tracking.
        self.log({f"model/{model_class_name}_parameters": total_params}, step=0)

        # Register model with wandb.watch for gradient tracking.
        # Guard against duplicate registration using object id.
        model_id = id(model)
        if self.wandb_run is not None and model_id not in self._watched_models:
            try:
                wandb.watch(  # type: ignore[union-attr]
                    model,
                    log="gradients",
                    log_freq=100,
                )
                self._watched_models.add(model_id)
            except Exception as exc:
                warnings.warn(
                    f"[Logger] wandb.watch failed for {model_class_name}: {exc}. "
                    "Gradient tracking disabled for this model.",
                    UserWarning,
                    stacklevel=2,
                )

    def save_config(self, config: Any) -> None:
        """Persist the experiment configuration as config.json.

        Writes the full hydra config (or plain dict) to the log directory
        as a JSON file. This ensures every experiment has a complete record
        of its hyperparameters, including all values from config.yaml:
        rwm_training, mbpo_ppo, rwm architecture, policy, reward weights,
        seed, experiment type, robot, and env_backend.

        If wandb is enabled, uploads the config file to the wandb run
        artifacts for centralized experiment tracking.

        Args:
            config: Configuration to persist. Accepts OmegaConf DictConfig,
                plain Python dict, or any JSON-serializable object.
                Non-serializable values are converted to strings via
                default=str in json.dump.
        """
        config_dict: Dict[str, Any] = _to_plain_dict(config)
        config_path: Path = Path(self.log_dir) / "config.json"

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, default=str)
        except (OSError, TypeError) as exc:
            warnings.warn(
                f"[Logger] Failed to save config to {config_path}: {exc}. "
                "Experiment configuration will not be persisted.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Upload config file to wandb artifacts for centralized tracking.
        if self.wandb_run is not None:
            try:
                wandb.save(str(config_path))  # type: ignore[union-attr]
            except Exception as exc:
                warnings.warn(
                    f"[Logger] wandb.save failed for config file: {exc}.",
                    UserWarning,
                    stacklevel=2,
                )

    def close(self) -> None:
        """Flush and finalize both logging backends.

        Must be called at the end of training (ideally in a try/finally
        block in main.py) to ensure all buffered metrics are written and
        wandb runs are properly finalized. Incomplete wandb runs appear as
        "running" indefinitely in the dashboard if finish() is not called.

        Example usage in main.py:
            logger = Logger(...)
            try:
                train(logger)
            finally:
                logger.close()
        """
        # Flush and close TensorBoard writer.
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception as exc:
                warnings.warn(
                    f"[Logger] Error closing TensorBoard writer: {exc}.",
                    UserWarning,
                    stacklevel=2,
                )

        # Finalize wandb run — uploads remaining data and marks run as finished.
        if self.wandb_run is not None:
            try:
                wandb.finish()  # type: ignore[union-attr]
            except Exception as exc:
                warnings.warn(
                    f"[Logger] Error finishing wandb run: {exc}.",
                    UserWarning,
                    stacklevel=2,
                )

        print(f"[Logger] Closed. Logs saved to {self.log_dir}")
