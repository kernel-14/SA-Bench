## utils/logging_utils.py
"""Centralized experiment tracking and visualization for MA-RLHF.

This module provides LoggingUtils, the single logging interface used by all
three training stages (SFT, RM, MA-PPO) and the evaluation pipeline. It
bridges training loops with wandb, TensorBoard, matplotlib visualizations,
and Python's standard logging module.

Key responsibilities:
  - Scalar metric logging to wandb and TensorBoard simultaneously.
  - RM score distribution tracking (Figures 2, 13, 15 from the paper).
  - RM score distribution histogram generation (Figures 3, 10, 14, 16).
  - L2 norm tracking for advantages and Q-values (Figure 11).
  - Persistent file-based logging for reproducibility.

Dependencies:
    External: wandb, torch.utils.tensorboard, matplotlib, numpy, logging
    Internal: config.py (Config, LoggingConfig)

Design note: All external library calls are guarded so that training never
crashes due to logging failures. wandb and TensorBoard are optional.
"""

import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from config import Config


class LoggingUtils:
    """Centralized logging and visualization for MA-RLHF experiments.

    Provides a unified interface for logging scalar metrics, RM score
    distributions, and L2 norms of advantage/Q-value tensors to wandb,
    TensorBoard, and disk simultaneously.

    This class is instantiated once per training run in main.py and passed
    to all trainers. It has no dependencies on other project modules except
    config.py.

    Attributes:
        config: The full Config object with all hyperparameters.
        run_name: Identifier for this training run, used as wandb run name
            and for naming output files and directories.
        logger: Python logging.Logger instance for console and file output.
        writer: TensorBoard SummaryWriter, or None if TensorBoard is
            disabled or unavailable.
        wandb_run: Active wandb run object, or None if wandb is disabled
            or unavailable.
        output_dir: pathlib.Path to the run-specific output directory.
            Created during __init__ if it does not exist.
    """

    def __init__(self, config: Config, run_name: str) -> None:
        """Initialize all logging backends and create the output directory.

        Sets up Python logging (console + file), wandb, and TensorBoard.
        All external library initializations are guarded with try/except
        so that training proceeds even if a backend is unavailable.

        The wandb run is initialized with config.to_dict() so that all
        hyperparameters are recorded for reproducibility.

        Args:
            config: Full Config object. The logging section controls which
                backends are active. The output_dir field determines where
                logs and artifacts are saved.
            run_name: Human-readable identifier for this run, e.g.,
                "tldr_gemma2b_mappo_n5". Used as the wandb run name and
                as a subdirectory under config.output_dir.
        """
        self.config: Config = config
        self.run_name: str = run_name

        # --- Output directory ---
        # Create a run-specific subdirectory under the global output_dir.
        self.output_dir: pathlib.Path = (
            pathlib.Path(config.output_dir) / run_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Python logging ---
        self.logger: logging.Logger = self._setup_python_logger(run_name)

        # --- TensorBoard ---
        self.writer: Optional[Any] = self._setup_tensorboard(config, run_name)

        # --- wandb ---
        self.wandb_run: Optional[Any] = self._setup_wandb(config, run_name)

        self.logger.info(
            "LoggingUtils initialized for run: '%s'. Output dir: '%s'.",
            run_name,
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Primary logging methods
    # ------------------------------------------------------------------

    def log_metrics(self, metrics_dict: Dict[str, float], step: int) -> None:
        """Write arbitrary scalar metrics to all active logging backends.

        Logs to Python logger (DEBUG), wandb, and TensorBoard simultaneously.
        Failures in wandb or TensorBoard are caught and logged as warnings
        so that training is never interrupted by logging errors.

        Called by MAPPOTrainer._log_metrics(), SFTTrainer.train(), and
        RMTrainer.train() at every log_interval steps.

        Typical keys in metrics_dict:
            "train/policy_loss", "train/critic_loss", "train/kl_penalty",
            "train/rm_score", "train/advantages_mean", "train/returns_mean"

        Args:
            metrics_dict: Dictionary mapping metric names to scalar float
                values. All values must be finite floats (not NaN or inf).
            step: Global training step number, used as the x-axis for all
                logged metrics.
        """
        if not metrics_dict:
            return

        # --- Python logger (DEBUG to avoid flooding stdout) ---
        msg_parts: List[str] = [
            f"{k}={v:.4f}" for k, v in metrics_dict.items()
        ]
        self.logger.debug("Step %d: %s", step, ", ".join(msg_parts))

        # --- wandb ---
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics_dict, step=step)
            except Exception as exc:
                self.logger.warning(
                    "wandb.log failed at step %d: %s", step, exc
                )

        # --- TensorBoard ---
        if self.writer is not None:
            try:
                for key, value in metrics_dict.items():
                    self.writer.add_scalar(key, value, global_step=step)
            except Exception as exc:
                self.logger.warning(
                    "TensorBoard add_scalar failed at step %d: %s", step, exc
                )

    def log_rm_scores(self, scores: List[float], step: int) -> None:
        """Log summary statistics of the RM score distribution.

        Computes mean, std, min, max, and median of the provided RM scores
        and logs them to all active backends. Also logs a histogram to
        TensorBoard if log_score_distribution is enabled.

        This feeds the training curves in Figures 2, 13, 15 of the paper.
        Called by MAPPOTrainer._evaluate_rm() at every eval_interval steps.

        Args:
            scores: List of scalar RM scores from the evaluation set.
                Typically 2000 scores for TL;DR and HH-RLHF (Section 4.1).
            step: Global training step number.
        """
        if not scores:
            self.logger.warning(
                "log_rm_scores called with empty scores list at step %d.",
                step,
            )
            return

        arr: np.ndarray = np.array(scores, dtype=np.float64)

        mean: float = float(arr.mean())
        std: float = float(arr.std())
        min_val: float = float(arr.min())
        max_val: float = float(arr.max())
        median: float = float(np.median(arr))

        # Log summary statistics to all backends.
        self.log_metrics(
            {
                "eval/rm_score_mean": mean,
                "eval/rm_score_std": std,
                "eval/rm_score_min": min_val,
                "eval/rm_score_max": max_val,
                "eval/rm_score_median": median,
            },
            step=step,
        )

        # Console output at INFO level for visibility during training.
        self.logger.info(
            "Step %d | RM Score: mean=%.4f, std=%.4f, "
            "min=%.4f, max=%.4f, median=%.4f",
            step,
            mean,
            std,
            min_val,
            max_val,
            median,
        )

        # TensorBoard histogram for distribution shape visualization.
        if (
            self.writer is not None
            and self.config.logging.log_score_distribution
        ):
            try:
                self.writer.add_histogram(
                    "eval/rm_score_distribution",
                    arr,
                    global_step=step,
                )
            except Exception as exc:
                self.logger.warning(
                    "TensorBoard add_histogram failed at step %d: %s",
                    step,
                    exc,
                )

    def save_score_distribution(
        self,
        scores_a: List[float],
        scores_b: List[float],
        labels: List[str],
        step: int,
        output_path: str,
    ) -> None:
        """Generate and save a histogram comparing two RM score distributions.

        Produces overlapping normalized histograms with mean lines, matching
        the visual style of Figures 3, 10, 14, and 16 from the paper. The
        figure is saved to disk and optionally logged to wandb and TensorBoard.

        Args:
            scores_a: RM scores from model A (e.g., vanilla PPO). Must be
                non-empty.
            scores_b: RM scores from model B (e.g., MA-PPO). Must be
                non-empty.
            labels: List of exactly two strings for the legend, e.g.,
                ["Vanilla PPO", "MA-PPO (n=5)"]. labels[0] corresponds to
                scores_a and labels[1] to scores_b.
            step: Training step number, used in the filename and figure title.
            output_path: Directory where the PNG figure will be saved.
                Created if it does not exist.
        """
        # Guard: skip if distribution logging is disabled.
        if not self.config.logging.log_score_distribution:
            return

        if not scores_a or not scores_b:
            self.logger.warning(
                "save_score_distribution called with empty scores at step %d.",
                step,
            )
            return

        if len(labels) < 2:
            self.logger.warning(
                "save_score_distribution requires exactly 2 labels, "
                "got %d. Using defaults.",
                len(labels),
            )
            labels = ["Model A", "Model B"]

        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend for server use.
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning(
                "matplotlib not available; skipping score distribution plot."
            )
            return

        arr_a: np.ndarray = np.array(scores_a, dtype=np.float64)
        arr_b: np.ndarray = np.array(scores_b, dtype=np.float64)

        # Shared bin range for fair visual comparison.
        all_scores: np.ndarray = np.concatenate([arr_a, arr_b])
        bin_min: float = float(all_scores.min())
        bin_max: float = float(all_scores.max())

        # Guard against degenerate case where all scores are identical.
        if bin_min == bin_max:
            bin_min -= 0.5
            bin_max += 0.5

        # 50 bins matches the visual resolution of the paper's figures.
        bins: np.ndarray = np.linspace(bin_min, bin_max, 51)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Overlapping histograms with transparency (alpha=0.6).
        # density=True normalizes to probability density (paper's y-axis).
        ax.hist(
            arr_a,
            bins=bins,
            alpha=0.6,
            label=labels[0],
            color="steelblue",
            density=True,
        )
        ax.hist(
            arr_b,
            bins=bins,
            alpha=0.6,
            label=labels[1],
            color="coral",
            density=True,
        )

        # Vertical mean lines for quick comparison.
        mean_a: float = float(arr_a.mean())
        mean_b: float = float(arr_b.mean())

        ax.axvline(
            mean_a,
            color="steelblue",
            linestyle="--",
            linewidth=1.5,
            label=f"{labels[0]} mean={mean_a:.2f}",
        )
        ax.axvline(
            mean_b,
            color="coral",
            linestyle="--",
            linewidth=1.5,
            label=f"{labels[1]} mean={mean_b:.2f}",
        )

        ax.set_xlabel("RM Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"RM Score Distribution at Step {step}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Save to disk.
        save_dir: pathlib.Path = pathlib.Path(output_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path: pathlib.Path = (
            save_dir / f"rm_distribution_step{step}.png"
        )

        try:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            self.logger.info(
                "Saved RM score distribution plot to '%s'.", save_path
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to save RM distribution plot: %s", exc
            )
            plt.close(fig)
            return

        plt.close(fig)  # Prevent memory leak from unclosed figures.

        # Log to wandb as an image artifact.
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(
                    {
                        "eval/rm_score_distribution": wandb.Image(
                            str(save_path)
                        )
                    },
                    step=step,
                )
            except Exception as exc:
                self.logger.warning(
                    "wandb image logging failed at step %d: %s", step, exc
                )

        # Log to TensorBoard as an image.
        if self.writer is not None:
            try:
                from PIL import Image
                import torchvision.transforms.functional as TF

                pil_img: Image.Image = Image.open(str(save_path)).convert("RGB")
                img_tensor: torch.Tensor = TF.to_tensor(pil_img)
                self.writer.add_image(
                    "eval/rm_score_distribution",
                    img_tensor,
                    global_step=step,
                )
            except ImportError:
                # PIL/torchvision not available; skip TensorBoard image logging.
                self.logger.debug(
                    "PIL or torchvision not available; "
                    "skipping TensorBoard image logging."
                )
            except Exception as exc:
                self.logger.warning(
                    "TensorBoard image logging failed at step %d: %s",
                    step,
                    exc,
                )

    def log_l2_norms(
        self,
        advantages: torch.Tensor,
        q_values: torch.Tensor,
        step: int,
    ) -> None:
        """Compute and log L2 norms of advantage and Q-value tensors.

        Reproduces Figure 11 from the paper, which shows that MA-PPO
        achieves lower and more stable L2 norms compared to vanilla PPO,
        contributing to faster and more stable learning.

        Tensors are detached and cast to float32 before norm computation
        to ensure numerical accuracy regardless of training dtype (bf16/fp16).

        Called by MAPPOTrainer._update_policy() at every training step,
        guarded by config.logging.log_l2_norms.

        Args:
            advantages: Advantage estimates from GAE, shape
                [batch_size, num_macro_actions]. May be in bf16/fp16.
            q_values: Q-value estimates (returns = advantages + values),
                shape [batch_size, num_macro_actions]. May be in bf16/fp16.
            step: Global training step number.
        """
        # Guard: skip if L2 norm logging is disabled.
        if not self.config.logging.log_l2_norms:
            return

        # Detach from computation graph and cast to float32 for accuracy.
        # bf16 has limited precision that can distort norm computations.
        adv_detached: torch.Tensor = advantages.detach().float()
        qval_detached: torch.Tensor = q_values.detach().float()

        # Frobenius norm over the entire batch tensor (scalar output).
        adv_l2: float = float(torch.norm(adv_detached, p=2).item())
        qval_l2: float = float(torch.norm(qval_detached, p=2).item())

        # Per-element statistics for richer diagnostics.
        adv_mean: float = float(adv_detached.mean().item())
        adv_std: float = float(adv_detached.std().item())
        qval_mean: float = float(qval_detached.mean().item())
        qval_std: float = float(qval_detached.std().item())

        self.log_metrics(
            {
                "train/advantages_l2_norm": adv_l2,
                "train/q_values_l2_norm": qval_l2,
                "train/advantages_mean": adv_mean,
                "train/advantages_std": adv_std,
                "train/q_values_mean": qval_mean,
                "train/q_values_std": qval_std,
            },
            step=step,
        )

    def close(self) -> None:
        """Flush and close all logging backends.

        Must be called at the end of training in main.py to ensure all
        buffered metrics are written to disk and remote backends before
        the process exits.

        Safe to call multiple times (idempotent).
        """
        # Close TensorBoard writer.
        if self.writer is not None:
            try:
                self.writer.close()
                self.logger.info("TensorBoard writer closed.")
            except Exception as exc:
                self.logger.warning(
                    "Error closing TensorBoard writer: %s", exc
                )

        # Finish wandb run.
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
                self.logger.info("wandb run finished.")
            except Exception as exc:
                self.logger.warning(
                    "Error finishing wandb run: %s", exc
                )

        self.logger.info(
            "LoggingUtils closed for run: '%s'.", self.run_name
        )

    # ------------------------------------------------------------------
    # Private setup helpers
    # ------------------------------------------------------------------

    def _setup_python_logger(self, run_name: str) -> logging.Logger:
        """Configure and return a named Python logger.

        Creates a logger with both a StreamHandler (console) and a
        FileHandler (training.log in the output directory). Guards against
        duplicate handlers if __init__ is called multiple times.

        Args:
            run_name: Used as part of the logger name for namespacing.

        Returns:
            A configured logging.Logger instance.
        """
        logger_name: str = f"ma_rlhf.{run_name}"
        logger: logging.Logger = logging.getLogger(logger_name)

        # Avoid adding duplicate handlers on repeated initialization.
        if logger.handlers:
            return logger

        logger.setLevel(logging.DEBUG)

        # Formatter: timestamp + level + message.
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler at INFO level (avoids DEBUG flooding stdout).
        console_handler: logging.StreamHandler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler at DEBUG level (captures all messages for debugging).
        log_file: pathlib.Path = self.output_dir / "training.log"
        try:
            file_handler: logging.FileHandler = logging.FileHandler(
                str(log_file), mode="a", encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            # Non-fatal: console logging still works.
            logger.warning(
                "Could not create log file '%s': %s. "
                "Continuing with console logging only.",
                log_file,
                exc,
            )

        # Prevent propagation to the root logger to avoid duplicate output.
        logger.propagate = False

        return logger

    def _setup_tensorboard(
        self, config: Config, run_name: str
    ) -> Optional[Any]:
        """Initialize TensorBoard SummaryWriter.

        Creates a run-specific subdirectory under config.logging.tensorboard_dir.
        Returns None if TensorBoard is disabled or unavailable.

        Args:
            config: Full Config object.
            run_name: Used as a subdirectory name under tensorboard_dir.

        Returns:
            A SummaryWriter instance, or None.
        """
        if not config.logging.use_tensorboard:
            return None

        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir: pathlib.Path = (
                pathlib.Path(config.logging.tensorboard_dir) / run_name
            )
            tb_dir.mkdir(parents=True, exist_ok=True)

            writer = SummaryWriter(log_dir=str(tb_dir))
            self.logger.info(
                "TensorBoard initialized. Log dir: '%s'.", tb_dir
            )
            return writer

        except ImportError:
            self.logger.warning(
                "tensorboard not installed; TensorBoard logging disabled."
            )
            return None
        except Exception as exc:
            self.logger.warning(
                "Failed to initialize TensorBoard: %s. "
                "TensorBoard logging disabled.",
                exc,
            )
            return None

    def _setup_wandb(
        self, config: Config, run_name: str
    ) -> Optional[Any]:
        """Initialize a wandb run.

        Passes config.to_dict() as the run config so all hyperparameters
        are recorded for reproducibility. Returns None if wandb is disabled,
        unavailable, or if initialization fails (e.g., no API key).

        Args:
            config: Full Config object. config.to_dict() is passed to
                wandb.init() as the run configuration.
            run_name: Used as the wandb run name.

        Returns:
            The active wandb.run object, or None.
        """
        if not config.logging.use_wandb:
            return None

        try:
            import wandb

            # entity=None uses the default wandb entity when the config
            # value is an empty string.
            entity: Optional[str] = (
                config.logging.wandb_entity
                if config.logging.wandb_entity
                else None
            )

            wandb.init(
                project=config.logging.wandb_project,
                entity=entity,
                name=run_name,
                config=config.to_dict(),
                # resume="allow" lets wandb resume a crashed run with the
                # same run_name rather than creating a duplicate.
                resume="allow",
            )

            self.logger.info(
                "wandb initialized: project='%s', run='%s'.",
                config.logging.wandb_project,
                run_name,
            )
            return wandb.run

        except ImportError:
            self.logger.warning(
                "wandb not installed; wandb logging disabled."
            )
            return None
        except Exception as exc:
            self.logger.warning(
                "Failed to initialize wandb: %s. wandb logging disabled.",
                exc,
            )
            return None
