## utils/logging_utils.py
"""Experiment logging utilities for gated attention reproduction pipeline.

This module implements ExperimentLogger, the central observability component
for all experiments in the paper "Gated Attention for Large Language Models:
Non-linearity, Sparsity, and Attention-Sink-Free".

Three primary functions:
    1. Real-time training monitoring — logs step-level metrics (loss, grad norm,
       LR) to TensorBoard and optionally W&B during the training loop.
    2. Evaluation result recording — logs PPL per domain and benchmark scores
       at evaluation checkpoints.
    3. Offline visualization — generates smoothed training loss curve plots
       matching Figure 1 (right) of the paper, which uses EMA smoothing with
       coefficient 0.9 (config.logging.smoothing_coeff: 0.9).

Config values used (from config.yaml):
    logging.log_dir: 'outputs/logs' — directory for TensorBoard events and plots
    logging.use_wandb: false — whether to enable W&B logging
    logging.smoothing_coeff: 0.9 — EMA alpha for loss curve smoothing
        Paper Fig. 1 right caption: "smoothed, 0.9 coeff."

Integration points:
    Trainer.train_step() → log_step(step, metrics)  [every step]
    Trainer.train()      → log_eval(step, metrics)  [every eval_interval steps]
    Main.run_training()  → log_training_curve(loss_history, path)  [post-training]
    Main.run_evaluation() → log_eval(0, metrics)    [standalone eval]
    Main.run_full_experiment() → close()            [in finally block]
"""

import os
from typing import Dict, List, Optional

# Use non-interactive Agg backend for headless training servers.
# Must be set before any other matplotlib imports.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Central logging component for training monitoring and result visualization.

    Provides a unified interface over TensorBoard and optionally W&B for
    logging training metrics, evaluation results, and generating publication-
    quality training curve plots.

    The EMA smoothing applied in log_step matches the paper's Figure 1 (right):
    "Training loss comparison (smoothed, 0.9 coeff.) over 3.5T tokens between
    baseline and SDPA-gated 1.7B dense models under identical hyperparameters."

    Attributes:
        log_dir: Directory for TensorBoard event files and saved plots.
            From config.logging.log_dir = 'outputs/logs'.
        use_wandb: Whether W&B logging is enabled.
            From config.logging.use_wandb = false.
        smoothing_coeff: EMA alpha for loss smoothing.
            From config.logging.smoothing_coeff = 0.9.
        writer: TensorBoard SummaryWriter instance.
        _smoothed_loss: Running EMA state for real-time loss smoothing.
            None until the first log_step call (first value initializes it).
        _step_losses: Accumulated raw per-step loss values for log_training_curve.
    """

    def __init__(
        self,
        log_dir: str = "outputs/logs",
        use_wandb: bool = False,
        smoothing_coeff: float = 0.9,
    ) -> None:
        """Initialize ExperimentLogger and set up logging backends.

        Creates the log directory, initializes TensorBoard SummaryWriter,
        and optionally initializes W&B. W&B import is deferred to avoid
        hard dependency when use_wandb=False (config default).

        Args:
            log_dir: Directory for TensorBoard event files and saved plots.
                Created with exist_ok=True if it does not exist.
                From config.logging.log_dir = 'outputs/logs'.
            use_wandb: Whether to enable W&B logging. Default False matches
                config.logging.use_wandb: false. When True, wandb.init() is
                called only if no run is already active (handles multi-process
                scenarios where rank 0 initializes W&B).
            smoothing_coeff: EMA alpha for training loss smoothing.
                Default 0.9 matches config.logging.smoothing_coeff: 0.9 and
                the paper's Fig. 1 caption: "smoothed, 0.9 coeff."
                Must be in [0, 1). Values close to 1 produce smoother curves.
        """
        self.log_dir: str = log_dir
        self.use_wandb: bool = use_wandb
        self.smoothing_coeff: float = smoothing_coeff

        # Create log directory (and any missing parent directories)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize TensorBoard SummaryWriter
        # Events are written to log_dir; view with: tensorboard --logdir=log_dir
        self.writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

        # EMA state for real-time loss smoothing in log_step.
        # None until the first log_step call — the first raw loss value
        # initializes the EMA directly (no bias correction needed for monitoring).
        self._smoothed_loss: Optional[float] = None

        # Accumulated raw per-step loss values for log_training_curve.
        # Stores raw (unsmoothed) values; log_training_curve recomputes EMA
        # from scratch over the full history for accurate offline plotting.
        self._step_losses: List[float] = []

        # Initialize W&B if requested
        # Deferred import avoids ImportError on systems without wandb installed.
        if self.use_wandb:
            try:
                import wandb  # type: ignore[import]

                # Only call wandb.init if no run is already active.
                # This handles multi-process scenarios where rank 0 initializes W&B
                # and other ranks should not create duplicate runs.
                if wandb.run is None:
                    wandb.init(dir=log_dir)
            except ImportError:
                # W&B not installed — disable silently and continue with TensorBoard only
                self.use_wandb = False

    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Log training step metrics to TensorBoard and optionally W&B.

        Called every training step from Trainer.train_step(). Applies EMA
        smoothing to the loss value for real-time monitoring, matching the
        paper's Figure 1 (right) smoothing with coefficient 0.9.

        The EMA is updated incrementally:
            smoothed_t = α * smoothed_{t-1} + (1-α) * raw_t
        where α = self.smoothing_coeff = 0.9.

        Both raw and smoothed loss are logged to TensorBoard under separate
        tags ('train/loss_raw' and 'train/loss_smoothed') to allow comparison.

        Args:
            step: Current optimization step (0-based). Used as the x-axis
                value in TensorBoard and W&B plots.
            metrics: Dict of metric name → float value. Expected keys:
                - 'loss' (required): Raw cross-entropy loss for this step.
                - 'grad_norm' (optional): Gradient norm after clipping.
                  From config training.grad_clip: 1.0.
                - 'lr' (optional): Current learning rate from scheduler.
                  Ranges from 0 to max_lr (e.g., 2e-3 for MoE, 4e-3 for dense).
                - 'lbl_loss' (optional): MoE load balance loss (MoE models only).
                  From config moe.lbl_loss_coeff: 1.0e-2.
                - 'z_loss' (optional): MoE Z-loss (MoE models only).
                  From config moe.z_loss_coeff: 1.0e-4.
                All values should be Python floats (not tensors).
        """
        raw_loss: float = float(metrics.get("loss", 0.0))

        # Update EMA smoothed loss
        # First call: initialize directly with the raw loss (no bias correction)
        if self._smoothed_loss is None:
            self._smoothed_loss = raw_loss
        else:
            self._smoothed_loss = (
                self.smoothing_coeff * self._smoothed_loss
                + (1.0 - self.smoothing_coeff) * raw_loss
            )

        # Accumulate raw loss for offline log_training_curve plotting
        self._step_losses.append(raw_loss)

        # --- TensorBoard logging ---
        # Log both raw and smoothed loss for comparison in TensorBoard
        self.writer.add_scalar("train/loss_raw", raw_loss, step)
        self.writer.add_scalar("train/loss_smoothed", self._smoothed_loss, step)

        # Log gradient norm (from Trainer's gradient clipping step)
        grad_norm: float = float(metrics.get("grad_norm", 0.0))
        self.writer.add_scalar("train/grad_norm", grad_norm, step)

        # Log current learning rate (from WarmupCosineScheduler)
        lr: float = float(metrics.get("lr", 0.0))
        self.writer.add_scalar("train/lr", lr, step)

        # Log MoE auxiliary losses if present (MoE-15A2B experiments)
        if "lbl_loss" in metrics:
            self.writer.add_scalar(
                "train/lbl_loss", float(metrics["lbl_loss"]), step
            )
        if "z_loss" in metrics:
            self.writer.add_scalar(
                "train/z_loss", float(metrics["z_loss"]), step
            )

        # Log any additional metrics not covered above
        _known_keys = frozenset({"loss", "grad_norm", "lr", "lbl_loss", "z_loss"})
        for key, value in metrics.items():
            if key not in _known_keys:
                self.writer.add_scalar(f"train/{key}", float(value), step)

        # --- W&B logging ---
        if self.use_wandb:
            try:
                import wandb  # type: ignore[import]

                wandb_metrics: Dict[str, float] = {
                    "train/loss_raw": raw_loss,
                    "train/loss_smoothed": self._smoothed_loss,
                    "train/grad_norm": grad_norm,
                    "train/lr": lr,
                }
                # Add all additional metrics with train/ prefix
                for key, value in metrics.items():
                    if key not in _known_keys:
                        wandb_metrics[f"train/{key}"] = float(value)
                # Add MoE auxiliary losses if present
                if "lbl_loss" in metrics:
                    wandb_metrics["train/lbl_loss"] = float(metrics["lbl_loss"])
                if "z_loss" in metrics:
                    wandb_metrics["train/z_loss"] = float(metrics["z_loss"])

                wandb.log(wandb_metrics, step=step)
            except Exception:
                # Silently ignore W&B errors to avoid disrupting training
                pass

    def log_eval(self, step: int, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics to TensorBoard and optionally W&B.

        Called from Trainer.train() at eval_interval steps and from
        Main.run_evaluation() for standalone evaluation runs. No EMA
        smoothing is applied — evaluation metrics are point estimates.

        All metrics are logged under the 'eval/' prefix in TensorBoard
        to separate them from training metrics in the tag hierarchy.

        Args:
            step: Current optimization step (0-based). Used as the x-axis
                value in TensorBoard and W&B plots. Pass 0 for standalone
                evaluation runs (Main.run_evaluation()).
            metrics: Dict of metric name → float value. Expected structure:
                PPL metrics (from PerplexityEvaluator.compute_all_domain_ppl):
                    'ppl/english': float  — English test set PPL
                    'ppl/chinese': float  — Chinese test set PPL
                    'ppl/code': float     — Code test set PPL
                    'ppl/math': float     — Math test set PPL
                    'ppl/law': float      — Law test set PPL
                    'ppl/literature': float — Literature test set PPL
                    'ppl/avg': float      — Average PPL across all domains
                        Paper reports this as "Avg PPL" in Tables 1, 2, 3, 4.
                Benchmark scores (from BenchmarkEvaluator.format_results):
                    'bench/hellaswag': float  — Hellaswag accuracy (10-shot)
                    'bench/mmlu': float       — MMLU accuracy (5-shot)
                    'bench/gsm8k': float      — GSM8k accuracy (5-shot)
                    'bench/humaneval': float  — HumanEval pass@1 (0-shot)
                    'bench/ceval-valid': float — C-eval accuracy (5-shot)
                    'bench/cmmlu': float      — CMMLU accuracy (5-shot)
                All values should be Python floats (not tensors).
        """
        # --- TensorBoard logging ---
        # Log all evaluation metrics under 'eval/' prefix
        for key, value in metrics.items():
            self.writer.add_scalar(f"eval/{key}", float(value), step)

        # --- W&B logging ---
        if self.use_wandb:
            try:
                import wandb  # type: ignore[import]

                wandb_metrics: Dict[str, float] = {
                    f"eval/{k}": float(v) for k, v in metrics.items()
                }
                wandb.log(wandb_metrics, step=step)
            except Exception:
                # Silently ignore W&B errors to avoid disrupting evaluation
                pass

    def log_training_curve(
        self,
        loss_history: List[float],
        save_path: str,
    ) -> None:
        """Generate and save a training loss curve plot matching Figure 1 (right).

        Creates a matplotlib figure showing both raw and EMA-smoothed training
        loss over training steps. The smoothing is recomputed from scratch over
        the full loss_history using self.smoothing_coeff (0.9), matching the
        paper's Figure 1 (right): "Training loss comparison (smoothed, 0.9 coeff.)
        over 3.5T tokens between baseline and SDPA-gated 1.7B dense models."

        The EMA is recomputed from scratch (not using self._smoothed_loss) to:
            1. Avoid initialization bias from the first value
            2. Allow plotting any externally provided history (e.g., combining
               baseline and gated model histories for side-by-side comparison)

        Args:
            loss_history: List of raw per-step loss values. Typically
                trainer.loss_history (passed from Main.run_training()).
                Can also be an externally provided list for post-hoc plotting
                or combining multiple runs for comparison (as in Fig. 1 right,
                which shows both baseline and gated model curves).
            save_path: Full path where the plot image will be saved.
                Typically os.path.join(self.log_dir, 'training_curve.png').
                The directory must exist (created by __init__ for log_dir).
                Supports any matplotlib-compatible format (.png, .pdf, .svg).

        Note:
            This method saves the figure and closes it immediately to free
            memory. It does not display the figure (Agg backend is non-interactive).
        """
        if not loss_history:
            # Nothing to plot — return silently
            return

        # Compute EMA-smoothed loss from scratch over the full history.
        # This is more accurate than using self._smoothed_loss (which has
        # initialization bias from the first value) and allows plotting
        # any externally provided history.
        smoothed_history: List[float] = self._apply_ema(
            loss_history, self.smoothing_coeff
        )

        # Build x-axis as step indices
        steps: List[int] = list(range(len(loss_history)))

        # --- Create figure ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # Raw loss: light, semi-transparent, thin line
        # Provides context for the smoothed curve without dominating the plot
        ax.plot(
            steps,
            loss_history,
            alpha=0.2,
            color="steelblue",
            linewidth=0.5,
            label="Raw loss",
        )

        # Smoothed loss: solid, prominent line — the main visual element
        # Matches the paper's Fig. 1 right which shows smoothed curves
        ax.plot(
            steps,
            smoothed_history,
            color="steelblue",
            linewidth=1.5,
            label=f"Smoothed (α={self.smoothing_coeff})",
        )

        # Axis labels and title
        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(
            f"Training Loss Curve (EMA smoothing α={self.smoothing_coeff})",
            fontsize=13,
        )

        # Legend and grid
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Tight layout to avoid clipping labels
        plt.tight_layout()

        # Save figure to disk
        # dpi=150 provides good resolution for publication-quality figures
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        # Close figure immediately to free memory
        # (important for long training runs that call this multiple times)
        plt.close(fig)

    def close(self) -> None:
        """Flush and close all logging backends.

        Called in Main.run_full_experiment() in a finally block to ensure
        resources are released even if training fails with an exception.

        Operations:
            1. Flush TensorBoard writer (writes any buffered events to disk)
            2. Close TensorBoard writer (releases file handles)
            3. Finish W&B run if active (uploads remaining data, marks run complete)

        This method is idempotent — calling it multiple times is safe because
        SummaryWriter.close() is idempotent and wandb.finish() checks for an
        active run before finishing.
        """
        # Flush and close TensorBoard writer
        # flush() ensures any buffered events are written to disk before close()
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            # Silently ignore errors during cleanup to avoid masking the
            # original exception in the finally block
            pass

        # Finish W&B run if active
        if self.use_wandb:
            try:
                import wandb  # type: ignore[import]

                # Only call finish() if a run is currently active
                # (avoids errors if W&B was never successfully initialized)
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                # Silently ignore W&B cleanup errors
                pass

    # ---------------------------------------------------------------------------
    # Private helper methods
    # ---------------------------------------------------------------------------

    @staticmethod
    def _apply_ema(values: List[float], alpha: float) -> List[float]:
        """Apply exponential moving average smoothing to a list of values.

        Computes the EMA from scratch over the full input list:
            smoothed_0 = values[0]  (initialize with first value)
            smoothed_t = alpha * smoothed_{t-1} + (1-alpha) * values[t]

        This is a static method to allow use in log_training_curve without
        depending on instance state, enabling post-hoc plotting of any
        externally provided loss history.

        Args:
            values: List of raw float values to smooth. Must be non-empty.
            alpha: EMA smoothing coefficient in [0, 1). Higher values produce
                smoother curves (more weight on history). Default in the paper
                is 0.9 (config.logging.smoothing_coeff: 0.9).

        Returns:
            List of smoothed float values, same length as input.
            The first element equals values[0] (no initialization bias).

        Example:
            >>> ExperimentLogger._apply_ema([1.0, 2.0, 3.0], alpha=0.9)
            [1.0, 1.1, 1.29]  # approximately
        """
        if not values:
            return []

        smoothed: List[float] = []
        running: float = values[0]  # Initialize with first value

        for i, v in enumerate(values):
            if i == 0:
                # First value: initialize EMA directly (no prior state)
                running = v
            else:
                # Standard EMA update: α * prev + (1-α) * current
                running = alpha * running + (1.0 - alpha) * v
            smoothed.append(running)

        return smoothed

    def get_smoothed_loss(self) -> Optional[float]:
        """Return the current EMA-smoothed loss value.

        Provides access to the running smoothed loss for external monitoring
        (e.g., early stopping logic in Trainer or progress bar display).

        Returns:
            Current EMA-smoothed loss as a float, or None if log_step has
            never been called (no loss values have been logged yet).
        """
        return self._smoothed_loss

    def get_step_losses(self) -> List[float]:
        """Return a copy of the accumulated raw per-step loss history.

        Provides access to the full loss history for external use, such as
        combining multiple run histories for comparison plots (Fig. 1 right
        shows both baseline and gated model curves on the same axes).

        Returns:
            Copy of self._step_losses as a new list. Modifying the returned
            list does not affect the internal state.
        """
        return list(self._step_losses)

    def __repr__(self) -> str:
        """Return a human-readable string representation of the logger.

        Returns:
            String summarizing the logger configuration and current state.
        """
        return (
            f"ExperimentLogger("
            f"log_dir='{self.log_dir}', "
            f"use_wandb={self.use_wandb}, "
            f"smoothing_coeff={self.smoothing_coeff}, "
            f"steps_logged={len(self._step_losses)}, "
            f"current_smoothed_loss={self._smoothed_loss}"
            f")"
        )
