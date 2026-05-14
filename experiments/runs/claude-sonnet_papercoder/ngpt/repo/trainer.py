## trainer.py
"""Training orchestration for GPT and nGPT experiments.

This module implements the Trainer class that manages the complete training
loop for both the baseline GPT and the normalized nGPT models. It handles:

    - Training loop with gradient accumulation
    - Mixed precision (bfloat16) via torch.autocast
    - Post-step weight normalization (nGPT-specific, critical correctness requirement)
    - Learning rate scheduling (cosine annealing with optional warmup)
    - Validation and checkpointing
    - TensorBoard logging
    - DDP (DistributedDataParallel) support

Critical ordering invariant (nGPT):
    optimizer.step() → _post_step_normalize() → next forward()
    Violating this order trains nGPT with un-normalized weights.

Typical usage:
    from config import Config
    from data import OpenWebTextDataset
    from model import nGPTModel
    from trainer import Trainer

    config = Config.ngpt_500m(context_length=4096)
    dataset = OpenWebTextDataset(config)
    model = nGPTModel(config).to(device)
    trainer = Trainer(config, model, dataset)
    trainer.train()
"""

import contextlib
import logging
import math
import os
import pathlib
from typing import Dict
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
import torch.nn.utils
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]

from config import Config
from data import OpenWebTextDataset
from model import GPTModel
from model import nGPTModel
from utils import AverageMeter
from utils import load_checkpoint
from utils import save_checkpoint
from utils import setup_logger


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = setup_logger("trainer")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the training loop for GPT and nGPT models.

    Manages the full training lifecycle: optimizer setup, LR scheduling,
    gradient accumulation, mixed-precision forward/backward passes, the
    critical post-step normalization hook for nGPT, validation, checkpointing,
    and TensorBoard logging.

    The most important correctness requirement is the ordering guarantee for
    nGPT: ``optimizer.step()`` → ``_post_step_normalize()`` → next
    ``forward()``. This is enforced by calling ``_post_step_normalize()``
    immediately after ``_train_step()`` returns in the main training loop.

    Attributes:
        config: Experiment configuration.
        model: The GPT or nGPT model (may be DDP-wrapped).
        dataset: OpenWebText dataset providing training and validation batches.
        optimizer: Configured Adam or AdamW optimizer.
        scheduler: Cosine annealing LR scheduler with optional warmup.
        scaler: GradScaler for mixed precision (disabled for bfloat16).
        step: Current training step (0-indexed, restored from checkpoint).
        best_val_loss: Best validation loss seen so far.
        writer: TensorBoard SummaryWriter (main process only).
        is_ngpt: True if the model is an nGPTModel instance.
        is_main_process: True if this is the main (rank 0) process.
        is_distributed: True if running under DDP.
        device: The compute device (cuda or cpu).
    """

    def __init__(
        self,
        config: Config,
        model: Union[GPTModel, nGPTModel],
        dataset: OpenWebTextDataset,
    ) -> None:
        """Initialize the Trainer.

        Sets up the optimizer, LR scheduler, mixed-precision scaler,
        TensorBoard writer, and all mutable training state. Detects DDP
        environment via the LOCAL_RANK environment variable.

        Args:
            config: Experiment configuration. All hyperparameters are sourced
                from this object (no hardcoded values).
            model: The model to train. May be a raw GPTModel/nGPTModel or a
                DDP-wrapped version. Must already be moved to the target device.
            dataset: OpenWebTextDataset providing get_batch() and
                get_val_loader() methods.
        """
        self.config: Config = config
        self.model: Union[GPTModel, nGPTModel] = model
        self.dataset: OpenWebTextDataset = dataset

        # ----------------------------------------------------------------
        # DDP detection
        # ----------------------------------------------------------------
        local_rank_str: Optional[str] = os.environ.get("LOCAL_RANK")
        local_rank: int = int(local_rank_str) if local_rank_str is not None else -1
        self.is_distributed: bool = local_rank >= 0
        # Main process: rank 0 in distributed, or the only process otherwise
        self.is_main_process: bool = local_rank <= 0

        # ----------------------------------------------------------------
        # Device detection
        # ----------------------------------------------------------------
        # Infer device from model parameters (works for both DDP and non-DDP)
        try:
            self.device: torch.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----------------------------------------------------------------
        # Unwrap DDP to access the raw model
        # ----------------------------------------------------------------
        self._raw_model: Union[GPTModel, nGPTModel] = (
            model.module  # type: ignore[union-attr]
            if hasattr(model, "module")
            else model
        )

        # ----------------------------------------------------------------
        # Model type detection (must use unwrapped model)
        # ----------------------------------------------------------------
        self.is_ngpt: bool = isinstance(self._raw_model, nGPTModel)

        # ----------------------------------------------------------------
        # Optimizer setup
        # ----------------------------------------------------------------
        if self.is_ngpt:
            # nGPT: Adam with weight_decay=0.0 (paper Table 3)
            self.optimizer: Optimizer = self._raw_model.configure_optimizer(
                lr=config.learning_rate,
                betas=config.betas,
            )
        else:
            # GPT: AdamW with weight_decay=0.1 (paper Table 3)
            self.optimizer = self._raw_model.configure_optimizer(
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas,
            )

        # ----------------------------------------------------------------
        # LR scheduler
        # ----------------------------------------------------------------
        self.scheduler: LambdaLR = self._build_scheduler()

        # ----------------------------------------------------------------
        # Mixed precision scaler
        # ----------------------------------------------------------------
        # bfloat16 does not require loss scaling (wider dynamic range than
        # float16), so we disable the scaler. The autocast context still
        # provides bfloat16 compute. We keep the GradScaler API for
        # uniformity across float16 and bfloat16 code paths.
        use_cuda: bool = self.device.type == "cuda"
        scaler_enabled: bool = (
            use_cuda and config.dtype == "float16"
        )
        self.scaler: torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler(
            enabled=scaler_enabled
        )

        # ----------------------------------------------------------------
        # Training state
        # ----------------------------------------------------------------
        self.step: int = 0
        self.best_val_loss: float = float("inf")

        # ----------------------------------------------------------------
        # Logging (main process only)
        # ----------------------------------------------------------------
        if self.is_main_process:
            _ensure_dir(config.log_dir)
            _ensure_dir(config.checkpoint_dir)
            self.writer: Optional[SummaryWriter] = SummaryWriter(
                log_dir=config.log_dir
            )
        else:
            self.writer = None

        self._logger: logging.Logger = setup_logger("trainer", config.log_dir)

        # ----------------------------------------------------------------
        # Running loss meter for logging
        # ----------------------------------------------------------------
        self._train_loss_meter: AverageMeter = AverageMeter("train_loss")

        self._logger.info(
            "Trainer initialized: model_type=%s, is_ngpt=%s, "
            "is_distributed=%s, is_main_process=%s, device=%s, "
            "n_params=%s",
            config.model_type,
            self.is_ngpt,
            self.is_distributed,
            self.is_main_process,
            self.device,
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M",
        )

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop from self.step to config.max_steps.

        Implements the main training loop with the following per-step sequence:
            1. Sample training batch (with gradient accumulation)
            2. Forward + backward pass (bfloat16 autocast)
            3. Gradient clipping
            4. Optimizer step
            5. [nGPT only] Post-step weight normalization  ← critical ordering
            6. LR scheduler step
            7. Logging (every config.log_interval steps)
            8. Validation + checkpointing (every config.eval_interval steps)

        The training loop resumes from self.step if a checkpoint was loaded
        via _load_checkpoint() before calling train().
        """
        self._logger.info(
            "Starting training from step %d to %d.",
            self.step,
            self.config.max_steps,
        )

        # Run initial validation to establish baseline loss
        if self.step == 0 and self.is_main_process:
            self._logger.info("Running initial validation at step 0...")
            val_loss: float = self._validate()
            self._log_metrics(
                {"val/loss": val_loss, "val/perplexity": math.exp(val_loss)},
                step=0,
            )
            self._logger.info("Initial val_loss=%.4f", val_loss)

        # Set model to training mode
        self.model.train()

        while self.step < self.config.max_steps:
            # ----------------------------------------------------------------
            # Training step (with gradient accumulation)
            # ----------------------------------------------------------------
            train_loss, grad_norm = self._train_step()

            # ----------------------------------------------------------------
            # CRITICAL: Post-step normalization for nGPT
            # Must happen immediately after optimizer.step() (inside
            # _train_step) and before the next forward pass.
            # ----------------------------------------------------------------
            if self.is_ngpt:
                self._post_step_normalize()

            # ----------------------------------------------------------------
            # Update running loss meter
            # ----------------------------------------------------------------
            self._train_loss_meter.update(train_loss)

            # ----------------------------------------------------------------
            # Logging
            # ----------------------------------------------------------------
            if self.step % self.config.log_interval == 0 and self.is_main_process:
                current_lr: float = self.scheduler.get_last_lr()[0]
                self._log_metrics(
                    {
                        "train/loss": train_loss,
                        "train/loss_avg": self._train_loss_meter.avg,
                        "train/lr": current_lr,
                        "train/grad_norm": grad_norm,
                        "train/perplexity": math.exp(
                            min(train_loss, 20.0)
                        ),  # cap to avoid overflow
                    },
                    step=self.step,
                )

            # ----------------------------------------------------------------
            # Validation and checkpointing
            # ----------------------------------------------------------------
            if self.step % self.config.eval_interval == 0 and self.step > 0:
                val_loss = self._validate()

                if self.is_main_process:
                    self._log_metrics(
                        {
                            "val/loss": val_loss,
                            "val/perplexity": math.exp(min(val_loss, 20.0)),
                        },
                        step=self.step,
                    )
                    self._logger.info(
                        "step=%d | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
                        self.step,
                        self._train_loss_meter.avg,
                        val_loss,
                        self.scheduler.get_last_lr()[0],
                    )

                    # Save checkpoint if validation loss improved
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(val_loss)

                # Reset running loss meter after each eval
                self._train_loss_meter.reset()

                # Restore training mode after validation
                self.model.train()

            self.step += 1

        # ----------------------------------------------------------------
        # Final validation at end of training
        # ----------------------------------------------------------------
        self._logger.info("Training complete at step %d.", self.step)
        final_val_loss: float = self._validate()

        if self.is_main_process:
            self._log_metrics(
                {
                    "val/loss": final_val_loss,
                    "val/perplexity": math.exp(min(final_val_loss, 20.0)),
                },
                step=self.step,
            )
            self._logger.info(
                "Final val_loss=%.4f (best=%.4f)",
                final_val_loss,
                self.best_val_loss,
            )
            # Save final checkpoint regardless of whether it's the best
            self._save_checkpoint(final_val_loss, is_final=True)

            if self.writer is not None:
                self.writer.close()

    def load_checkpoint(self, path: str) -> None:
        """Public interface to restore training state from a checkpoint.

        Loads model weights, optimizer state, scheduler state, and training
        step from a checkpoint file. For nGPT, re-normalizes weights after
        loading to ensure consistency.

        Args:
            path: Full path to the checkpoint file (e.g.,
                "outputs/checkpoints/best.pt").
        """
        self._load_checkpoint(path)

    # -----------------------------------------------------------------------
    # Private training methods
    # -----------------------------------------------------------------------

    def _train_step(self) -> tuple:
        """Execute one training step with gradient accumulation.

        Accumulates gradients over config.gradient_accumulation_steps
        micro-batches, then performs a single optimizer step. Uses DDP's
        no_sync() context for all but the last micro-step to avoid redundant
        all-reduce operations.

        Returns:
            A tuple (loss, grad_norm) where:
                - loss: Mean training loss over all micro-batches (float).
                - grad_norm: Gradient norm before clipping (float).
        """
        grad_accum_steps: int = self.config.gradient_accumulation_steps
        device_type: str = self.device.type

        # Determine autocast dtype
        amp_dtype: torch.dtype = (
            torch.bfloat16
            if self.config.dtype == "bfloat16"
            else torch.float16
            if self.config.dtype == "float16"
            else torch.float32
        )
        use_autocast: bool = device_type == "cuda" and self.config.use_amp

        # Zero gradients before accumulation
        self.optimizer.zero_grad(set_to_none=True)

        accumulated_loss: float = 0.0

        for micro_step in range(grad_accum_steps):
            # Sample a micro-batch
            tokens, targets = self.dataset.get_batch(
                split="train",
                device=str(self.device),
                batch_size=self.config.micro_batch_size,
                context_length=self.config.context_length,
            )

            # DDP gradient sync: use no_sync() for all but the last micro-step
            # to avoid redundant all-reduce operations during accumulation.
            is_last_micro_step: bool = micro_step == grad_accum_steps - 1
            sync_context = (
                contextlib.nullcontext()
                if (not self.is_distributed or is_last_micro_step)
                else self.model.no_sync()  # type: ignore[union-attr]
            )

            with sync_context:
                # Forward pass with mixed precision
                if use_autocast:
                    with torch.autocast(device_type=device_type, dtype=amp_dtype):
                        _, loss = self.model(tokens, targets)
                else:
                    _, loss = self.model(tokens, targets)

                # Normalize loss by accumulation steps for correct gradient scale
                loss_normalized: torch.Tensor = loss / grad_accum_steps

                # Backward pass
                self.scaler.scale(loss_normalized).backward()

            accumulated_loss += loss.item()

        # Average loss over micro-batches
        mean_loss: float = accumulated_loss / grad_accum_steps

        # NaN/Inf detection — log warning but continue (don't crash training)
        if not math.isfinite(mean_loss):
            self._logger.warning(
                "Non-finite loss detected at step %d: loss=%.4f. "
                "Skipping optimizer step.",
                self.step,
                mean_loss,
            )
            self.optimizer.zero_grad(set_to_none=True)
            return mean_loss, 0.0

        # Unscale gradients before clipping (required when using GradScaler)
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping (config.yaml training.grad_clip: 1.0)
        grad_norm: float = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.grad_clip,
        ).item()

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # LR scheduler step (once per training step, not per micro-step)
        self.scheduler.step()

        return mean_loss, grad_norm

    def _post_step_normalize(self) -> None:
        """Normalize all nGPT weight matrices after an optimizer step.

        This is the critical nGPT-specific hook that must be called
        immediately after every optimizer.step() call. It snaps all
        NormLinear and NormEmbedding weight tensors back to the unit
        hypersphere, preventing accumulated floating-point drift.

        Implementation details:
            - Calls nGPTModel.normalize_all_weights() on the unwrapped model.
            - normalize_all_weights() uses param.data.copy_() for in-place
              modification, preserving the optimizer's momentum buffer
              references to the same storage.
            - Wrapped in torch.no_grad() inside normalize_all_weights() to
              prevent autograd tracking.
            - In DDP, each process normalizes its own replica independently.
              Since all replicas start from the same parameters (DDP syncs
              gradients), they remain synchronized after normalization.

        This method is a no-op if self.is_ngpt is False.
        """
        if not self.is_ngpt:
            return

        # Use the unwrapped model to access nGPTModel.normalize_all_weights()
        # isinstance(ddp_model, nGPTModel) would be False, so we must unwrap.
        raw_model: nGPTModel = self._raw_model  # type: ignore[assignment]
        raw_model.normalize_all_weights()

    def _validate(self) -> float:
        """Evaluate the model on the validation set.

        Runs the model in eval mode over config.eval_steps validation batches
        using a fixed random seed (via dataset.get_val_loader) for
        reproducible evaluation. Restores train mode in a finally block.

        In distributed training, validation is run on all processes but only
        the main process logs the result. For simplicity, we do not all-reduce
        the validation loss across processes — the main process result is
        sufficient for monitoring.

        Returns:
            Mean cross-entropy validation loss over config.eval_steps batches.
        """
        device_type: str = self.device.type
        amp_dtype: torch.dtype = (
            torch.bfloat16
            if self.config.dtype == "bfloat16"
            else torch.float16
            if self.config.dtype == "float16"
            else torch.float32
        )
        use_autocast: bool = device_type == "cuda" and self.config.use_amp

        # Set eval mode on the unwrapped model
        self._raw_model.eval()

        total_loss: float = 0.0
        n_batches: int = 0

        try:
            with torch.no_grad():
                for tokens, targets in self.dataset.get_val_loader(
                    steps=self.config.eval_steps,
                    device=str(self.device),
                    batch_size=self.config.micro_batch_size,
                    context_length=self.config.context_length,
                ):
                    if use_autocast:
                        with torch.autocast(
                            device_type=device_type, dtype=amp_dtype
                        ):
                            _, loss = self.model(tokens, targets)
                    else:
                        _, loss = self.model(tokens, targets)

                    total_loss += loss.item()
                    n_batches += 1

        finally:
            # Always restore training mode, even if an exception occurs
            self._raw_model.train()

        if n_batches == 0:
            self._logger.warning(
                "Validation produced zero batches. Returning inf loss."
            )
            return float("inf")

        return total_loss / n_batches

    # -----------------------------------------------------------------------
    # Checkpoint methods
    # -----------------------------------------------------------------------

    def _save_checkpoint(
        self,
        val_loss: float,
        is_final: bool = False,
    ) -> None:
        """Save training state to a checkpoint file.

        Only the main process saves checkpoints (guarded by is_main_process).
        Saves the unwrapped model's state_dict to ensure the checkpoint is
        loadable without DDP.

        Two files are saved:
            1. A step-specific checkpoint:
               checkpoint_step{step}_loss{val_loss:.4f}.pt
            2. A "best.pt" copy (overwritten on each improvement) for easy
               access to the best model.

        Args:
            val_loss: Current validation loss, included in the filename and
                state dict for reference.
            is_final: If True, saves as "final.pt" in addition to the
                step-specific file.
        """
        if not self.is_main_process:
            return

        _ensure_dir(self.config.checkpoint_dir)

        # Build state dict with all training state needed for resumption
        state: dict = {
            "step": self.step,
            "model_state_dict": self._raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
        }

        # Step-specific checkpoint filename
        ckpt_filename: str = (
            f"checkpoint_step{self.step:07d}_loss{val_loss:.4f}.pt"
        )
        ckpt_path: str = os.path.join(self.config.checkpoint_dir, ckpt_filename)
        save_checkpoint(state, ckpt_path)
        self._logger.info(
            "Saved checkpoint: %s (val_loss=%.4f)", ckpt_path, val_loss
        )

        # Always overwrite "best.pt" when this method is called from the
        # improvement branch in train()
        best_path: str = os.path.join(self.config.checkpoint_dir, "best.pt")
        save_checkpoint(state, best_path)
        self._logger.info("Updated best checkpoint: %s", best_path)

        # Save "final.pt" at end of training
        if is_final:
            final_path: str = os.path.join(
                self.config.checkpoint_dir, "final.pt"
            )
            save_checkpoint(state, final_path)
            self._logger.info("Saved final checkpoint: %s", final_path)

    def _load_checkpoint(self, path: str) -> None:
        """Restore training state from a checkpoint file.

        Loads model weights, optimizer state, scheduler state, and training
        step. For nGPT, re-normalizes weights after loading to ensure
        consistency (in case the checkpoint was saved before normalization).

        Args:
            path: Full path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        self._logger.info("Loading checkpoint from '%s'...", path)

        # load_checkpoint from utils always loads to CPU first
        checkpoint: dict = load_checkpoint(path)

        # Restore model weights (unwrapped model)
        self._raw_model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler state
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training step and best loss
        self.step = checkpoint.get("step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        # Move model back to the correct device (load_checkpoint loads to CPU)
        self._raw_model.to(self.device)

        # nGPT-specific: re-normalize weights after loading
        # This is a safety measure in case the checkpoint was saved before
        # normalization, or if there was any floating-point drift.
        if self.is_ngpt:
            raw_model: nGPTModel = self._raw_model  # type: ignore[assignment]
            raw_model.normalize_all_weights()
            self._logger.info(
                "Re-normalized nGPT weights after checkpoint load."
            )

        self._logger.info(
            "Checkpoint loaded: step=%d, best_val_loss=%.4f",
            self.step,
            self.best_val_loss,
        )

    # -----------------------------------------------------------------------
    # Scheduler builder
    # -----------------------------------------------------------------------

    def _build_scheduler(self) -> LambdaLR:
        """Build the learning rate scheduler.

        Implements the paper's LR schedule (Table 3):
            - GPT: Linear warmup for warmup_steps=2000, then cosine decay to 0.
            - nGPT: No warmup (warmup_steps=0), pure cosine decay from step 0.

        The schedule decays the LR to exactly 0 at max_steps, matching
        config.yaml training.gpt.final_lr: 0.0 and training.ngpt.final_lr: 0.0.

        Returns:
            A LambdaLR scheduler that applies the computed multiplier to the
            optimizer's base learning rate at each step.
        """
        warmup_steps: int = self.config.warmup_steps
        max_steps: int = self.config.max_steps

        def lr_lambda(current_step: int) -> float:
            """Compute LR multiplier for the given step.

            Args:
                current_step: The current training step (0-indexed).

            Returns:
                LR multiplier in [0, 1]. The actual LR is
                base_lr * lr_lambda(step).
            """
            # Phase 1: Linear warmup (GPT only; nGPT has warmup_steps=0)
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            # Phase 2: Cosine annealing to 0
            # progress ∈ [0, 1]: 0 at start of cosine phase, 1 at max_steps
            decay_steps: int = max_steps - warmup_steps
            progress: float = float(current_step - warmup_steps) / float(
                max(1, decay_steps)
            )
            # Clamp to [0, 1] to handle steps beyond max_steps gracefully
            progress = min(1.0, max(0.0, progress))
            # Cosine decay: 1.0 at progress=0, 0.0 at progress=1
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Write metrics to TensorBoard and the Python logger.

        Only the main process writes metrics (guarded by is_main_process).
        TensorBoard events are written via self.writer.add_scalar().

        Args:
            metrics: Dictionary mapping metric names to float values.
                Example: {"train/loss": 2.34, "train/lr": 1e-3}.
            step: The training step to associate with these metrics. If None,
                uses self.step.
        """
        if not self.is_main_process:
            return

        log_step: int = step if step is not None else self.step

        # Write to TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                if math.isfinite(value):
                    self.writer.add_scalar(key, value, log_step)

        # Write to Python logger (concise format for console/file)
        metric_str: str = " | ".join(
            f"{k}={v:.6f}" for k, v in metrics.items()
        )
        self._logger.info("step=%07d | %s", log_step, metric_str)


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create a directory and all parent directories if they do not exist.

    Args:
        path: The directory path to create.
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
