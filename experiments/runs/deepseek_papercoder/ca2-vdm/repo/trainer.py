## trainer.py
import contextlib
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from config import Config
from model.ca2_vdm import Ca2VDM

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for Ca2‑VDM and its baselines (OS‑Fix, OS‑Ext).

    Handles optimizer, mixed precision, gradient accumulation, learning‑rate
    scheduling, checkpointing and logging.  The actual loss computation is
    delegated to ``Ca2VDM.training_step()``, which must return a dict with
    at least the key ``"loss"``.

    Attributes:
        model:          The complete Ca2VDM model (VAE, text encoder, transformer).
        config:         Frozen configuration dataclass.
        train_loader:   DataLoader yielding batches with keys ``latents``,
                        ``text_emb`` (optional), ``prefix_length`` and
                        ``loss_mask`` (plus optional padding mask).
        val_loader:     Optional validation DataLoader (unused in reproduction).
        optimizer:      AdamW optimizer.
        scheduler:      Optional cosine‑annealing LR scheduler.
        scaler:         ``GradScaler`` for float16 AMP; ``None`` otherwise.
        use_amp:        Boolean indicating whether mixed precision is active.
        amp_dtype:      The ``torch.dtype`` used by ``autocast``.
        accumulation_steps: Number of micro‑batches to accumulate before an
                            optimizer step.
        global_step:    Number of optimizer steps taken so far.
    """

    def __init__(
        self,
        model: Ca2VDM,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Args:
            model:          Ca2VDM instance (will be moved to the configured device).
            config:         Global configuration object.
            train_loader:   DataLoader supplying training batches.
            val_loader:     Optional DataLoader for validation (currently unused).
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Device
        self.device = torch.device(config.system.device)
        self.model = self.model.to(self.device)

        # Determine active training stage and its parameters
        self._set_stage_info()

        # Optimizer – only trainable parameters (VAE/T5 are frozen)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Optional LR scheduler (disabled by default, matching the paper’s setup)
        self.scheduler: Optional[CosineAnnealingLR] = None
        if getattr(self.config.training, "use_lr_scheduler", False):
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_steps)

        # Mixed precision
        self.use_amp, self.amp_dtype = self._configure_amp()
        self.scaler: Optional[GradScaler] = None
        if self.use_amp and self.config.training.mixed_precision == "fp16":
            self.scaler = GradScaler()

        # Gradient accumulation
        self.accumulation_steps = getattr(
            self.config.training, "gradient_accumulation_steps", 1
        )
        if self.accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")

        # Internal counters
        self.global_step: int = 0          # number of optimizer updates
        self._accum_batches: int = 0       # micro‑batches since last optimizer step

        # Logging & checkpointing
        self.log_interval = getattr(config.training, "log_interval", 50)
        self.ckpt_interval = getattr(config.training, "checkpoint_interval", 1000)
        os.makedirs(config.system.checkpoint_dir, exist_ok=True)
        os.makedirs(config.system.log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _set_stage_info(self) -> None:
        """
        Extract max steps and batch size from the active training stage
        (stage1, stage2 or video_prediction).  Raises an error if no
        stage is enabled.
        """
        tcfg = self.config.training
        if tcfg.stage1 is not None and tcfg.stage1.enabled:
            self.max_steps = tcfg.stage1.steps
            self.batch_size = tcfg.stage1.batch_size
        elif tcfg.stage2 is not None and tcfg.stage2.enabled:
            self.max_steps = tcfg.stage2.steps
            self.batch_size = tcfg.stage2.batch_size
        elif tcfg.video_prediction is not None:
            self.max_steps = tcfg.video_prediction.steps
            self.batch_size = tcfg.video_prediction.batch_size
        else:
            raise RuntimeError(
                "No enabled training stage found in config. "
                "Set stage1.enabled, stage2.enabled or provide a video_prediction sub‑config."
            )

    def _configure_amp(self) -> Tuple[bool, Optional[torch.dtype]]:
        """Translate the config’s mixed_precision string into a flag and a dtype."""
        prec = self.config.training.mixed_precision
        if prec == "fp16":
            return True, torch.float16
        elif prec == "bf16":
            return True, torch.bfloat16
        elif prec == "no":
            return False, None
        else:
            raise ValueError(f"Unsupported mixed_precision: {prec}")

    @staticmethod
    def _to_device(obj: Any, device: torch.device) -> Any:
        """Recursively move tensors (and dicts/lists/tuples) to the given device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        elif isinstance(obj, dict):
            return {k: Trainer._to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Trainer._to_device(v, device) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(Trainer._to_device(v, device) for v in obj)
        else:
            return obj

    # ------------------------------------------------------------------
    # Low‑level training step (accumulate gradients)
    # ------------------------------------------------------------------
    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Perform a single forward‑backward pass for one micro‑batch, using
        mixed precision if configured.  Gradients are **accumulated**;
        the optimizer is stepped only when the accumulated count reaches
        ``self.accumulation_steps`` (handled by :meth:`_maybe_optimizer_step`).

        Args:
            batch:  Dict with tensors as returned by the dataset collate function.

        Returns:
            The unscaled loss (averaged over the micro‑batch) as a Python float.
        """
        # Move input to device
        batch = self._to_device(batch, self.device)

        # Autocast context for mixed precision
        ctx = (
            autocast(device_type=self.device.type, dtype=self.amp_dtype)
            if self.use_amp
            else contextlib.nullcontext()
        )
        with ctx:
            loss_dict = self.model.training_step(batch)
            loss = loss_dict["loss"] / self.accumulation_steps

        # Backward pass (scaled if using float16)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._accum_batches += 1
        # Possibly perform an optimizer step
        self._maybe_optimizer_step()

        # Return the original unscaled loss for logging
        return loss_dict["loss"].item()

    def _maybe_optimizer_step(self) -> None:
        """
        If the number of accumulated micro‑batches is a multiple of
        ``self.accumulation_steps``, call ``optimizer.step()`` (with
        gradient unscaling if using FP16), zero gradients, update the LR
        scheduler, and increment ``self.global_step``.
        """
        if self._accum_batches % self.accumulation_steps != 0:
            return

        # Gradient unscaling (if using GradScaler)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        # Step the scheduler (if present)
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1
        self._accum_batches = 0   # reset for the next accumulation cycle

    # ------------------------------------------------------------------
    # Epoch‑based helper (optional, used by run_training)
    # ------------------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Iterate over one epoch of the given ``DataLoader``, calling
        :meth:`train_step` for each batch.  Stops early if
        ``self.global_step`` reaches ``self.max_steps``.

        Args:
            dataloader:  Training DataLoader for one epoch.

        Returns:
            Average loss over all (micro‑)batches processed in this epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
            n_batches += 1

            if self.global_step >= self.max_steps:
                break

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def run_training(self) -> None:
        """
        Main entry point for training.  Loops over steps (or epochs) until
        ``self.global_step`` reaches ``self.max_steps``.  Logs progress and
        saves checkpoints periodically.
        """
        logger.info("Starting training for %d steps.", self.max_steps)
        logger.info("Mixed precision: %s, accumulation steps: %d",
                     self.config.training.mixed_precision, self.accumulation_steps)

        self.model.train()
        pbar = tqdm(total=self.max_steps, desc="Training", unit="step")

        # Create an infinite iterator over the training data
        train_iter = iter(self.train_loader)

        while self.global_step < self.max_steps:
            # Fetch next batch; restart the iterator when exhausted
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Process one micro‑batch (gradient accumulation is internal)
            loss = self.train_step(batch)

            # Progress bar is only updated after an optimizer step,
            # because self.global_step changes there.
            if self._accum_batches == 0:
                pbar.update(1)
                pbar.set_postfix(
                    loss=loss,
                    lr=self.optimizer.param_groups[0]["lr"],
                )

            # Log to console / file at intervals
            if self.global_step % self.log_interval == 0 and self._accum_batches == 0:
                self._log_metrics(self.global_step, loss)

            # Save checkpoint
            if self.global_step % self.ckpt_interval == 0 and self._accum_batches == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

        pbar.close()
        # Final checkpoint
        self.save_checkpoint("final.pt")
        logger.info("Training finished.")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, filename: str) -> None:
        """
        Save the current training state to the checkpoint directory.

        Args:
            filename: Name of the checkpoint file (e.g., ``step_1000.pt``).
        """
        path = os.path.join(self.config.system.checkpoint_dir, filename)
        checkpoint: Dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config_yaml": yaml.dump(asdict(self.config)),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Restore model, optimizer, scheduler, scaler and global step from a
        previously saved checkpoint.

        Args:
            checkpoint_path: Path to the ``.pt`` file.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info("Resumed from checkpoint at step %d", self.global_step)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_metrics(self, step: int, loss: float) -> None:
        """Simple console logger; can be extended to TensorBoard/WandB."""
        msg = f"Step {step}: loss={loss:.6f}, lr={self.optimizer.param_groups[0]['lr']:.2e}"
        logger.info(msg)
        # Optionally write to a file
        if hasattr(self, "_log_file") and self._log_file is not None:
            self._log_file.write(msg + "\n")
            self._log_file.flush()

