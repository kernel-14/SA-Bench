## trainer.py
"""
Trainer for reproducing Gated Attention LLM training experiments.

Implements a robust training loop with FSDP-support through HuggingFace Accelerate,
BF16 mixed precision, cosine learning rate schedule with linear warmup, gradient
accumulation, loss spike monitoring, and checkpointing.

All hyperparameters are driven by the configuration dictionary provided by the
global `config.yaml`, faithfully following the paper's dense‑model recipes
(Tables 2–3, Section 3).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, Optional, Union, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from accelerate import Accelerator
from accelerate.utils import set_seed

# Local imports – assume these are available in the project
from utils import set_seed as set_seed_utils  # alternative if needed
from data import DataModule
from model import GPTModel


# ------------------------------------------------------------------
# Logger interface (simple implementation)
# ------------------------------------------------------------------
class Logger:
    """
    Abstract interface for logging training metrics and events.

    Subclasses may write to console, TensorBoard, W&B, etc.
    """

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Log scalar metrics at a given training step.

        Args:
            step: Global optimizer step.
            metrics: Dictionary of metric names and their values.
        """
        raise NotImplementedError

    def log_loss_spike(self, step: int, loss: float) -> None:
        """
        Record an anomalous loss spike.

        Args:
            step: Global optimizer step.
            loss: The aberrant loss value.
        """
        raise NotImplementedError

    def finalize(self) -> None:
        """Called at the end of training to flush/close resources."""
        pass


class ConsoleLogger(Logger):
    """Minimal logger that prints metrics to stdout and optionally logs to W&B."""

    def __init__(self, use_wandb: bool = False, wandb_project: Optional[str] = None):
        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb:
            wandb.init(project=wandb_project or "gated-attention-llm")

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        log_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[Step {step}] {log_str}")
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_loss_spike(self, step: int, loss: float) -> None:
        msg = f"🚨 Loss spike at step {step}: loss={loss:.2f}"
        print(msg)
        if self.use_wandb:
            wandb.log({"train/loss_spike": loss}, step=step)

    def finalize(self) -> None:
        if self.use_wandb:
            wandb.finish()


# ------------------------------------------------------------------
# Trainer class
# ------------------------------------------------------------------
class Trainer:
    """
    Distributed trainer for the GPT model with optional gating.

    Args:
        model: The full GPTModel instance (already built with the correct
               architecture, including possible gate modules and adjusted
               intermediate size).
        datamodule: DataModule providing training dataloaders.
        config: A dictionary for the 'training' section of config.yaml (or
                any mapping with fields: max_lr, min_lr, warmup_steps,
                max_steps, global_batch_size, seq_length, gradient_clip_val,
                weight_decay, adam_beta1, adam_beta2, adam_eps, mixed_precision,
                seed, and optionally save_every_steps).
        logger: A Logger instance for recording metrics.
    """

    def __init__(
        self,
        model: GPTModel,
        datamodule: DataModule,
        config: Dict,
        logger: Logger,
    ):
        self.model = model
        self.datamodule = datamodule
        self.config = config
        self.logger = logger

        self.accelerator: Optional[Accelerator] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[LambdaLR] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.global_step: int = 0

    def train(self) -> None:
        """
        Run the full training loop.

        Prepares the accelerator, creates optimizer and scheduler, and iterates
        over `max_steps` training steps. Checkpoints are saved periodically.
        """
        # ------------------------------------------------------------------
        # 1. Set seed for reproducibility
        # ------------------------------------------------------------------
        seed = self.config.get("seed", 42)
        set_seed(seed)

        # ------------------------------------------------------------------
        # 2. Retrieve training dataloader
        # ------------------------------------------------------------------
        self.train_dataloader = self.datamodule.get_train_dataloader()
        # The dataloader is expected to yield batches of {'input_ids', 'labels'}
        # We'll infer per‑device batch size from the dataloader for GA calculation.
        per_device_batch_size = self.train_dataloader.batch_size
        if per_device_batch_size is None:
            raise ValueError(
                "Training dataloader must have a non‑None `batch_size` attribute "
                "to compute gradient accumulation steps."
            )

        # ------------------------------------------------------------------
        # 3. Gradient accumulation setup
        # ------------------------------------------------------------------
        # We must know world size to compute total parallel tokens per step.
        # Accelerator will provide the number of processes after init.
        # So we compute GA after instantiating accelerator.
        # We'll first create the Accelerator and then adjust.
        # But Accelerator wants model, optimizer, etc. for preparation.
        # We'll create it without gradient_accumulation_steps and set it after.
        # Actually Accelerate allows passing gradient_accumulation_steps.
        # We'll compute after determining world_size (via torch.distributed if available).
        # Better: use a temporary distributed check?
        # Let's use `Accelerator()` to get world_size even before preparing.
        # We'll instantiate a preliminary accelerator to query world_size.
        # However, the main accelerator must be the same instance.
        # Standard approach: compute GA steps outside, then pass to Accelerator.
        # We'll use torch.distributed.get_world_size() after init.
        # For now, we'll use `torch.distributed.is_initialized()` after we init.
        # In the __init__ we don't have distributed init yet. We'll do init_process_group()?
        # Usually, the main process initializes distributed before Trainer creation.
        # We'll assume that torch.distributed is initialized (via torchrun or accelerate launch).
        # Thus we can query world_size now.
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1  # single‑GPU fallback

        global_batch_size = self.config.get("global_batch_size", 1024)
        gradient_accumulation_steps = max(
            1, global_batch_size // (per_device_batch_size * world_size)
        )
        # Warn if global_batch_size isn't perfectly divisible
        if global_batch_size % (per_device_batch_size * world_size) != 0:
            print(
                f"Warning: global_batch_size {global_batch_size} not divisible by "
                f"(per_device_batch_size {per_device_batch_size} * world_size {world_size}). "
                f"Effective batch size will be {per_device_batch_size * world_size * gradient_accumulation_steps}."
            )

        # ------------------------------------------------------------------
        # 4. Initialize Accelerator
        # ------------------------------------------------------------------
        self.accelerator = Accelerator(
            mixed_precision=self.config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # ------------------------------------------------------------------
        # 5. Optimizer & scheduler
        # ------------------------------------------------------------------
        # Collect all trainable parameters
        param_groups = self._get_optimizer_param_groups()

        self.optimizer = AdamW(
            param_groups,
            lr=self.config.get("max_lr", 4e-3),
            betas=(self.config.get("adam_beta1", 0.9), self.config.get("adam_beta2", 0.95)),
            eps=self.config.get("adam_eps", 1e-8),
            weight_decay=self.config.get("weight_decay", 0.1),
        )

        # Learning rate schedule: linear warmup + cosine decay
        max_steps = self.config.get("max_steps", 100000)
        warmup_steps = self.config.get("warmup_steps", 1000)
        min_lr = self.config.get("min_lr", 3e-5)

        def lr_lambda(current_step: int) -> float:
            """Cosine with warmup."""
            if current_step < warmup_steps:
                # Linear warmup from 0 to max_lr
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return min_lr / self.config["max_lr"] + 0.5 * (
                1.0 - min_lr / self.config["max_lr"]
            ) * (1.0 + math.cos(progress * math.pi))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # ------------------------------------------------------------------
        # 6. Prepare everything with Accelerator
        # ------------------------------------------------------------------
        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.scheduler
        )

        self.accelerator.wait_for_everyone()

        # ------------------------------------------------------------------
        # 7. Training loop
        # ------------------------------------------------------------------
        self.model.train()

        # For logging loss per optimizer step
        step_loss = 0.0

        # Create an infinite iterator over the dataloader
        train_iter = iter(self.train_dataloader)

        for batch_step in range(max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                # Data exhausted – reset iterator (in case dataset is finite)
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            # Perform one micro‑batch within gradient accumulation
            with self.accelerator.accumulate(self.model):
                loss = self._training_step(batch)
                # Accumulate loss for logging (scaled internally by Accelerate)
                step_loss += loss.detach().item()

            # After the accumulate block, if gradients were synced (i.e., an optimizer step occurred),
            # log metrics and step the global counter.
            if self.accelerator.sync_gradients:
                self.global_step += 1
                avg_loss = step_loss / gradient_accumulation_steps
                step_loss = 0.0

                # Log learning rate
                lr = self.scheduler.get_last_lr()[0]

                self.logger.log_metrics(
                    self.global_step,
                    {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                    },
                )

                # Checkpoint saving
                save_every = self.config.get("save_every_steps", 5000)
                if self.global_step % save_every == 0 or self.global_step >= max_steps:
                    self._save_checkpoint(self.global_step)

        # ------------------------------------------------------------------
        # 8. Finish
        # ------------------------------------------------------------------
        self.accelerator.wait_for_everyone()
        self._save_checkpoint(final=True)
        self.logger.finalize()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single forward/backward pass on a micro‑batch.

        Args:
            batch: A dictionary with at least 'input_ids' and 'labels'.

        Returns:
            The computed loss tensor (scalar).
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask", None)

        # The model's forward method returns (logits, loss) when labels are provided.
        # We ignore logits here.
        _, loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Loss spike detection
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.log_loss_spike(self.global_step, loss.item() if not loss.isnan() else float("nan"))
            raise RuntimeError(f"Training diverged: loss is NaN/Inf at step {self.global_step}.")

        self.accelerator.backward(loss)

        # Gradient clipping (only if a clip value is set)
        clip_val = self.config.get("gradient_clip_val", None)
        if clip_val is not None and clip_val > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), clip_val)

        # Optimizer and scheduler steps are handled by the accumulate context manager,
        # so we don't call them here.
        return loss

    def _get_optimizer_param_groups(self) -> Union[list, Dict[str, List[nn.Parameter]]]:
        """
        Create parameter groups for the optimizer. By default, all parameters are in a single
        group. Subclasses can override to implement weight decay exceptions.

        Returns:
            A list of dictionaries suitable for AdamW constructor.
        """
        # Standard: all parameters, no special weight decay handling.
        # If we wanted to separate biases/LayerNorm from weight decay, we could implement that.
        return self.model.parameters()

    def _save_checkpoint(self, step: Optional[int] = None, final: bool = False) -> None:
        """
        Save the current training state.

        Args:
            step: Global step number; used for naming if not final.
            final: If True, saves the final checkpoint regardless of step.
        """
        output_dir = self.config.get("output_dir", "./checkpoints")
        os.makedirs(output_dir, exist_ok=True)

        # Use accelerator's save_state which is FSDP‑safe.
        if final:
            save_path = os.path.join(output_dir, "final")
        else:
            save_path = os.path.join(output_dir, f"step-{step:07d}")

        self.accelerator.save_state(save_path)

        # Additionally save the unwrapped model's state dict for easy evaluation.
        # This may be large; use with care.
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "model_state_dict.pt"))

        if self.accelerator.is_main_process:
            print(f"Checkpoint saved to {save_path}")

