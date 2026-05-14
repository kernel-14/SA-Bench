# training/trainer.py

"""
Training orchestrator for NaViL using DeepSpeed.

This module implements the ``Trainer`` class, which manages one complete
training stage (S1.1, S1.2 or S2).  It handles:

* Parameter freezing according to stage‑specific patterns
* AdamW optimizer and learning rate scheduler construction
* DeepSpeed integration (ZeRO, mixed precision, gradient accumulation)
* Efficient data loading with correct per‑device batch size derived from
  the global batch size
* Checkpointing via DeepSpeed’s native mechanism
* Logging of loss, learning rate and throughput

All hyper‑parameters are drawn from the ``TrainingConfig`` and its nested
``StageConfig`` instances, as defined in ``config.yaml``.

Usage (from ``main.py``)::

    trainer = Trainer(model=navil_model, config=training_config, device="cuda")
    trainer.train_stage("s1_1", dataset_s1_1)
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import deepspeed
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import TrainingConfig, StageConfig
from data.dataset import MultiModalDataset
from models.navil_model import NaViLModel
from training.stages import apply_freeze_pattern
from utils.logging import get_logger  # assumed in design


# ---------------------------------------------------------------------------
# Helper: build a minimal DeepSpeed configuration
# ---------------------------------------------------------------------------

def _get_default_deepspeed_config(
    stage_config: StageConfig,
    training_config: TrainingConfig,
    world_size: int,
) -> Dict[str, Any]:
    """
    Build a reasonable DeepSpeed configuration dict when the user does not
    supply an external one.  The configuration enables:

    * bfloat16 mixed precision (matching the paper’s default)
    * ZeRO stage 2 (good trade‑off for models up to ~10B)
    * Gradient accumulation from the training config
    * No pipeline parallelism
    """
    grad_accum = training_config.gradient_accumulation_steps

    return {
        "train_batch_size": stage_config.global_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": stage_config.learning_rate,
                "betas": [training_config.optimizer.beta1, training_config.optimizer.beta2],
                "eps": training_config.optimizer.epsilon,
                "weight_decay": stage_config.weight_decay,
            },
        },
        "scheduler": {
            "type": "WarmupLR",  # will be overridden by our custom scheduler
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": stage_config.learning_rate,
                "warmup_num_steps": stage_config.warmup_steps,
            },
        },
        "wall_clock_breakdown": False,
    }


# ---------------------------------------------------------------------------
# Learning rate schedule builders
# ---------------------------------------------------------------------------

def _build_constant_warmup_schedule(
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
) -> Callable[[int], float]:
    """
    Returns a lambda ``step -> multiplier`` for constant‑with‑warmup schedule.
    """
    def schedule(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return schedule


def _build_cosine_decay_schedule(
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float = 0.0,
) -> Callable[[int], float]:
    """Warmup then cosine decay from peak_lr to min_lr."""
    def schedule(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return (min_lr + (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress)) / 2.0) / peak_lr
    return schedule


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Handles a single training stage for the NaViL model.

    Args:
        model: The fully assembled NaViL model (before DeepSpeed wrapping).
        config: Training configuration loaded from ``config.yaml``.
        device: Device string (e.g. ``"cuda"``).  Used mainly before
            DeepSpeed initialisation; afterwards device management is
            delegated to the engine.
        deepspeed_config_path: Optional path to a DeepSpeed configuration
            JSON file.  If ``None`` (default), a sensible built‑in
            configuration is used.
    """

    def __init__(
        self,
        model: NaViLModel,
        config: TrainingConfig,
        device: str,
        deepspeed_config_path: Optional[str] = None,
    ) -> None:
        self.original_model = model
        self.config = config
        self.device = device
        self.deepspeed_config_path = deepspeed_config_path

        # Logger – shared across all stages, tag with process rank if distributed.
        self.logger = get_logger(__name__)
        self.rank = 0  # updated after distributed init
        self.world_size = 1

        # These will be set during train_stage().
        self.engine: Optional[deepspeed.DeepSpeedEngine] = None
        self.current_stage: Optional[str] = None
        self.stage_config: Optional[StageConfig] = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def train_stage(
        self,
        stage: str,
        dataset: MultiModalDataset,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Run the full training pipeline for a single stage.

        Args:
            stage: Stage identifier – one of ``"s1_1"``, ``"s1_2"``, ``"s2"``.
            dataset: Pre‑configured ``MultiModalDataset`` providing the
                training data for this stage.
            checkpoint_path: If given, resume training from a DeepSpeed
                checkpoint directory.
        """
        if stage not in self.config.stages:
            raise ValueError(f"Unknown stage '{stage}'.  Available: {list(self.config.stages.keys())}")
        self.current_stage = stage
        self.stage_config = self.config.stages[stage]

        # ------------------------------------------------------------------
        # 1. Parameter freezing
        # ------------------------------------------------------------------
        self.logger.info("Applying freeze pattern for stage '%s': %s", stage, self.stage_config.freeze_pattern)
        apply_freeze_pattern(self.original_model, self.stage_config.freeze_pattern)

        # ------------------------------------------------------------------
        # 2. Build optimizer (only trainable parameters)
        # ------------------------------------------------------------------
        trainable_params = [p for p in self.original_model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=self.stage_config.learning_rate,
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
            eps=self.config.optimizer.epsilon,
            weight_decay=self.stage_config.weight_decay,
        )

        # ------------------------------------------------------------------
        # 3. Build learning rate scheduler
        # ------------------------------------------------------------------
        total_steps = self.stage_config.steps
        warmup_steps = self.stage_config.warmup_steps
        schedule_type = self.stage_config.lr_schedule

        if schedule_type == "constant_with_warmup":
            lr_lambda = _build_constant_warmup_schedule(warmup_steps, total_steps, self.stage_config.learning_rate)
        elif schedule_type == "cosine_decay":
            lr_lambda = _build_cosine_decay_schedule(warmup_steps, total_steps, self.stage_config.learning_rate)
        else:
            raise ValueError(f"Unknown LR schedule '{schedule_type}'")

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

        # ------------------------------------------------------------------
        # 4. Prepare DeepSpeed configuration and initialise engine
        # ------------------------------------------------------------------
        # Determine world size (if not already initialised)
        try:
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        except RuntimeError:
            self.world_size = 1
            self.rank = 0

        ds_config = self._resolve_deepspeed_config()

        self.logger.info("Initialising DeepSpeed engine …")
        self.engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.original_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config_params=ds_config,
            model_parameters=trainable_params,
        )

        # The engine now manages the optimizer and scheduler; the objects
        # returned by `initialize` are the same instances updated.
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # ------------------------------------------------------------------
        # 5. Optionally resume from checkpoint
        # ------------------------------------------------------------------
        start_step = 0
        if checkpoint_path is not None:
            self.logger.info("Resuming from checkpoint: %s", checkpoint_path)
            load_path, client_state = self.engine.load_checkpoint(checkpoint_path)
            start_step = client_state.get("step", 0) if client_state else 0
            self.logger.info("Resumed at step %d", start_step)

        # ------------------------------------------------------------------
        # 6. Build DataLoader
        # ------------------------------------------------------------------
        per_device_batch_size = (
            self.stage_config.global_batch_size
            // self.world_size
            // self.config.gradient_accumulation_steps
        )
        if per_device_batch_size <= 0:
            raise ValueError(
                f"global_batch_size ({self.stage_config.global_batch_size}) too small "
                f"for world_size={self.world_size} and grad_accum={self.config.gradient_accumulation_steps}"
            )

        self.logger.info("Per‑device batch size: %d", per_device_batch_size)

        dataloader = DataLoader(
            dataset,
            batch_size=per_device_batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            collate_fn=MultiModalDataset.collate_fn,
        )

        # ------------------------------------------------------------------
        # 7. Training loop
        # ------------------------------------------------------------------
        self.logger.info(
            "Starting training for stage '%s' (%d steps).",
            stage,
            total_steps,
        )
        self._train_loop(dataloader, total_steps, warmup_steps, start_step)

        # ------------------------------------------------------------------
        # 8. Final checkpoint
        # ------------------------------------------------------------------
        final_step = total_steps  # we break when step reaches total_steps
        self.save_checkpoint(final_step)
        self.logger.info("Stage '%s' completed.", stage)

    def save_checkpoint(self, step: int) -> None:
        """
        Save a DeepSpeed checkpoint at the given step.

        Args:
            step: Current training step (used in the directory name).
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialised; cannot save checkpoint.")
        save_dir = Path(self.config.save_dir or "./checkpoints") / f"stage-{self.current_stage}-step-{step}"
        os.makedirs(save_dir, exist_ok=True)
        self.engine.save_checkpoint(
            save_dir=str(save_dir),
            client_state={
                "step": step,
                "stage": self.current_stage,
                "seed": self.config.seed,
            },
        )
        self.logger.info("Checkpoint saved to %s", save_dir)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _resolve_deepspeed_config(self) -> Dict[str, Any]:
        """
        Return the DeepSpeed configuration dictionary.

        If an external JSON file was provided, load and return it.
        Otherwise return a built‑in default that matches the paper’s setup.
        """
        if self.deepspeed_config_path is not None:
            import json
            with open(self.deepspeed_config_path, "r") as fh:
                return json.load(fh)

        return _get_default_deepspeed_config(
            stage_config=self.stage_config,
            training_config=self.config,
            world_size=self.world_size,
        )

    def _train_loop(
        self,
        dataloader: DataLoader,
        total_steps: int,
        warmup_steps: int,
        start_step: int = 0,
    ) -> None:
        """
        Core training loop using the DeepSpeed engine.

        Args:
            dataloader: DataLoader providing batches.
            total_steps: Total number of training steps for this stage.
            warmup_steps: Number of warmup steps (used only for logging).
            start_step: Step index from which to resume (used to skip
                batches when resuming from checkpoint).
        """
        engine = self.engine  # type: deepspeed.DeepSpeedEngine
        device = engine.device
        log_interval = getattr(self.config, "log_interval", 10)  # steps between logs

        # Prepare iterator; we need to fast‑forward it if resuming.
        data_iter = iter(dataloader)
        if start_step > 0:
            self.logger.info("Skipping %d batches to resume at step %d", start_step, start_step)
            for _ in range(start_step):
                try:
                    next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    next(data_iter)

        total_loss = 0.0
        total_samples = 0
        global_step = start_step

        while global_step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch tensors to the engine’s device.
            batch = self._to_device(batch, device)

            # Forward pass: model is available as engine.module.
            outputs = engine.module(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                image_token_mask=batch["image_token_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]
            if loss is None:
                # Should never happen when labels are provided.
                raise ValueError("Model returned no loss despite labels being present.")

            # Backward and step.
            engine.backward(loss)
            engine.step()

            # Accumulate loss for logging.
            total_loss += loss.item()
            total_samples += 1
            global_step += 1

            # Logging.
            if global_step % log_interval == 0 or global_step == 1 or global_step == total_steps:
                avg_loss = total_loss / total_samples
                lr = engine.get_lr()[0]
                self.logger.info(
                    "Stage %s | Step %d / %d | Loss: %.4f | LR: %.2e",
                    self.current_stage,
                    global_step,
                    total_steps,
                    avg_loss,
                    lr,
                )
                total_loss = 0.0
                total_samples = 0

            # Periodic checkpointing (every 5000 steps or as configured).
            save_every = getattr(self.config, "checkpoint_interval", 5000)
            if global_step % save_every == 0 and global_step > 0:
                self.save_checkpoint(global_step)

        self.logger.info("Training loop finished at step %d", global_step)

    @staticmethod
    def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Recursively move all tensors in *batch* to *device*.

        Args:
            batch: A dictionary potentially containing tensors, lists of
                tensors, or other nested structures.
            device: Target device.

        Returns:
            The same structure with tensors moved to *device*.
        """
        def move(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device, non_blocking=True)
            elif isinstance(obj, list):
                return [move(el) for el in obj]
            else:
                return obj

        return {k: move(v) for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Deprecated internal method kept only for design compatability.
    # ------------------------------------------------------------------
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        param_filter: Optional[Callable[[nn.Parameter], bool]] = None,
    ) -> float:
        """
        Legacy wrapper that calls the internal DeepSpeed loop.
        *(Not used in current implementation; retained for API symmetry.)*
        """
        # This method is overridden by the deepspeed loop; we simply raise
        # an error to avoid confusion.
        raise NotImplementedError(
            "Use `train_stage()` instead; this method is not compatible with DeepSpeed."
        )

