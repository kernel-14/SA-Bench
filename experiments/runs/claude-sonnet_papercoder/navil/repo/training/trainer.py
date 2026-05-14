```python
## training/trainer.py
"""Central training orchestrator for NaViL's three-stage training pipeline.

This module implements ``NaViLTrainer``, which manages the full training
lifecycle across three stages (S1.1 → S1.2 → S2):

- S1.1: Multi-modal Generative Pre-training on 500M web-scale image-caption
  pairs. Only visual encoder, connector, and MoE visual experts are trainable.
- S1.2: High-quality alignment pre-training on 185M samples. Additionally
  unfreezes linguistic expert attention projections.
- S2: Supervised Fine-tuning on 68M high-quality instruction data. All
  parameters trainable.

Key design decisions:
- HuggingFace Accelerate handles distributed training, mixed precision (bf16),
  and gradient accumulation transparently.
- A fresh AdamW optimizer is created per stage to avoid stale momentum states
  for parameters that change trainability between stages.
- The model is prepared with Accelerate once; the optimizer is re-prepared
  per stage.
- ``itertools.cycle`` allows training for an exact step count regardless of
  dataset size.
- Gradient clipping uses ``accelerator.clip_grad_norm_`` for correct behavior
  under mixed precision (handles gradient unscaling before norm computation).

Config alignment (configs/navil_2b.yaml):
    training.s1_1.{steps, peak_lr, weight_decay, lr_schedule, warmup_steps}
    training.s1_2.{steps, peak_lr, weight_decay, lr_schedule, warmup_steps}
    training.s2.{steps, peak_lr, weight_decay, lr_schedule, warmup_steps}
    training.optimizer.{beta1, beta2, eps, gradient_accumulation_steps, max_grad_norm}
    training.precision: "bfloat16"
    output.{checkpoint_dir, max_checkpoints, save_every_steps, log_every_steps, log_dir}

Dependencies:
    - model/navil_model.py: NaViLModel
    - training/loss.py: NTPLoss
    - training/scheduler.py: LRScheduler
    - utils/checkpoint.py: CheckpointManager
    - utils/logging_utils.py: setup_logger, log_metrics, AverageMeter
"""

import itertools
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as torch_data
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from model.navil_model import NaViLModel
from training.loss import NTPLoss
from training.scheduler import LRScheduler
from utils.checkpoint import CheckpointManager
from utils.logging_utils import AverageMeter, log_metrics, setup_logger


class NaViLTrainer:
    """Orchestrates NaViL's three-stage training pipeline.

    Manages parameter freezing, optimizer/scheduler lifecycle, distributed
    training via Accelerate, gradient clipping, logging, and checkpointing
    across all three training stages.

    Args:
        model:  A ``NaViLModel`` instance (not yet prepared by Accelerate).
                The model's parameters will be frozen/unfrozen per stage
                via ``model.freeze_params_for_stage``.
        config: OmegaConf DictConfig loaded from ``configs/navil_2b.yaml``
                or ``configs/navil_9b.yaml``. All hyperparameters are read
                from this config.

    Attributes:
        model:               The NaViLModel (may be wrapped by Accelerate
                             after ``setup_stage`` is called).
        config:              Stored OmegaConf DictConfig.
        accelerator:         HuggingFace Accelerate instance for distributed
                             training and mixed precision.
        optimizer:           Current stage's AdamW optimizer. ``None`` until
                             ``setup_stage`` is called.
        scheduler:           Current stage's ``LRScheduler``. ``None`` until
                             ``setup_stage`` is called.
        loss_fn:             ``NTPLoss`` instance (shared across all stages).
        checkpoint_manager:  ``CheckpointManager`` for saving/loading state.
        current_stage:       String identifier of the active training stage
                             (``"s1_1"``, ``"s1_2"``, or ``"s2"``). ``None``
                             before the first ``setup_stage`` call.
        global_step:         Global step counter across all stages. Not reset
                             between stages. Used for checkpoint naming and
                             logging.
        log_every_steps:     Logging frequency in steps (from config).
        save_every_steps:    Checkpoint saving frequency in steps (from config).

    Example::

        from omegaconf import OmegaConf
        config = OmegaConf.load("configs/navil_2b.yaml")
        model = NaViLModel(config)
        trainer = NaViLTrainer(model, config)
        trainer.run_full_training({
            "s1_1": dataloader_s1_1,
            "s1_2": dataloader_s1_2,
            "s2":   dataloader_s2,
        })
    """

    def __init__(
        self,
        model: NaViLModel,
        config: DictConfig,
    ) -> None:
        """Initialise the trainer with all stateful components.

        Args:
            model:  NaViLModel instance (not yet Accelerate-prepared).
            config: OmegaConf DictConfig from configs/navil_2b.yaml or
                    configs/navil_9b.yaml.
        """
        self.model: NaViLModel = model
        self.config: DictConfig = config

        # ------------------------------------------------------------------ #
        # Read shared training hyperparameters from config                    #
        # ------------------------------------------------------------------ #
        gradient_accumulation_steps: int = int(
            config.training.optimizer.gradient_accumulation_steps
        )
        precision: str = str(config.training.precision)  # "bfloat16"

        # Map config precision string to Accelerate mixed_precision argument
        # "bfloat16" → "bf16", "float16" → "fp16", "float32" → "no"
        _precision_map: Dict[str, str] = {
            "bfloat16": "bf16",
            "float16": "fp16",
            "float32": "no",
            "bf16": "bf16",
            "fp16": "fp16",
        }
        mixed_precision: str = _precision_map.get(precision.lower(), "bf16")

        # ------------------------------------------------------------------ #
        # Accelerate — created once, reused across all stages                 #
        # ------------------------------------------------------------------ #
        self.accelerator: Accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # ------------------------------------------------------------------ #
        # Training state                                                       #
        # ------------------------------------------------------------------ #
        self.current_stage: Optional[str] = None
        self.global_step: int = 0

        # Optimizer and scheduler are created per-stage in setup_stage()
        self.optimizer: Optional[torch.optim.AdamW] = None
        self.scheduler: Optional[LRScheduler] = None

        # ------------------------------------------------------------------ #
        # Loss function — shared across all stages                            #
        # ------------------------------------------------------------------ #
        self.loss_fn: NTPLoss = NTPLoss(ignore_index=-100)

        # ------------------------------------------------------------------ #
        # Checkpoint manager                                                   #
        # ------------------------------------------------------------------ #
        checkpoint_dir: str = str(config.output.checkpoint_dir)
        max_checkpoints: int = int(config.output.max_checkpoints)
        self.checkpoint_manager: CheckpointManager = CheckpointManager(
            output_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
        )

        # ------------------------------------------------------------------ #
        # Logging configuration                                                #
        # ------------------------------------------------------------------ #
        self.log_every_steps: int = int(config.output.log_every_steps)
        self.save_every_steps: int = int(config.output.save_every_steps)

        log_dir: str = str(config.output.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_file: Optional[str] = (
            os.path.join(log_dir, "train.log")
            if self.accelerator.is_local_main_process
            else None
        )
        self._logger: logging.Logger = setup_logger(
            "navil.trainer",
            log_file=log_file,
            level=logging.INFO,
        )

        # ------------------------------------------------------------------ #
        # Track whether the model has been prepared by Accelerate             #
        # The model is prepared once; the optimizer is re-prepared per stage. #
        # ------------------------------------------------------------------ #
        self._model_prepared: bool = False

        self._logger.info(
            "NaViLTrainer initialised: mixed_precision=%s, "
            "gradient_accumulation_steps=%d, "
            "log_every_steps=%d, save_every_steps=%d",
            mixed_precision,
            gradient_accumulation_steps,
            self.log_every_steps,
            self.save_every_steps,
        )

    # ---------------------------------------------------------------------- #
    # Optimizer setup                                                          #
    # ---------------------------------------------------------------------- #

    def setup_optimizer(self) -> None:
        """Create a fresh AdamW optimizer for the current training stage.

        Uses only the parameters with ``requires_grad=True`` (set by
        ``model.freeze_params_for_stage`` before this call). Splits
        parameters into two groups:
        - Weight-decayed group: 2D+ tensors (weight matrices).
        - Non-decayed group: 1D tensors (biases, RMSNorm weights).

        This prevents weight decay from shrinking bias terms and normalization
        scale parameters, which would harm training stability.

        Must be called after ``model.freeze_params_for_stage(stage)`` and
        after ``self.current_stage`` is set.

        Raises:
            RuntimeError: If ``current_stage`` is None (setup_stage not called).
        """
        if self.current_stage is None:
            raise RuntimeError(
                "setup_optimizer() called before current_stage was set. "
                "Call setup_stage(stage) instead, which sets current_stage "
                "and then calls setup_optimizer()."
            )

        # Read stage-specific hyperparameters
        stage_cfg: DictConfig = getattr(self.config.training, self.current_stage)
        weight_decay: float = float(stage_cfg.weight_decay)
        peak_lr: float = float(stage_cfg.peak_lr)

        # Read shared optimizer hyperparameters
        beta1: float = float(self.config.training.optimizer.beta1)
        beta2: float = float(self.config.training.optimizer.beta2)
        eps: float = float(self.config.training.optimizer.eps)

        # ------------------------------------------------------------------ #
        # Collect trainable parameters and split into two groups              #
        # ------------------------------------------------------------------ #
        # Unwrap model if already prepared by Accelerate to access raw params
        raw_model: NaViLModel = self.accelerator.unwrap_model(self.model)

        decay_params: List[nn.Parameter] = []
        no_decay_params: List[nn.Parameter] = []

        param: nn.Parameter
        name: str
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue

            # Split criterion: 2D+ tensors get weight decay (weight matrices);
            # 1D tensors do not (biases, RMSNorm/LayerNorm scale parameters).
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        # Build parameter groups
        param_groups: List[Dict[str, Any]] = []

        if decay_params:
            param_groups.append(
                {
                    "params": decay_params,
                    "weight_decay": weight_decay,
                    "lr": peak_lr,
                }
            )

        if no_decay_params:
            param_groups.append(
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "lr": peak_lr,
                }
            )

        if not param_groups:
            self._logger.warning(
                "Stage '%s': no trainable parameters found. "
                "Creating optimizer with empty parameter groups. "
                "Check model.freeze_params_for_stage('%s').",
                self.current_stage,
                self.current_stage,
            )
            # Create a minimal optimizer to avoid downstream errors
            param_groups = [{"params": [], "weight_decay": weight_decay, "lr": peak_lr}]

        # ------------------------------------------------------------------ #
        # Create AdamW optimizer                                               #
        # ------------------------------------------------------------------ #
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=peak_lr,
            betas=(beta1, beta2),
            eps=eps,
        )

        total_trainable: int = sum(
            p.numel() for group in param_groups for p in group["params"]
        )
        self._logger.info(
            "Stage '%s' optimizer: AdamW, lr=%.2e, weight_decay=%.4f, "
            "beta1=%.2f, beta2=%.3f, eps=%.1e, "
            "trainable_params=%d (decay=%d, no_decay=%d)",
            self.current_stage,
            peak_lr,
            weight_decay,
            beta1,
            beta2,
            eps,
            total_trainable,
            sum(p.numel() for p in decay_params),
            sum(p.numel() for p in no_decay_params),
        )

    # ---------------------------------------------------------------------- #
    # Stage setup                                                              #
    # ---------------------------------------------------------------------- #

    def setup_stage(self, stage: str) -> None:
        """Fully configure the trainer for a given training stage.

        Performs in order:
        1. Set ``current_stage``.
        2. Freeze/unfreeze model parameters via ``model.freeze_params_for_stage``.
        3. Create fresh AdamW optimizer via ``setup_optimizer``.
        4. Create ``LRScheduler`` for this stage.
        5. Prepare model (once) and optimizer with Accelerate.

        Args:
            stage: Training stage identifier: ``"s1_1"``, ``"s1_2"``,
                   or ``"s2"``.

        Raises:
            ValueError: If ``stage`` is not one of the three valid stages.
        """
        valid_stages: Tuple[str, ...] = ("s1_1", "s1_2", "s2")
        if stage not in valid_stages:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of {valid_stages}."
            )

        self._logger.info("Setting up training stage: %s", stage)
        self.current_stage = stage

        # ------------------------------------------------------------------ #
        # Step 1: Freeze/unfreeze model parameters for this stage             #
        # ------------------------------------------------------------------ #
        # Unwrap model if already prepared to access freeze_params_for_stage
        raw_model: NaViLModel = self.accelerator.unwrap_model(self.model)
        raw_model.freeze_params_for_stage(stage)

        # ------------------------------------------------------------------ #
        # Step 2: Create fresh optimizer with stage-specific settings         #
        # ------------------------------------------------------------------ #
        self.setup_optimizer()

        # ------------------------------------------------------------------ #
        # Step 3: Create LR scheduler for this stage                          #
        # ------------------------------------------------------------------ #
        stage_cfg: DictConfig = getattr(self.config.training, stage)
        lr_schedule: str = str(stage_cfg.lr_schedule)
        peak_lr: float = float(stage_cfg.peak_lr)
        warmup_steps: int = int(stage_cfg.warmup_steps)
        total_steps: int = int(stage_cfg.steps)

        self.scheduler = LRScheduler(
            optimizer=self.optimizer,
            schedule_type=lr_schedule,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=0.0,
        )

        self._logger.info(
            "Stage '%s' scheduler: type=%s, peak_lr=%.2e, "
            "warmup_steps=%d, total_steps=%d",
            stage,
            lr_schedule,
            peak_lr,
            warmup_steps,
            total_steps,
        )

        # ------------------------------------------------------------------ #
        # Step 4: Prepare with Accelerate                                     #
        # ------------------------------------------------------------------ #
        # The model is prepared only once to avoid double-wrapping with DDP.
        # The optimizer is re-prepared each stage since it changes.
        if not self._model_prepared:
            self.model = self.accelerator.prepare(self.model)
            self._model_prepared = True
            self._logger.info("Model prepared with Accelerate (first time).")

        # Always re-prepare the optimizer (it is recreated each stage)
        self.optimizer = self.accelerator.prepare(self.optimizer)

        self._logger.info("Stage '%s' setup complete.", stage)

    # ---------------------------------------------------------------------- #
    # Single training step                                                     #
    # ---------------------------------------------------------------------- #

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute one forward-backward-optimizer step.

        Handles mixed precision, gradient accumulation, gradient clipping,
        and LR scheduling. Returns the scalar loss value for logging.

        Args:
            batch: Dict from ``NaViLDataset._collate_fn`` with keys:
                   - ``"input_ids"``:      LongTensor (B, L)
                   - ``"labels"``:         LongTensor (B, L) with -100 at masked positions
                   - ``"attention_mask"``: LongTensor (B, L)
                   - ``"modality_mask"``:  LongTensor (B, L) with 0=visual, 1=text
                   - ``"pixel_values"``:   List[List[Tensor]] or None
                   - ``"grid_sizes"``:     List[List[Tuple[int,int]]] or None

        Returns:
            Scalar float loss value for the current step.

        Note:
            With ``gradient_accumulation_steps=1`` (from config), the
            ``accelerator.accumulate`` context manager is a no-op but is
            included for correctness if the config is changed.
        """
        # ------------------------------------------------------------------ #
        # Unpack batch                                                         #
        # ------------------------------------------------------------------ #
        input_ids: torch.Tensor = batch["input_ids"]
        labels: torch.Tensor = batch["labels"]
        attention_mask: torch.Tensor = batch["attention_mask"]
        modality_mask: torch.Tensor = batch["modality_mask"]
        pixel_values: Optional[List[Any]] = batch.get("pixel_values", None)
        grid_sizes: Optional[List[Any]] = batch.get("grid_sizes", None)

        # ------------------------------------------------------------------ #
        # Forward + backward within accumulate context                        #
        # ------------------------------------------------------------------ #
        with self.accelerator.accumulate(self.model):
            # Forward pass — pass labels=None to get raw logits back.
            # Loss is computed separately via NTPLoss for explicit control.
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    modality_mask=modality_mask,
                    attention_mask=attention_mask,
                    labels=None,
                )
            except Exception as exc:
                self._logger.warning(
                    "Forward pass failed at global_step=%d: %s. "
                    "Skipping this batch.",
                    self.global_step,
                    exc,
                )
                return 0.0

            # Extract logits from model output
            # NaViLModel.forward returns CausalLMOutputWithPast or similar
            if hasattr(outputs, "logits"):
                logits: torch.Tensor = outputs.logits
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                logits = outputs[0]
            else:
                self._logger.warning(
                    "Unexpected model output type %s at global_step=%d. "
                    "Skipping batch.",
                    type(outputs).__name__,
                    self.global_step,
                )
                return 0.0

            # Compute NTP loss with shift-by-one and ignore_index=-100 masking
            try:
                loss: torch.Tensor = self.loss_fn(logits, labels)
            except Exception as exc:
                self._logger.warning(
                    "Loss computation failed at global_step=%d: %s. "
                    "Skipping batch.",
                    self.global_step,
                    exc,
                )
                return 0.0

            # Check for NaN/Inf loss (can occur with corrupted batches)
            if not torch.isfinite(loss):
                self._logger.warning(
                    "Non-finite loss (%.4f) at global_step=%d. "
                    "Skipping backward pass.",
                    loss.item(),
                    self.global_step,
                )
                return 0.0

            # Backward pass — Accelerate handles mixed precision gradient scaling
            self.accelerator.backward(loss)

            # Gradient clipping — only at actual update steps (not accumulation sub-steps)
            if self.accelerator.sync_gradients:
                max_grad_norm: float = float(
                    self.config.training.optimizer.max_grad_norm
                )
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=max_grad_norm,
                )

            # Optimizer step
            self.optimizer.step()

            # LR scheduler step — updates LR for next optimizer step
            self.scheduler.step()

            # Zero gradients for next step
            self.optimizer.zero_grad()

        return loss.item()

    # ---------------------------------------------------------------------- #
    # Stage training loop                                                      #
    # ---------------------------------------------------------------------- #

    def train_stage(
        self,
        dataloader: torch_data.DataLoader,
        stage: str,
        steps: int,
    ) -> None:
        """Run the full training loop for one stage.

        Sets up the stage (parameter freezing, optimizer, scheduler),
        prepares the dataloader with Accelerate, then iterates for exactly
        ``steps`` steps using an infinite cycling iterator.

        Args:
            dataloader: PyTorch DataLoader for this stage's dataset.
                        Will be prepared with Accelerate for distributed
                        training and device placement.
            stage:      Training stage identifier: ``"s1_1"``, ``"s1_2"``,
                        or ``"s2"``.
            steps:      Number of training steps to run for this stage.
                        From config: s1_1=70000, s1_2=40000, s2=30000
                        (NaViL-2B).

        Note:
            ``self.global_step`` is NOT reset between stages. It accumulates
            across all stages for consistent checkpoint naming.
        """
        self._logger.info(
            "Starting training stage '%s' for %d steps "
            "(global_step starts at %d).",
            stage,
            steps,
            self.global_step,
        )

        # ------------------------------------------------------------------ #
        # Stage setup: freeze params, create optimizer/scheduler, prepare     #
        # ------------------------------------------------------------------ #
        self.setup_stage(stage)

        # ------------------------------------------------------------------ #
        # Prepare dataloader with Accelerate                                  #
        # ------------------------------------------------------------------ #
        prepared_dataloader: torch_data.DataLoader = self.accelerator.prepare(
            dataloader
        )

        # ------------------------------------------------------------------ #
        # Create infinite cycling iterator over the dataloader                #
        # ------------------------------------------------------------------ #
        # itertools.cycle creates an infinite iterator that restarts from the
        # beginning when the dataloader is exhausted. This allows training for
        # an exact number of steps regardless of dataset size.
        data_iter: Iterator[Dict[str, Any]] = itertools.cycle(prepared_dataloader)

        # ------------------------------------------------------------------ #
        # Loss tracking                                                        #
        # ------------------------------------------------------------------ #
        loss_meter: AverageMeter = AverageMeter()

        # ------------------------------------------------------------------ #
        # Training loop                                                        #
        # ------------------------------------------------------------------ #
        # tqdm progress bar — only shown on the main process in distributed
        # training to avoid duplicate output.
        progress_bar = tqdm(
            range(steps),
            desc=f"Stage {stage}",
            disable=not self.accelerator.is_local_main_process,
            dynamic_ncols=True,
        )

        step_idx: int
        for step_idx in progress_bar:
            # ---------------------------------------------------------------- #
            # Get next batch                                                    #
            # ---------------------------------------------------------------- #
            try:
                batch: Dict[str, Any] = next(data_iter)
            except StopIteration:
                # Should not happen with itertools.cycle, but guard defensively
                self._logger.warning(
                    "Data iterator exhausted at stage '%s' step %d/%d. "
                    "This should not happen with itertools.cycle.",
                    stage,
                    step_idx,
                    steps,
                )
                break

            # ---------------------------------------------------------------- #
            # Execute training step                                             #
            # ---------------------------------------------------------------- #
            loss_val: float = self.train_step(batch)

            # ---------------------------------------------------------------- #
            # Update loss meter and global step counter                        #
            # ---------------------------------------------------------------- #
            loss_meter.update(loss_val)
            self.global_step += 1

            # ---------------------------------------------------------------- #
            # Update progress bar postfix                                      #
            # ---------------------------------------------------------------- #
            if self.accelerator.is_local_main_process:
                current_lr: float = self.scheduler.get_lr(
                    self.scheduler.current_step - 1
                )
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss_val:.4f}",
                        "avg_loss": f"{loss_meter.avg:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": self.global_step,
                    }
                )

            # ---------------------------------------------------------------- #
            # Logging (every log_every_steps, main process only)              #
            # ---------------------------------------------------------------- #
            if (
                self.accelerator.is_local_main_process
                and self.global_step % self.log_every_steps == 0
            ):
                current_lr = self.scheduler.get_lr(
                    self.scheduler.current_step - 1
                )
                metrics: Dict[str, Any] = {
                    "loss": loss_meter.avg,
                    "lr": current_lr,
                    "stage": stage,
                    "global_step": self.global_step,
                }
                log_metrics(
                    self._logger,
                    metrics,
                    step=self.global_step,
                    prefix=f"train/{stage}",
                )
                # Reset meter for the next logging window
                loss_meter.reset()

            # ---------------------------------------------------------------- #
            # Checkpointing (every save_every_steps, main process only)       #
            # ---------------------------------------------------------------- #
            if (
                self.accelerator.is_local_main_process
                and self.global_step % self.save_every_steps == 0
            ):
                self.save_checkpoint(self.global_step)

        # ------------------------------------------------------------------ #
        # End-of-stage: save final checkpoint                                 #
        # ------------------------------------------------------------------ #
        if self.accelerator.is_local_main_process:
            self._logger.info(
                "Stage '%s' complete. Saving final checkpoint at "
                "global_step=%d.",
                stage,
                self.global_step,
            )
            self.save_checkpoint(self.global_step)

        # Synchronize all processes before moving to the next stage
        self.accelerator.wait_for_everyone()

        self._logger.info(
            "Stage '%s' finished: %d steps completed, "
            "global_step=%d.",
            stage,
            steps,
            self.global_step,
        )

    # ---------------------------------------------------------------------- #
    # Full training pipeline                                                   #
    # ---------------------------------------------------------------------- #

    def run_full_training(
        self,
        dataloaders: Dict[str, torch_data.DataLoader],
    ) -> None:
        """Orchestrate all three training stages sequentially.

        Runs S1.1 → S1.2 → S2 in order. Each stage calls ``train_stage``,
        which handles setup, training loop, and checkpointing.

        Args:
            dataloaders: Dict mapping stage names to DataLoader instances:
                         - ``"s1_1"``: DataLoader for 500M web-scale data
                         - ``"s1_2"``: DataLoader for 185M high-quality data
                         - ``"s2"``:   DataLoader for 68M SFT data

        Raises:
            KeyError: If any required stage key is missing from ``dataloaders``.

        Note:
            ``global_step`` accumulates across all stages. After full training:
            - NaViL-2B: global_step = 70000 + 40000 + 30000 = 140000
            - NaViL-9B: global_step = 50000 + 33000 + 6000 = 89000
        """
        # Validate that all required dataloaders are present
        required_stages: Tuple[str, ...] = ("s1_1", "s1_2", "s2")
        for required_stage in required_stages:
            if required_stage not in dataloaders:
                raise KeyError(
                    f"Missing dataloader for stage '{required_stage}'. "
                    f"dataloaders must contain keys: {list(required_stages)}. "
                    f"Got keys: {list(dataloaders.keys())}."
                )

        self._logger.info(
            "Starting full NaViL training pipeline: S1.1 → S1.2 → S2. "
            "Total stages: 3."
        )

        # ------------------------------------------------------------------ #
        # Stage 1.1: Multi-modal Generative Pre-training (web-scale)          #
        # ------------------------------------------------------------------ #
        s1_1_steps: int = int(self.config.training.s1_1.steps)
        self._logger.info(
            "=== Stage S1.1: Multi-modal Generative Pre-training "
            "(web-scale, %d steps) ===",
            s1_1_steps,
        )
        self.train_stage(
            dataloader=dataloaders["s1_1"],
            stage="s1_1",
            steps=s1_1_steps,
        )

        # ------------------------------------------------------------------ #
        # Stage 1.2: Multi-modal Generative Pre-training (high-quality)       #
        # ------------------------------------------------------------------ #
        s1_2_steps: int = int(self.config.training.s1_2.steps)
        self._logger.info(
            "=== Stage S1.2: Multi-modal Generative Pre-training "
            "(high-quality, %d steps) ===",
            s1_2_steps,
        )
        self.train_stage(
            dataloader=dataloaders["s1_2"],
            stage="s1_2",
            steps=s1_2_steps,
        )

        # ------------------------------------------------------------------ #
        # Stage 2: Supervised Fine-tuning                                     #
        # ------------------------------------------------------------------ #
        s2_steps: int = int(self.config.training.s2.steps)
        self._logger.info(
            "=== Stage S2: Supervised Fine-tuning (%d steps) ===",