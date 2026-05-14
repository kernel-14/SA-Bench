## Code: training/finetune.py

```python
## training/finetune.py
"""
Adapter-only fine-tuning loop for the multi-physics neural operator
pretraining framework described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Implements the Finetuner class that drives the fine-tuning phase where the
shared FNO backbone (θ_F) is frozen and only the new problem-specific
lifting and projection adapters (θ_{L_ft}, θ_{P_ft}) are trained.

From the paper (Section 3):
    "In the fine-tuning stage we fix the parameters θ_F both to highlight
    the generalizing properties of the operator and to reduce training costs:
    only the new adapter parameters (θ_{P_ft}, θ_{L_ft}) are trained."

This module is used in all three experiments:
  - Exp1: Fine-tune on out-of-sample parameter values (same physics,
          different coefficients)
  - Exp2: Fine-tune with extended input adapter (new input functions added)
  - Exp3: Fine-tune on entirely different physics (reaction-diffusion after
          advection+Burgers pretraining)

It also serves the scratch baseline path: when config.freeze_backbone=False,
the same class trains all parameters from random initialization, providing
the comparison baseline for Tables 1 and 2.

Design contract (Data structures and interfaces):
  Finetuner:
    __init__(model: AdapterFramework, train_dataset: Dataset,
             val_dataset: Dataset, config: FinetuneConfig,
             physics_id: str, device: torch.device)
    finetune() -> Dict[str, List[float]]
    _train_epoch(epoch: int) -> float
    _validate() -> float
    save_checkpoint(epoch: int, val_loss: float) -> None

Config alignment (config.yaml):
  training.finetune.lr: 1.0e-4           -> Adam learning rate
  training.finetune.batch_size: 16        -> DataLoader batch size
  training.finetune.n_epochs: 100         -> Training loop iterations
  training.finetune.freeze_backbone: true -> Backbone frozen during fine-tuning
  training.finetune.checkpoint_dir: "checkpoints/finetune"
  training.finetune.save_every: 10        -> Periodic checkpoint interval
  training.loss: "mse"                    -> Training loss function
  evaluation.epoch_time_warmup: 1         -> Warmup epochs excluded from timing

Tensor layout convention (Shared Knowledge #1):
  Channel-first: [B, C, L] for 1D, [B, C, H, W] for 2D.

Physics ID strings (Shared Knowledge #2):
  No dots — use 'p' for decimal point.
  Received as constructor argument; not created here.

Epoch time measurement (Shared Knowledge #6):
  Wall-clock time per epoch measured with time.perf_counter().
  Timing covers forward+backward pass only, not DataLoader iteration.
  First epoch (warmup) excluded from average per config.yaml
  evaluation.epoch_time_warmup: 1.

Dependencies:
  torch, torch.nn, torch.optim, torch.utils.data
  time, logging, os, math, typing
  models/adapter_framework.py -> AdapterFramework
  training/losses.py          -> MSELoss, get_loss_fn
  utils/config.py             -> FinetuneConfig
  utils/logging_utils.py      -> get_logger
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from models.adapter_framework import AdapterFramework
from training.losses import get_loss_fn
from utils.config import FinetuneConfig
from utils.logging_utils import get_logger

# ---------------------------------------------------------------------------
# Module-level logger (fallback; Finetuner creates its own instance logger)
# ---------------------------------------------------------------------------

_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default number of DataLoader worker processes.
# 0 = load data in the main process (safe default; avoids multiprocessing
# issues with HDF5 files and CUDA tensors).
_DEFAULT_NUM_WORKERS: int = 0

# Minimum number of batches required before reporting epoch statistics.
# Prevents division-by-zero if a dataset is unexpectedly empty.
_MIN_BATCHES_FOR_STATS: int = 1

# Scheduler type string constants (match config.yaml training.finetune.scheduler)
_SCHEDULER_COSINE: str = "cosine"
_SCHEDULER_PLATEAU: str = "plateau"

# Checkpoint filename template for best model.
# physics_id is embedded to distinguish checkpoints for different fine-tuning
# targets (e.g., 'finetune_burgers_nu0p001_best.pt').
_CKPT_BEST_TEMPLATE: str = "finetune_{physics_id}_best.pt"

# Periodic checkpoint filename template.
_CKPT_PERIODIC_TEMPLATE: str = "finetune_{physics_id}_epoch_{epoch:04d}.pt"

# Default weight decay for Adam optimizer.
# FinetuneConfig does not include weight_decay in the design contract;
# use the same value as pretraining (config.yaml training.pretrain.weight_decay).
_DEFAULT_WEIGHT_DECAY: float = 1.0e-4

# Default minimum learning rate for schedulers.
_DEFAULT_ETA_MIN: float = 1.0e-6

# Number of warmup epochs excluded from average epoch time computation.
# From config.yaml: evaluation.epoch_time_warmup: 1.
_EPOCH_TIME_WARMUP: int = 1

# Default save_every interval (epochs between periodic checkpoints).
# From config.yaml: training.finetune.save_every: 10.
_DEFAULT_SAVE_EVERY: int = 10


# ---------------------------------------------------------------------------
# Finetuner
# ---------------------------------------------------------------------------


class Finetuner:
    """Adapter-only fine-tuning loop for the AdapterFramework.

    Drives the fine-tuning phase where the shared FNO backbone (θ_F) is
    frozen and only the new problem-specific lifting and projection adapters
    (θ_{L_ft}, θ_{P_ft}) are trained. This is the core transfer learning
    mechanism of the paper.

    Also serves the scratch baseline path: when config.freeze_backbone=False,
    all parameters are trained from random initialization, providing the
    comparison baseline for Tables 1 and 2.

    Key differences from Pretrainer:
      1. Single-physics DataLoader (no multi-physics routing, no collate_fn).
      2. Optimizer receives only adapter parameters (when freeze_backbone=True).
      3. physics_id is fixed for the entire fine-tuning run.
      4. Backbone is frozen (no gradient computation through backbone).

    Attributes:
        model: AdapterFramework with the target adapter registered.
        train_dataset: Single-physics training dataset.
        val_dataset: Single-physics validation dataset.
        config: FinetuneConfig with all hyperparameters.
        physics_id: Physics identifier string for the fine-tuning target.
        _device: Target device (CPU or CUDA).
        _optimizer: Adam optimizer (adapter params only, or all params).
        _scheduler: LR scheduler (CosineAnnealingLR or ReduceLROnPlateau).
        _scheduler_type: String identifying the scheduler type.
        _loss_fn: Training loss function (MSELoss).
        _train_loader: DataLoader for training data.
        _val_loader: DataLoader for validation data.
        _logger: Instance-specific logger.
        _best_val_loss: Best validation loss seen so far (for checkpointing).
        _epoch_times: Wall-clock time per epoch (for "Avg. epoch (s)" metric).
        _train_history: Accumulated training loss history.
        _val_history: Accumulated validation loss history.

    Example::

        from models.fno_backbone import FNOBackbone
        from models.adapter_framework import AdapterFramework
        from utils.config import FinetuneConfig

        # Load pretrained model
        backbone = FNOBackbone(hidden_dim=64, n_modes=16, n_layers=4)
        model = AdapterFramework(backbone=backbone, hidden_dim=64)
        model.load_checkpoint('checkpoints/pretrain/pretrain_best.pt')

        # Register new adapter for fine-tuning target
        model.register_adapter('burgers_nu0p001', n_in=1, n_out=1)

        config = FinetuneConfig(
            target_physics='burgers_nu0p001',
            n_epochs=100, lr=1e-4, batch_size=16,
            pretrained_checkpoint='checkpoints/pretrain/pretrain_best.pt',
            freeze_backbone=True,
        )

        finetuner = Finetuner(
            model=model,
            train_dataset=finetune_train_ds,
            val_dataset=finetune_val_ds,
            config=config,
            physics_id='burgers_nu0p001',
            device=torch.device('cuda'),
        )
        history = finetuner.finetune()
    """

    def __init__(
        self,
        model: AdapterFramework,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: FinetuneConfig,
        physics_id: str,
        device: torch.device,
    ) -> None:
        """Initialise Finetuner.

        Sets up backbone freezing (if configured), optimizer (adapter params
        only or all params), LR scheduler, training loss, DataLoaders, logger,
        and checkpoint directory.

        Args:
            model: AdapterFramework with the target adapter already registered
                via register_adapter(physics_id, n_in, n_out). The model is
                moved to ``device`` immediately.
            train_dataset: Single-physics training dataset. Must implement
                __len__ and __getitem__ returning (input_tensor, target_tensor).
                All samples must have the same n_in (no multi-physics routing
                needed here).
            val_dataset: Single-physics validation dataset. Same interface as
                train_dataset. Used to select the best checkpoint.
            config: FinetuneConfig populated from config.yaml. All
                hyperparameters (lr, batch_size, n_epochs, freeze_backbone)
                are read from this object.
            physics_id: Physics identifier string for the fine-tuning target.
                Must be registered in the model via register_adapter() before
                constructing Finetuner. Must follow Shared Knowledge #2
                convention (no dots, use 'p' for decimal point).
                Examples: 'burgers_nu0p001', 'heat_conv_alpha0p01'.
            device: Target device. The model and all batches are moved to
                this device. Use torch.device('cuda') for GPU training
                (required for MambaFNO) or torch.device('cpu') for testing.

        Raises:
            ValueError: If physics_id is not registered in the model.
            ValueError: If config.n_epochs <= 0.
            ValueError: If config.batch_size <= 0.
            ValueError: If config.lr <= 0.
            OSError: If config.checkpoint_dir cannot be created.
        """
        # ── Validate config ───────────────────────────────────────────────
        if config.n_epochs <= 0:
            raise ValueError(
                f"config.n_epochs must be positive, got {config.n_epochs}."
            )
        if config.batch_size <= 0:
            raise ValueError(
                f"config.batch_size must be positive, got {config.batch_size}."
            )
        if config.lr <= 0.0:
            raise ValueError(
                f"config.lr must be positive, got {config.lr}."
            )

        # ── Validate physics_id is registered ────────────────────────────
        # AdapterFramework.forward() will raise KeyError if not registered,
        # but we validate early here to provide a clearer error message.
        if physics_id not in model._adapter_registry:
            registered: List[str] = sorted(model._adapter_registry.keys())
            raise ValueError(
                f"physics_id='{physics_id}' is not registered in the "
                f"AdapterFramework. "
                f"Call model.register_adapter('{physics_id}', n_in, n_out) "
                f"before constructing Finetuner. "
                f"Currently registered physics IDs: {registered}."
            )

        # ── Store references ──────────────────────────────────────────────
        self.model: AdapterFramework = model
        self.train_dataset: Dataset = train_dataset
        self.val_dataset: Dataset = val_dataset
        self.config: FinetuneConfig = config
        self.physics_id: str = physics_id
        self._device: torch.device = device

        # ── Move model to device ──────────────────────────────────────────
        self.model.to(device)

        # ── Tracking state ────────────────────────────────────────────────
        self._best_val_loss: float = float("inf")
        self._epoch_times: List[float] = []
        self._train_history: List[float] = []
        self._val_history: List[float] = []

        # ── Checkpoint directory ──────────────────────────────────────────
        # Derive checkpoint_dir from config. FinetuneConfig has
        # pretrained_checkpoint but not checkpoint_dir explicitly.
        # Use a sensible default derived from the pretrained checkpoint path,
        # or fall back to "checkpoints/finetune".
        self._checkpoint_dir: str = self._resolve_checkpoint_dir(config)
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        # ── Logger ────────────────────────────────────────────────────────
        log_file: str = os.path.join(
            self._checkpoint_dir,
            f"finetune_{physics_id}.log",
        )
        self._logger: logging.Logger = get_logger(
            f"Finetuner[{physics_id}]",
            log_file=log_file,
        )

        # ── Backbone freezing (paper's core mechanism) ────────────────────
        # config.freeze_backbone=True  -> adapter-only fine-tuning (paper method)
        # config.freeze_backbone=False -> scratch baseline (all params trained)
        if config.freeze_backbone:
            self.model.freeze_backbone()
            self._logger.info(
                "Fine-tuning mode: adapter-only (backbone frozen). "
                "Only adapter parameters for physics_id='%s' will be updated.",
                physics_id,
            )
        else:
            # Ensure backbone is unfrozen for scratch baseline.
            self.model.unfreeze_backbone()
            self._logger.info(
                "Training from scratch (all params unfrozen). "
                "This is the scratch baseline — all parameters will be updated.",
            )

        # ── Optimizer setup ───────────────────────────────────────────────
        # Adapter-only: optimizer receives only the adapter pair parameters.
        # Scratch baseline: optimizer receives all model parameters.
        #
        # The double enforcement (freeze_backbone + adapter-only optimizer)
        # ensures no accidental backbone updates even if requires_grad is
        # somehow bypassed.
        if config.freeze_backbone:
            # get_adapter_params returns parameters of the lifting and
            # projection adapters for physics_id only.
            adapter_params: List[nn.Parameter] = self.model.get_adapter_params(
                physics_id
            )
            n_adapter_params: int = sum(p.numel() for p in adapter_params)

            if len(adapter_params) == 0:
                raise ValueError(
                    f"model.get_adapter_params('{physics_id}') returned an "
                    f"empty list. Ensure the adapter is registered and has "
                    f"trainable parameters."
                )

            self._optimizer: Adam = Adam(
                adapter_params,
                lr=config.lr,
                weight_decay=_DEFAULT_WEIGHT_DECAY,
            )

            self._logger.info(
                "Optimizer: Adam over adapter params only. "
                "n_adapter_params=%d, lr=%.2e, weight_decay=%.2e.",
                n_adapter_params,
                config.lr,
                _DEFAULT_WEIGHT_DECAY,
            )
        else:
            # Scratch baseline: all parameters.
            all_params: List[nn.Parameter] = list(self.model.parameters())
            n_all_params: int = sum(p.numel() for p in all_params)

            self._optimizer = Adam(
                all_params,
                lr=config.lr,
                weight_decay=_DEFAULT_WEIGHT_DECAY,
            )

            self._logger.info(
                "Optimizer: Adam over ALL parameters (scratch baseline). "
                "n_total_params=%d, lr=%.2e, weight_decay=%.2e.",
                n_all_params,
                config.lr,
                _DEFAULT_WEIGHT_DECAY,
            )

        # ── LR Scheduler ─────────────────────────────────────────────────
        # Default to CosineAnnealingLR matching the pretrain scheduler pattern
        # from config.yaml training.pretrain.scheduler: "cosine".
        # FinetuneConfig does not have a scheduler field; default to cosine.
        self._scheduler_type: str = _SCHEDULER_COSINE

        self._scheduler: object = CosineAnnealingLR(
            self._optimizer,
            T_max=config.n_epochs,
            eta_min=_DEFAULT_ETA_MIN,
        )

        self._logger.info(
            "LR scheduler: CosineAnnealingLR(T_max=%d, eta_min=%.2e).",
            config.n_epochs,
            _DEFAULT_ETA_MIN,
        )

        # ── Loss function ─────────────────────────────────────────────────
        # Training loss from config.yaml training.loss: "mse".
        # NMAE is only used for evaluation (Evaluator), not training.
        self._loss_fn: nn.Module = get_loss_fn("mse")

        # ── DataLoaders ───────────────────────────────────────────────────
        # Single-physics fine-tuning: no custom collate_fn needed.
        # All samples have the same n_in, so standard batching works.
        pin_memory: bool = (device.type == "cuda")

        self._train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=_DEFAULT_NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=True,  # Drop last incomplete batch for stable training
        )

        self._val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=_DEFAULT_NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=False,
        )

        # ── Log initialization summary ────────────────────────────────────
        n_total_params: int = sum(p.numel() for p in self.model.parameters())
        n_trainable_params: int = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        self._logger.info(
            "Finetuner initialized: "
            "physics_id='%s', n_epochs=%d, lr=%.2e, batch_size=%d, "
            "freeze_backbone=%s, device='%s'. "
            "Model: total_params=%d, trainable_params=%d. "
            "Train samples=%d, Val samples=%d. "
            "Checkpoint dir: '%s'.",
            physics_id,
            config.n_epochs,
            config.lr,
            config.batch_size,
            config.freeze_backbone,
            str(device),
            n_total_params,
            n_trainable_params,
            len(train_dataset),
            len(val_dataset),
            self._checkpoint_dir,
        )

    # -----------------------------------------------------------------------
    # Private: checkpoint directory resolution
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_checkpoint_dir(config: FinetuneConfig) -> str:
        """Derive the checkpoint directory from FinetuneConfig.

        FinetuneConfig has ``pretrained_checkpoint`` (path to the pretrained
        model checkpoint) but not an explicit ``checkpoint_dir`` field.
        This method derives a sensible checkpoint directory:

        1. If ``pretrained_checkpoint`` is set, use its parent directory
           with a 'finetune' subdirectory.
        2. Otherwise, fall back to 'checkpoints/finetune'.

        Args:
            config: FinetuneConfig instance.

        Returns:
            Checkpoint directory path string.
        """
        if config.pretrained_checkpoint:
            pretrain_dir: str = os.path.dirname(
                os.path.abspath(config.pretrained_checkpoint)
            )
            # Replace 'pretrain' with 'finetune' in the path if present,
            # otherwise append 'finetune' as a sibling directory.
            if "pretrain" in pretrain_dir:
                checkpoint_dir: str = pretrain_dir.replace("pretrain", "finetune")
            else:
                parent_dir: str = os.path.dirname(pretrain_dir)
                checkpoint_dir = os.path.join(parent_dir, "finetune")
        else:
            checkpoint_dir = "checkpoints/finetune"

        return checkpoint_dir

    # -----------------------------------------------------------------------
    # Public: main fine-tuning loop
    # -----------------------------------------------------------------------

    def finetune(self) -> Dict[str, List[float]]:
        """Run the complete adapter-only fine-tuning loop.

        Executes ``config.n_epochs`` training epochs. Each epoch consists of:
          1. Training pass over the full training dataset (_train_epoch).
          2. Validation pass over the full validation dataset (_validate).
          3. LR scheduler step (cosine: unconditional).
          4. Logging of epoch statistics (loss, LR, epoch time).
          5. Checkpoint saving if val_loss improved (best model).
          6. Periodic checkpoint saving every save_every epochs.

        The training history is accumulated in self._train_history and
        self._val_history and returned as a dict for downstream use by
        experiment scripts and the Evaluator.

        Returns:
            Dict with keys:
              - 'train_loss': List[float] of mean training loss per epoch.
              - 'val_loss': List[float] of mean validation loss per epoch.
              - 'epoch_time': List[float] of wall-clock time per epoch (s).
            All lists have length config.n_epochs.

        Note:
            The best model checkpoint is saved to
            ``{checkpoint_dir}/finetune_{physics_id}_best.pt`` whenever
            val_loss improves. This is the checkpoint used by the Evaluator
            for final test-set evaluation.

            The "Avg. epoch (s)" metric reported in Tables 1 and 2 of the
            paper is computed as the mean of epoch_time[_EPOCH_TIME_WARMUP:],
            excluding the first warmup epoch per config.yaml
            evaluation.epoch_time_warmup: 1.
        """
        self._logger.info(
            "Starting fine-tuning: %d epochs, %d train batches/epoch, "
            "%d val batches/epoch. physics_id='%s', freeze_backbone=%s.",
            self.config.n_epochs,
            len(self._train_loader),
            len(self._val_loader),
            self.physics_id,
            self.config.freeze_backbone,
        )

        # Determine save_every interval.
        # FinetuneConfig does not have save_every; use module-level default.
        save_every: int = _DEFAULT_SAVE_EVERY

        total_start_time: float = time.perf_counter()

        for epoch in range(1, self.config.n_epochs + 1):
            # ── Training pass ─────────────────────────────────────────────
            self.model.train()
            train_loss: float = self._train_epoch(epoch)
            self._train_history.append(train_loss)

            # ── Validation pass ───────────────────────────────────────────
            self.model.eval()
            val_loss: float = self._validate()
            self._val_history.append(val_loss)

            # ── LR scheduler step ─────────────────────────────────────────
            # CosineAnnealingLR: step unconditionally once per epoch.
            if self._scheduler_type == _SCHEDULER_COSINE:
                self._scheduler.step()  # type: ignore[union-attr]
            elif self._scheduler_type == _SCHEDULER_PLATEAU:
                self._scheduler.step(val_loss)  # type: ignore[union-attr]

            # ── Get current LR for logging ────────────────────────────────
            current_lr: float = self._optimizer.param_groups[0]["lr"]

            # ── Get epoch time (stored by _train_epoch) ───────────────────
            epoch_time: float = (
                self._epoch_times[-1] if self._epoch_times else 0.0
            )

            # ── Log epoch summary ─────────────────────────────────────────
            self._logger.info(
                "Epoch [%d/%d] | train_loss=%.6e | val_loss=%.6e | "
                "lr=%.2e | epoch_time=%.2fs",
                epoch,
                self.config.n_epochs,
                train_loss,
                val_loss,
                current_lr,
                epoch_time,
            )

            # ── Save best checkpoint ──────────────────────────────────────
            self.save_checkpoint(epoch, val_loss)

            # ── Periodic checkpoint ───────────────────────────────────────
            if epoch % save_every == 0:
                self._save_periodic_checkpoint(epoch, val_loss)

        # ── Training complete: log summary ────────────────────────────────
        total_elapsed: float = time.perf_counter() - total_start_time

        # Compute average epoch time excluding warmup epochs.
        avg_epoch_time: float = self._compute_avg_epoch_time()

        self._logger.info(
            "Fine-tuning complete. "
            "Best val_loss=%.6e. "
            "Avg. epoch time (excl. %d warmup)=%.2fs. "
            "Total training time=%.2fs. "
            "Best checkpoint: %s",
            self._best_val_loss,
            _EPOCH_TIME_WARMUP,
            avg_epoch_time,
            total_elapsed,
            os.path.join(
                self._checkpoint_dir,
                _CKPT_BEST_TEMPLATE.format(physics_id=self.physics_id),
            ),
        )

        return {
            "train_loss": list(self._train_history),
            "val_loss": list(self._val_history),
            "epoch_time": list(self._epoch_times),
        }

    # -----------------------------------------------------------------------
    # Private: single training epoch
    # -----------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch over the fine-tuning dataset.

        Iterates over the training DataLoader, routes each batch through
        the correct adapter pair via AdapterFramework.forward(input, physics_id),
        computes MSE loss, backpropagates, and steps the optimizer.

        Since the backbone is frozen (requires_grad=False), gradients only
        flow through the adapter parameters. The backward pass is therefore
        much cheaper than full-model training.

        Epoch timing (Shared Knowledge #6):
            Wall-clock time is measured with time.perf_counter() around the
            inner loop body (forward + loss + backward + optimizer step),
            excluding DataLoader iteration time. This reproduces the
            "Avg. epoch (s)" column in Tables 1 and 2 of the paper.

        NaN guard:
            If the loss is NaN or Inf, the optimizer step is skipped and a
            warning is logged. This prevents corrupting adapter weights from
            a single bad batch.

        Args:
            epoch: Current epoch number (1-indexed). Used for logging only.

        Returns:
            Mean training loss over all batches in the epoch.
            Returns 0.0 if no batches were processed (empty dataset).
        """
        total_loss: float = 0.0
        n_batches: int = 0
        total_compute_time: float = 0.0  # forward+backward time only

        for batch_input, batch_target in self._train_loader:
            # ── Move data to device ───────────────────────────────────────
            batch_input = batch_input.to(self._device, non_blocking=True)
            batch_target = batch_target.to(self._device, non_blocking=True)

            # ── Zero gradients ────────────────────────────────────────────
            self._optimizer.zero_grad(set_to_none=True)

            # ── Timed forward + backward pass ─────────────────────────────
            # Timer wraps the compute-intensive part only (not data loading).
            # This matches the "Avg. epoch (s)" measurement methodology
            # described in Shared Knowledge #6.
            compute_start: float = time.perf_counter()

            # Forward pass: routes through LiftingAdapter[physics_id] ->
            # backbone (frozen) -> ProjectionAdapter[physics_id].
            pred: Tensor = self.model.forward(batch_input, self.physics_id)

            # Compute MSE loss against target.
            # Target may have more channels than pred if the dataset was
            # constructed with extra channels; slice to match pred.
            n_out: int = pred.shape[1]
            target_sliced: Tensor = batch_target[:, :n_out]

            loss: Tensor = self._loss_fn(pred, target_sliced)

            # ── NaN/Inf guard ─────────────────────────────────────────────
            loss_val: float = loss.item()
            if not math.isfinite(loss_val):
                self._logger.warning(
                    "Epoch %d: non-finite loss=%.6e detected. "
                    "Skipping optimizer step for this batch. "
                    "Check data normalization and adapter initialization.",
                    epoch,
                    loss_val,
                )
                compute_end: float = time.perf_counter()
                total_compute_time += compute_end - compute_start
                continue

            # ── Backward pass ─────────────────────────────────────────────
            # Only adapter parameters have requires_grad=True (when backbone
            # is frozen), so only they accumulate gradients.
            loss.backward()

            # ── Optimizer step ────────────────────────────────────────────
            # Updates only the parameters in the optimizer's param_groups
            # (adapter params only when freeze_backbone=True).
            self._optimizer.step()

            compute_end = time.perf_counter()
            total_compute_time += compute_end - compute_start

            total_loss += loss_val
            n_batches += 1

        # ── Store epoch time ──────────────────────────────────────────────
        # Total compute time for this epoch (sum of per-batch compute times).
        # This is the wall-clock time for the training loop body, excluding
        # DataLoader iteration overhead.
        self._epoch_times.append(total_compute_time)

        if n_batches < _MIN_BATCHES_FOR_STATS:
            self._logger.warning(
                "Epoch %d: only %d batches processed. "
                "Training dataset may be too small or drop_last=True "
                "dropped all batches.",
                epoch,
                n_batches,
            )
            return 0.