## Code: training/pretrain.py

```python
## training/pretrain.py
"""
Multi-physics pretraining loop for the adapter-based neural operator
framework described in:
  "Towards Universal Neural Operators through Multiphysics Pretraining"

Implements the Pretrainer class that drives the pretraining phase where ALL
parameters are jointly optimized:
  (θ_{P_1}, ..., θ_{P_N}, θ_F, θ_{L_1}, ..., θ_{L_N})

This is in direct contrast to fine-tuning (Finetuner in training/finetune.py)
where θ_F is frozen and only new adapter parameters are updated.

Design contract (Data structures and interfaces):
  Pretrainer:
    __init__(model: AdapterFramework, train_dataset: MultiPhysicsDataset,
             val_dataset: MultiPhysicsDataset, config: PretrainConfig,
             device: torch.device)
    train() -> Dict[str, List[float]]
    _train_epoch(epoch: int) -> float
    _validate() -> float
    save_checkpoint(epoch: int, val_loss: float, tag: str = 'best') -> None
    load_checkpoint(path: str) -> None

Config alignment (config.yaml):
  training.pretrain.lr: 1.0e-3          -> Adam learning rate
  training.pretrain.weight_decay: 1.0e-4 -> Adam weight decay
  training.pretrain.batch_size: 16       -> DataLoader batch size
  training.pretrain.n_epochs: 200        -> Training loop iterations
  training.pretrain.scheduler: "cosine"  -> CosineAnnealingLR
  training.pretrain.checkpoint_dir: "checkpoints/pretrain"
  training.pretrain.save_every: 10       -> Periodic checkpoint interval
  training.loss: "mse"                   -> Training loss function

Tensor layout convention (Shared Knowledge #1):
  Channel-first: [B, C, L] for 1D, [B, C, H, W] for 2D.

Physics ID strings (Shared Knowledge #2):
  No dots — use 'p' for decimal point.
  Received from MultiPhysicsDataset; not created here.

Epoch time measurement (Shared Knowledge #6):
  Wall-clock time per epoch measured with time.perf_counter().
  Timing covers forward+backward pass only, not DataLoader iteration.

Dependencies:
  torch, torch.nn, torch.optim, torch.utils.data
  time, logging, os, typing, collections
  models/adapter_framework.py  -> AdapterFramework
  data/multiphysics_dataset.py -> MultiPhysicsDataset (collate_fn)
  training/losses.py           -> MSELoss, get_loss_fn
  utils/config.py              -> PretrainConfig
  utils/logging_utils.py       -> get_logger
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.multiphysics_dataset import MultiPhysicsDataset
from models.adapter_framework import AdapterFramework
from training.losses import get_loss_fn
from utils.config import PretrainConfig
from utils.logging_utils import get_logger

# ---------------------------------------------------------------------------
# Module-level logger (fallback; Pretrainer creates its own instance logger)
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

# Scheduler type string constants (match config.yaml training.pretrain.scheduler)
_SCHEDULER_COSINE: str = "cosine"
_SCHEDULER_PLATEAU: str = "plateau"

# Checkpoint filename templates
_CKPT_BEST_FILENAME: str = "pretrain_best.pt"
_CKPT_META_SUFFIX: str = "_meta.pt"
_CKPT_PERIODIC_TEMPLATE: str = "pretrain_epoch_{epoch:04d}.pt"


# ---------------------------------------------------------------------------
# Pretrainer
# ---------------------------------------------------------------------------


class Pretrainer:
    """Multi-physics pretraining loop for the AdapterFramework.

    Drives the pretraining phase where ALL parameters are jointly optimized:
    the shared FNO backbone (θ_F) plus all problem-specific lifting and
    projection adapters (θ_{L_1},...,θ_{L_N}, θ_{P_1},...,θ_{P_N}).

    The training loop iterates over a MultiPhysicsDataset that interleaves
    samples from N physics problems. Each batch is routed through the
    correct adapter pair via AdapterFramework.forward(a, physics_id).

    Multi-physics routing strategy:
        The collate_fn zero-pads inputs to the maximum n_in in each batch.
        If all samples in a batch share the same physics_id (round-robin
        sampling, the default per config.yaml), a single forward pass
        suffices. If a batch contains mixed physics, samples are grouped
        by physics_id and forwarded separately, then losses are averaged.

    Attributes:
        model: AdapterFramework with all adapters registered.
        train_dataset: MultiPhysicsDataset for training.
        val_dataset: MultiPhysicsDataset for validation.
        config: PretrainConfig with all hyperparameters.
        device: Target device (CPU or CUDA).
        _optimizer: Adam optimizer over all model parameters.
        _scheduler: LR scheduler (CosineAnnealingLR or ReduceLROnPlateau).
        _scheduler_type: String identifying the scheduler type.
        _loss_fn: Training loss function (MSELoss by default).
        _train_loader: DataLoader for training data.
        _val_loader: DataLoader for validation data.
        _logger: Instance-specific logger.
        _best_val_loss: Best validation loss seen so far (for checkpointing).
        _train_history: Accumulated training loss history.
        _val_history: Accumulated validation loss history.

    Example::

        from models.fno_backbone import FNOBackbone
        from models.adapter_framework import AdapterFramework
        from data.multiphysics_dataset import MultiPhysicsDataset
        from utils.config import PretrainConfig

        backbone = FNOBackbone(hidden_dim=64, n_modes=16, n_layers=4)
        model = AdapterFramework(backbone=backbone, hidden_dim=64)
        model.register_adapter('burgers_nu0p01', n_in=1, n_out=1)

        config = PretrainConfig(
            physics_list=['burgers_nu0p01'],
            n_epochs=200, lr=1e-3, batch_size=16,
            hidden_dim=64, n_modes=16, n_layers=4,
            weight_decay=1e-4, scheduler='cosine',
            checkpoint_dir='checkpoints/pretrain',
        )

        pretrainer = Pretrainer(
            model=model,
            train_dataset=multi_train_ds,
            val_dataset=multi_val_ds,
            config=config,
            device=torch.device('cuda'),
        )
        history = pretrainer.train()
    """

    def __init__(
        self,
        model: AdapterFramework,
        train_dataset: MultiPhysicsDataset,
        val_dataset: MultiPhysicsDataset,
        config: PretrainConfig,
        device: torch.device,
    ) -> None:
        """Initialise Pretrainer.

        Sets up the optimizer (Adam over all parameters), LR scheduler
        (CosineAnnealingLR or ReduceLROnPlateau), training loss (MSE),
        DataLoaders (with MultiPhysicsDataset.collate_fn), logger, and
        checkpoint directory.

        Args:
            model: AdapterFramework with all pretraining adapters already
                registered via register_adapter(). The model is moved to
                ``device`` immediately. All parameters must have
                requires_grad=True (backbone is NOT frozen during pretrain).
            train_dataset: MultiPhysicsDataset for training. Must implement
                __len__, __getitem__ (returning (input, target, physics_id)),
                and expose the static collate_fn.
            val_dataset: MultiPhysicsDataset for validation. Same interface
                as train_dataset. Used to select the best checkpoint.
            config: PretrainConfig populated from config.yaml. All
                hyperparameters (lr, batch_size, n_epochs, etc.) are read
                from this object — no hardcoded values.
            device: Target device. The model and all batches are moved to
                this device. Use torch.device('cuda') for GPU training
                (required for MambaFNO) or torch.device('cpu') for testing.

        Raises:
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

        # ── Store references ──────────────────────────────────────────────
        self.model: AdapterFramework = model.to(device)
        self.train_dataset: MultiPhysicsDataset = train_dataset
        self.val_dataset: MultiPhysicsDataset = val_dataset
        self.config: PretrainConfig = config
        self.device: torch.device = device

        # ── Tracking state ────────────────────────────────────────────────
        self._best_val_loss: float = float("inf")
        self._train_history: List[float] = []
        self._val_history: List[float] = []

        # ── Checkpoint directory ──────────────────────────────────────────
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # ── Logger ────────────────────────────────────────────────────────
        log_file: str = os.path.join(config.checkpoint_dir, "pretrain.log")
        self._logger: logging.Logger = get_logger("Pretrainer", log_file=log_file)

        # ── Verify backbone is NOT frozen (pretraining requires all params) ─
        n_frozen: int = sum(
            1 for p in self.model._backbone.parameters()
            if not p.requires_grad
        )
        if n_frozen > 0:
            self._logger.warning(
                "Pretrainer: %d backbone parameters have requires_grad=False. "
                "During pretraining, ALL parameters should be trainable. "
                "Call model.unfreeze_backbone() before constructing Pretrainer "
                "if the backbone was previously frozen.",
                n_frozen,
            )

        # ── Optimizer: Adam over ALL parameters ───────────────────────────
        # Pretraining phase: backbone + all adapters are jointly optimized.
        # lr and weight_decay from config.yaml training.pretrain section.
        self._optimizer: Adam = Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # ── LR Scheduler ─────────────────────────────────────────────────
        # Branch on config.scheduler string (from config.yaml
        # training.pretrain.scheduler: "cosine").
        self._scheduler_type: str = config.scheduler.strip().lower()

        if self._scheduler_type == _SCHEDULER_COSINE:
            # CosineAnnealingLR: decays lr from initial to eta_min over T_max
            # epochs. T_max = n_epochs (one full cosine cycle per training run).
            # eta_min = 1e-6 from config.yaml training.pretrain.scheduler_params.
            self._scheduler: object = CosineAnnealingLR(
                self._optimizer,
                T_max=config.n_epochs,
                eta_min=1.0e-6,
            )
        elif self._scheduler_type == _SCHEDULER_PLATEAU:
            # ReduceLROnPlateau: reduces lr when val_loss stops improving.
            # patience=10, factor=0.5 are reasonable defaults.
            self._scheduler = ReduceLROnPlateau(
                self._optimizer,
                mode="min",
                patience=10,
                factor=0.5,
                min_lr=1.0e-6,
            )
        else:
            self._logger.warning(
                "Unknown scheduler type '%s'. Defaulting to CosineAnnealingLR.",
                config.scheduler,
            )
            self._scheduler_type = _SCHEDULER_COSINE
            self._scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=config.n_epochs,
                eta_min=1.0e-6,
            )

        # ── Loss function ─────────────────────────────────────────────────
        # Training loss from config.yaml training.loss: "mse".
        # NMAE is only used for evaluation (Evaluator), not training.
        # get_loss_fn handles the "mse" -> MSELoss mapping.
        self._loss_fn: nn.Module = get_loss_fn("mse")

        # ── DataLoaders ───────────────────────────────────────────────────
        # CRITICAL: collate_fn=MultiPhysicsDataset.collate_fn must be passed
        # explicitly. This zero-pads inputs to max_n_in in each batch,
        # enabling batching across physics with different input cardinalities.
        pin_memory: bool = (device.type == "cuda")

        self._train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=MultiPhysicsDataset.collate_fn,
            num_workers=_DEFAULT_NUM_WORKERS,
            pin_memory=pin_memory,
            drop_last=False,
        )

        self._val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=MultiPhysicsDataset.collate_fn,
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
            "Pretrainer initialized: "
            "n_epochs=%d, lr=%.2e, weight_decay=%.2e, batch_size=%d, "
            "scheduler='%s', loss='mse', device='%s'. "
            "Model: total_params=%d, trainable_params=%d. "
            "Train samples=%d, Val samples=%d.",
            config.n_epochs,
            config.lr,
            config.weight_decay,
            config.batch_size,
            self._scheduler_type,
            str(device),
            n_total_params,
            n_trainable_params,
            len(train_dataset),
            len(val_dataset),
        )

    # -----------------------------------------------------------------------
    # Public: main training loop
    # -----------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        """Run the full multi-physics pretraining loop.

        Executes ``config.n_epochs`` training epochs. Each epoch consists of:
          1. Training pass over the full training dataset (_train_epoch).
          2. Validation pass over the full validation dataset (_validate).
          3. LR scheduler step (cosine: unconditional; plateau: with val_loss).
          4. Logging of epoch statistics (loss, LR, epoch time).
          5. Checkpoint saving if val_loss improved (best model).
          6. Periodic checkpoint saving every config.save_every epochs.

        The training history is accumulated in self._train_history and
        self._val_history and returned as a dict for downstream use by
        experiment scripts.

        Returns:
            Dict with keys:
              - 'train_loss': List[float] of mean training loss per epoch.
              - 'val_loss': List[float] of mean validation loss per epoch.
            Both lists have length config.n_epochs.

        Note:
            The best model checkpoint is saved to
            ``config.checkpoint_dir/pretrain_best.pt`` whenever val_loss
            improves. This is the checkpoint loaded by Finetuner.
        """
        self._logger.info(
            "Starting pretraining: %d epochs, %d train batches/epoch, "
            "%d val batches/epoch.",
            self.config.n_epochs,
            len(self._train_loader),
            len(self._val_loader),
        )

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
            # ReduceLROnPlateau: step with current val_loss.
            if self._scheduler_type == _SCHEDULER_COSINE:
                self._scheduler.step()  # type: ignore[union-attr]
            elif self._scheduler_type == _SCHEDULER_PLATEAU:
                self._scheduler.step(val_loss)  # type: ignore[union-attr]

            # ── Get current LR for logging ────────────────────────────────
            current_lr: float = self._optimizer.param_groups[0]["lr"]

            # ── Log epoch summary ─────────────────────────────────────────
            self._logger.info(
                "Epoch [%d/%d] | train_loss=%.6e | val_loss=%.6e | lr=%.2e",
                epoch,
                self.config.n_epochs,
                train_loss,
                val_loss,
                current_lr,
            )

            # ── Save best checkpoint ──────────────────────────────────────
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, tag="best")
                self._logger.info(
                    "New best val_loss=%.6e at epoch %d. Checkpoint saved.",
                    val_loss,
                    epoch,
                )

            # ── Periodic checkpoint ───────────────────────────────────────
            # Save every config.save_every epochs (from config.yaml
            # training.pretrain.save_every: 10).
            save_every: int = getattr(self.config, "save_every", 10)
            if epoch % save_every == 0:
                periodic_tag: str = f"epoch_{epoch:04d}"
                self.save_checkpoint(epoch, val_loss, tag=periodic_tag)
                self._logger.debug(
                    "Periodic checkpoint saved at epoch %d.", epoch
                )

        self._logger.info(
            "Pretraining complete. Best val_loss=%.6e. "
            "Checkpoint: %s",
            self._best_val_loss,
            os.path.join(self.config.checkpoint_dir, _CKPT_BEST_FILENAME),
        )

        return {
            "train_loss": list(self._train_history),
            "val_loss": list(self._val_history),
        }

    # -----------------------------------------------------------------------
    # Private: single training epoch
    # -----------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch over the full training dataset.

        Iterates over the training DataLoader, routes each batch through
        the correct adapter pair(s) via AdapterFramework.forward(), computes
        MSE loss, backpropagates, and steps the optimizer.

        Multi-physics routing:
            If all samples in a batch share the same physics_id (round-robin
            sampling, the default), a single forward pass is used. If a batch
            contains mixed physics, samples are grouped by physics_id and
            forwarded separately; losses are averaged weighted by group size.

        Epoch timing (Shared Knowledge #6):
            Wall-clock time is measured with time.perf_counter() around the
            forward+backward pass only (inside the batch loop), excluding
            DataLoader iteration time. This reproduces the "Avg. epoch (s)"
            column in Tables 1 and 2 of the paper.

        NaN guard:
            If the loss is NaN or Inf, the optimizer step is skipped and a
            warning is logged. This prevents corrupting model weights from
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

        for batch_input, batch_target, physics_ids in self._train_loader:
            # ── Move data to device ───────────────────────────────────────
            batch_input = batch_input.to(self.device, non_blocking=True)
            batch_target = batch_target.to(self.device, non_blocking=True)

            # ── Zero gradients ────────────────────────────────────────────
            self._optimizer.zero_grad(set_to_none=True)

            # ── Forward + loss (timed) ────────────────────────────────────
            compute_start: float = time.perf_counter()

            loss: Tensor = self._forward_multiphysics_batch(
                batch_input, batch_target, physics_ids
            )

            # ── NaN/Inf guard ─────────────────────────────────────────────
            loss_val: float = loss.item()
            if not math.isfinite(loss_val):
                self._logger.warning(
                    "Epoch %d: non-finite loss=%.6e detected. "
                    "Skipping optimizer step for this batch. "
                    "Check data normalization and model initialization.",
                    epoch,
                    loss_val,
                )
                continue

            # ── Backward pass ─────────────────────────────────────────────
            loss.backward()
            self._optimizer.step()

            compute_end: float = time.perf_counter()
            total_compute_time += compute_end - compute_start

            total_loss += loss_val
            n_batches += 1

        if n_batches < _MIN_BATCHES_FOR_STATS:
            self._logger.warning(
                "Epoch %d: only %d batches processed. "
                "Training dataset may be too small.",
                epoch,
                n_batches,
            )
            return 0.0

        mean_loss: float = total_loss / n_batches
        avg_batch_compute_time: float = total_compute_time / n_batches

        self._logger.debug(
            "Epoch %d train: mean_loss=%.6e, n_batches=%d, "
            "avg_batch_compute_time=%.4fs, total_compute_time=%.2fs.",
            epoch,
            mean_loss,
            n_batches,
            avg_batch_compute_time,
            total_compute_time,
        )

        return mean_loss

    # -----------------------------------------------------------------------
    # Private: validation pass
    # -----------------------------------------------------------------------

    def _validate(self) -> float:
        """Run one validation pass over the full validation dataset.

        Identical routing logic to _train_epoch but with torch.no_grad()
        context manager to disable gradient computation. This reduces memory
        usage and speeds up validation, especially for large Perceiver/CoDA-NO
        models.

        Returns:
            Mean validation loss over all batches.
            Returns float('inf') if no batches were processed (empty dataset).
        """
        total_loss: float = 0.0
        n_batches: int = 0

        with torch.no_grad():
            for batch_input, batch_target, physics_ids in self._val_loader:
                # ── Move data to device ───────────────────────────────────
                batch_input = batch_input.to(self.device, non_blocking=True)
                batch_target = batch_target.to(self.device, non_blocking=True)

                # ── Forward + loss (no backward) ──────────────────────────
                loss: Tensor = self._forward_multiphysics_batch(
                    batch_input, batch_target, physics_ids
                )

                loss_val: float = loss.item()
                if not math.isfinite(loss_val):
                    self._logger.warning(
                        "Validation: non-finite loss=%.6e detected. "
                        "Skipping this batch.",
                        loss_val,
                    )
                    continue

                total_loss += loss_val
                n_batches += 1

        if n_batches < _MIN_BATCHES_FOR_STATS:
            self._logger.warning(
                "Validation: only %d batches processed. "
                "Validation dataset may be too small.",
                n_batches,
            )
            return float("inf")

        mean_loss: float = total_loss / n_batches

        self._logger.debug(
            "Validation: mean_loss=%.6e, n_batches=%d.",
            mean_loss,
            n_batches,
        )

        return mean_loss

    # -----------------------------------------------------------------------
    # Private: multi-physics batch routing
    # -----------------------------------------------------------------------

    def _forward_multiphysics_batch(
        self,
        batch_input: Tensor,
        batch_target: Tensor,
        physics_ids: List[str],
    ) -> Tensor:
        """Route a multi-physics batch through the AdapterFramework.

        Handles two cases:
          1. Homogeneous batch (all physics_ids identical): single forward
             pass through AdapterFramework.forward(batch_input, physics_id).
             This is the common case with round-robin sampling.
          2. Heterogeneous batch (mixed physics_ids): group samples by
             physics_id, forward each group separately, compute weighted
             average loss. This handles interleaved sampling strategies.

        The collate_fn zero-pads inputs to max_n_in in the batch. The
        LiftingAdapter for each physics_id reads only its first n_in channels
        (ignoring padding), so the routing is correct regardless of padding.

        Loss weighting:
            For heterogeneous batches, the loss from each physics group is
            weighted by the number of samples in that group (proportional
            weighting). This gives equal weight per sample across physics,
            regardless of how many samples each physics contributes to the
            batch.

        Args:
            batch_input: Zero-padded input tensor of shape
                [B, max_n_in, *spatial]. Produced by collate_fn.
            batch_target: Target tensor of shape [B, max_n_out, *spatial].
                Produced by collate_fn.
            physics_ids: List of physics ID strings of length B. Each
                element identifies the physics problem for the corresponding
                sample in the batch.

        Returns:
            Scalar loss tensor (MSE). Differentiable with respect to all
            model parameters (for backward pass in _train_epoch).

        Raises:
            KeyError: If a physics_id in the batch is not registered in the
                AdapterFramework (propagated from AdapterFramework.forward).
        """
        batch_size: int = len(physics_ids)

        # ── Case 1: Homogeneous batch (all same physics_id) ───────────────
        # Check if all physics_ids are identical — this is the common case
        # with round-robin single-physics batches (default per config.yaml).
        first_pid: str = physics_ids[0]
        is_homogeneous: bool = all(pid == first_pid for pid in physics_ids)

        if is_homogeneous:
            # Single forward pass: all samples use the same adapter pair.
            # batch_input: [B, max_n_in, *spatial]
            # pred:        [B, n_out, *spatial]
            pred: Tensor = self.model.forward(batch_input, first_pid)

            # Compute loss against the first n_out channels of batch_target.
            # batch_target may be padded to max_n_out; we use only the
            # channels relevant to this physics.
            n_out: int = pred.shape[1]
            target_sliced: Tensor = batch_target[:, :n_out]

            loss: Tensor = self._loss_fn(pred, target_sliced)
            return loss

        # ── Case 2: Heterogeneous batch (mixed physics_ids) ───────────────
        # Group sample indices by physics_id.
        # defaultdict(list) maps physics_id -> list of batch indices.
        physics_groups: Dict[str, List[int]] = defaultdict(list)
        for sample_idx, pid in enumerate(physics_ids):
            physics_groups[pid].append(sample_idx)

        # Accumulate weighted loss across physics groups.
        # Use a list of (loss_tensor, weight) pairs to avoid in-place ops
        # that could interfere with autograd.
        weighted_losses: List[Tuple[Tensor, float]] = []

        for pid, indices in physics_groups.items():
            # Convert indices to a tensor for advanced indexing.
            idx_tensor: Tensor = torch.tensor(
                indices, dtype=torch.long, device=self.device
            )

            # Slice sub-batch for this physics group.
            # sub_input:  [sub_B, max_n_in, *spatial]
            # sub_target: [sub_B, max_n_out, *spatial]
            sub_input: Tensor = batch_input[idx_tensor]
            sub_target: Tensor = batch_target[idx_tensor]

            # Forward through the correct adapter pair.
            # LiftingAdapter[pid] reads only its first n_in channels.
            # sub_pred: [sub_B, n_out_pid, *spatial]
            sub_pred: Tensor = self.model.forward(sub_input, pid)

            # Slice target to match prediction channel count.
            n_out_pid: int = sub_pred.shape[1]
            sub_target_sliced: Tensor = sub_target[:, :n_out_pid]

            # Compute MSE loss for this physics group.
            sub_loss: Tensor = self._loss_fn(sub_pred, sub_target_sliced)

            # Weight by group size (number of samples from this physics).
            group_weight: float = float(len(indices)) / float(batch_size)
            weighted