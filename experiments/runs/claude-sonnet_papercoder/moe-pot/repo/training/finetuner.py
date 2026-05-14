# training/finetuner.py
"""Fine-tuning loop for the MoE-POT architecture.

Implements the Finetuner class that handles two distinct fine-tuning scenarios
described in the paper:

1. Dataset fine-tuning (Appendix B.3): 200 epochs, warmup=40, on each of the
   6 pre-training datasets individually. The router-gating network is frozen
   to preserve the expert assignment strategy learned during pre-training.

2. Downstream task fine-tuning (Appendix B.3): 500 epochs, warmup=100, on
   NS(1e-4), CNS(1,0.01), and PDEArena datasets.

Key behavioral differences from Trainer:
  - No noise injection (config.yaml finetuning.noise_injection: false)
  - Router-gating network frozen (config.yaml finetuning.freeze_router: true)
  - Single dataset DataLoader (no BalancedMultiDatasetSampler)
  - Load balancing loss discarded (router frozen, balance loss irrelevant)
  - Only L2RelativeLoss used as the training objective

From the paper (Appendix B.3):
    "we freeze the parameters of the router-gating network during fine-tuning
    to preserve the expert assignment strategy obtained from the joint training
    stage. Only the expert networks are updated to adapt to the target dataset."

From config.yaml (finetuning section):
    num_epochs: 200
    learning_rate: 1.0e-3
    warmup_epochs: 40
    freeze_router: true
    noise_injection: false

From config.yaml (downstream section):
    num_epochs: 500
    learning_rate: 1.0e-3
    warmup_epochs: 100
    freeze_router: true
    noise_injection: false
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.moe_pot import MoEPOT
from training.losses import L2RelativeLoss
from utils.checkpoint import Checkpointer
from utils.logger import Logger


class Finetuner:
    """Orchestrates the MoE-POT fine-tuning phase on a single PDE dataset.

    Handles both dataset fine-tuning (200 epochs) and downstream task
    fine-tuning (500 epochs) by reading the appropriate epoch count and
    warmup schedule from the config object passed by the caller.

    The router-gating network is frozen immediately in __init__ before
    optimizer creation, ensuring frozen parameters are excluded from
    optimizer state (no wasted memory on momentum/variance for frozen params).

    Only the L2RelativeLoss is used as the training objective. The load
    balancing loss returned by MoEPOT.forward() is discarded because:
      1. The frozen router cannot be updated by the balance loss gradient.
      2. During single-dataset fine-tuning, routing collapse toward the
         relevant experts is actually desirable behavior.

    Attributes:
        model: The model used for forward/backward passes. May be a
            DistributedDataParallel wrapper around raw_model when DDP
            is active.
        raw_model: The unwrapped MoEPOT instance. Always used for
            checkpointing to avoid the 'module.' prefix in DDP state dicts.
        optimizer: Adam optimizer over trainable (non-frozen) parameters only.
            Router parameters are excluded via requires_grad=False filtering.
        scheduler: OneCycleLR scheduler stepped once per batch.
        train_loader: DataLoader for the single fine-tuning dataset.
        val_loader: DataLoader for the validation/test split of the same
            dataset. May be None if no validation is desired.
        config: Configuration object exposing hyperparameters from config.yaml
            finetuning or downstream sections as attributes.
        logger: Logger instance for file/stdout/wandb logging.
        checkpointer: Checkpointer instance for saving .pt files.
        criterion: L2RelativeLoss instance — the sole training objective.
        device: Target device (cuda:local_rank or cpu).
        is_distributed: Whether torch.distributed is initialized.
        local_rank: Local GPU rank (0 for single-GPU or non-distributed).
        is_main_process: True if this process should log and checkpoint.
        best_val_l2re: Best validation L2RE seen so far. Used to determine
            when to save the 'best.pt' checkpoint.
    """

    def __init__(
        self,
        model: MoEPOT,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Any,
        logger: Logger,
        checkpointer: Checkpointer,
    ) -> None:
        """Initializes the Finetuner, freezing the router before optimizer creation.

        The order of operations is critical:
          1. freeze_router() MUST be called before Adam() so that frozen
             parameters are excluded from optimizer parameter groups.
          2. The model is moved to the correct device before DDP wrapping.
          3. OneCycleLR is configured with pct_start derived from config
             warmup_epochs / num_epochs.

        Args:
            model: Initialized MoEPOT model, already loaded with pre-trained
                weights via Checkpointer.load_model_only() in main.py.
                Will be moved to the correct device and optionally wrapped
                in DistributedDataParallel if torch.distributed is active.
            train_loader: DataLoader for the fine-tuning training split.
                Typically a single-dataset loader (no BalancedMultiDatasetSampler).
                May return 2-tuples (u_input, u_target) or 3-tuples
                (u_input, u_target, dataset_idx) — both are handled.
            val_loader: DataLoader for the validation/test split. May be
                None if no validation is desired (e.g., during ablation
                studies where only training metrics are needed).
            config: Configuration object with attributes matching config.yaml
                finetuning or downstream sections. Required attributes:
                  - config.learning_rate: float (1e-3)
                  - config.weight_decay: float (1e-6)
                  - config.beta1: float (0.9)
                  - config.beta2: float (0.9)
                  - config.num_epochs: int (200 for finetune, 500 for downstream)
                  - config.warmup_epochs: int (40 for finetune, 100 for downstream)
                  - config.save_interval: int (50, from logging section)
                  - config.log_interval: int (10, from logging section)
            logger: Logger instance for writing metrics to file, stdout,
                and optionally wandb.
            checkpointer: Checkpointer instance for saving model state dicts
                to disk. Saves periodic checkpoints and the best model.
        """
        # ----------------------------------------------------------------
        # Step 1: Distributed training setup
        # ----------------------------------------------------------------
        # Mirror the Trainer's DDP handling for consistency. Fine-tuning
        # can also benefit from multi-GPU acceleration, especially for
        # larger datasets like CFDBench (9000 training samples).
        self.is_distributed: bool = (
            dist.is_available() and dist.is_initialized()
        )

        if self.is_distributed:
            self.local_rank: int = dist.get_rank()
            self.is_main_process: bool = self.local_rank == 0
        else:
            self.local_rank = 0
            self.is_main_process = True

        # ----------------------------------------------------------------
        # Step 2: Device setup
        # ----------------------------------------------------------------
        if torch.cuda.is_available():
            self.device: torch.device = torch.device(
                f"cuda:{self.local_rank}"
            )
        else:
            self.device = torch.device("cpu")

        # ----------------------------------------------------------------
        # Step 3: Move model to device
        # ----------------------------------------------------------------
        model = model.to(self.device)

        # ----------------------------------------------------------------
        # Step 4: Freeze the router-gating network BEFORE optimizer creation
        # ----------------------------------------------------------------
        # This is the most critical step in Finetuner.__init__. Calling
        # freeze_router() sets requires_grad=False on all RouterGating
        # parameters across all MoEBlocks. The subsequent Adam() call
        # uses filter(lambda p: p.requires_grad, ...) to exclude these
        # frozen parameters from optimizer state, saving memory and
        # preventing any accidental router updates.
        #
        # From paper Appendix B.3:
        #   "we freeze the parameters of the router-gating network during
        #   fine-tuning to preserve the expert assignment strategy obtained
        #   from the joint training stage."
        model.freeze_router()

        # ----------------------------------------------------------------
        # Step 5: Store raw model reference (before DDP wrapping)
        # ----------------------------------------------------------------
        # Always keep a reference to the unwrapped model for:
        #   - Checkpointing (avoids 'module.' prefix in DDP state dict keys)
        #   - Calling unfreeze_router() after fine-tuning if needed
        #   - count_parameters() and get_router_weights() for analysis
        self.raw_model: MoEPOT = model

        # ----------------------------------------------------------------
        # Step 6: Optionally wrap in DistributedDataParallel
        # ----------------------------------------------------------------
        if self.is_distributed:
            # find_unused_parameters=False: safe because all non-frozen
            # parameters are used in every forward pass. Frozen router
            # parameters are excluded from DDP's gradient synchronization
            # automatically since they have requires_grad=False.
            self.model: nn.Module = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        else:
            self.model = model

        # ----------------------------------------------------------------
        # Step 7: Store data loaders and config
        # ----------------------------------------------------------------
        self.train_loader: DataLoader = train_loader
        self.val_loader: Optional[DataLoader] = val_loader
        self.config: Any = config
        self.logger: Logger = logger
        self.checkpointer: Checkpointer = checkpointer

        # ----------------------------------------------------------------
        # Step 8: Loss function
        # ----------------------------------------------------------------
        # L2RelativeLoss is the sole training objective during fine-tuning.
        # The load balancing loss from MoEPOT.forward() is discarded.
        self.criterion: L2RelativeLoss = L2RelativeLoss()

        # ----------------------------------------------------------------
        # Step 9: Optimizer over trainable parameters only
        # ----------------------------------------------------------------
        # filter(lambda p: p.requires_grad, ...) excludes frozen router
        # parameters from the optimizer's parameter groups. This is more
        # robust than relying on zero gradients and reduces optimizer memory.
        #
        # Adam hyperparameters from config.yaml (inherited from pretraining):
        #   learning_rate: 1.0e-3
        #   weight_decay: 1.0e-6
        #   beta1: 0.9
        #   beta2: 0.9
        trainable_params = filter(
            lambda p: p.requires_grad,
            self.model.parameters(),
        )
        self.optimizer: Adam = Adam(
            trainable_params,
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
            betas=(float(config.beta1), float(config.beta2)),
        )

        # ----------------------------------------------------------------
        # Step 10: OneCycleLR scheduler (stepped per batch)
        # ----------------------------------------------------------------
        # pct_start = warmup_epochs / num_epochs:
        #   Finetune:   40 / 200 = 0.2
        #   Downstream: 100 / 500 = 0.2
        # Both happen to be 0.2, but computed dynamically from config.
        #
        # steps_per_epoch = number of batches in train_loader.
        # OneCycleLR requires total_steps = num_epochs * steps_per_epoch.
        steps_per_epoch: int = max(len(train_loader), 1)
        pct_start: float = float(config.warmup_epochs) / float(config.num_epochs)

        self.scheduler: OneCycleLR = OneCycleLR(
            self.optimizer,
            max_lr=float(config.learning_rate),
            epochs=int(config.num_epochs),
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            # PyTorch defaults matching Trainer configuration:
            # div_factor=25 → initial_lr = max_lr / 25 = 4e-5
            # final_div_factor=1e4 → min_lr = initial_lr / 1e4 = 4e-9
            div_factor=25.0,
            final_div_factor=1e4,
            anneal_strategy="cos",
        )

        # ----------------------------------------------------------------
        # Step 11: Training state tracking
        # ----------------------------------------------------------------
        # Tracks the best validation L2RE for best.pt checkpointing.
        # Initialized to infinity so the first validation result always
        # triggers a best-model save.
        self.best_val_l2re: float = float("inf")

    def _prepare_batch(
        self,
        batch: Tuple,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Moves a batch from DataLoader to the target device without noise.

        Handles both 2-tuple (u_input, u_target) and 3-tuple
        (u_input, u_target, dataset_idx) batch formats. Single-dataset
        fine-tuning loaders typically return 2-tuples, but MultiPDEDataset
        returns 3-tuples. Both are supported for flexibility.

        No noise injection is applied — this is the critical behavioral
        difference from Trainer._prepare_batch(). The config.yaml setting
        finetuning.noise_injection: false enforces this.

        Args:
            batch: Tuple from DataLoader. Either:
                - 2-tuple: (u_input, u_target) from single-dataset loader
                - 3-tuple: (u_input, u_target, dataset_idx) from
                  MultiPDEDataset loader

        Returns:
            Tuple (u_input, u_target) with both tensors on self.device:
              - u_input: shape (B, T, C, H, W) — T=10 input frames
              - u_target: shape (B, C, H, W) — next frame to predict
        """
        # Unpack batch — handle both 2-tuple and 3-tuple formats.
        if len(batch) == 3:
            u_input: torch.Tensor = batch[0]
            u_target: torch.Tensor = batch[1]
            # dataset_idx (batch[2]) is discarded — not needed for fine-tuning
        elif len(batch) == 2:
            u_input = batch[0]
            u_target = batch[1]
        else:
            raise ValueError(
                f"Unexpected batch format: expected 2 or 3 elements, "
                f"got {len(batch)}. Batch element types: "
                f"{[type(b).__name__ for b in batch]}"
            )

        # Move tensors to target device with non_blocking=True for async
        # CPU→GPU transfer when DataLoader uses pin_memory=True.
        u_input = u_input.to(self.device, non_blocking=True)
        u_target = u_target.to(self.device, non_blocking=True)

        return u_input, u_target

    def finetune_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one full fine-tuning epoch over the single-dataset training split.

        Iterates over all batches in train_loader, computes the L2 relative
        error loss (no noise, no balance loss), and updates model parameters
        via backpropagation. Only non-frozen parameters (expert networks and
        other non-router components) are updated.

        The OneCycleLR scheduler is stepped once per batch (not per epoch),
        which is required for correct LR behavior with OneCycleLR.

        Args:
            epoch: Current epoch number (1-indexed). Used for tqdm display
                and logging.

        Returns:
            Dictionary with training metrics averaged over all batches:
              - 'train_loss': Mean L2 relative error (same as train_l2re
                since balance loss is discarded during fine-tuning)
              - 'train_l2re': Mean L2 relative error
              - 'lr': Current learning rate after the last scheduler step
        """
        self.model.train()

        # Metric accumulators.
        total_loss: float = 0.0
        total_l2re: float = 0.0
        num_batches: int = 0

        # Log interval from config.yaml logging.log_interval: 10
        log_interval: int = int(getattr(self.config, "log_interval", 10))

        # tqdm progress bar (only on main process to avoid duplicate output).
        loader_iter = tqdm(
            self.train_loader,
            desc=f"Finetune Epoch {epoch}/{self.config.num_epochs}",
            disable=not self.is_main_process,
            leave=False,
        )

        batch: Tuple
        for batch in loader_iter:
            # ----------------------------------------------------------
            # Step 1: Prepare batch — move to device, no noise injection
            # ----------------------------------------------------------
            u_input: torch.Tensor
            u_target: torch.Tensor
            u_input, u_target = self._prepare_batch(batch)
            # u_input shape:  (B, T, C, H, W)
            # u_target shape: (B, C, H, W)

            # ----------------------------------------------------------
            # Step 2: Zero gradients
            # ----------------------------------------------------------
            self.optimizer.zero_grad()

            # ----------------------------------------------------------
            # Step 3: Forward pass
            # ----------------------------------------------------------
            # MoEPOT.forward() always returns (u_pred, total_balance_loss).
            # The balance_loss is discarded during fine-tuning because:
            #   1. The frozen router cannot be updated by balance loss gradients.
            #   2. Single-dataset routing collapse toward relevant experts
            #      is desirable behavior during fine-tuning.
            u_pred: torch.Tensor
            _balance_loss: torch.Tensor
            u_pred, _balance_loss = self.model(u_input)
            # u_pred shape: (B, C, H, W)

            # ----------------------------------------------------------
            # Step 4: Compute prediction loss (L2 relative error only)
            # ----------------------------------------------------------
            # L2RelativeLoss returns mean L2RE over the batch:
            #   L2RE = ||pred - target||_2 / ||target||_2
            # This is the sole training objective during fine-tuning.
            # No balance loss is added (router is frozen, balance loss
            # would only add noise to expert network gradients).
            l2re_loss: torch.Tensor = self.criterion(u_pred, u_target)
            loss: torch.Tensor = l2re_loss

            # ----------------------------------------------------------
            # Step 5: NaN/Inf guard
            # ----------------------------------------------------------
            # Guard against numerical instability, especially at the start
            # of fine-tuning when the model is adapting to a new dataset.
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    f"Finetune Epoch {epoch}, batch {num_batches}: "
                    f"NaN/Inf loss detected (l2re={l2re_loss.item():.4f}). "
                    f"Skipping batch."
                )
                self.optimizer.zero_grad()
                continue

            # ----------------------------------------------------------
            # Step 6: Backward pass
            # ----------------------------------------------------------
            # Gradients flow only to non-frozen parameters (expert networks,
            # patchify, temporal_agg, output_proj). Router parameters have
            # requires_grad=False so they receive no gradient updates.
            # In DDP mode, gradients are automatically averaged across
            # processes via all-reduce during backward().
            loss.backward()

            # ----------------------------------------------------------
            # Step 7: Optimizer step
            # ----------------------------------------------------------
            # Updates only the non-frozen parameters in the optimizer's
            # parameter groups (router params were excluded at init time).
            self.optimizer.step()

            # ----------------------------------------------------------
            # Step 8: Scheduler step (per-batch for OneCycleLR)
            # ----------------------------------------------------------
            # OneCycleLR MUST be stepped once per batch, not per epoch.
            # Stepping per epoch would give incorrect LR behavior.
            self.scheduler.step()

            # ----------------------------------------------------------
            # Step 9: Accumulate metrics
            # ----------------------------------------------------------
            total_loss += loss.item()
            total_l2re += l2re_loss.item()
            num_batches += 1

            # Update tqdm postfix with current batch metrics.
            if num_batches % log_interval == 0 and self.is_main_process:
                loader_iter.set_postfix(
                    l2re=f"{l2re_loss.item():.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                )

        # Guard against empty loader (should not happen in practice).
        if num_batches == 0:
            self.logger.warning(
                f"Finetune Epoch {epoch}: No batches processed. "
                f"Check DataLoader configuration."
            )
            return {
                "train_loss": float("nan"),
                "train_l2re": float("nan"),
                "lr": self.scheduler.get_last_lr()[0],
            }

        return {
            "train_loss": total_loss / num_batches,
            "train_l2re": total_l2re / num_batches,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def validate(self) -> Dict[str, float]:
        """Evaluates the fine-tuned model on the validation split.

        Runs the model in eval mode over val_loader without gradient
        computation, computing the mean L2 relative error. No noise
        injection is applied. The load balancing loss is discarded.

        Handles the case where val_loader is None by returning a dict
        with NaN values and logging a warning.

        Returns:
            Dictionary with validation metrics:
              - 'val_l2re': Mean L2 relative error on the validation split.
                Returns float('nan') if val_loader is None or empty.
        """
        # Handle missing val_loader gracefully.
        if self.val_loader is None:
            self.logger.warning(
                "val_loader is None. Skipping validation. "
                "Returning val_l2re=nan."
            )
            return {"val_l2re": float("nan")}

        self.model.eval()

        total_l2re: float = 0.0
        num_batches: int = 0

        with torch.no_grad():
            val_batch: Tuple
            for val_batch in self.val_loader:
                # Prepare batch — move to device, no noise injection.
                u_input: torch.Tensor
                u_target: torch.Tensor
                u_input, u_target = self._prepare_batch(val_batch)
                # u_input shape:  (B, T, C, H, W)
                # u_target shape: (B, C, H, W)

                # Forward pass — discard balance_loss (irrelevant for eval).
                u_pred: torch.Tensor
                _balance: torch.Tensor
                u_pred, _balance = self.model(u_input)
                # u_pred shape: (B, C, H, W)

                # Compute L2 relative error.
                l2re: torch.Tensor = self.criterion(u_pred, u_target)
                total_l2re += l2re.item()
                num_batches += 1

        if num_batches == 0:
            self.logger.warning(
                "val_loader produced no batches. "
                "Returning val_l2re=nan."
            )
            return {"val_l2re": float("nan")}

        return {"val_l2re": total_l2re / num_batches}

    def finetune(self) -> None:
        """Runs the full fine-tuning loop for config.num_epochs epochs.

        For each epoch:
          1. Runs finetune_epoch() to update expert network parameters.
          2. Runs validate() to compute validation L2RE.
          3. Logs combined metrics (main process only).
          4. Saves periodic checkpoints every save_interval epochs.
          5. Saves 'best.pt' whenever validation L2RE improves.

        Checkpointing always uses raw_model (not DDP-wrapped) to avoid
        the 'module.' prefix in state dict keys, ensuring compatibility
        with load_model_only() for subsequent fine-tuning or evaluation.

        The fine-tuning loop is designed to be robust:
          - Handles NaN validation metrics gracefully (no best-model save)
          - Logs progress at every epoch (fine-tuning is shorter than pre-training)
          - Saves final checkpoint at the end of training regardless of
            whether it is the best model
        """
        self.logger.info(
            f"Starting fine-tuning for {self.config.num_epochs} epochs "
            f"on device {self.device}. "
            f"Distributed: {self.is_distributed}. "
            f"Router frozen: True."
        )

        # Log configuration once at the start of fine-tuning.
        if self.is_main_process:
            self.logger.log_config(self.config)

        # Retrieve save_interval from config.yaml logging.save_interval: 50
        save_interval: int = int(getattr(self.config, "save_interval", 50))

        epoch: int
        for epoch in range(1, int(self.config.num_epochs) + 1):

            # ----------------------------------------------------------
            # DistributedSampler epoch setting
            # ----------------------------------------------------------
            # When using DistributedSampler, set_epoch() must be called
            # before each epoch to ensure different shuffling per epoch
            # across all processes.
            if self.is_distributed and hasattr(
                self.train_loader.sampler, "set_epoch"
            ):
                self.train_loader.sampler.set_epoch(epoch)

            # ----------------------------------------------------------
            # Fine-tuning epoch
            # ----------------------------------------------------------
            train_metrics: Dict[str, float] = self.finetune_epoch(epoch)

            # ----------------------------------------------------------
            # Validation
            # ----------------------------------------------------------
            val_metrics: Dict[str, float] = self.validate()

            # ----------------------------------------------------------
            # Logging (main process only)
            # ----------------------------------------------------------
            if self.is_main_process:
                # Combine train and val metrics into a single flat dict.
                all_metrics: Dict[str, float] = {
                    **train_metrics,
                    **val_metrics,
                    "epoch": float(epoch),
                }

                # Log to file/stdout/wandb.
                self.logger.log_metrics(all_metrics, step=epoch)

                # Human-readable progress line.
                val_l2re_val: float = val_metrics.get("val_l2re", float("nan"))
                self.logger.info(
                    f"Finetune Epoch {epoch}/{self.config.num_epochs} | "
                    f"Train L2RE: {train_metrics.get('train_l2re', float('nan')):.6f} | "
                    f"Val L2RE: {val_l2re_val:.6f} | "
                    f"LR: {train_metrics.get('lr', 0.0):.2e}"
                )

            # ----------------------------------------------------------
            # Checkpointing (main process only)
            # ----------------------------------------------------------
            if self.is_main_process:
                val_l2re: float = val_metrics.get("val_l2re", float("nan"))

                # Determine if this is a new best model.
                # NaN validation metrics do not trigger best-model saves.
                is_best: bool = (
                    val_l2re == val_l2re  # NaN check: NaN != NaN is True
                    and val_l2re < self.best_val_l2re
                )

                if is_best:
                    self.best_val_l2re = val_l2re

                # Build metrics dict for checkpoint metadata.
                checkpoint_metrics: Dict[str, float] = {
                    **train_metrics,
                    **val_metrics,
                }

                # Periodic checkpoint: save every save_interval epochs.
                # config.yaml logging.save_interval: 50
                # For 200-epoch fine-tuning: saves at epochs 50, 100, 150, 200.
                if epoch % save_interval == 0:
                    self.checkpointer.save(
                        model=self.raw_model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=checkpoint_metrics,
                        filename=f"finetune_epoch_{epoch}.pt",
                    )
                    self.logger.info(
                        f"Checkpoint saved at finetune epoch {epoch}."
                    )

                # Best model checkpoint: save when val L2RE improves.
                # Uses explicit 'best.pt' filename to ensure it's written
                # even if epoch % save_interval != 0.
                if is_best:
                    self.checkpointer.save(
                        model=self.raw_model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=checkpoint_metrics,
                        filename="best.pt",
                    )
                    self.logger.info(
                        f"New best fine-tuned model at epoch {epoch}: "
                        f"val_l2re={val_l2re:.6f} "
                        f"(previous best: {self.best_val_l2re:.6f})"
                    )

        # ----------------------------------------------------------
        # Final checkpoint at the end of fine-tuning
        # ----------------------------------------------------------
        # Always save the final model state, regardless of whether it
        # is the best model. This ensures the fully fine-tuned model
        # is always available for evaluation.
        if self.is_main_process:
            final_metrics: Dict[str, float] = {
                "train_l2re": float("nan"),  # placeholder
                "val_l2re": self.best_val_l2re,
                "num_epochs": float(self.config.num_epochs),
            }
            self.checkpointer.save(
                model=self.raw_model,
                optimizer=self.optimizer,
                epoch=int(self.config.num_epochs),
                metrics=final_metrics,
                filename="finetune_final.pt",
            )
            self.logger.info(
                f"Fine-tuning complete. Final checkpoint saved. "
                f"Best val_l2re: {self.best_val_l2re:.6f}"
            )
