## Code: training/trainer.py

```python
# training/trainer.py
"""Pre-training loop for the MoE-POT architecture.

Implements the Trainer class that orchestrates the full pre-training phase:
auto-regressive denoising on 6 mixed PDE datasets, OneCycleLR scheduling,
multi-GPU DDP support, periodic validation, and checkpointing.

From the paper (Section 5, Training and Evaluation):
    "We employed the Adam optimizer with a learning rate of 1×10⁻³ and
    trained the models for 1000 epochs. Training was conducted on servers
    equipped with 8 RTX 4090 GPUs, each with 24GB of memory."

From config.yaml (pretraining section):
    num_epochs: 1000
    learning_rate: 1.0e-3
    weight_decay: 1.0e-6
    beta1: 0.9
    beta2: 0.9
    batch_size: 20
    warmup_epochs: 200
    noise_injection: true
    noise_scale: 0.01

From config.yaml (logging section):
    save_interval: 50
    log_interval: 10
"""

import os
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


class Trainer:
    """Orchestrates the MoE-POT pre-training phase.

    Handles the full training loop including:
      - Noise injection for denoising pre-training (Section 2.2)
      - Combined prediction loss + load balancing loss (Section 4)
      - OneCycleLR scheduling with 200-epoch warmup (Appendix B.3)
      - Multi-GPU training via DistributedDataParallel
      - Periodic validation across all 6 PDE datasets
      - Checkpoint saving (periodic + best model)

    The Trainer is responsible for pre-training only. Fine-tuning on
    individual datasets is handled by the separate Finetuner class, which
    disables noise injection and freezes the router-gating network.

    Attributes:
        model: The model used for forward/backward passes. May be a
            DistributedDataParallel wrapper around raw_model when DDP
            is active.
        raw_model: The unwrapped MoEPOT instance. Always used for
            checkpointing, freeze_router(), and count_parameters() to
            avoid the 'module.' prefix in DDP state dicts.
        optimizer: Adam optimizer with paper-specified hyperparameters.
        scheduler: OneCycleLR scheduler stepped once per batch.
        train_loader: DataLoader backed by BalancedMultiDatasetSampler
            for cross-dataset balanced sampling.
        val_loaders: Dict mapping dataset name to its test DataLoader.
            Used for per-dataset validation after each epoch.
        config: Configuration object exposing all hyperparameters from
            config.yaml as attributes.
        logger: Logger instance for file/stdout/wandb logging.
        checkpointer: Checkpointer instance for saving .pt files.
        criterion: L2RelativeLoss instance for prediction loss computation.
        device: Target device (cuda:local_rank or cpu).
        is_distributed: Whether torch.distributed is initialized.
        local_rank: Local GPU rank (0 for single-GPU or non-distributed).
        is_main_process: True if this process should log and checkpoint.
        best_val_loss: Best average validation L2RE seen so far. Used to
            determine when to save the 'best.pt' checkpoint.
    """

    def __init__(
        self,
        model: MoEPOT,
        train_loader: DataLoader,
        val_loaders: Dict[str, DataLoader],
        config: Any,
        logger: Logger,
        checkpointer: Checkpointer,
    ) -> None:
        """Initializes the Trainer with model, data, and training configuration.

        Sets up the device, optionally wraps the model in DDP, creates the
        Adam optimizer and OneCycleLR scheduler, and initializes the loss
        function and tracking state.

        Args:
            model: Initialized MoEPOT model on CPU or the target device.
                Will be moved to the correct device and optionally wrapped
                in DistributedDataParallel if torch.distributed is active.
            train_loader: DataLoader for the combined multi-PDE training set.
                Should use BalancedMultiDatasetSampler for proper cross-dataset
                sampling. The number of batches per epoch determines the
                OneCycleLR total_steps.
            val_loaders: Dictionary mapping dataset name strings (e.g.,
                'fno_ns_1e5', 'pdebench_swe') to their respective test
                DataLoaders. Used for per-dataset validation after each epoch.
                May be an empty dict if no validation is desired.
            config: Configuration object with attributes matching config.yaml
                pretraining and logging sections. Required attributes:
                  - config.learning_rate: float (1e-3)
                  - config.weight_decay: float (1e-6)
                  - config.beta1: float (0.9)
                  - config.beta2: float (0.9)
                  - config.num_epochs: int (1000)
                  - config.warmup_epochs: int (200)
                  - config.noise_scale: float (0.01)
                  - config.noise_injection: bool (True)
                  - config.save_interval: int (50)
                  - config.log_interval: int (10)
            logger: Logger instance for writing metrics to file, stdout,
                and optionally wandb.
            checkpointer: Checkpointer instance for saving model state dicts
                to disk. Saves periodic checkpoints and the best model.
        """
        # ----------------------------------------------------------------
        # Step 1: Distributed training setup
        # ----------------------------------------------------------------
        self.is_distributed: bool = dist.is_available() and dist.is_initialized()

        if self.is_distributed:
            self.local_rank: int = dist.get_rank()
            self.is_main_process: bool = self.local_rank == 0
        else:
            self.local_rank = 0
            self.is_main_process = True

        # ----------------------------------------------------------------
        # Step 2: Device setup
        # ----------------------------------------------------------------
        # Use the current CUDA device if available, otherwise CPU.
        # In DDP mode, each process has already called torch.cuda.set_device()
        # in main.py before constructing the Trainer.
        if torch.cuda.is_available():
            self.device: torch.device = torch.device(
                f"cuda:{self.local_rank}"
            )
        else:
            self.device = torch.device("cpu")

        # ----------------------------------------------------------------
        # Step 3: Move model to device and optionally wrap in DDP
        # ----------------------------------------------------------------
        # Always keep a reference to the raw (unwrapped) model for:
        #   - Checkpointing (avoids 'module.' prefix in state dict keys)
        #   - Calling freeze_router(), unfreeze_router(), count_parameters()
        #   - get_router_weights() for interpretability analysis
        model = model.to(self.device)
        self.raw_model: MoEPOT = model

        if self.is_distributed:
            # find_unused_parameters=False: safe because all parameters are
            # used in every forward pass. The router always runs (for load
            # balancing loss), shared experts always run, and the top-K
            # routed experts are selected but their gradients flow through
            # the routing weights. Setting False avoids the overhead of
            # scanning for unused parameters.
            self.model: nn.Module = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        else:
            self.model = model

        # ----------------------------------------------------------------
        # Step 4: Store data loaders and config
        # ----------------------------------------------------------------
        self.train_loader: DataLoader = train_loader
        self.val_loaders: Dict[str, DataLoader] = val_loaders
        self.config: Any = config
        self.logger: Logger = logger
        self.checkpointer: Checkpointer = checkpointer

        # ----------------------------------------------------------------
        # Step 5: Loss function
        # ----------------------------------------------------------------
        self.criterion: L2RelativeLoss = L2RelativeLoss()

        # ----------------------------------------------------------------
        # Step 6: Optimizer
        # ----------------------------------------------------------------
        # Adam with paper-specified hyperparameters (Appendix B.3):
        #   lr=1e-3, weight_decay=1e-6, beta1=0.9, beta2=0.9
        # Note: beta2=0.9 is unusual (standard Adam uses 0.999) but the
        # paper explicitly specifies (beta1, beta2) = (0.9, 0.9).
        self.optimizer: Adam = Adam(
            self.model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
            betas=(float(config.beta1), float(config.beta2)),
        )

        # ----------------------------------------------------------------
        # Step 7: OneCycleLR scheduler
        # ----------------------------------------------------------------
        # Stepped once per batch (not per epoch). Total steps =
        # num_epochs × steps_per_epoch.
        # pct_start = warmup_epochs / num_epochs = 200/1000 = 0.2
        # This means the LR rises for the first 20% of total steps
        # (the warmup phase), then decays for the remaining 80%.
        steps_per_epoch: int = max(len(train_loader), 1)
        pct_start: float = float(config.warmup_epochs) / float(config.num_epochs)

        self.scheduler: OneCycleLR = OneCycleLR(
            self.optimizer,
            max_lr=float(config.learning_rate),
            epochs=int(config.num_epochs),
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            # PyTorch defaults for div_factor and final_div_factor:
            # div_factor=25 → initial_lr = max_lr / 25 = 4e-5
            # final_div_factor=1e4 → min_lr = initial_lr / 1e4 = 4e-9
            div_factor=25.0,
            final_div_factor=1e4,
            anneal_strategy="cos",
        )

        # ----------------------------------------------------------------
        # Step 8: Training state tracking
        # ----------------------------------------------------------------
        # Tracks the best average validation L2RE for best.pt checkpointing.
        self.best_val_loss: float = float("inf")

    def _prepare_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Moves a batch from DataLoader to the target device.

        Unpacks the (input_frames, target_frame, dataset_idx) tuple
        returned by MultiPDEDataset.__getitem__ and moves all tensors
        to self.device with non_blocking=True for async CPU→GPU transfer.

        Args:
            batch: Tuple of (u_input, u_target, dataset_idx) where:
                - u_input: shape (B, T, C, H, W) — T=10 input frames
                - u_target: shape (B, C, H, W) — next frame to predict
                - dataset_idx: shape (B,) — integer dataset labels

        Returns:
            Tuple (u_input, u_target, dataset_idx) with all tensors on
            self.device. Shapes are unchanged from the input.
        """
        u_input: torch.Tensor
        u_target: torch.Tensor
        dataset_idx: torch.Tensor
        u_input, u_target, dataset_idx = batch

        # non_blocking=True enables asynchronous CPU→GPU transfer when
        # the DataLoader uses pin_memory=True. Falls back to synchronous
        # transfer gracefully when pin_memory=False.
        u_input = u_input.to(self.device, non_blocking=True)
        u_target = u_target.to(self.device, non_blocking=True)
        dataset_idx = dataset_idx.to(self.device, non_blocking=True)

        return u_input, u_target, dataset_idx

    def _inject_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Adds scaled Gaussian noise to input frames for denoising pre-training.

        Implements the noise injection from Section 2.2 and Appendix B.1:
            ε ~ N(0, noise_scale · ||u^{<t}|| · I)

        The noise standard deviation is proportional to the per-sample,
        per-timestep L2 norm of the input, computed over the (C, H, W)
        dimensions. This ensures that higher-magnitude frames receive
        proportionally more noise, improving robustness to scale variations
        across different PDE datasets.

        This method is ONLY called during pre-training (config.yaml:
        pretraining.noise_injection: true). The Finetuner class does not
        call this method (config.yaml: finetuning.noise_injection: false).

        Args:
            x: Input tensor of shape (B, T, C, H, W) representing a batch
                of T-frame temporal windows. Typically u_input after
                _prepare_batch(), before the model forward pass.

        Returns:
            Noisy tensor of shape (B, T, C, H, W) with additive Gaussian
            noise. The noise has no gradient (detached), so gradients flow
            only through the original x values.
        """
        noise_scale: float = float(self.config.noise_scale)

        # Compute per-sample, per-timestep L2 norm over (C, H, W) dimensions.
        # x shape: (B, T, C, H, W)
        # norm shape: (B, T, 1, 1, 1) — keepdim for broadcasting
        # dim=(-3, -2, -1) covers the (C, H, W) dimensions.
        norm: torch.Tensor = x.detach().norm(p=2, dim=(-3, -2, -1), keepdim=True)
        # Shape: (B, T, 1, 1, 1)

        # Compute per-sample, per-timestep noise standard deviation.
        # std shape: (B, T, 1, 1, 1) — broadcasts over (C, H, W)
        std: torch.Tensor = noise_scale * norm

        # Sample isotropic Gaussian noise with the computed std.
        # torch.randn_like ensures same device, dtype, and shape as x.
        # The noise tensor has no gradient (randn_like produces leaf tensors
        # with requires_grad=False by default).
        epsilon: torch.Tensor = torch.randn_like(x) * std

        return x + epsilon

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one full training epoch over the combined multi-PDE dataset.

        Iterates over all batches in train_loader, applies noise injection,
        computes the combined prediction + load balancing loss, and updates
        model parameters via backpropagation.

        The OneCycleLR scheduler is stepped once per batch (not per epoch),
        which is required for correct LR behavior with OneCycleLR.

        Args:
            epoch: Current epoch number (1-indexed). Used for tqdm display
                and logging.

        Returns:
            Dictionary with training metrics averaged over all batches:
              - 'train_loss': Mean total loss (L_pred + L_balance)
              - 'train_l2re': Mean prediction L2 relative error
              - 'train_balance_loss': Mean load balancing loss
              - 'lr': Current learning rate after the last scheduler step
        """
        self.model.train()

        # Metric accumulators.
        total_loss: float = 0.0
        total_l2re: float = 0.0
        total_balance: float = 0.0
        num_batches: int = 0

        # Log interval from config.yaml logging.log_interval: 10
        log_interval: int = int(getattr(self.config, "log_interval", 10))

        # tqdm progress bar (only on main process to avoid duplicate output).
        loader_iter = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.num_epochs}",
            disable=not self.is_main_process,
            leave=False,
        )

        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        for batch in loader_iter:
            # ----------------------------------------------------------
            # Step 1: Prepare batch — move to device
            # ----------------------------------------------------------
            u_input: torch.Tensor
            u_target: torch.Tensor
            _dataset_idx: torch.Tensor
            u_input, u_target, _dataset_idx = self._prepare_batch(batch)
            # u_input shape:  (B, T, C, H, W)
            # u_target shape: (B, C, H, W)

            # ----------------------------------------------------------
            # Step 2: Inject noise (pre-training only)
            # ----------------------------------------------------------
            # config.yaml pretraining.noise_injection: true
            noise_injection: bool = bool(
                getattr(self.config, "noise_injection", True)
            )
            if noise_injection:
                u_noisy: torch.Tensor = self._inject_noise(u_input)
            else:
                u_noisy = u_input
            # u_noisy shape: (B, T, C, H, W)

            # ----------------------------------------------------------
            # Step 3: Zero gradients
            # ----------------------------------------------------------
            self.optimizer.zero_grad()

            # ----------------------------------------------------------
            # Step 4: Forward pass
            # ----------------------------------------------------------
            # MoEPOT.forward() always returns (u_pred, total_balance_loss).
            # total_balance_loss = Σ_l L_balance^l (sum over N blocks).
            # The load_balance_weight=0.1 is already applied inside MoELayer.
            u_pred: torch.Tensor
            total_balance_loss: torch.Tensor
            u_pred, total_balance_loss = self.model(u_noisy)
            # u_pred shape: (B, C, H, W)

            # ----------------------------------------------------------
            # Step 5: Compute prediction loss (L2 relative error)
            # ----------------------------------------------------------
            # L2RelativeLoss returns mean L2RE over the batch.
            # This is the primary prediction objective from Section 4:
            #   L_pred = ||G_w(u^{<t} + ε) - u^t||_2 / ||u^t||_2
            l2re_loss: torch.Tensor = self.criterion(u_pred, u_target)

            # ----------------------------------------------------------
            # Step 6: Combined loss
            # ----------------------------------------------------------
            # L = L_pred + Σ_l L_balance^l  (Section 4, Loss Function)
            # The balance loss is already scaled by w_bal=0.1 inside MoELayer.
            loss: torch.Tensor = l2re_loss + total_balance_loss

            # ----------------------------------------------------------
            # Step 7: NaN/Inf guard
            # ----------------------------------------------------------
            # With very small per-GPU batch sizes (2-3 samples), occasional
            # NaN losses can occur at the start of training. Skip the batch
            # rather than crashing, and log a warning.
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    f"Epoch {epoch}, batch {num_batches}: "
                    f"NaN/Inf loss detected (l2re={l2re_loss.item():.4f}, "
                    f"balance={total_balance_loss.item():.4f}). "
                    f"Skipping batch."
                )
                # Zero gradients to avoid accumulating NaN gradients.
                self.optimizer.zero_grad()
                continue

            # ----------------------------------------------------------
            # Step 8: Backward pass
            # ----------------------------------------------------------
            # DDP automatically averages gradients across all processes
            # via all-reduce during backward(). No manual synchronization
            # is needed.
            loss.backward()

            # ----------------------------------------------------------
            # Step 9: Gradient clipping
            # ----------------------------------------------------------
            # Not explicitly stated in the paper but standard practice for
            # stability with small batch sizes (2-3 per GPU) and the MoE
            # routing mechanism. max_norm=1.0 is a conservative safe default.
            nn_utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            # ----------------------------------------------------------
            # Step 10: Optimizer step
            # ----------------------------------------------------------
            self.optimizer.step()

            # ----------------------------------------------------------
            # Step 11: Scheduler step (per-batch for OneCycleLR)
            # ----------------------------------------------------------
            # OneCycleLR MUST be stepped once per batch, not per epoch.
            # Stepping per epoch would give incorrect LR behavior.
            self.scheduler.step()

            # ----------------------------------------------------------
            # Step 12: Accumulate metrics
            # ----------------------------------------------------------
            total_loss += loss.item()
            total_l2re += l2re_loss.item()
            total_balance += total_balance_loss.item()
            num_batches += 1

            # Update tqdm postfix with current batch metrics.
            if num_batches % log_interval == 0 and self.is_main_process:
                loader_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l2re=f"{l2re_loss.item():.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                )

        # Guard against empty loader (should not happen in practice).
        if num_batches == 0:
            self.logger.warning(
                f"Epoch {epoch}: No batches processed. "
                f"Check DataLoader configuration."
            )
            return {
                "train_loss": float("nan"),
                "train_l2re": float("nan"),
                "train_balance_loss": float("nan"),
                "lr": self.scheduler.get_last_lr()[0],
            }

        return {
            "train_loss": total_loss / num_batches,
            "train_l2re": total_l2re / num_batches,
            "train_balance_loss": total_balance / num_batches,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def validate(self) -> Dict[str, float]:
        """Evaluates the model on all validation datasets without gradient computation.

        Runs the model in eval mode over each dataset's test DataLoader,
        computing the mean L2 relative error per dataset. No noise injection
        is applied during validation.

        The load balancing loss is ignored during validation — only the
        prediction quality (L2RE) is reported.

        Returns:
            Dictionary mapping dataset name to mean L2RE on that dataset's
            test split. Example:
                {
                    'fno_ns_1e5': 0.0682,
                    'pdebench_swe': 0.00640,
                    ...
                }
            Returns an empty dict if val_loaders is empty.
        """
        self.model.eval()
        results: Dict[str, float] = {}

        with torch.no_grad():
            dataset_name: str
            loader: DataLoader
            for dataset_name, loader in self.val_loaders.items():
                total_l2re: float = 0.0
                num_batches: int = 0

                val_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                for val_batch in loader:
                    # Prepare batch — move to device.
                    u_input: torch.Tensor
                    u_target: torch.Tensor
                    u_input, u_target, _ = self._prepare_batch(val_batch)
                    # u_input shape:  (B, T, C, H, W)
                    # u_target shape: (B, C, H, W)

                    # Forward pass — no noise injection during validation.
                    # Ignore balance_loss (not relevant for evaluation).
                    u_pred: torch.Tensor
                    _balance: torch.Tensor
                    u_pred, _balance = self.model(u_input)
                    # u_pred shape: (B, C, H, W)

                    # Compute L2 relative error.
                    l2re: torch.Tensor = self.criterion(u_pred, u_target)
                    total_l2re += l2re.item()
                    num_batches += 1

                if num_batches > 0:
                    results[dataset_name] = total_l2re / num_batches
                else:
                    results[dataset_name] = float("nan")
                    self.logger.warning(
                        f"Validation dataset '{dataset_name}' has no batches."
                    )

        return results

    def train(self) -> None:
        """Runs the full pre-training loop for config.num_epochs epochs.

        For each epoch:
          1. Sets the DistributedSampler epoch (if applicable) for proper
             shuffling across processes.
          2. Runs train_epoch() to update model parameters.
          3. Runs validate() to compute per-dataset L2RE.
          4. Logs combined metrics (main process only).
          5. Saves periodic checkpoints every save_interval epochs.
          6. Saves 'best.pt' whenever average validation L2RE improves.

        Checkpointing always uses raw_model (not DDP-wrapped) to avoid
        the 'module.' prefix in state dict keys, ensuring compatibility
        with load_model_only() during fine-tuning initialization.

        The training loop is robust to empty val_loaders (no validation
        checkpointing in that case) and handles DDP rank filtering for
        logging and checkpointing.
        """
        self.logger.info(
            f"Starting pre-training for {self.config.num_epochs} epochs "
            f"on device {self.device}. "
            f"Distributed: {self.is_distributed}."
        )

        # Log configuration once at the start of training.
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
            # across all processes. Without this, all processes would see
            # the same data order every epoch.
            if self.is_distributed and hasattr(
                self.train_loader.sampler, "set_epoch"
            ):
                self.train_loader.sampler.set_epoch(epoch)

            # ----------------------------------------------------------
            # Training epoch
            # ----------------------------------------------------------
            train_metrics: Dict[str, float] = self.train_epoch(epoch)

            # ----------------------------------------------------------
            # Validation
            # ----------------------------------------------------------
            val_metrics: Dict[str, float] = self.validate()

            # ----------------------------------------------------------
            # Logging (main process only)
            # ----------------------------------------------------------
            if self.is_main_process:
                # Combine train and val metrics into a single flat dict.
                # Prefix val metrics with 'val_' to distinguish from train.
                all_metrics: Dict[str, float] = {**train_metrics}
                for k, v in val_metrics.items():
                    all_metrics[f"val_{k}"] = v

                # Log to file/stdout/wandb.
                self.logger.log_metrics(all_metrics, step=epoch)

                # Human-readable summary line.
                val_summary: str = ", ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items()
                )
                self.logger.info(
                    f"Epoch {epoch}/{self.config.num_epochs} | "
                    f"train_loss={train_metrics['train_loss']:.4f} | "
                    f"train_l2re={train_metrics['train_l2re']:.4f} | "
                    f"lr={train_metrics['lr']:.2e} | "
                    f"val: {val_summary}"
                )

            # ----------------------------------------------------------
            # Checkpointing (main process only)
            # ----------------------------------------------------------
            if self.is_main_process:
                # Compute average validation L2RE across all datasets.
                # Used to determine whether this is the best model so far.
                avg_val_l2re: Optional[float] = None
                if val_metrics:
                    valid_vals = [
                        v for v in val_metrics.values()
                        if not (v != v)  # filter NaN (NaN != NaN is True)
                    ]
                    if valid_vals:
                        avg_val_l2re = sum(valid_vals) / len(valid_vals)

                # Build metrics dict for checkpoint metadata.
                checkpoint_metrics: Dict[str, float] = {
                    **train_metrics,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                if avg_val_l2re is not None:
                    checkpoint_metrics["val_l2re"] = avg_val_l2re

                # Periodic checkpoint: save every save_interval epochs.
                # config.yaml logging.save_interval: 50
                if epoch % save_interval == 0:
                    self.checkpointer.save(
                        model=self.raw_model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=checkpoint_metrics,
                        filename=f"epoch_{epoch}.pt",
                    )
                    self.logger.info(
                        f"Checkpoint saved at epoch {epoch}."
                    )

                # Best model checkpoint: save when avg val L2RE improves.
                # The Checkpointer.save() method handles best.pt internally
                # when metrics['val_l2re'] is provided and is a new best.
                # We also track it here for the logger message.
                if avg_val_l2re is not None and avg_val_l2re < self.best_val_loss:
                    self.best_val_loss = avg_val_l2re
                    # Save with explicit 'best.pt' filename to ensure it's
                    # written even if epoch % save_interval != 0.
                    self.checkpointer.save(
                        model=self.raw_model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        metrics=checkpoint_metrics,
                        filename="best.pt",
                    )
                    self.logger.info(
                        f"New best model at epoch {epoch}: "
                        f"avg_val_l2re={avg_val_l2re:.6f} "
                        f"(previous best: {self.best_val_loss:.6f})"
                    )

        # ----------------------------------------------------------
        # Final checkpoint at the end of training
        # ----------------------------------------------------------
        if self.is_main_process:
            final_metrics: Dict[str, float] = {
                "train_loss": 0.0,  # placeholder, actual value from last epoch
                "val_l2re": self.best_val_loss,
            }
            self.checkpointer.save(
                model=self.raw_model,
                optimizer=self.optimizer,
                epoch=int(self.config.num_epochs),
                metrics=final_metrics,
                filename="final