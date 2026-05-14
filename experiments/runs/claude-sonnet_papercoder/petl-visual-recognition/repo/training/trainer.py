## training/trainer.py
"""Training engine for the PEFT Visual Recognition reproduction study.

This module provides the Trainer class — the core training loop used by all
three experiment types (VTAB-1K, many-shot, robustness). It handles a single
training run with a fixed configuration. HyperparamSearch calls it repeatedly
with different configs for grid search.

Paper reference: "We employ AdamW optimizer with a batch size of 64 and utilize
the cosine decay learning rate scheduler. We train all methods with 100 epochs."
(Appendix A.1)

Config references (config.yaml):
    vtab.training.optimizer: adamw
    vtab.training.epochs: 100
    vtab.training.lr_scheduler: cosine_decay
    vtab.training.gradient_clip_max_norm: 1.0
    manyshot.training.epochs: 40
    compute.mixed_precision: true
    output.log_interval: 10

Typical usage (called by HyperparamSearch and main.py):
    trainer = Trainer(
        model=peft_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        checkpoint=checkpoint,
    )
    best_val_acc = trainer.train()
"""

import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.metrics import Metrics
from utils.checkpoint import Checkpoint
from utils.logger import Logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default training hyperparameters (from config.yaml)
# ---------------------------------------------------------------------------

# config.yaml: vtab.training.gradient_clip_max_norm: 1.0
_DEFAULT_GRADIENT_CLIP_MAX_NORM: float = 1.0

# config.yaml: output.log_interval: 10
_DEFAULT_LOG_INTERVAL: int = 10

# config.yaml: vtab.training.epochs: 100
_DEFAULT_EPOCHS: int = 100

# config.yaml: compute.mixed_precision: true
_DEFAULT_MIXED_PRECISION: bool = True


class Trainer:
    """Core training engine for a single PEFT experiment run.

    Handles the full training loop: optimizer construction, cosine LR
    scheduling, mixed-precision forward/backward passes, gradient clipping,
    validation, and best-model checkpointing.

    Designed to be stateless with respect to model architecture and data —
    it receives a fully constructed PEFTModel and DataLoaders, making it
    reusable across all hyperparameter trials in HyperparamSearch.

    Paper: "We employ AdamW optimizer with a batch size of 64 and utilize
    the cosine decay learning rate scheduler." (Appendix A.1)

    Attributes:
        model: The PEFTModel (or any nn.Module) being trained.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        config: Flattened experiment configuration with lr, weight_decay,
            epochs, mixed_precision, and log_interval attributes.
        logger: Logger instance for metric and loss logging.
        checkpoint: Checkpoint instance for saving best model state.
        device: Torch device inferred from model parameters.
        optimizer: AdamW optimizer over trainable parameters only.
        scheduler: CosineAnnealingLR scheduler stepping once per epoch.
        scaler: GradScaler for mixed-precision training, or None if disabled.
        best_val_acc: Best validation accuracy seen during training.
        best_epoch: Epoch index at which best_val_acc was achieved.
        metrics: Metrics instance for top1_accuracy computation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        logger: Logger,
        checkpoint: Checkpoint,
    ) -> None:
        """Initialises the Trainer with all required components.

        Builds the AdamW optimizer over trainable parameters only, constructs
        the CosineAnnealingLR scheduler, and optionally initialises a
        GradScaler for mixed-precision training.

        Args:
            model: The PEFTModel (or any nn.Module) to train. Must already be
                on the correct device. Must implement get_trainable_params()
                returning a list of nn.Parameter objects with requires_grad=True.
                The head parameters must be included in get_trainable_params().
            train_loader: DataLoader returning (images, labels) batches for
                the training split. Images should be preprocessed and normalized.
            val_loader: DataLoader returning (images, labels) batches for the
                validation split. Used in validate() after each training epoch.
            config: Flattened experiment configuration object. Must expose:
                - config.lr (float): learning rate, e.g. 1e-3 or 1e-2
                - config.weight_decay (float): weight decay, e.g. 1e-4 or 1e-3
                - config.epochs (int): total training epochs, e.g. 100 or 40
                - config.mixed_precision (bool): enable AMP, default True
                - config.log_interval (int): log every N batches, default 10
                These are resolved from config.yaml by HyperparamSearch or
                main.py before constructing Trainer.
            logger: Logger instance for writing metrics to TensorBoard and CSV.
                Receives per-epoch train_loss and val_acc via log_metrics().
            checkpoint: Checkpoint instance for saving best model state.
                save() is called only when val_acc improves.
        """
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.config: Any = config
        self.logger: Logger = logger
        self.checkpoint: Checkpoint = checkpoint

        # ------------------------------------------------------------------
        # Infer device from model parameters.
        # This avoids device mismatch when config.device and model.device differ.
        # ------------------------------------------------------------------
        try:
            self.device: torch.device = next(model.parameters()).device
        except StopIteration:
            # Model has no parameters (edge case) — fall back to CPU.
            _logger.warning(
                "Model has no parameters. Defaulting device to CPU."
            )
            self.device = torch.device("cpu")

        # ------------------------------------------------------------------
        # Read config attributes with safe defaults.
        # All values come from config.yaml as documented above.
        # ------------------------------------------------------------------
        self.lr: float = float(getattr(config, "lr", 1e-3))
        self.weight_decay: float = float(getattr(config, "weight_decay", 1e-4))
        self.epochs: int = int(getattr(config, "epochs", _DEFAULT_EPOCHS))
        self.gradient_clip_max_norm: float = float(
            getattr(config, "gradient_clip_max_norm", _DEFAULT_GRADIENT_CLIP_MAX_NORM)
        )
        self.mixed_precision: bool = bool(
            getattr(config, "mixed_precision", _DEFAULT_MIXED_PRECISION)
        )
        self.log_interval: int = int(
            getattr(config, "log_interval", _DEFAULT_LOG_INTERVAL)
        )

        # ------------------------------------------------------------------
        # Build optimizer and scheduler.
        # ------------------------------------------------------------------
        self.optimizer: optim.Optimizer = self._build_optimizer()
        self.scheduler: lr_scheduler.LRScheduler = self._build_scheduler(
            self.optimizer
        )

        # ------------------------------------------------------------------
        # Initialise GradScaler for mixed-precision training.
        # Only enabled when CUDA is available AND mixed_precision is True.
        # config.yaml: compute.mixed_precision: true
        # ------------------------------------------------------------------
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if self.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            _logger.info(
                "Mixed-precision training enabled (GradScaler initialised)."
            )
        else:
            if self.mixed_precision and not torch.cuda.is_available():
                _logger.warning(
                    "mixed_precision=True but CUDA is not available. "
                    "Falling back to full-precision training."
                )

        # ------------------------------------------------------------------
        # Best model tracking state.
        # ------------------------------------------------------------------
        self.best_val_acc: float = 0.0
        self.best_epoch: int = -1

        # ------------------------------------------------------------------
        # Metrics instance for top1_accuracy computation in validate().
        # ------------------------------------------------------------------
        self.metrics: Metrics = Metrics()

        _logger.info(
            "Trainer initialised: device=%s, lr=%.2e, weight_decay=%.2e, "
            "epochs=%d, mixed_precision=%s, log_interval=%d, "
            "trainable_params=%d",
            self.device,
            self.lr,
            self.weight_decay,
            self.epochs,
            self.mixed_precision and torch.cuda.is_available(),
            self.log_interval,
            self._count_trainable_params(),
        )

    # ------------------------------------------------------------------
    # Public training methods
    # ------------------------------------------------------------------

    def train(self) -> float:
        """Runs the full training loop for config.epochs epochs.

        For each epoch:
        1. Runs train_epoch() to update model parameters.
        2. Runs validate() to compute validation accuracy.
        3. Steps the LR scheduler (once per epoch, after validation).
        4. Logs train_loss and val_acc to Logger.
        5. Saves a checkpoint if val_acc improves.

        Paper: "We train all methods with 100 epochs." (VTAB-1K, Appendix A.1)
        Paper: "We train all methods with 40 epochs." (many-shot, Appendix A.1)

        Returns:
            Best validation accuracy achieved across all epochs, as a float
            in [0.0, 1.0]. Used by HyperparamSearch to compare configurations.
            Returns 0.0 if training fails or epochs=0.
        """
        _logger.info(
            "Starting training: %d epochs, device=%s",
            self.epochs,
            self.device,
        )

        for epoch in range(self.epochs):
            # ------------------------------------------------------------------
            # Step 1: Training epoch.
            # ------------------------------------------------------------------
            train_loss: float = self.train_epoch(epoch)

            # ------------------------------------------------------------------
            # Step 2: Validation.
            # ------------------------------------------------------------------
            val_acc: float = self.validate()

            # ------------------------------------------------------------------
            # Step 3: Step the LR scheduler (once per epoch, after validation).
            # CosineAnnealingLR steps from lr to 0 over T_max=epochs epochs.
            # ------------------------------------------------------------------
            self.scheduler.step()

            # ------------------------------------------------------------------
            # Step 4: Log metrics.
            # config.yaml: output.tensorboard: true
            # ------------------------------------------------------------------
            self.logger.log_metrics(
                metrics={
                    "train/loss": train_loss,
                    "val/top1_acc": val_acc * 100.0,  # Log as percentage
                    "train/lr": self._get_current_lr(),
                },
                step=epoch,
            )

            _logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_acc=%.4f (%.2f%%), lr=%.2e",
                epoch + 1,
                self.epochs,
                train_loss,
                val_acc,
                val_acc * 100.0,
                self._get_current_lr(),
            )

            # ------------------------------------------------------------------
            # Step 5: Save checkpoint if val_acc improved.
            # config.yaml: output.save_checkpoints: true
            # ------------------------------------------------------------------
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.checkpoint.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    val_acc=val_acc,
                    config=self.config,
                )
                _logger.info(
                    "New best val_acc=%.4f (%.2f%%) at epoch %d. Checkpoint saved.",
                    val_acc,
                    val_acc * 100.0,
                    epoch,
                )

        _logger.info(
            "Training complete: best_val_acc=%.4f (%.2f%%) at epoch %d.",
            self.best_val_acc,
            self.best_val_acc * 100.0,
            self.best_epoch,
        )

        return self.best_val_acc

    def train_epoch(self, epoch: int) -> float:
        """Runs one training epoch over the full training DataLoader.

        Iterates all batches in train_loader, computes cross-entropy loss,
        performs backward pass with optional mixed precision, clips gradients,
        and steps the optimizer. Logs loss every log_interval batches.

        Paper: "AdamW optimizer with a batch size of 64" (Appendix A.1)
        Config: vtab.training.gradient_clip_max_norm: 1.0

        Args:
            epoch: Current epoch index (0-based). Used for tqdm description
                and per-batch logging messages.

        Returns:
            Average cross-entropy loss over all batches in the epoch.
            Returns 0.0 if the training loader is empty.
        """
        self.model.train()

        total_loss: float = 0.0
        num_batches: int = 0

        # Wrap train_loader with tqdm for progress display.
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # ------------------------------------------------------------------
            # Extract images and labels from batch.
            # Handle both (images, labels) and (images, labels, *extra) formats.
            # ------------------------------------------------------------------
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images: torch.Tensor = batch[0]
                labels: torch.Tensor = batch[1]
            else:
                raise ValueError(
                    f"Unexpected batch format from DataLoader at batch {batch_idx}. "
                    f"Expected (images, labels) tuple, got type {type(batch)}."
                )

            # Move to device.
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # ------------------------------------------------------------------
            # Zero gradients before forward pass.
            # set_to_none=True is slightly more memory-efficient than zero_grad().
            # ------------------------------------------------------------------
            self.optimizer.zero_grad(set_to_none=True)

            # ------------------------------------------------------------------
            # Forward pass + loss computation + backward pass.
            # Two paths: mixed precision (CUDA + GradScaler) or full precision.
            # ------------------------------------------------------------------
            if self.scaler is not None:
                # ------------------------------------------------------------------
                # Mixed-precision path (config.yaml: compute.mixed_precision: true).
                # ------------------------------------------------------------------
                with torch.cuda.amp.autocast():
                    logits: torch.Tensor = self.model(images)
                    loss: torch.Tensor = F.cross_entropy(logits, labels)

                # Check for NaN loss before scaling.
                if torch.isnan(loss):
                    _logger.warning(
                        "NaN loss detected at epoch %d, batch %d. "
                        "Skipping this batch.",
                        epoch,
                        batch_idx,
                    )
                    continue

                # Scale loss and compute gradients.
                self.scaler.scale(loss).backward()

                # Unscale gradients before clipping (required by GradScaler).
                # This converts scaled gradients back to full precision for clipping.
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping on trainable parameters only.
                # config.yaml: vtab.training.gradient_clip_max_norm: 1.0
                trainable_params: List[nn.Parameter] = self._get_trainable_params()
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=self.gradient_clip_max_norm,
                    )

                # Optimizer step (GradScaler handles the actual step).
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # ------------------------------------------------------------------
                # Full-precision path (CPU or mixed_precision=False).
                # ------------------------------------------------------------------
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)

                # Check for NaN loss.
                if torch.isnan(loss):
                    _logger.warning(
                        "NaN loss detected at epoch %d, batch %d. "
                        "Skipping this batch.",
                        epoch,
                        batch_idx,
                    )
                    continue

                loss.backward()

                # Gradient clipping on trainable parameters only.
                trainable_params = self._get_trainable_params()
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=self.gradient_clip_max_norm,
                    )

                self.optimizer.step()

            # ------------------------------------------------------------------
            # Accumulate loss for epoch average.
            # ------------------------------------------------------------------
            loss_value: float = loss.item()
            total_loss += loss_value
            num_batches += 1

            # ------------------------------------------------------------------
            # Update tqdm progress bar with current loss.
            # ------------------------------------------------------------------
            progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

            # ------------------------------------------------------------------
            # Per-batch logging at log_interval.
            # config.yaml: output.log_interval: 10
            # ------------------------------------------------------------------
            if batch_idx % self.log_interval == 0:
                _logger.debug(
                    "Epoch %d, Batch %d/%d, Loss: %.4f, LR: %.2e",
                    epoch,
                    batch_idx,
                    len(self.train_loader),
                    loss_value,
                    self._get_current_lr(),
                )

        # ------------------------------------------------------------------
        # Compute and return average loss for the epoch.
        # ------------------------------------------------------------------
        if num_batches == 0:
            _logger.warning(
                "train_epoch: no batches processed in epoch %d. "
                "Returning loss=0.0.",
                epoch,
            )
            return 0.0

        avg_loss: float = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """Runs inference on the validation set and computes Top-1 accuracy.

        Sets model to eval mode, iterates val_loader without gradients,
        collects predictions and labels on CPU, and delegates accuracy
        computation to Metrics.top1_accuracy().

        Paper: "The reported TOP-1 ACCURACY is obtained after training over
        the 1000 images and evaluating on the original test set." (Section 3)
        For hyperparameter search, validation uses the 80/20 split of the
        1000 training samples.

        Returns:
            Top-1 accuracy on the validation set as a float in [0.0, 1.0].
            Returns 0.0 if val_loader is empty.
        """
        self.model.eval()

        all_preds: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Extract images and labels.
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images: torch.Tensor = batch[0]
                    labels: torch.Tensor = batch[1]
                else:
                    raise ValueError(
                        f"Unexpected batch format from val_loader. "
                        f"Expected (images, labels) tuple, got type {type(batch)}."
                    )

                images = images.to(self.device, non_blocking=True)
                # Labels stay on CPU for accumulation efficiency.

                # Forward pass with optional mixed precision.
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits: torch.Tensor = self.model(images)
                else:
                    logits = self.model(images)

                # Compute predictions via argmax and move to CPU.
                preds: torch.Tensor = logits.argmax(dim=1).cpu()

                all_preds.append(preds)
                all_labels.append(labels.cpu())

        # ------------------------------------------------------------------
        # Handle empty validation loader.
        # ------------------------------------------------------------------
        if len(all_preds) == 0:
            _logger.warning(
                "validate: val_loader yielded 0 batches. Returning accuracy=0.0."
            )
            return 0.0

        # ------------------------------------------------------------------
        # Concatenate all batches and compute accuracy.
        # ------------------------------------------------------------------
        final_preds: torch.Tensor = torch.cat(all_preds, dim=0)
        final_labels: torch.Tensor = torch.cat(all_labels, dim=0)

        val_acc: float = self.metrics.top1_accuracy(final_preds, final_labels)

        return val_acc

    def save_best_checkpoint(self, val_acc: float, epoch: int) -> None:
        """Explicitly saves a checkpoint for the given val_acc and epoch.

        This is a public convenience method for callers that manage their own
        best-model tracking (e.g., main.py after final training). The internal
        train() method calls checkpoint.save() directly when val_acc improves.

        Args:
            val_acc: Validation accuracy to record in the checkpoint metadata.
            epoch: Epoch index to record in the checkpoint metadata.
        """
        self.checkpoint.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            val_acc=val_acc,
            config=self.config,
        )
        _logger.info(
            "Checkpoint saved explicitly: val_acc=%.4f (%.2f%%), epoch=%d.",
            val_acc,
            val_acc * 100.0,
            epoch,
        )

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> optim.Optimizer:
        """Builds the AdamW optimizer over trainable parameters only.

        Calls model.get_trainable_params() to retrieve only parameters with
        requires_grad=True. This is critical — passing all model parameters
        would accidentally update frozen backbone weights.

        Paper: "We employ AdamW optimizer" (Appendix A.1)
        Config: vtab.hyperparam_search.learning_rate: [0.001, 0.01]
                vtab.hyperparam_search.weight_decay: [0.0001, 0.001]

        Returns:
            torch.optim.AdamW optimizer over trainable parameters.

        Raises:
            ValueError: If model has no trainable parameters (would cause
                AdamW to raise an error on empty parameter list).
        """
        trainable_params: List[nn.Parameter] = self._get_trainable_params()

        if len(trainable_params) == 0:
            raise ValueError(
                "Model has no trainable parameters. Cannot build AdamW optimizer. "
                "Ensure PEFTFactory.build() correctly sets requires_grad=True "
                "on PEFT-specific parameters and the classification head."
            )

        optimizer: optim.AdamW = optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_trainable: int = sum(p.numel() for p in trainable_params)
        _logger.info(
            "AdamW optimizer built: lr=%.2e, weight_decay=%.2e, "
            "trainable_params=%d (%.4fM)",
            self.lr,
            self.weight_decay,
            total_trainable,
            total_trainable / 1_000_000,
        )

        return optimizer

    def _build_scheduler(
        self,
        optimizer: optim.Optimizer,
    ) -> lr_scheduler.LRScheduler:
        """Builds the CosineAnnealingLR scheduler.

        The LR decays from config.lr to 0 over T_max=config.epochs epochs.
        The scheduler steps once per epoch (called in train() after validate()).

        Paper: "cosine decay learning rate scheduler" (Appendix A.1)
        Config: vtab.training.lr_scheduler: cosine_decay

        Args:
            optimizer: The AdamW optimizer to schedule.

        Returns:
            torch.optim.lr_scheduler.CosineAnnealingLR instance with
            T_max=config.epochs and eta_min=0.
        """
        scheduler: lr_scheduler.CosineAnnealingLR = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=0.0,
        )

        _logger.info(
            "CosineAnnealingLR scheduler built: T_max=%d, eta_min=0.0",
            self.epochs,
        )

        return scheduler

    def _get_trainable_params(self) -> List[nn.Parameter]:
        """Returns the list of trainable parameters from the model.

        Calls model.get_trainable_params() if available (PEFTModel interface),
        otherwise falls back to filtering model.parameters() by requires_grad.

        This method is called in both _build_optimizer() and train_epoch()
        (for gradient clipping) to ensure consistency.

        Returns:
            List of nn.Parameter objects with requires_grad=True.
        """
        if hasattr(self.model, "get_trainable_params") and callable(
            self.model.get_trainable_params
        ):
            return self.model.get_trainable_params()
        else:
            # Fallback for non-PEFTModel nn.Module instances.
            return [p for p in self.model.parameters() if p.requires_grad]

    def _count_trainable_params(self) -> int:
        """Returns the total number of trainable parameters.

        Used for logging in __init__ to verify the PEFT parameter count
        matches the expected range from config.yaml peft_methods.*.params_range_M.

        Returns:
            Integer count of trainable parameters (sum of numel() for all
            parameters with requires_grad=True).
        """
        return sum(p.numel() for p in self._get_trainable_params())

    def _get_current_lr(self) -> float:
        """Returns the current learning rate from the optimizer.

        Reads the LR from the first parameter group of the optimizer.
        Used for logging in train() and train_epoch().

        Returns:
            Current learning rate as a Python float.
        """
        return float(self.optimizer.param_groups[0]["lr"])
