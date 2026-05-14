## training/trainer.py
"""Trainer class for WDNO Base-Resolution Model (BRM) and Super-Resolution Model (SRM).

This module implements the Trainer class that orchestrates the full training
lifecycle for both BRM and SRM diffusion models. The same Trainer class serves
all experiments (1D Burgers', advection, compressible NS, 2D fluid, ERA5) and
both simulation and control tasks.

Key design decisions:
    - Step-based training loop (not epoch-based): paper specifies train_steps=190000
    - CosineAnnealingLR with T_max=train_steps: paper specifies "cosine annealing"
    - Gradient clipping at max_norm=1.0 for training stability
    - Batch format auto-detection: len(batch)==2 → BRM, len(batch)==3 → SRM
    - Optional on-the-fly wavelet transform for raw data batches
    - DataParallel support for 2D experiments (2× A100)
    - TensorBoard logging every 100 steps, validation every 1000 steps

Paper sources:
    - Training hyperparameters: Table 18 (1D Burgers'), Table 19 (compressible NS),
      Table 20 (2D fluid)
    - Optimizer: Adam, lr=1e-4 (all experiments)
    - LR scheduler: cosine annealing (all experiments)
    - Batch size: 16 (1D), 4 (2D)
    - Training steps: 190000 (all experiments)
    - Hardware: 1× A100 (1D), 2× A100 (2D)

Config references:
    - training.<experiment>.learning_rate: 1e-4
    - training.<experiment>.train_steps: 190000
    - training.<experiment>.batch_size: 16 (1D) / 4 (2D)
    - training.<experiment>.lr_scheduler: cosine_annealing
    - training.<experiment>.num_gpus: 1 (1D) / 2 (2D)
    - diffusion.cfg_dropout_prob: 0.1
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from models.diffusion import Diffusion
from utils.helpers import load_checkpoint, make_dirs, save_checkpoint
from wavelet.wavelet_transform import WaveletTransform

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for WDNO BRM and SRM diffusion models.

    Handles the full training lifecycle: data iteration, loss computation,
    backpropagation, gradient clipping, optimizer/scheduler stepping,
    checkpointing, TensorBoard logging, and optional validation.

    Supports two batch formats:
        BRM batch: (x0_wavelet, cond_wavelet) — 2-tuple
        SRM batch: (W_high, W_low_duplicated, W_cond_high) — 3-tuple

    The batch format is auto-detected by checking len(batch). For SRM,
    the conditioning input is torch.cat([W_low_duplicated, W_cond_high], dim=1).

    Attributes:
        diffusion: Diffusion model wrapping UNet + noise schedule buffers.
            For 2D experiments with num_gpus=2, diffusion.model is wrapped
            in nn.DataParallel after initialization.
        optimizer: Adam optimizer wrapping diffusion.model.parameters().
            lr=config.lr (1e-4 for all experiments).
        scheduler: CosineAnnealingLR with T_max=config.train_steps.
            Decays lr from config.lr to 0 over the full training run.
        train_loader: DataLoader yielding training batches. Cycled
            indefinitely until train_steps gradient steps are completed.
        config: Experiment configuration. Drives all hyperparameters.
        val_loader: Optional DataLoader for periodic validation.
            If None, validation is skipped.
        wavelet_transform: Optional WaveletTransform for on-the-fly
            transformation of raw data batches. If None, batches are
            assumed to already contain wavelet coefficients.
        global_step: Total number of gradient steps completed. Used for
            resuming training from checkpoints.
        best_val_loss: Best validation loss seen so far. Used for
            best-model checkpointing.
        writer: TensorBoard SummaryWriter logging to
            config.checkpoint_dir/logs/.
        device: Compute device string from config.device.
        cfg_dropout_prob: Classifier-free guidance dropout probability
            from config.cfg_dropout_prob (0.1).
        _use_data_parallel: Whether DataParallel is active (2D experiments
            with num_gpus >= 2 and multiple CUDA devices available).
    """

    def __init__(
        self,
        diffusion: Diffusion,
        train_loader: DataLoader,
        config: Config,
        val_loader: Optional[DataLoader] = None,
        wavelet_transform: Optional[WaveletTransform] = None,
    ) -> None:
        """Initialize the Trainer.

        Constructs the Adam optimizer and CosineAnnealingLR scheduler,
        optionally wraps the model in DataParallel for multi-GPU training,
        and initializes the TensorBoard writer.

        Args:
            diffusion: Diffusion model instance (BRM or SRM). Contains the
                UNet denoising network and all noise schedule buffers.
                For 2D experiments, diffusion.model will be wrapped in
                nn.DataParallel if multiple GPUs are available.
            train_loader: DataLoader yielding training batches. Format:
                - BRM: (x0_wavelet, cond_wavelet) — 2-tuple of tensors
                - SRM: (W_high, W_low_dup, W_cond_high) — 3-tuple of tensors
                The format is auto-detected by len(batch) in _train_step.
            config: Experiment configuration. Reads:
                - config.lr (1e-4): Adam learning rate
                - config.train_steps (190000): total gradient steps
                - config.num_gpus (1 or 2): for DataParallel decision
                - config.spatial_dim (1 or 2): for DataParallel decision
                - config.cfg_dropout_prob (0.1): CFG dropout rate
                - config.device: compute device
                - config.checkpoint_dir: for TensorBoard logs and checkpoints
            val_loader: Optional DataLoader for periodic validation every
                1000 steps. If None, validation is skipped entirely.
                Config: not explicitly specified; standard practice.
            wavelet_transform: Optional WaveletTransform for on-the-fly
                transformation of raw PDE data batches. If None, batches
                are assumed to already contain pre-transformed wavelet
                coefficients (as produced by MultiResolutionDataset).
                For BurgersGenerator data stored in raw format, pass the
                WaveletTransform instance here.
        """
        self.diffusion: Diffusion = diffusion
        self.train_loader: DataLoader = train_loader
        self.config: Config = config
        self.val_loader: Optional[DataLoader] = val_loader
        self.wavelet_transform: Optional[WaveletTransform] = wavelet_transform

        self.device: str = config.device
        self.cfg_dropout_prob: float = config.cfg_dropout_prob
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")

        # --- Multi-GPU DataParallel setup ---
        # Paper Appendix C.6: 2D experiments use 2× A100
        # Config: training.fluid_2d.num_gpus=2
        self._use_data_parallel: bool = False
        if (
            config.num_gpus >= 2
            and torch.cuda.is_available()
            and torch.cuda.device_count() >= 2
        ):
            logger.info(
                "Wrapping model in DataParallel for %d GPUs.",
                torch.cuda.device_count(),
            )
            self.diffusion.model = nn.DataParallel(self.diffusion.model)
            self._use_data_parallel = True

        # Move diffusion model (and all schedule buffers) to device
        self.diffusion = self.diffusion.to(torch.device(self.device))

        # --- Optimizer ---
        # Paper Tables 18, 19, 20: Adam, lr=1e-4 for all experiments
        # Config: training.<experiment>.learning_rate=1e-4
        self.optimizer: Adam = Adam(
            self._get_model_parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

        # --- LR Scheduler ---
        # Paper Tables 18, 19, 20: cosine annealing
        # Config: training.<experiment>.lr_scheduler=cosine_annealing
        # T_max=train_steps: lr decays from config.lr to 0 over full run
        self.scheduler: CosineAnnealingLR = CosineAnnealingLR(
            self.optimizer,
            T_max=config.train_steps,
            eta_min=0.0,
        )

        # --- TensorBoard writer ---
        log_dir: str = os.path.join(config.checkpoint_dir, "logs")
        make_dirs(log_dir)
        self.writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

        logger.info(
            "Trainer initialized: experiment=%s, device=%s, lr=%.2e, "
            "train_steps=%d, batch_size=%d, cfg_dropout_prob=%.2f, "
            "data_parallel=%s, val_loader=%s",
            config.experiment,
            self.device,
            config.lr,
            config.train_steps,
            config.batch_size,
            self.cfg_dropout_prob,
            self._use_data_parallel,
            "enabled" if val_loader is not None else "disabled",
        )

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop for config.train_steps gradient steps.

        Implements a step-based training loop (not epoch-based) as required
        by the diffusion model training convention. The DataLoader is cycled
        indefinitely until train_steps gradient steps are completed.

        Per-step actions:
            1. Sample a batch from the DataLoader (cycle if exhausted)
            2. Call _train_step(batch) to compute loss and update parameters
            3. Every 100 steps: log metrics to TensorBoard and stdout
            4. Every 1000 steps: run validation if val_loader is provided
            5. Every 10000 steps: save a periodic checkpoint

        Supports resuming from a checkpoint: if global_step > 0 (set by
        load_checkpoint), the loop starts from global_step rather than 0.

        Paper: training.*.train_steps=190000 for all experiments.
        Config: config.train_steps.
        """
        train_steps: int = self.config.train_steps
        log_interval: int = 100       # log every 100 steps (standard practice)
        val_interval: int = 1000      # validate every 1000 steps
        save_interval: int = 10000    # save checkpoint every 10000 steps

        # Set model to training mode
        self.diffusion.model.train()

        # Create infinite DataLoader iterator
        data_iterator: Iterator = iter(self.train_loader)

        # Progress bar starting from current global_step (supports resume)
        pbar = tqdm(
            initial=self.global_step,
            total=train_steps,
            desc=f"Training [{self.config.experiment}]",
            dynamic_ncols=True,
        )

        logger.info(
            "Starting training: global_step=%d, target_steps=%d",
            self.global_step,
            train_steps,
        )

        while self.global_step < train_steps:
            # --- Sample batch (cycle DataLoader if exhausted) ---
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_loader)
                batch = next(data_iterator)

            # --- Training step ---
            loss: float = self._train_step(batch)

            # --- Logging ---
            if self.global_step % log_interval == 0:
                self._log_metrics(loss, self.global_step)
                pbar.set_postfix(
                    loss=f"{loss:.6f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                )

            # --- Validation ---
            if (
                self.val_loader is not None
                and self.global_step % val_interval == 0
                and self.global_step > 0
            ):
                val_loss: float = self._validate(self.global_step)
                logger.info(
                    "Step %d | Val loss: %.6f | Best: %.6f",
                    self.global_step,
                    val_loss,
                    self.best_val_loss,
                )

            # --- Periodic checkpoint ---
            if self.global_step % save_interval == 0 and self.global_step > 0:
                ckpt_path: str = os.path.join(
                    self.config.checkpoint_dir,
                    f"{self.config.experiment}_step{self.global_step}.pt",
                )
                self.save_checkpoint(ckpt_path)
                logger.info("Saved periodic checkpoint: %s", ckpt_path)

            self.global_step += 1
            pbar.update(1)

        pbar.close()

        # Save final checkpoint
        final_ckpt_path: str = os.path.join(
            self.config.checkpoint_dir,
            f"{self.config.experiment}_final.pt",
        )
        self.save_checkpoint(final_ckpt_path)
        logger.info(
            "Training complete. Final checkpoint saved: %s", final_ckpt_path
        )

        # Close TensorBoard writer
        self.writer.close()

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def _train_step(self, batch: Union[Tuple, List]) -> float:
        """Perform a single gradient update step.

        Auto-detects batch format by checking len(batch):
            len(batch) == 2: BRM batch → (x0_wavelet, cond_wavelet)
            len(batch) == 3: SRM batch → (W_high, W_low_dup, W_cond_high)

        For SRM batches, the conditioning input is constructed by
        concatenating W_low_dup and W_cond_high along the channel dimension:
            cond = torch.cat([W_low_dup, W_cond_high], dim=1)

        If wavelet_transform is provided and batch contains raw PDE data
        (detected by checking tensor dimensionality vs expected wavelet
        coefficient shape), applies the wavelet transform on-the-fly.

        Backpropagation sequence:
            1. optimizer.zero_grad()
            2. loss = diffusion.compute_loss(x0, cond, cfg_dropout_prob)
            3. loss.backward()
            4. clip_grad_norm_(model.parameters(), max_norm=1.0)
            5. optimizer.step()
            6. scheduler.step()

        Args:
            batch: Training batch. Either a 2-tuple (BRM) or 3-tuple (SRM).
                All tensors are moved to self.device before processing.

        Returns:
            Scalar loss value as a Python float. Used for logging.

        Raises:
            ValueError: If batch length is not 2 or 3.
        """
        batch_len: int = len(batch)

        if batch_len == 2:
            # BRM batch: (x0_wavelet, cond_wavelet)
            x0_raw: torch.Tensor = batch[0].to(self.device, non_blocking=True)
            cond_raw: torch.Tensor = batch[1].to(self.device, non_blocking=True)

            # Apply on-the-fly wavelet transform if needed
            if self.wavelet_transform is not None:
                x0: torch.Tensor = self._apply_wavelet_if_needed(x0_raw)
                cond: torch.Tensor = self._apply_wavelet_if_needed(cond_raw)
            else:
                x0 = x0_raw
                cond = cond_raw

        elif batch_len == 3:
            # SRM batch: (W_high, W_low_duplicated, W_cond_high)
            W_high: torch.Tensor = batch[0].to(self.device, non_blocking=True)
            W_low_dup: torch.Tensor = batch[1].to(self.device, non_blocking=True)
            W_cond_high: torch.Tensor = batch[2].to(self.device, non_blocking=True)

            # SRM target is W_high; conditioning is concat of W_low_dup and W_cond_high
            x0 = W_high
            cond = torch.cat([W_low_dup, W_cond_high], dim=1)

        else:
            raise ValueError(
                f"Unexpected batch length {batch_len}. "
                "Expected 2 (BRM: x0_wavelet, cond_wavelet) or "
                "3 (SRM: W_high, W_low_dup, W_cond_high)."
            )

        # --- Backpropagation ---
        self.optimizer.zero_grad()

        loss_tensor: torch.Tensor = self.diffusion.compute_loss(
            x0=x0,
            cond=cond,
            cfg_dropout_prob=self.cfg_dropout_prob,
        )

        loss_tensor.backward()

        # Gradient clipping for training stability (task specification: max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            self._get_model_parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()
        self.scheduler.step()

        return float(loss_tensor.item())

    # -----------------------------------------------------------------------
    # Checkpoint management
    # -----------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save a training checkpoint to disk.

        Saves model state, optimizer state, scheduler state, global step,
        best validation loss, and config. Handles DataParallel unwrapping
        to ensure checkpoints are portable across different GPU configurations.

        Checkpoint dict structure:
            {
                'global_step': int,
                'model_state_dict': OrderedDict,  # unwrapped from DataParallel
                'optimizer_state_dict': dict,
                'scheduler_state_dict': dict,
                'best_val_loss': float,
                'config': dict,
            }

        Args:
            path: Full file path where the checkpoint will be saved.
                Parent directories are created automatically via make_dirs.
        """
        # Unwrap DataParallel to get portable state_dict
        model: nn.Module = self.diffusion.model
        if isinstance(model, nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        state: Dict = {
            "global_step": self.global_step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
        }

        save_checkpoint(state, path)
        logger.debug("Checkpoint saved: %s (step=%d)", path, self.global_step)

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint from disk and restore all states.

        Restores model weights, optimizer state, scheduler state, global
        step, and best validation loss. Handles DataParallel wrapping:
        if the model is currently wrapped in DataParallel, loads into
        the underlying module.

        After calling this method, training can be resumed from the
        checkpoint's global_step by calling train() — the loop will
        start from self.global_step rather than 0.

        Args:
            path: Full file path to the saved checkpoint.

        Raises:
            FileNotFoundError: If path does not exist (raised by
                utils.helpers.load_checkpoint).
        """
        state: Dict = load_checkpoint(path, device=self.device)

        # Restore model weights (handle DataParallel)
        model: nn.Module = self.diffusion.model
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state["model_state_dict"])

        # Restore optimizer and scheduler states
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])

        # Restore training progress
        self.global_step = int(state.get("global_step", 0))
        self.best_val_loss = float(state.get("best_val_loss", float("inf")))

        logger.info(
            "Checkpoint loaded: %s (step=%d, best_val_loss=%.6f)",
            path,
            self.global_step,
            self.best_val_loss,
        )

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _log_metrics(self, loss: float, step: int) -> None:
        """Log training metrics to TensorBoard and stdout.

        Called every 100 steps during training. Logs:
            - train/loss: current training loss
            - train/lr: current learning rate from scheduler

        Args:
            loss: Current training loss value (Python float).
            step: Current global step index.
        """
        current_lr: float = self.scheduler.get_last_lr()[0]

        # TensorBoard logging
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/lr", current_lr, step)

        # Stdout logging for visibility
        logger.info(
            "Step %6d/%d | Loss: %.6f | LR: %.2e",
            step,
            self.config.train_steps,
            loss,
            current_lr,
        )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate(self, step: int) -> float:
        """Run validation on the val_loader and log results.

        Evaluates the model on the full validation set with no CFG dropout
        (always conditional). Updates best_val_loss and saves a best-model
        checkpoint if the current validation loss is lower.

        Called every 1000 steps when val_loader is not None.

        Args:
            step: Current global step index. Used for TensorBoard logging
                and checkpoint naming.

        Returns:
            Average validation loss over the full val_loader. Python float.
        """
        assert self.val_loader is not None, (
            "_validate called but val_loader is None."
        )

        # Switch to eval mode
        self.diffusion.model.eval()

        total_loss: float = 0.0
        num_batches: int = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch_len: int = len(batch)

                if batch_len == 2:
                    x0_raw = batch[0].to(self.device, non_blocking=True)
                    cond_raw = batch[1].to(self.device, non_blocking=True)

                    if self.wavelet_transform is not None:
                        x0 = self._apply_wavelet_if_needed(x0_raw)
                        cond = self._apply_wavelet_if_needed(cond_raw)
                    else:
                        x0 = x0_raw
                        cond = cond_raw

                elif batch_len == 3:
                    W_high = batch[0].to(self.device, non_blocking=True)
                    W_low_dup = batch[1].to(self.device, non_blocking=True)
                    W_cond_high = batch[2].to(self.device, non_blocking=True)
                    x0 = W_high
                    cond = torch.cat([W_low_dup, W_cond_high], dim=1)

                else:
                    logger.warning(
                        "Unexpected batch length %d in validation. Skipping batch.",
                        batch_len,
                    )
                    continue

                # Validation: no CFG dropout (always conditional)
                val_loss_tensor: torch.Tensor = self.diffusion.compute_loss(
                    x0=x0,
                    cond=cond,
                    cfg_dropout_prob=0.0,
                )
                total_loss += float(val_loss_tensor.item())
                num_batches += 1

        # Restore training mode
        self.diffusion.model.train()

        if num_batches == 0:
            logger.warning("Validation loader yielded no valid batches.")
            return float("inf")

        avg_val_loss: float = total_loss / num_batches

        # TensorBoard logging
        self.writer.add_scalar("val/loss", avg_val_loss, step)

        # Best model checkpointing
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            best_ckpt_path: str = os.path.join(
                self.config.checkpoint_dir,
                f"{self.config.experiment}_best.pt",
            )
            self.save_checkpoint(best_ckpt_path)
            logger.info(
                "New best model saved: %s (val_loss=%.6f)",
                best_ckpt_path,
                avg_val_loss,
            )

        return avg_val_loss

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_model_parameters(self):
        """Return the trainable parameters of the underlying model.

        Handles DataParallel wrapping: returns parameters from the
        underlying module, not the DataParallel wrapper.

        Returns:
            Iterator over trainable model parameters.
        """
        model: nn.Module = self.diffusion.model
        if isinstance(model, nn.DataParallel):
            return model.module.parameters()
        return model.parameters()

    def _apply_wavelet_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """Apply wavelet transform to a tensor if it appears to be raw PDE data.

        Heuristic: if the tensor has the expected number of dimensions for
        raw PDE data (3D for 1D PDEs: [B, T, X], or 4D for 2D PDEs:
        [B, T, H, W]) and wavelet_transform is available, apply the transform.

        If the tensor already has more dimensions than expected for raw data
        (e.g., has a channel dimension from pre-transformation), return as-is.

        This heuristic is conservative: when in doubt, return the tensor
        unchanged. The caller is responsible for ensuring correct data format.

        Args:
            x: Input tensor. May be raw PDE data or pre-transformed wavelet
                coefficients.

        Returns:
            Wavelet-transformed tensor if x appears to be raw PDE data and
            wavelet_transform is available. Otherwise returns x unchanged.
        """
        if self.wavelet_transform is None:
            return x

        # Determine expected raw data dimensionality based on spatial_dim
        # spatial_dim=1: raw data is [B, T, X] (3D)
        # spatial_dim=2: raw data is [B, T, H, W] (4D)
        expected_raw_ndim: int = self.config.spatial_dim + 2  # +1 for B, +1 for T

        if x.dim() == expected_raw_ndim:
            # Looks like raw PDE data — apply wavelet transform
            try:
                return self.wavelet_transform.forward(x)
            except Exception as exc:
                logger.warning(
                    "Wavelet transform failed on tensor of shape %s: %s. "
                    "Returning tensor unchanged.",
                    tuple(x.shape),
                    exc,
                )
                return x
        else:
            # Already has channel dimension — assume pre-transformed
            return x
