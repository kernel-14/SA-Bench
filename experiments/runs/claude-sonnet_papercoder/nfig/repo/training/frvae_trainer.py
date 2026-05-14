## Code: training/frvae_trainer.py

```python
## training/frvae_trainer.py
"""FR-VAE Trainer for Phase 1 of the NFIG framework.

Orchestrates adversarial training of the Frequency-guided Residual-quantized VAE
(FR-VAE) image tokenizer. Implements the VQ-GAN training paradigm adapted for
the frequency-residual quantization scheme described in the NFIG paper.

Loss function (paper Appendix B.1):
    L = ||I - Î||² + ||f̂ - f̂_reconstructed||² + L_p(I) + 0.5·L_g(I)

Training target: rFID = 0.85 (paper Table 2, config.frvae.target_rfid).

Config values used:
    frvae.learning_rate:           1e-4   (generator optimizer LR)
    frvae.disc_learning_rate:      1e-4   (discriminator optimizer LR)
    frvae.batch_size:              8      (per-GPU batch size)
    frvae.epochs:                  100    (total training epochs)
    frvae.warmup_steps:            1000   (linear LR warmup steps)
    frvae.grad_clip:               1.0    (gradient clipping norm)
    frvae.gan_loss_weight:         0.5    (GAN adversarial loss weight)
    frvae.lpips_weight:            1.0    (LPIPS perceptual loss weight)
    frvae.reconstruction_weight:   1.0    (pixel L2 reconstruction weight)
    frvae.freq_quantization_weight: 1.0  (feature L2 quantization weight)
    frvae.commitment_loss_weight:  0.25   (VQ commitment loss beta)
    training.mixed_precision:      true   (bf16 on H100)
    training.precision:            'bf16'
    training.distributed:          true   (multi-GPU via DDP)
    training.checkpoint_dir:       'checkpoints'
    training.log_dir:              'logs'
    training.log_every_steps:      100
    training.eval_every_epochs:    10
    training.save_every_epochs:    10
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from data.imagenet_dataset import ImageNetDataset
from models.frvae.discriminator import DINODiscriminator
from models.frvae.frvae import FRVAE
from utils.checkpoint import CheckpointManager
from utils.config import Config
from utils.losses import NFIGLosses


class FRVAETrainer:
    """Trainer for the Frequency-guided Residual-quantized VAE (FR-VAE).

    Manages the full adversarial training loop including:
    - Two-step GAN update (generator first, discriminator second)
    - Composite loss from paper Appendix B.1
    - Mixed precision training (bf16 on H100)
    - Distributed data parallel training
    - Learning rate warmup
    - Checkpoint saving/loading
    - TensorBoard logging
    - Validation with reconstruction loss monitoring

    Attributes:
        config: Root Config dataclass loaded from config.yaml.
        model: FRVAE model (possibly wrapped in DDP).
        discriminator: DINODiscriminator (possibly wrapped in DDP).
        losses: NFIGLosses module with all loss functions.
        optimizer_g: Adam optimizer for the generator (FRVAE).
        optimizer_d: Adam optimizer for the discriminator.
        scheduler_g: LambdaLR scheduler with linear warmup for generator.
        scheduler_d: LambdaLR scheduler with linear warmup for discriminator.
        train_loader: DataLoader for ImageNet training split.
        val_loader: DataLoader for ImageNet validation split.
        checkpoint_manager: CheckpointManager for saving/loading checkpoints.
        writer: TensorBoard SummaryWriter (None if tensorboard disabled or rank != 0).
        device: Target device (cuda or cpu).
        is_distributed: Whether distributed training is active.
        rank: Process rank (0 for single-GPU or master process).
        world_size: Total number of processes.
        current_epoch: Current training epoch (updated during train()).
        global_step: Global training step counter (updated during train_epoch()).
        best_rfid: Best reconstruction FID seen during validation.
        scaler_g: GradScaler for generator mixed precision.
        scaler_d: GradScaler for discriminator mixed precision.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the FRVAETrainer.

        Constructs all components in the following order:
          1. Device and distributed setup
          2. Dataset and DataLoaders
          3. FRVAE model
          4. DINODiscriminator
          5. NFIGLosses
          6. Optimizers (Adam for generator and discriminator)
          7. LR schedulers (linear warmup)
          8. Mixed precision scalers
          9. CheckpointManager
          10. TensorBoard writer

        Args:
            config: Root Config dataclass populated from config.yaml.
                All hyperparameters are read from this object.
        """
        self.config: Config = config

        # ------------------------------------------------------------------ #
        # 1. Device and distributed setup
        # ------------------------------------------------------------------ #
        self.is_distributed: bool = (
            config.training.distributed
            and dist.is_available()
            and dist.is_initialized()
        )

        if self.is_distributed:
            self.rank: int = dist.get_rank()
            self.world_size: int = dist.get_world_size()
            self.device: torch.device = torch.device(f"cuda:{self.rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device(
                config.training.device
                if torch.cuda.is_available()
                else "cpu"
            )

        # ------------------------------------------------------------------ #
        # 2. Dataset and DataLoaders
        # ------------------------------------------------------------------ #
        train_dataset = ImageNetDataset(
            root=config.data.train_dir,
            split="train",
            image_size=config.data.image_size,
        )
        val_dataset = ImageNetDataset(
            root=config.data.val_dir,
            split="val",
            image_size=config.data.image_size,
        )

        # Distributed sampler for training (handles per-rank data sharding).
        train_sampler: Optional[DistributedSampler] = None
        val_sampler: Optional[DistributedSampler] = None
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )

        self.train_loader: DataLoader = train_dataset.get_dataloader(
            batch_size=config.frvae.batch_size,
            num_workers=config.data.num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
        )
        self.val_loader: DataLoader = val_dataset.get_dataloader(
            batch_size=config.frvae.batch_size,
            num_workers=config.data.num_workers,
            shuffle=False,
            sampler=val_sampler,
        )

        # Store sampler reference for set_epoch() calls in train().
        self._train_sampler: Optional[DistributedSampler] = train_sampler

        # ------------------------------------------------------------------ #
        # 3. FRVAE model
        # ------------------------------------------------------------------ #
        self._raw_model: FRVAE = FRVAE(config.frvae).to(self.device)

        if self.is_distributed:
            self.model: nn.Module = DDP(
                self._raw_model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )
        else:
            self.model = self._raw_model

        # ------------------------------------------------------------------ #
        # 4. DINODiscriminator
        # ------------------------------------------------------------------ #
        self._raw_discriminator: DINODiscriminator = DINODiscriminator(
            pretrained=config.frvae.use_dino_discriminator
        ).to(self.device)

        if self.is_distributed:
            self.discriminator: nn.Module = DDP(
                self._raw_discriminator,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True,  # Backbone is frozen; some params unused
            )
        else:
            self.discriminator = self._raw_discriminator

        # ------------------------------------------------------------------ #
        # 5. NFIGLosses
        # ------------------------------------------------------------------ #
        self.losses: NFIGLosses = NFIGLosses(
            gan_weight=config.frvae.gan_loss_weight,
            lpips_weight=config.frvae.lpips_weight,
        ).to(self.device)

        # ------------------------------------------------------------------ #
        # 6. Optimizers
        # ------------------------------------------------------------------ #
        # Generator optimizer: covers all FRVAE parameters.
        # Paper Section 4.1 specifies Adam for the transformer; FR-VAE uses
        # the same optimizer family. LR from config.frvae.learning_rate = 1e-4.
        self.optimizer_g: optim.Adam = optim.Adam(
            self._raw_model.parameters(),
            lr=config.frvae.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        # Discriminator optimizer: covers only the trainable head parameters.
        # The DINOv2 backbone is frozen; only head parameters are updated.
        disc_trainable_params = [
            p for p in self._raw_discriminator.parameters()
            if p.requires_grad
        ]
        self.optimizer_d: optim.Adam = optim.Adam(
            disc_trainable_params,
            lr=config.frvae.disc_learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        # ------------------------------------------------------------------ #
        # 7. LR schedulers (linear warmup, then constant)
        # ------------------------------------------------------------------ #
        warmup_steps: int = config.frvae.warmup_steps  # 1000

        def _lr_lambda(current_step: int) -> float:
            """Linear warmup from 0 to 1 over warmup_steps, then constant 1."""
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        self.scheduler_g: optim.lr_scheduler.LambdaLR = (
            optim.lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda=_lr_lambda)
        )
        self.scheduler_d: optim.lr_scheduler.LambdaLR = (
            optim.lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda=_lr_lambda)
        )

        # ------------------------------------------------------------------ #
        # 8. Mixed precision scalers
        # ------------------------------------------------------------------ #
        # Use separate scalers for generator and discriminator to allow
        # independent gradient scaling across the two backward passes.
        # bf16 on H100 does not require loss scaling (unlike fp16), but
        # GradScaler is kept for compatibility and fp16 fallback.
        amp_enabled: bool = (
            config.training.mixed_precision
            and torch.cuda.is_available()
        )
        self.scaler_g: torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled
        )
        self.scaler_d: torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled
        )

        # Determine autocast dtype: bf16 on H100, fp16 otherwise.
        if config.training.precision == "bf16" and torch.cuda.is_bf16_supported():
            self._autocast_dtype: torch.dtype = torch.bfloat16
        else:
            self._autocast_dtype = torch.float16

        self._amp_enabled: bool = amp_enabled

        # ------------------------------------------------------------------ #
        # 9. CheckpointManager
        # ------------------------------------------------------------------ #
        self.checkpoint_manager: CheckpointManager = CheckpointManager(
            checkpoint_dir=config.training.checkpoint_dir
        )

        # ------------------------------------------------------------------ #
        # 10. TensorBoard writer (rank 0 only)
        # ------------------------------------------------------------------ #
        self.writer: Optional[SummaryWriter] = None
        if config.training.tensorboard and self.rank == 0:
            log_dir: str = os.path.join(config.training.log_dir, "frvae")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

        # ------------------------------------------------------------------ #
        # State tracking
        # ------------------------------------------------------------------ #
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_rfid: float = float("inf")

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def train(self) -> None:
        """Run the full FR-VAE training loop.

        Iterates over all epochs defined in config.frvae.epochs (100 by default).
        For each epoch:
          1. Sets the distributed sampler epoch (for proper shuffling).
          2. Runs train_epoch() to update model and discriminator.
          3. Runs validate() every eval_every_epochs epochs.
          4. Saves checkpoints every save_every_epochs epochs.
          5. Tracks best rFID and saves the best checkpoint.

        After training completes, closes the TensorBoard writer.
        """
        start_epoch: int = self.current_epoch
        total_epochs: int = self.config.frvae.epochs

        for epoch in range(start_epoch, total_epochs):
            self.current_epoch = epoch

            # Update distributed sampler epoch for proper per-epoch shuffling.
            if self._train_sampler is not None:
                self._train_sampler.set_epoch(epoch)

            # --- Training epoch ---
            train_metrics: Dict[str, float] = self.train_epoch(epoch)

            # --- Validation ---
            val_metrics: Dict[str, float] = {}
            if epoch % self.config.training.eval_every_epochs == 0:
                val_metrics = self.validate(epoch)

                # Track best rFID and save best checkpoint (rank 0 only).
                if self.rank == 0:
                    current_rfid: float = val_metrics.get("rfid", float("inf"))
                    if current_rfid < self.best_rfid:
                        self.best_rfid = current_rfid
                        self.save_checkpoint(
                            epoch=epoch,
                            metrics={**train_metrics, **val_metrics, "is_best": True},
                        )
                        if self.writer is not None:
                            self.writer.add_scalar(
                                "val/best_rfid", self.best_rfid, epoch
                            )

            # --- Periodic checkpoint saving (rank 0 only) ---
            if (
                self.rank == 0
                and epoch % self.config.training.save_every_epochs == 0
            ):
                self.save_checkpoint(
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics},
                )

        # Close TensorBoard writer.
        if self.writer is not None:
            self.writer.close()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch over the full training dataset.

        Iterates over all batches in train_loader. For each batch:
          1. Moves images to device.
          2. Runs generator training step (FRVAE forward + backward).
          3. Runs discriminator training step (DINODiscriminator forward + backward).
          4. Steps LR schedulers.
          5. Logs metrics to TensorBoard every log_every_steps steps.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Dictionary of mean loss values over the epoch:
                - 'loss_reconstruction': mean pixel L2 loss
                - 'loss_freq_quantization': mean feature L2 loss
                - 'loss_perceptual': mean LPIPS loss
                - 'loss_gan_generator': mean GAN generator loss
                - 'loss_commitment': mean VQ commitment loss
                - 'loss_total_generator': mean total generator loss
                - 'loss_discriminator': mean discriminator loss
        """
        # Set models to training mode.
        self.model.train()
        self.discriminator.train()

        # Accumulators for epoch-level mean metrics.
        epoch_metrics: Dict[str, float] = {
            "loss_reconstruction": 0.0,
            "loss_freq_quantization": 0.0,
            "loss_perceptual": 0.0,
            "loss_gan_generator": 0.0,
            "loss_commitment": 0.0,
            "loss_total_generator": 0.0,
            "loss_discriminator": 0.0,
        }
        num_batches: int = 0

        for batch_idx, (images, _labels) in enumerate(self.train_loader):
            images: Tensor = images.to(self.device, non_blocking=True)

            # ---------------------------------------------------------------- #
            # Generator step: update FRVAE parameters
            # ---------------------------------------------------------------- #
            x_hat: Tensor
            g_loss_dict: Dict[str, float]
            x_hat, g_loss_dict = self._train_generator_step(images)

            # ---------------------------------------------------------------- #
            # Discriminator step: update DINODiscriminator head parameters
            # ---------------------------------------------------------------- #
            _d_tensor: Tensor
            d_loss_dict: Dict[str, float]
            _d_tensor, d_loss_dict = self._train_discriminator_step(images, x_hat)

            # ---------------------------------------------------------------- #
            # LR scheduler steps (per optimizer step, not per epoch)
            # ---------------------------------------------------------------- #
            self.scheduler_g.step()
            self.scheduler_d.step()

            # ---------------------------------------------------------------- #
            # Accumulate epoch metrics
            # ---------------------------------------------------------------- #
            for key in epoch_metrics:
                if key in g_loss_dict:
                    epoch_metrics[key] += g_loss_dict[key]
                elif key in d_loss_dict:
                    epoch_metrics[key] += d_loss_dict[key]
            num_batches += 1

            # ---------------------------------------------------------------- #
            # TensorBoard logging (rank 0, every log_every_steps steps)
            # ---------------------------------------------------------------- #
            if (
                self.rank == 0
                and self.writer is not None
                and self.global_step % self.config.training.log_every_steps == 0
            ):
                self._log_train_step(g_loss_dict, d_loss_dict, epoch)

            self.global_step += 1

        # Compute epoch means.
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

        # Log epoch-level metrics.
        if self.rank == 0 and self.writer is not None:
            for key, value in epoch_metrics.items():
                self.writer.add_scalar(f"train_epoch/{key}", value, epoch)

        return epoch_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation over the full validation dataset.

        Computes:
          - Mean reconstruction loss (pixel L2) over all validation batches.
          - Reconstruction FID (rFID) by saving reconstructed images and
            computing FID against the real validation images.

        The model is set to eval mode during validation and restored to
        train mode afterward.

        Args:
            epoch: Current epoch index for logging.

        Returns:
            Dictionary with validation metrics:
                - 'rfid': Reconstruction FID (lower is better; target 0.85)
                - 'val_rec_loss': Mean pixel L2 reconstruction loss
        """
        # Set model to eval mode (disables dropout, uses running stats for BN).
        self.model.eval()
        self.discriminator.eval()

        val_rec_loss: float = 0.0
        num_batches: int = 0

        # Temporary directories for saving real and reconstructed images for FID.
        # Only rank 0 computes FID to avoid redundant computation.
        real_images_list: List[Tensor] = []
        recon_images_list: List[Tensor] = []

        with torch.no_grad():
            for images, _labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)

                # Forward pass: get reconstruction.
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self._autocast_dtype,
                    enabled=self._amp_enabled,
                ):
                    # FRVAE.forward() returns (x_hat, f, f_tilde).
                    raw_model = (
                        self.model.module
                        if hasattr(self.model, "module")
                        else self.model
                    )
                    x_hat, _f, _f_tilde = raw_model.forward(images)

                # Compute reconstruction loss (in float32 for accuracy).
                x_hat_f32: Tensor = x_hat.float()
                images_f32: Tensor = images.float()
                rec_loss: Tensor = self.losses.reconstruction_loss(
                    images_f32, x_hat_f32
                )
                val_rec_loss += rec_loss.item()
                num_batches += 1

                # Collect images for FID computation (CPU, float32, [-1, 1]).
                # Limit to first 5000 samples to keep validation fast.
                if len(real_images_list) * images.shape[0] < 5000:
                    real_images_list.append(images.cpu().float())
                    recon_images_list.append(x_hat.cpu().float())

        # Compute mean reconstruction loss.
        if num_batches > 0:
            val_rec_loss /= num_batches

        # Compute rFID using saved images.
        rfid: float = self._compute_rfid(real_images_list, recon_images_list)

        val_metrics: Dict[str, float] = {
            "rfid": rfid,
            "val_rec_loss": val_rec_loss,
        }

        # Log to TensorBoard (rank 0 only).
        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar("val/rfid", rfid, epoch)
            self.writer.add_scalar("val/reconstruction_loss", val_rec_loss, epoch)

            # Log image grid for qualitative inspection.
            if real_images_list and recon_images_list:
                self._log_image_grid(
                    real_images=real_images_list[0],
                    recon_images=recon_images_list[0],
                    epoch=epoch,
                )

        # Restore training mode.
        self.model.train()
        self.discriminator.train()

        return val_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model and discriminator checkpoints.

        Saves both the FRVAE generator and the DINODiscriminator to separate
        checkpoint files. Only called on rank 0 in distributed training.

        Args:
            epoch: Current epoch index (used in checkpoint filename).
            metrics: Dictionary of metrics to store alongside the checkpoint
                for tracking training progress.
        """
        if self.rank != 0:
            return

        # Access underlying model (unwrap DDP if needed).
        raw_model: FRVAE = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        raw_disc: DINODiscriminator = (
            self.discriminator.module
            if hasattr(self.discriminator, "module")
            else self.discriminator
        )

        # Save FRVAE generator checkpoint.
        self.checkpoint_manager.save(
            model=raw_model,
            optimizer=self.optimizer_g,
            epoch=epoch,
            metrics=metrics,
            name=f"frvae_epoch_{epoch:04d}",
        )

        # Save discriminator checkpoint.
        self.checkpoint_manager.save(
            model=raw_disc,
            optimizer=self.optimizer_d,
            epoch=epoch,
            metrics=metrics,
            name=f"disc_epoch_{epoch:04d}",
        )

    def load_checkpoint(self, path: str) -> int:
        """Load a FRVAE checkpoint and restore training state.

        Loads the model weights, optimizer state, and epoch number from a
        previously saved checkpoint. Also attempts to load the corresponding
        discriminator checkpoint if it exists.

        Args:
            path: Path to the FRVAE generator checkpoint file.
                The discriminator checkpoint is inferred by replacing
                'frvae_' with 'disc_' in the filename.

        Returns:
            The epoch number to resume training from (epoch + 1 of the
            saved checkpoint, so training continues from the next epoch).
        """
        # Access underlying model (unwrap DDP if needed).
        raw_model: FRVAE = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        raw_disc: DINODiscriminator = (
            self.discriminator.module
            if hasattr(self.discriminator, "module")
            else self.discriminator
        )

        # Load FRVAE generator checkpoint.
        start_epoch: int
        _metrics: Dict
        start_epoch, _metrics = self.checkpoint_manager.load(
            path=path,
            model=raw_model,
            optimizer=self.optimizer_g,
        )

        # Attempt to load discriminator checkpoint.
        # Infer discriminator checkpoint path from generator checkpoint path.
        disc_path: str = path.replace("frvae_", "disc_")
        if os.path.exists(disc_path):
            _disc_epoch: int
            _disc_metrics: Dict
            _disc_epoch, _disc_metrics = self.checkpoint_manager.load(
                path=disc_path,
                model=raw_disc,
                optimizer=self.optimizer_d,
            )

        # Update trainer state.
        self.current_epoch = start_epoch + 1
        self.global_step = (start_epoch + 1) * len(self.train_loader)

        return start_epoch + 1

    # ---------------------------------------------------------------------- #
    # Private training step methods
    # ---------------------------------------------------------------------- #

    def _train_generator_step(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Perform one generator (FRVAE) training step.

        Computes the full composite loss from paper Appendix B.1:
            L = reconstruction_weight * ||I - Î||²
              + freq_quantization_weight * ||f̂ - f̂_reconstructed||²
              + lpips_weight * L_p(I)
              + gan_loss_weight * L_g(I)
              + commitment_loss_weight * L_commit

        Updates FRVAE parameters via optimizer_g.

        Args:
            x: Input image batch of shape (B, 3, H, W), values in [-1, 1].
               On the correct device.

        Returns:
            Tuple of:
                - x_hat: Reconstructed image batch (detached), shape (B, 3, H, W).
                  Passed to _train_discriminator_step() for the discriminator update.
                - loss_dict: Dictionary of individual loss values (Python floats)
                  for logging. Keys:
                    'loss_reconstruction', 'loss_freq_quantization',
                    'loss_perceptual', 'loss_gan_generator', 'loss_commitment',
                    'loss_total_generator'
        """
        self.optimizer_g.zero_grad(set_to_none=True)

        # ------------------------------------------------------------------ #
        # Forward pass through FRVAE (with autocast for mixed precision)
        # ------------------------------------------------------------------ #
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self._autocast_dtype,
            enabled=self._amp_enabled,
        ):
            # FRVAE.forward() returns (x_hat, f, f_tilde).
            # Side effect: populates quantizer._last_v_pairs for commitment loss.
            raw_model: FRVAE = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            x_hat: Tensor
            f: Tensor
            f_tilde: Tensor
            x_hat, f, f_tilde = raw_model.forward(x)

            # ---------------------------------------------------------------- #
            # Compute all generator loss components
            # ---------------------------------------------------------------- #

            # 1. Pixel-space L2 reconstruction loss: ||I - Î||²
            # Cast to float32 for LPIPS and loss stability.
            x_f32: Tensor = x.float()
            x_hat_f32: Tensor = x_hat.float()
            f_f32: Tensor = f.float()
            f_tilde_f32: Tensor = f_tilde.float()

            l_rec: Tensor = self.losses.reconstruction_loss(x_f32, x_hat_f32)

            # 2. Feature-space L2 frequency quantization loss: ||f̂ - f̂_reconstructed||²
            l_freq: Tensor = self.losses.frequency_quantization_loss(
                f_f32, f_tilde_f32
            )

            # 3. LPIPS perceptual loss: L_p(I)
            # LPIPS expects float32 inputs in [-1, 1].
            l_lpips: Tensor = self.losses.perceptual_loss(x_f32, x_hat_f32)

            # 4. GAN generator loss: L_g(I)
            # Discriminator forward on reconstructed images (no detach — gradients
            # must flow through x_hat to update the generator).
            fake_logits: Tensor = self.discriminator(x_hat_f32)
            l_gen: Tensor = self.losses.gan_generator_loss(fake_logits)

            # 5. VQ commitment + codebook loss: L_commit
            # Aggregated across all 10 frequency bands via FRVAE convenience method.
            l_commit: Tensor = raw_model.get_last_codebook_loss()

            # ---------------------------------------------------------------- #
            # Assemble total generator loss (paper Appendix B.1 weights)
            # ---------------------------------------------------------------- #
            cfg = self.config.frvae
            total_g_loss: Tensor = (
                cfg.reconstruction_weight * l_rec
                + cfg.freq_quantization_weight * l_freq
                + l_lpips  # lpips_weight already applied inside losses.perceptual_loss
                + l_gen    # gan_loss_weight already applied inside losses.gan_generator_loss