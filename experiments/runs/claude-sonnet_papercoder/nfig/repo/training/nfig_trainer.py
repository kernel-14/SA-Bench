## Code: training/nfig_trainer.py

```python
## training/nfig_trainer.py
"""NFIG Transformer Trainer for Phase 2 of the NFIG framework.

Orchestrates autoregressive training of the NFIG Transformer using discrete
token sequences produced by the frozen FR-VAE tokenizer. The core learning
signal is cross-entropy over codebook indices across all 680 token positions.

Training setup (paper Section 4.1):
    - Optimizer: Adam, lr=8e-5, batch_size=768, 350 epochs
    - CFG dropout: 10% unconditional training (null class replacement)
    - Mixed precision: bf16 on H100
    - Distributed: multi-GPU via DDP

Data flow:
    ImageNet images → frozen FR-VAE.get_tokens() → 680 discrete token indices
    → NFIGTransformer.forward() → logits [B, 680, 4096]
    → cross-entropy loss → backward → optimizer step

Config values used (config.yaml nfig section):
    learning_rate:     8e-5   (explicitly stated in paper Section 4.1)
    batch_size:        768    (explicitly stated in paper Section 4.1)
    epochs:            350    (explicitly stated in paper Section 4.1)
    warmup_steps:      5000
    grad_clip:         1.0
    weight_decay:      0.0
    cfg_dropout_prob:  0.1
    null_class_id:     1000
    scale_factors:     [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    total_tokens:      680
"""

import math
import os
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
from models.frvae.frvae import FRVAE
from models.transformer.nfig_transformer import NFIGTransformer
from utils.checkpoint import CheckpointManager
from utils.config import Config
from utils.losses import NFIGLosses


class NFIGTrainer:
    """Trainer for the NFIG Transformer (Phase 2 of the NFIG pipeline).

    Manages the full autoregressive transformer training loop including:
    - Frozen FR-VAE tokenization of input images
    - CFG dropout for classifier-free guidance training
    - Cross-entropy loss over all 680 token positions
    - Mixed precision training (bf16 on H100)
    - Distributed data parallel training
    - Learning rate warmup + cosine annealing
    - Checkpoint saving/loading with full state restoration
    - TensorBoard logging

    Attributes:
        config: Root Config dataclass loaded from config.yaml.
        frvae: Frozen FRVAE tokenizer (eval mode, no gradients).
        transformer: NFIGTransformer model being trained (possibly DDP-wrapped).
        losses: NFIGLosses module (only transformer_ce_loss is used here).
        optimizer: Adam optimizer for transformer parameters only.
        scheduler: LambdaLR with linear warmup + cosine annealing.
        train_loader: DataLoader for ImageNet training split.
        val_loader: DataLoader for ImageNet validation split.
        checkpoint_manager: CheckpointManager for saving/loading checkpoints.
        writer: TensorBoard SummaryWriter (None if disabled or rank != 0).
        device: Target device (cuda or cpu).
        is_distributed: Whether distributed training is active.
        rank: Process rank (0 for single-GPU or master process).
        world_size: Total number of processes.
        current_epoch: Current training epoch (updated during train()).
        global_step: Global training step counter.
        best_val_loss: Best validation cross-entropy loss seen so far.
        scaler: GradScaler for mixed precision (bf16/fp16).
    """

    def __init__(self, config: Config, frvae_checkpoint: str) -> None:
        """Initialize the NFIGTrainer.

        Constructs all components in the following order:
          1. Device and distributed setup
          2. Load and freeze FR-VAE tokenizer
          3. Instantiate NFIGTransformer
          4. Dataset and DataLoaders
          5. NFIGLosses
          6. Adam optimizer (transformer parameters only)
          7. LR scheduler (linear warmup + cosine annealing)
          8. Mixed precision scaler
          9. CheckpointManager
          10. TensorBoard writer

        Args:
            config: Root Config dataclass populated from config.yaml.
                All hyperparameters are read from this object.
            frvae_checkpoint: Path to the trained FR-VAE checkpoint file.
                The FR-VAE is loaded, frozen, and used exclusively as a
                tokenizer throughout NFIG Transformer training.

        Raises:
            FileNotFoundError: If frvae_checkpoint does not exist.
            ValueError: If config validation fails (propagated from Config).
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
        # 2. Load and freeze FR-VAE tokenizer
        # ------------------------------------------------------------------ #
        # Instantiate FR-VAE and load pretrained weights.
        # The FR-VAE is frozen throughout NFIG Transformer training:
        #   - eval() mode: disables dropout, uses running stats for BN
        #   - requires_grad_(False): no gradient computation through FR-VAE
        #   - torch.no_grad() in _tokenize_batch: no gradient tape allocation
        self._raw_frvae: FRVAE = FRVAE(config.frvae).to(self.device)

        # Load FR-VAE checkpoint weights.
        # CheckpointManager.load() handles state_dict loading.
        # We pass optimizer=None since we don't need to restore FR-VAE optimizer.
        _temp_checkpoint_manager = CheckpointManager(config.training.checkpoint_dir)
        _start_epoch, _metrics = _temp_checkpoint_manager.load(
            path=frvae_checkpoint,
            model=self._raw_frvae,
            optimizer=None,
        )

        # Freeze FR-VAE: no gradients, eval mode.
        self._raw_frvae.freeze()  # calls requires_grad_(False) and eval()

        # Expose as self.frvae for external access (e.g., by Evaluator).
        # FR-VAE is never wrapped in DDP since it has no trainable parameters.
        self.frvae: FRVAE = self._raw_frvae

        # ------------------------------------------------------------------ #
        # 3. Instantiate NFIGTransformer
        # ------------------------------------------------------------------ #
        # This is the only model with trainable parameters in Phase 2.
        # depth=16, hidden_dim=1024, num_heads=16 → ~310M parameters.
        self._raw_transformer: NFIGTransformer = NFIGTransformer(
            config.nfig
        ).to(self.device)

        # Wrap in DDP for distributed training.
        if self.is_distributed:
            self.transformer: nn.Module = DDP(
                self._raw_transformer,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=False,
            )
        else:
            self.transformer = self._raw_transformer

        # ------------------------------------------------------------------ #
        # 4. Dataset and DataLoaders
        # ------------------------------------------------------------------ #
        # Per-GPU batch size: global batch_size=768 divided by world_size.
        # For single-GPU: per_gpu_batch = 768.
        # For 8-GPU: per_gpu_batch = 768 // 8 = 96.
        per_gpu_batch_size: int = max(
            1, config.nfig.batch_size // self.world_size
        )

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

        # Distributed samplers for proper per-rank data sharding.
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
            batch_size=per_gpu_batch_size,
            num_workers=config.data.num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
        )
        self.val_loader: DataLoader = val_dataset.get_dataloader(
            batch_size=per_gpu_batch_size,
            num_workers=config.data.num_workers,
            shuffle=False,
            sampler=val_sampler,
        )

        # Store sampler reference for set_epoch() calls in train().
        self._train_sampler: Optional[DistributedSampler] = train_sampler

        # ------------------------------------------------------------------ #
        # 5. NFIGLosses
        # ------------------------------------------------------------------ #
        # Only transformer_ce_loss is used in Phase 2.
        # GAN and LPIPS losses are not needed here.
        # The module is kept for API consistency with FRVAETrainer.
        self.losses: NFIGLosses = NFIGLosses(
            gan_weight=config.frvae.gan_loss_weight,
            lpips_weight=config.frvae.lpips_weight,
        ).to(self.device)

        # ------------------------------------------------------------------ #
        # 6. Adam optimizer (transformer parameters only)
        # ------------------------------------------------------------------ #
        # Paper Section 4.1: "Adam optimizer, learning rate 8e-5"
        # weight_decay=0.0 per config.nfig.weight_decay
        # betas=(0.9, 0.999): PyTorch Adam defaults (not specified in paper)
        self.optimizer: optim.Adam = optim.Adam(
            self._raw_transformer.parameters(),
            lr=config.nfig.learning_rate,       # 8e-5 from paper
            betas=(0.9, 0.999),
            weight_decay=config.nfig.weight_decay,  # 0.0
            eps=1e-8,
        )

        # ------------------------------------------------------------------ #
        # 7. LR scheduler: linear warmup + cosine annealing
        # ------------------------------------------------------------------ #
        # Paper does not specify the schedule; cosine annealing with linear
        # warmup is standard for transformer training at this scale.
        # Warmup: linearly ramp from 0 to lr over warmup_steps steps.
        # After warmup: cosine decay from lr to lr_min = lr * 0.1.
        total_steps: int = config.nfig.epochs * len(self.train_loader)
        warmup_steps: int = config.nfig.warmup_steps  # 5000

        def _lr_lambda(current_step: int) -> float:
            """Combined linear warmup + cosine annealing LR schedule.

            Args:
                current_step: Current global training step (0-indexed).

            Returns:
                LR multiplier in [0, 1]. Actual LR = base_lr * multiplier.
            """
            if current_step < warmup_steps:
                # Linear warmup: 0 → 1 over warmup_steps steps.
                return float(current_step) / float(max(1, warmup_steps))

            # Cosine annealing after warmup.
            # Progress in [0, 1] over the post-warmup training period.
            progress: float = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            # Cosine decay from 1.0 to min_lr_ratio=0.1.
            min_lr_ratio: float = 0.1
            cosine_factor: float = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        self.scheduler: optim.lr_scheduler.LambdaLR = (
            optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lr_lambda)
        )

        # ------------------------------------------------------------------ #
        # 8. Mixed precision scaler
        # ------------------------------------------------------------------ #
        # bf16 on H100 typically doesn't require loss scaling, but GradScaler
        # is included for compatibility and fp16 fallback.
        amp_enabled: bool = (
            config.training.mixed_precision
            and torch.cuda.is_available()
        )
        self.scaler: torch.cuda.amp.GradScaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled
        )

        # Determine autocast dtype: bf16 on H100, fp16 otherwise.
        if (
            config.training.precision == "bf16"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
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
            log_dir: str = os.path.join(config.training.log_dir, "nfig_transformer")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

        # ------------------------------------------------------------------ #
        # State tracking
        # ------------------------------------------------------------------ #
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def train(self) -> None:
        """Run the full NFIG Transformer training loop.

        Iterates over all epochs defined in config.nfig.epochs (350 by default).
        For each epoch:
          1. Sets the distributed sampler epoch (for proper shuffling).
          2. Runs train_epoch() to update transformer parameters.
          3. Runs validate() every eval_every_epochs epochs.
          4. Saves checkpoints every save_every_epochs epochs.
          5. Tracks best validation loss and saves the best checkpoint.

        After training completes, closes the TensorBoard writer.
        """
        start_epoch: int = self.current_epoch
        total_epochs: int = self.config.nfig.epochs

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

                # Track best validation loss and save best checkpoint (rank 0 only).
                if self.rank == 0:
                    current_val_loss: float = val_metrics.get(
                        "val_loss", float("inf")
                    )
                    if current_val_loss < self.best_val_loss:
                        self.best_val_loss = current_val_loss
                        self.save_checkpoint(
                            epoch=epoch,
                            metrics={
                                **train_metrics,
                                **val_metrics,
                                "is_best": True,
                            },
                        )
                        if self.writer is not None:
                            self.writer.add_scalar(
                                "val/best_loss", self.best_val_loss, epoch
                            )

            # --- Periodic checkpoint saving (rank 0 only) ---
            if (
                self.rank == 0
                and (
                    epoch % self.config.training.save_every_epochs == 0
                    or epoch == total_epochs - 1
                )
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

        For each batch:
          1. Tokenize images via frozen FR-VAE (no_grad).
          2. Apply CFG dropout to class labels.
          3. Forward pass through transformer (autocast bf16).
          4. Compute cross-entropy loss over all 680 token positions.
          5. Backward pass with gradient clipping.
          6. Optimizer and scheduler step.
          7. Log metrics to TensorBoard.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Dictionary of mean metrics over the epoch:
                - 'loss': mean cross-entropy loss
                - 'lr': final learning rate for this epoch
                - 'grad_norm': mean gradient norm before clipping
                - 'perplexity': exp(mean_loss) as interpretable metric
        """
        # Set transformer to training mode.
        # FR-VAE remains in eval mode (frozen).
        self._raw_transformer.train()

        # Accumulators for epoch-level mean metrics.
        epoch_loss: float = 0.0
        epoch_grad_norm: float = 0.0
        num_batches: int = 0

        for batch_idx, (images, class_labels) in enumerate(self.train_loader):
            images: Tensor = images.to(self.device, non_blocking=True)
            class_labels: Tensor = class_labels.to(self.device, non_blocking=True)

            # ---------------------------------------------------------------- #
            # Step 1: Tokenize images via frozen FR-VAE
            # ---------------------------------------------------------------- #
            # Returns List[10 × Tensor[B, h_i*w_i]] of integer token indices.
            # Wrapped in no_grad inside _tokenize_batch.
            token_seqs: List[Tensor] = self._tokenize_batch(images)

            # ---------------------------------------------------------------- #
            # Step 2: Apply CFG dropout to class labels
            # ---------------------------------------------------------------- #
            # Randomly replace 10% of class labels with null_class_id=1000.
            class_labels_dropped: Tensor = self._apply_cfg_dropout(class_labels)

            # ---------------------------------------------------------------- #
            # Step 3: Forward pass through transformer
            # ---------------------------------------------------------------- #
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self._autocast_dtype,
                enabled=self._amp_enabled,
            ):
                # transformer.forward(token_seqs, class_labels) returns
                # logits of shape [B, total_tokens, codebook_size] = [B, 680, 4096].
                # The block-wise causal mask inside the transformer ensures that
                # logits at band i positions are conditioned only on bands 0..i-1.
                logits: Tensor = self.transformer(token_seqs, class_labels_dropped)

                # ---------------------------------------------------------------- #
                # Step 4: Compute cross-entropy loss
                # ---------------------------------------------------------------- #
                # Targets: concatenate all token sequences along the token dimension.
                # token_seqs[i]: [B, h_i*w_i] → cat → [B, total_tokens=680]
                # dtype must be torch.long for cross-entropy target.
                targets: Tensor = torch.cat(token_seqs, dim=1).long()  # [B, 680]

                # Cross-entropy over all 680 token positions.
                # losses.transformer_ce_loss handles reshaping internally.
                loss: Tensor = self.losses.transformer_ce_loss(logits, targets)

            # ---------------------------------------------------------------- #
            # Step 5: Backward pass with gradient clipping
            # ---------------------------------------------------------------- #
            # Scale loss for mixed precision; backward through scaled loss.
            self.scaler.scale(loss).backward()

            # Unscale gradients before clipping (required for GradScaler).
            self.scaler.unscale_(self.optimizer)

            # Clip gradient norm to config.nfig.grad_clip = 1.0.
            # Compute grad_norm before clipping for logging.
            grad_norm: float = torch.nn.utils.clip_grad_norm_(
                self._raw_transformer.parameters(),
                max_norm=self.config.nfig.grad_clip,
            ).item()

            # ---------------------------------------------------------------- #
            # Step 6: Optimizer and scheduler step
            # ---------------------------------------------------------------- #
            # scaler.step() calls optimizer.step() only if gradients are finite.
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler step per optimizer step (not per epoch) for warmup.
            self.scheduler.step()

            # ---------------------------------------------------------------- #
            # Accumulate epoch metrics
            # ---------------------------------------------------------------- #
            loss_val: float = loss.item()
            epoch_loss += loss_val
            epoch_grad_norm += grad_norm
            num_batches += 1

            # ---------------------------------------------------------------- #
            # Step 7: TensorBoard logging (rank 0, every log_every_steps steps)
            # ---------------------------------------------------------------- #
            if (
                self.rank == 0
                and self.writer is not None
                and self.global_step % self.config.training.log_every_steps == 0
            ):
                current_lr: float = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar("train/loss", loss_val, self.global_step)
                self.writer.add_scalar("train/lr", current_lr, self.global_step)
                self.writer.add_scalar(
                    "train/grad_norm", grad_norm, self.global_step
                )
                self.writer.add_scalar(
                    "train/perplexity",
                    math.exp(min(loss_val, 20.0)),  # clamp to avoid overflow
                    self.global_step,
                )

            self.global_step += 1

        # Compute epoch means.
        mean_loss: float = epoch_loss / max(1, num_batches)
        mean_grad_norm: float = epoch_grad_norm / max(1, num_batches)
        current_lr: float = self.scheduler.get_last_lr()[0]

        epoch_metrics: Dict[str, float] = {
            "loss": mean_loss,
            "lr": current_lr,
            "grad_norm": mean_grad_norm,
            "perplexity": math.exp(min(mean_loss, 20.0)),
        }

        # Log epoch-level metrics to TensorBoard (rank 0 only).
        if self.rank == 0 and self.writer is not None:
            for key, value in epoch_metrics.items():
                self.writer.add_scalar(f"train_epoch/{key}", value, epoch)

        return epoch_metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation over the full validation dataset.

        Computes mean cross-entropy loss on the validation set without CFG
        dropout (real class labels used throughout). No FID/IS computation
        here — that is handled by the Evaluator class.

        The transformer is set to eval mode during validation and restored
        to train mode afterward. FR-VAE remains in eval mode throughout.

        Args:
            epoch: Current epoch index for TensorBoard logging.

        Returns:
            Dictionary with validation metrics:
                - 'val_loss': Mean cross-entropy loss (lower is better)
                - 'val_perplexity': exp(val_loss) as interpretable metric
        """
        # Set transformer to eval mode (disables dropout if any).
        self._raw_transformer.eval()

        val_loss_accum: float = 0.0
        num_batches: int = 0

        with torch.no_grad():
            for images, class_labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                class_labels = class_labels.to(self.device, non_blocking=True)

                # Tokenize images via frozen FR-VAE.
                # _tokenize_batch is already wrapped in no_grad internally,
                # but the outer no_grad context here is redundant-safe.
                token_seqs: List[Tensor] = self._tokenize_batch(images)

                # Forward pass without CFG dropout (use real class labels).
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self._autocast_dtype,
                    enabled=self._amp_enabled,
                ):
                    logits: Tensor = self.transformer(token_seqs, class_labels)

                    # Compute cross-entropy loss.
                    targets: Tensor = torch.cat(token_seqs, dim=1).long()  # [B, 680]
                    loss: Tensor = self.losses.transformer_ce_loss(logits, targets)

                val_loss_accum += loss.item()
                num_batches += 1

        # Compute mean validation loss.
        mean_val_loss: float = val_loss_accum / max(1, num_batches)
        val_perplexity: float = math.exp(min(mean_val_loss, 20.0))

        val_metrics: Dict[str, float] = {
            "val_loss": mean_val_loss,
            "val_perplexity": val_perplexity,
        }

        # Log to TensorBoard (rank 0 only).
        if self.rank == 0 and self.writer is not None:
            self.writer.add_scalar("val/loss", mean_val_loss, epoch)
            self.writer.add_scalar("val/perplexity", val_perplexity, epoch)

        # Restore transformer to training mode.
        self._raw_transformer.train()

        return val_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save transformer checkpoint with full training state.

        Saves the transformer model weights, optimizer state, and scheduler
        state to enable exact training resumption. Only called on rank 0 in
        distributed training.

        The checkpoint filename encodes the epoch number for easy identification.
        The CheckpointManager handles the actual file I/O and tracks the best
        checkpoint based on the provided metrics.

        Args:
            epoch: Current epoch index (used in checkpoint filename).
            metrics: Dictionary of metrics to store alongside the checkpoint.
                Typically includes 'loss', 'val_loss', 'lr', 'perplexity'.
                Stored in the checkpoint for tracking training progress.
        """
        if self.rank != 0:
            return

        # Access underlying model (unwrap DDP if needed).
        raw_transformer: NFIGTransformer = (
            self.transformer.module
            if hasattr(self.transformer, "module")
            else self.transformer
        )

        # Save transformer checkpoint via CheckpointManager.
        # The checkpoint includes: model state_dict, optimizer state_dict,
        # epoch number, and metrics dict.
        self.checkpoint_manager.save(
            model=raw_transformer,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=metrics,
            name=f"nfig_transformer_epoch_{epoch:04d}",
        )

        # Save scheduler state separately alongside the checkpoint.
        # This ensures the LR schedule resumes correctly after loading.
        scheduler_path: str = os.path.join(
            self.config.training.checkpoint_dir,
            f"nfig_scheduler_epoch_{epoch:04d}.pt",
        )
        torch.save(
            {
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
            },
            scheduler_path,
        )

    def load_checkpoint(self, path: str) -> int:
        """Load a transformer checkpoint and restore full training state.

        Loads model weights, optimizer state, and epoch number from a
        previously saved checkpoint. Also attempts to restore the scheduler
        state from the corresponding scheduler checkpoint file.

        Args:
            path: Path to the NFIG Transformer checkpoint file.
                The scheduler checkpoint is inferred by replacing
                'nfig_transformer_' with 'nfig_scheduler_' in the filename.

        Returns:
            The epoch number to resume training from (epoch + 1 of the
            saved checkpoint, so training continues from the next epoch).
        """
        # Access underlying model (unwrap DDP if needed).
        raw_transformer: NFIGTransformer = (
            self.transformer.module
            if hasattr(self.transformer, "module")
            else self.transformer
        )

        # Load transformer checkpoint (model weights + optimizer state).
        start_epoch: int
        _metrics: Dict
        start_epoch, _metrics = self.checkpoint_manager.load(
            path=path,
            model=raw_transformer,
            optimizer=self.optimizer,
        )

        # Attempt to restore scheduler state from the corresponding file.
        # The scheduler checkpoint is stored alongside the model checkpoint.
        scheduler_path: str = path.replace(
            "nfig_transformer_", "nfig_scheduler_"
        )
        if os.path.exists(scheduler_path):
            scheduler_state: Dict = torch.load(
                scheduler_path,
                map_location=self.device,
            )
            self.scheduler.load_state_dict(
                scheduler_state["scheduler_state_dict"]
            )
            self.global_step = scheduler_state.get("global_step", 0)
            self.best_val_loss = scheduler_state.get(
                "best_val_loss", float("inf")
            )

        # Update trainer state: resume from the next epoch.
        self.current_epoch = start_epoch + 1

        return start_epoch + 1

    # ---------------------------------------------------------------------- #
    # Private helper methods
    # ---------------------------------------------------------------------- #

    @torch.no_