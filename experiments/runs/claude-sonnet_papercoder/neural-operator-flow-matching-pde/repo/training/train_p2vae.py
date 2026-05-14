## training/train_p2vae.py
"""Stage 1 training loop for P2VAE (Pretrained Physics Variational Autoencoder).

Trains the SD-VAE-based autoencoder to compress PDE field snapshots from
c3p128 (3 channels, 128×128) to c16p16 (16 channels, 16×16) latent grids.

Training configuration (from config.yaml, p2vae.training):
    - AdamW optimizer: beta1=0.9, beta2=0.995, weight_decay=1e-4
    - Cosine LR schedule with 10% linear warmup over 100k steps
    - Base LR = 1e-4 at batch_size=256 (linearly scaled for other sizes)
    - KL weight beta = 1e-3 (applied inside P2VAE.compute_loss)
    - AMP float16 mixed precision on H-100 GPUs
    - DDP across 4 GPUs (config: p2vae.hardware.num_gpus = 4)

The checkpoint produced by this trainer is consumed by FMTTrainer (Stage 2)
with frozen P2VAE weights. The checkpoint format follows the shared knowledge
specification: {'model', 'optimizer', 'scheduler', 'scaler', 'step', 'config'}.
"""

import logging
import os
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from evaluation.metrics import Metrics
from models.p2vae import P2VAE
from utils.distributed import is_main_process, reduce_tensor
from utils.lr_scheduler import CosineWarmupScheduler, get_lr_scale

logger = logging.getLogger(__name__)


def _infinite_loader(loader: DataLoader) -> Iterator[Dict[str, object]]:
    """Yield batches from a DataLoader indefinitely.

    Wraps the DataLoader in an infinite loop so the fixed-step training
    loop (100k steps) does not need to track epochs explicitly. When the
    DataLoader is exhausted, it restarts from the beginning.

    For DDP training with DistributedSampler, the caller is responsible for
    calling sampler.set_epoch(epoch) to ensure proper shuffling across epochs.
    This is handled inside P2VAETrainer.train().

    Args:
        loader: PyTorch DataLoader to iterate over indefinitely.

    Yields:
        Batches from the DataLoader, cycling indefinitely.
    """
    epoch: int = 0
    while True:
        # Notify DistributedSampler of the current epoch for proper shuffling.
        # The sampler attribute may not exist for non-distributed loaders.
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)  # type: ignore[union-attr]
        for batch in loader:
            yield batch
        epoch += 1


class P2VAETrainer:
    """Trainer for Stage 1: P2VAE autoencoder pretraining.

    Handles DDP multi-GPU training, AMP mixed precision, cosine LR scheduling
    with linear warmup, gradient clipping, periodic validation, and checkpoint
    saving/loading.

    The trainer processes trajectory batches of shape (B, 4, 3, 128, 128) by
    reshaping to (B*4, 3, 128, 128) so each frame is treated as an independent
    sample by the spatial autoencoder.

    Attributes:
        rank: DDP rank of this process (0 to world_size - 1).
        world_size: Total number of DDP processes.
        config: Full configuration dictionary loaded from config.yaml.
        device: CUDA device assigned to this rank.
        model: P2VAE model, DDP-wrapped when world_size > 1.
        optimizer: AdamW optimizer with paper-specified hyperparameters.
        scheduler: CosineWarmupScheduler for LR decay.
        scaler: GradScaler for AMP float16 training.
        grad_clip: Maximum gradient norm for clipping.
        log_every: Log training metrics every this many steps.
        val_every: Run validation every this many steps.
        save_every: Save checkpoint every this many steps.
        metrics: Metrics instance for L2RE and VRMSE computation.
        best_val_l2re: Best validation L2RE seen so far (for save_best).
        save_dir: Directory for checkpoint files.
        use_wandb: Whether to log to Weights & Biases.
    """

    def __init__(
        self,
        model: P2VAE,
        config: Dict,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Initialize P2VAETrainer.

        Sets up device assignment, DDP wrapping, AdamW optimizer with
        paper-specified hyperparameters, cosine LR scheduler, AMP GradScaler,
        and optional WandB logging.

        Args:
            model: Instantiated P2VAE model (not yet moved to device).
            config: Full configuration dictionary from config.yaml. Must
                contain 'p2vae', 'logging', and 'checkpointing' top-level keys.
            rank: DDP rank of this process. 0 for single-GPU or main process.
                From config: p2vae.hardware.num_gpus = 4 → ranks 0-3.
            world_size: Total number of DDP processes. 1 for single-GPU.
                From config: p2vae.hardware.num_gpus = 4.
        """
        self.rank: int = rank
        self.world_size: int = world_size
        self.config: Dict = config

        # ------------------------------------------------------------------
        # Device assignment: bind this process to its corresponding GPU.
        # ------------------------------------------------------------------
        if torch.cuda.is_available():
            self.device: torch.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")
            logger.warning(
                "CUDA not available. Training on CPU (not recommended for production)."
            )

        # Move model to device before DDP wrapping.
        model = model.to(self.device)

        # ------------------------------------------------------------------
        # DDP wrapping (only when world_size > 1).
        # find_unused_parameters=False: all P2VAE parameters are used in
        # every forward pass (no conditional computation paths).
        # ------------------------------------------------------------------
        if world_size > 1:
            self.model: nn.Module = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
            )
        else:
            self.model = model

        # ------------------------------------------------------------------
        # Optimizer: AdamW with paper-specified hyperparameters.
        # From config.yaml: p2vae.training.beta1=0.9, beta2=0.995,
        # weight_decay=1e-4, base_lr=1e-4 at base_batch_size=256.
        # ------------------------------------------------------------------
        training_cfg: Dict = config["p2vae"]["training"]
        base_lr: float = float(training_cfg["base_lr"])  # 1e-4
        base_batch_size: int = int(training_cfg["base_batch_size"])  # 256
        beta1: float = float(training_cfg["beta1"])  # 0.9
        beta2: float = float(training_cfg["beta2"])  # 0.995
        weight_decay: float = float(training_cfg["weight_decay"])  # 1e-4

        # Infer actual batch size from config for LR scaling.
        # The DataLoader batch_size is set externally; we use the config value
        # as the reference for linear LR scaling.
        actual_batch_size: int = int(training_cfg.get("batch_size", base_batch_size))
        scaled_lr: float = get_lr_scale(base_lr, actual_batch_size, base_batch_size)

        logger.info(
            "P2VAETrainer: base_lr=%.2e, actual_batch_size=%d, "
            "scaled_lr=%.2e (linear scaling)",
            base_lr,
            actual_batch_size,
            scaled_lr,
        )

        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.model.parameters(),
            lr=scaled_lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=1e-8,
        )

        # ------------------------------------------------------------------
        # LR scheduler: cosine decay with 10% linear warmup.
        # From config.yaml: p2vae.training.total_steps=100000,
        # warmup_ratio=0.1, min_lr_ratio=0.0.
        # ------------------------------------------------------------------
        total_steps: int = int(training_cfg["total_steps"])  # 100000
        warmup_ratio: float = float(training_cfg["warmup_ratio"])  # 0.1
        min_lr_ratio: float = float(training_cfg.get("min_lr_ratio", 0.0))

        self.scheduler: CosineWarmupScheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            total_steps=total_steps,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
        )

        # ------------------------------------------------------------------
        # AMP GradScaler for float16 mixed precision training.
        # Enabled only on CUDA; no-op on CPU.
        # ------------------------------------------------------------------
        self.scaler: GradScaler = GradScaler(enabled=torch.cuda.is_available())

        # ------------------------------------------------------------------
        # Gradient clipping value.
        # From config.yaml: p2vae.training.gradient_clip = 1.0 [ESTIMATED].
        # ------------------------------------------------------------------
        self.grad_clip: float = float(training_cfg.get("gradient_clip", 1.0))

        # ------------------------------------------------------------------
        # Logging and checkpointing cadence from config.yaml.
        # ------------------------------------------------------------------
        logging_cfg: Dict = config["logging"]
        self.log_every: int = int(logging_cfg.get("log_every_n_steps", 100))
        self.val_every: int = int(logging_cfg.get("val_every_n_steps", 1000))
        self.save_every: int = int(logging_cfg.get("save_every_n_steps", 10000))
        self.use_wandb: bool = bool(logging_cfg.get("use_wandb", False))

        ckpt_cfg: Dict = config.get("checkpointing", {})
        self.save_dir: str = str(ckpt_cfg.get("save_dir", "checkpoints"))
        self.save_best: bool = bool(ckpt_cfg.get("save_best", True))

        # ------------------------------------------------------------------
        # Metrics for validation.
        # ------------------------------------------------------------------
        self.metrics: Metrics = Metrics()

        # Track best validation L2RE for save_best checkpointing.
        self.best_val_l2re: float = float("inf")

        # ------------------------------------------------------------------
        # WandB initialization (rank 0 only).
        # ------------------------------------------------------------------
        if self.use_wandb and is_main_process():
            try:
                import wandb  # type: ignore[import]

                wandb.init(
                    project=logging_cfg.get(
                        "project_name", "generative_pde_foundation_model"
                    ),
                    name="p2vae_training",
                    config=config,
                    resume="allow",
                )
                logger.info("WandB initialized for P2VAE training.")
            except ImportError:
                logger.warning(
                    "wandb not installed. Disabling WandB logging."
                )
                self.use_wandb = False
            except Exception as exc:
                logger.warning("WandB initialization failed: %s. Disabling.", exc)
                self.use_wandb = False

        logger.info(
            "P2VAETrainer initialized: rank=%d, world_size=%d, device=%s, "
            "total_steps=%d, scaled_lr=%.2e",
            rank,
            world_size,
            self.device,
            total_steps,
            scaled_lr,
        )

    @property
    def raw_model(self) -> P2VAE:
        """Return the unwrapped P2VAE model (without DDP wrapper).

        Used for checkpoint saving (saves raw state_dict, not DDP state_dict)
        and for direct method calls during validation (get_latent, decode).

        Returns:
            The underlying P2VAE instance, regardless of DDP wrapping.
        """
        if self.world_size > 1:
            return self.model.module  # type: ignore[return-value]
        return self.model  # type: ignore[return-value]

    def train_step(self, batch: Dict[str, object]) -> Dict[str, float]:
        """Execute one forward + backward pass on a single batch.

        Reshapes the (B, 4, 3, 128, 128) trajectory batch to (B*4, 3, 128, 128)
        so each frame is treated as an independent sample by the spatial
        autoencoder. Computes the VAE loss (reconstruction + KL), performs
        the backward pass with AMP and gradient clipping, and steps the
        optimizer and scheduler.

        Args:
            batch: Dictionary from PDEUnifiedDataset.__getitem__ with keys:
                'frames': Tensor of shape (B, 4, 3, 128, 128), dtype float32.
                'dataset_name': List of str (not used in training step).

        Returns:
            Dictionary with scalar loss values and current LR:
                'recon_loss': Reconstruction loss (0.5 * ||x - x_hat||^2).
                'kl_loss': KL divergence loss (beta * KL).
                'total_loss': Sum of recon_loss and kl_loss.
                'lr': Current learning rate from scheduler.
        """
        # Extract frames tensor from batch dict.
        frames: Tensor = batch["frames"]  # type: ignore[assignment]

        # Reshape (B, 4, 3, 128, 128) → (B*4, 3, 128, 128).
        # P2VAE is a spatial autoencoder — processes each frame independently.
        b, t, c, h, w = frames.shape
        x: Tensor = frames.reshape(b * t, c, h, w)

        # Move to device with non-blocking transfer for overlap with compute.
        x = x.to(self.device, non_blocking=True)

        # Zero gradients before forward pass.
        self.optimizer.zero_grad(set_to_none=True)

        # AMP forward pass: activations computed in float16 for efficiency.
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            loss_dict: Dict[str, Tensor] = self.raw_model.compute_loss(x)
            total_loss: Tensor = loss_dict["total_loss"]

        # Backward pass with gradient scaling to prevent float16 underflow.
        self.scaler.scale(total_loss).backward()

        # Unscale gradients before clipping (required by GradScaler API).
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping: prevents exploding gradients during early training.
        # From config: p2vae.training.gradient_clip = 1.0 [ESTIMATED].
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.grad_clip,
        )

        # Optimizer step (skipped if gradients contain inf/nan after unscaling).
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # LR scheduler step: called every training step (not every epoch).
        self.scheduler.step()

        # Return scalar losses for logging.
        return {
            "recon_loss": loss_dict["recon_loss"].item(),
            "kl_loss": loss_dict["kl_loss"].item(),
            "total_loss": loss_dict["total_loss"].item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_step: int = 0,
    ) -> None:
        """Run the full P2VAE training loop for 100k steps.

        Iterates over the training DataLoader indefinitely (cycling through
        epochs as needed), logging metrics every 100 steps, validating every
        1000 steps, and saving checkpoints every 10000 steps.

        Args:
            train_loader: DataLoader over PDEUnifiedDataset (train split).
                Should use DistributedSampler when world_size > 1.
            val_loader: DataLoader over PDEUnifiedDataset (val split).
                Used for periodic L2RE and VRMSE evaluation.
            start_step: Step to resume from. Pass 0 for fresh training or
                the step loaded from a checkpoint for resumption.
        """
        total_steps: int = int(
            self.config["p2vae"]["training"]["total_steps"]
        )  # 100000

        # Set model to training mode.
        self.model.train()

        # Create infinite data iterator that handles epoch-based shuffling.
        data_iter: Iterator[Dict[str, object]] = _infinite_loader(train_loader)

        # Skip steps already completed when resuming from checkpoint.
        # This advances the data iterator to the correct position.
        # Note: for large start_step values this can be slow; a more efficient
        # approach would use a stateful sampler, but this is correct.
        if start_step > 0:
            logger.info(
                "Resuming from step %d. Advancing data iterator...", start_step
            )
            for _ in range(start_step):
                next(data_iter)

        logger.info(
            "Starting P2VAE training: steps %d → %d",
            start_step,
            total_steps,
        )

        # Running loss accumulators for periodic logging.
        running_recon: float = 0.0
        running_kl: float = 0.0
        running_total: float = 0.0
        log_count: int = 0

        for step in range(start_step, total_steps):
            # Fetch next batch from the infinite iterator.
            batch: Dict[str, object] = next(data_iter)

            # Execute one training step.
            loss_dict: Dict[str, float] = self.train_step(batch)

            # Accumulate losses for averaged logging.
            running_recon += loss_dict["recon_loss"]
            running_kl += loss_dict["kl_loss"]
            running_total += loss_dict["total_loss"]
            log_count += 1

            # ------------------------------------------------------------------
            # Periodic logging (rank 0 only, every log_every=100 steps).
            # ------------------------------------------------------------------
            if (step + 1) % self.log_every == 0 and is_main_process():
                avg_recon: float = running_recon / log_count
                avg_kl: float = running_kl / log_count
                avg_total: float = running_total / log_count
                current_lr: float = loss_dict["lr"]

                logger.info(
                    "Step %6d/%d | recon=%.4f | kl=%.6f | total=%.4f | lr=%.2e",
                    step + 1,
                    total_steps,
                    avg_recon,
                    avg_kl,
                    avg_total,
                    current_lr,
                )

                if self.use_wandb:
                    try:
                        import wandb  # type: ignore[import]

                        wandb.log(
                            {
                                "train/recon_loss": avg_recon,
                                "train/kl_loss": avg_kl,
                                "train/total_loss": avg_total,
                                "train/lr": current_lr,
                                "step": step + 1,
                            },
                            step=step + 1,
                        )
                    except Exception as exc:
                        logger.warning("WandB log failed at step %d: %s", step + 1, exc)

                # Reset accumulators.
                running_recon = 0.0
                running_kl = 0.0
                running_total = 0.0
                log_count = 0

            # ------------------------------------------------------------------
            # Periodic validation (rank 0 only, every val_every=1000 steps).
            # ------------------------------------------------------------------
            if (step + 1) % self.val_every == 0:
                val_metrics: Dict[str, float] = self.validate(val_loader)

                if is_main_process():
                    logger.info(
                        "Step %6d/%d | val_l2re=%.4f | val_vrmse=%.4f",
                        step + 1,
                        total_steps,
                        val_metrics["val_l2re"],
                        val_metrics["val_vrmse"],
                    )

                    if self.use_wandb:
                        try:
                            import wandb  # type: ignore[import]

                            wandb.log(
                                {
                                    "val/l2re": val_metrics["val_l2re"],
                                    "val/vrmse": val_metrics["val_vrmse"],
                                    "step": step + 1,
                                },
                                step=step + 1,
                            )
                        except Exception as exc:
                            logger.warning(
                                "WandB val log failed at step %d: %s", step + 1, exc
                            )

                    # Save best checkpoint based on validation L2RE.
                    if self.save_best and val_metrics["val_l2re"] < self.best_val_l2re:
                        self.best_val_l2re = val_metrics["val_l2re"]
                        best_path: str = os.path.join(self.save_dir, "p2vae_best.pt")
                        self.save_checkpoint(best_path, step + 1)
                        logger.info(
                            "New best val_l2re=%.4f. Saved to %s",
                            self.best_val_l2re,
                            best_path,
                        )

                # Synchronize all DDP ranks after validation.
                if self.world_size > 1:
                    torch.distributed.barrier()

                # Restore training mode after validation.
                self.model.train()

            # ------------------------------------------------------------------
            # Periodic checkpoint saving (rank 0 only, every save_every=10000).
            # ------------------------------------------------------------------
            if (step + 1) % self.save_every == 0 and is_main_process():
                step_ckpt_path: str = os.path.join(
                    self.save_dir, f"p2vae_step{step + 1}.pt"
                )
                self.save_checkpoint(step_ckpt_path, step + 1)

                # Always keep a 'latest' checkpoint for easy resumption.
                latest_path: str = os.path.join(self.save_dir, "p2vae_latest.pt")
                self.save_checkpoint(latest_path, step + 1)

                logger.info(
                    "Checkpoint saved at step %d: %s", step + 1, step_ckpt_path
                )

                # Synchronize after checkpoint save.
                if self.world_size > 1:
                    torch.distributed.barrier()

        # ------------------------------------------------------------------
        # Final checkpoint after training completes.
        # ------------------------------------------------------------------
        if is_main_process():
            final_path: str = os.path.join(self.save_dir, "p2vae_final.pt")
            self.save_checkpoint(final_path, total_steps)
            logger.info(
                "P2VAE training complete. Final checkpoint: %s", final_path
            )

            if self.use_wandb:
                try:
                    import wandb  # type: ignore[import]

                    wandb.finish()
                except Exception:
                    pass

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Compute reconstruction L2RE and VRMSE on the validation set.

        Evaluates the P2VAE reconstruction quality by encoding each frame
        to its posterior mean (deterministic latent, no sampling) and
        decoding back to pixel space. This matches how FMT will consume
        P2VAE latents during Stage 2 training.

        Metrics are averaged over all validation batches. In DDP mode,
        metrics are reduced across all ranks via all-reduce.

        Args:
            val_loader: DataLoader over PDEUnifiedDataset (val split).

        Returns:
            Dictionary with keys:
                'val_l2re': Mean L2 relative error over the validation set.
                'val_vrmse': Mean variance-normalized RMSE over the validation set.
        """
        # Switch to evaluation mode (disables dropout, uses running BN stats).
        self.raw_model.eval()

        all_l2re: List[float] = []
        all_vrmse: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                frames: Tensor = batch["frames"]  # type: ignore[assignment]

                # Reshape (B, 4, 3, 128, 128) → (B*4, 3, 128, 128).
                b, t, c, h, w = frames.shape
                x: Tensor = frames.reshape(b * t, c, h, w).to(
                    self.device, non_blocking=True
                )

                # AMP inference for consistency with training.
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    # get_latent returns mu (posterior mean, no sampling).
                    # This is the deterministic latent used for FMT training.
                    z: Tensor = self.raw_model.get_latent(x)  # (B*4, 16, 16, 16)

                    # Decode latent back to pixel space.
                    x_hat: Tensor = self.raw_model.decode(z)  # (B*4, 3, 128, 128)

                # Compute metrics in float32 for numerical stability.
                x_f32: Tensor = x.float()
                x_hat_f32: Tensor = x_hat.float()

                batch_l2re: Tensor = self.metrics.l2_relative_error(
                    x_hat_f32, x_f32
                )
                batch_vrmse: Tensor = self.metrics.vrmse(x_hat_f32, x_f32)

                all_l2re.append(batch_l2re.item())
                all_vrmse.append(batch_vrmse.item())

        # Compute mean over all validation batches.
        mean_l2re: float = sum(all_l2re) / max(len(all_l2re), 1)
        mean_vrmse: float = sum(all_vrmse) / max(len(all_vrmse), 1)

        # In DDP mode, average metrics across all ranks.
        if self.world_size > 1:
            l2re_tensor: Tensor = torch.tensor(
                mean_l2re, device=self.device, dtype=torch.float32
            )
            vrmse_tensor: Tensor = torch.tensor(
                mean_vrmse, device=self.device, dtype=torch.float32
            )
            l2re_tensor = reduce_tensor(l2re_tensor, self.world_size)
            vrmse_tensor = reduce_tensor(vrmse_tensor, self.world_size)
            mean_l2re = l2re_tensor.item()
            mean_vrmse = vrmse_tensor.item()

        return {
            "val_l2re": mean_l2re,
            "val_vrmse": mean_vrmse,
        }

    def save_checkpoint(self, path: str, step: int) -> None:
        """Save full training state to a checkpoint file.

        Saves the raw model state_dict (without DDP wrapper), optimizer,
        scheduler, scaler, current step, and config for full reproducibility.
        The checkpoint format is compatible with FMTTrainer.load_checkpoint
        for loading the frozen P2VAE in Stage 2.

        Only called from rank 0 (is_main_process() guard in train()).

        Args:
            path: Full path to the checkpoint file (e.g.,
                'checkpoints/p2vae_step10000.pt').
            step: Current training step (1-indexed, i.e., steps completed).
        """
        # Create parent directory if it doesn't exist.
        parent_dir: str = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Build checkpoint dict following the shared knowledge specification:
        # {'model', 'optimizer', 'scheduler', 'scaler', 'step', 'config'}
        checkpoint: Dict[str, object] = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": step,
            "config": self.config,
        }

        # Atomic save: write to temp file then rename to avoid corruption.
        tmp_path: str = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

        logger.debug("Checkpoint saved: %s (step=%d)", path, step)

    def load_checkpoint(self, path: str) -> int:
        """Load training state from a checkpoint file.

        Restores model weights, optimizer state, scheduler state, and scaler
        state. Returns the step number so the training loop can resume from
        the correct position.

        Args:
            path: Path to the checkpoint file to load.

        Returns:
            The training step at which the checkpoint was saved. Pass this
            as start_step to train() for seamless resumption.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {path}. "
                "Check the path or train from scratch."
            )

        # Load to the correct device for this DDP rank.
        checkpoint: Dict[str, object] = torch.load(
            path,
            map_location=self.device,
        )

        # Restore model weights (raw model, not DDP wrapper).
        self.raw_model.load_state_dict(checkpoint["model"])  # type: ignore[arg-type]

        # Restore optimizer state (includes momentum buffers, adaptive LR).
        self.optimizer.load_state_dict(checkpoint["optimizer"])  # type: ignore[arg-type]

        # Restore scheduler state (restores last_epoch for correct LR).
        self.scheduler.load_state_dict(checkpoint["scheduler"])  # type: ignore[arg-type]

        # Restore GradScaler state (restores loss scale and growth interval).
        self.scaler.load_state_dict(checkpoint["scaler"])  # type: ignore[arg-type]

        # Extract and return the saved step.
        step: int = int(checkpoint["step"])  # type: ignore[arg-type]

        logger.info(
            "Checkpoint loaded from %s (step=%d, device=%s)",
            path,
            step,
            self.device,
        )

        return step
