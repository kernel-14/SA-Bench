## Code: training/finetune.py

```python
## training/finetune.py
"""Few-shot fine-tuning of pretrained P2VAE + FMT to unseen dynamical systems.

Implements the REPA-E style joint fine-tuning approach described in the paper
(Section 4.4): "we finetune the pretrained model (P2VAE + FMT) to adapt to an
unseen system with a stop-gradient operation after the generation of latent
states y, so that the conditional flow matching loss won't deteriorate the
autoencoder."

The joint fine-tuning loss (paper Eq. 12):
    L(θ, φ, ω) = L_CFM(θ, φ) + λ_VAE * L_VAE(ω)

where:
    - L_CFM is computed on DETACHED latents (stop-gradient on encoder output)
    - L_VAE is computed with full gradients through encoder and decoder
    - λ_VAE = 1.0 (config: finetune.lambda_vae)

Experimental setup (paper Section 4.4):
    - Model: FMT-B-42M + P2VAE-16M
    - Dataset: Kolmogorov turbulence at Re=222 (u, v fields)
    - Training: 200 trajectories, 5k steps
    - Test: 500 trajectories
    - Metrics: L2RE and VRMSE

The stop-gradient asymmetry creates two separate computational graphs:
    Graph A (CFM path):  x → encode → mu → detach → FMT → L_CFM
                                              ↑ gradient stops here
    Graph B (VAE path):  x → encode → mu, logvar → decode → L_VAE
Both graphs share the same backward() call on total_loss = L_CFM + λ_VAE * L_VAE.
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
from models.fmt import FMT, FlowMarchingKernel
from models.p2vae import P2VAE
from utils.lr_scheduler import CosineWarmupScheduler

logger = logging.getLogger(__name__)


def _cycling_loader(loader: DataLoader) -> Iterator[Dict[str, object]]:
    """Yield batches from a DataLoader indefinitely, cycling when exhausted.

    With only 200 training trajectories, the DataLoader will be exhausted
    quickly (e.g., after ~6 batches at batch_size=32). This iterator
    restarts the DataLoader automatically so the 5k-step training loop
    can proceed without manual epoch management.

    Args:
        loader: PyTorch DataLoader to cycle over indefinitely.

    Yields:
        Batches from the DataLoader, cycling indefinitely.
    """
    while True:
        for batch in loader:
            yield batch


class FinetuneTrainer:
    """Trainer for few-shot fine-tuning of P2VAE + FMT to unseen PDE systems.

    Implements joint fine-tuning with the REPA-E stop-gradient trick:
    the CFM loss sees detached latents (encoder not updated via CFM),
    while the VAE loss has full gradients through the autoencoder.

    Both P2VAE and FMT are trainable — neither is frozen during fine-tuning.
    A single joint AdamW optimizer covers all parameters from both models.

    Attributes:
        fmt: FMT model (trainable, not DDP-wrapped for fine-tuning).
        p2vae: P2VAE model (trainable, not DDP-wrapped for fine-tuning).
        optimizer: Joint AdamW optimizer over fmt + p2vae parameters.
        scheduler: CosineWarmupScheduler for LR decay over 5k steps.
        scaler: GradScaler for AMP float16 training.
        lambda_vae: Weight for VAE loss in joint objective (1.0 from config).
        config: Full configuration dictionary loaded from config.yaml.
        device: CUDA device inferred from model parameters.
        grad_clip: Maximum gradient norm for clipping (1.0 from config).
        log_every: Log training metrics every this many steps.
        val_every: Run validation every this many steps.
        save_every: Save checkpoint every this many steps.
        save_dir: Directory for checkpoint files.
        use_wandb: Whether to log to Weights & Biases.
        best_val_l2re: Best validation L2RE seen so far (for save_best).
        metrics: Metrics instance for L2RE and VRMSE computation.
        kernel: FlowMarchingKernel for interpolation (used in validate).
    """

    def __init__(
        self,
        fmt: FMT,
        p2vae: P2VAE,
        config: Dict,
    ) -> None:
        """Initialize FinetuneTrainer.

        Sets up device assignment, joint optimizer over both models,
        cosine LR scheduler, AMP GradScaler, and optional WandB logging.
        Neither model is frozen — both are updated by the joint loss.

        Args:
            fmt: Pretrained FMT model loaded from Stage 2 checkpoint.
                Will be set to train mode and updated by L_CFM gradients.
            p2vae: Pretrained P2VAE model loaded from Stage 1 checkpoint.
                Will be set to train mode and updated by L_VAE gradients.
                The encoder is NOT updated by L_CFM (stop-gradient trick).
            config: Full configuration dictionary from config.yaml. Must
                contain a 'finetune' top-level key with the hyperparameters
                specified in config.yaml under the finetune section.
        """
        self.config: Dict = config

        # ------------------------------------------------------------------
        # Device assignment: infer from model parameters.
        # Fine-tuning is designed for single-process use (no DDP).
        # ------------------------------------------------------------------
        try:
            self.device: torch.device = next(fmt.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to device if not already there.
        self.fmt: FMT = fmt.to(self.device)
        self.p2vae: P2VAE = p2vae.to(self.device)

        # ------------------------------------------------------------------
        # Read fine-tuning hyperparameters from config.yaml finetune section.
        # All values have defaults matching config.yaml to avoid KeyError.
        # ------------------------------------------------------------------
        finetune_cfg: Dict = config.get("finetune", {})

        self.lambda_vae: float = float(finetune_cfg.get("lambda_vae", 1.0))
        self.stop_gradient_latent: bool = bool(
            finetune_cfg.get("stop_gradient_latent", True)
        )

        base_lr: float = float(finetune_cfg.get("base_lr", 1e-4))
        beta1: float = float(finetune_cfg.get("beta1", 0.9))
        beta2: float = float(finetune_cfg.get("beta2", 0.95))
        weight_decay: float = float(finetune_cfg.get("weight_decay", 0.01))
        total_steps: int = int(finetune_cfg.get("total_steps", 5000))
        warmup_ratio: float = float(finetune_cfg.get("warmup_ratio", 0.1))
        min_lr_ratio: float = float(finetune_cfg.get("min_lr_ratio", 0.0))
        self.grad_clip: float = float(finetune_cfg.get("gradient_clip", 1.0))

        # ------------------------------------------------------------------
        # Joint optimizer: covers ALL parameters from both fmt and p2vae.
        # The stop-gradient is enforced in train_step via .detach(), not by
        # excluding parameters from the optimizer. Both models are updated
        # by their respective loss terms.
        # ------------------------------------------------------------------
        all_params: List[nn.Parameter] = (
            list(self.fmt.parameters()) + list(self.p2vae.parameters())
        )

        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            all_params,
            lr=base_lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=1e-8,
        )

        logger.info(
            "FinetuneTrainer: base_lr=%.2e, beta1=%.3f, beta2=%.3f, "
            "weight_decay=%.4f, lambda_vae=%.2f, total_steps=%d",
            base_lr,
            beta1,
            beta2,
            weight_decay,
            self.lambda_vae,
            total_steps,
        )

        # ------------------------------------------------------------------
        # LR scheduler: cosine decay with 10% linear warmup over 5k steps.
        # From config.yaml: finetune.total_steps=5000, warmup_ratio=0.1.
        # ------------------------------------------------------------------
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
        # Logging and checkpointing cadence from config.yaml logging section.
        # ------------------------------------------------------------------
        logging_cfg: Dict = config.get("logging", {})
        self.log_every: int = int(logging_cfg.get("log_every_n_steps", 100))
        self.val_every: int = int(logging_cfg.get("val_every_n_steps", 1000))
        # With only 5k steps, save more frequently than the default 10k.
        self.save_every: int = int(
            logging_cfg.get("save_every_n_steps", min(10000, total_steps // 2))
        )
        self.use_wandb: bool = bool(logging_cfg.get("use_wandb", False))

        ckpt_cfg: Dict = config.get("checkpointing", {})
        self.save_dir: str = str(ckpt_cfg.get("save_dir", "checkpoints"))
        self.save_best: bool = bool(ckpt_cfg.get("save_best", True))

        # ------------------------------------------------------------------
        # Metrics and kernel for validation and loss computation.
        # ------------------------------------------------------------------
        self.metrics: Metrics = Metrics()
        self.kernel: FlowMarchingKernel = FlowMarchingKernel()

        # Track best validation L2RE for save_best checkpointing.
        self.best_val_l2re: float = float("inf")

        # ------------------------------------------------------------------
        # WandB initialization (fine-tuning is single-process, always init).
        # ------------------------------------------------------------------
        if self.use_wandb:
            try:
                import wandb  # type: ignore[import]

                wandb.init(
                    project=logging_cfg.get(
                        "project_name", "generative_pde_foundation_model"
                    ),
                    name="finetune_kolmogorov",
                    config=config,
                    resume="allow",
                )
                logger.info("WandB initialized for fine-tuning.")
            except ImportError:
                logger.warning("wandb not installed. Disabling WandB logging.")
                self.use_wandb = False
            except Exception as exc:
                logger.warning(
                    "WandB initialization failed: %s. Disabling.", exc
                )
                self.use_wandb = False

        logger.info(
            "FinetuneTrainer initialized: device=%s, total_steps=%d, "
            "lambda_vae=%.2f, stop_gradient=%s",
            self.device,
            total_steps,
            self.lambda_vae,
            self.stop_gradient_latent,
        )

    def train_step(self, batch: Dict[str, object]) -> Dict[str, float]:
        """Execute one forward + backward pass with the joint fine-tuning loss.

        Implements the REPA-E stop-gradient trick:
            - CFM path: encode → detach(mu) → FMT → L_CFM
              (encoder NOT updated by CFM gradients)
            - VAE path: encode → reparameterize → decode → L_VAE
              (encoder AND decoder updated by VAE gradients)
            - Total: L_CFM + lambda_vae * L_VAE

        The two paths share the same encoder forward pass output (mu, logvar),
        avoiding redundant computation. The detach() in the CFM path creates
        a gradient barrier at the encoder output for that loss term only.

        Args:
            batch: Dictionary from PDEUnifiedDataset.__getitem__ with keys:
                'frames': Tensor of shape (B, 4, 3, 128, 128), dtype float32.
                    4 consecutive frames: (x_0, x_1, x_2, x_3).
                'dataset_name': List of str (not used in training step).

        Returns:
            Dictionary with scalar loss values and current LR:
                'total_loss': Combined L_CFM + lambda_vae * L_VAE.
                'cfm_loss': Conditional flow matching loss.
                'vae_loss': VAE reconstruction + KL loss.
                'recon_loss': Reconstruction component of VAE loss.
                'kl_loss': KL divergence component of VAE loss.
                'lr': Current learning rate from scheduler.
        """
        # Extract frames tensor from batch dict.
        frames: Tensor = batch["frames"]  # type: ignore[assignment]

        # Move to device with non-blocking transfer.
        frames = frames.to(self.device, non_blocking=True)

        b: int = frames.shape[0]
        t_frames: int = frames.shape[1]  # 4 consecutive frames

        # Reshape (B, 4, 3, 128, 128) → (B*4, 3, 128, 128) for per-frame encoding.
        x_flat: Tensor = frames.view(b * t_frames, 3, 128, 128)

        # Zero gradients before forward pass.
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # ------------------------------------------------------------------
            # Shared encoder forward pass: get mu and logvar for all frames.
            # We call encode() once to avoid running the encoder twice.
            # mu shape: (B*4, 16, 16, 16), logvar shape: (B*4, 16, 16, 16)
            # ------------------------------------------------------------------
            mu: Tensor
            logvar: Tensor
            mu, logvar = self.p2vae.encode(x_flat)

            # ------------------------------------------------------------------
            # Graph A: CFM path with stop-gradient on encoder output.
            # y_detached = mu.detach() creates a gradient barrier:
            # - Gradients from L_CFM flow through FMT parameters only
            # - The encoder (P2VAE) is NOT updated by L_CFM
            # This prevents the generative loss from degrading reconstruction.
            # From paper Section 4.4 and config: finetune.stop_gradient_latent=true
            # ------------------------------------------------------------------
            if self.stop_gradient_latent:
                y_detached: Tensor = mu.detach()
            else:
                # Ablation mode: allow CFM gradients to flow through encoder.
                y_detached = mu

            # Reshape detached latents: (B*4, 16, 16, 16) → (B, 4, 16, 16, 16)
            y_for_cfm: Tensor = y_detached.view(b, t_frames, 16, 16, 16)

            # Compute L_CFM using FMT's internal loss computation.
            # This handles: t/k sampling, interpolation, pyramid, GRU, loss.
            cfm_loss_dict: Dict[str, Tensor] = self.fmt.compute_loss(
                latents=y_for_cfm
            )
            l_cfm: Tensor = cfm_loss_dict["total_loss"]

            # ------------------------------------------------------------------
            # Graph B: VAE path with full gradients through encoder + decoder.
            # p2vae.compute_loss() runs the full VAE forward pass:
            #   encode → reparameterize → decode → recon_loss + kl_loss
            # Gradients flow through both encoder and decoder.
            # ------------------------------------------------------------------
            vae_loss_dict: Dict[str, Tensor] = self.p2vae.compute_loss(x_flat)
            l_vae: Tensor = vae_loss_dict["total_loss"]

            # ------------------------------------------------------------------
            # Joint loss: L_CFM + lambda_vae * L_VAE
            # From paper Eq. 12 and config: finetune.lambda_vae = 1.0
            # ------------------------------------------------------------------
            total_loss: Tensor = l_cfm + self.lambda_vae * l_vae

        # ------------------------------------------------------------------
        # NaN/Inf guard: skip backward if loss is invalid.
        # This can happen early in fine-tuning with aggressive LR.
        # ------------------------------------------------------------------
        if not torch.isfinite(total_loss):
            logger.warning(
                "Non-finite total_loss detected (cfm=%.4f, vae=%.4f). "
                "Skipping backward pass.",
                l_cfm.item() if torch.isfinite(l_cfm) else float("nan"),
                l_vae.item() if torch.isfinite(l_vae) else float("nan"),
            )
            return {
                "total_loss": float("nan"),
                "cfm_loss": l_cfm.item() if torch.isfinite(l_cfm) else float("nan"),
                "vae_loss": l_vae.item() if torch.isfinite(l_vae) else float("nan"),
                "recon_loss": vae_loss_dict["recon_loss"].item(),
                "kl_loss": vae_loss_dict["kl_loss"].item(),
                "lr": self.scheduler.get_last_lr()[0],
            }

        # ------------------------------------------------------------------
        # Backward pass with AMP gradient scaling.
        # The single backward() call handles both Graph A and Graph B:
        # - FMT parameters receive gradients from L_CFM
        # - P2VAE encoder receives gradients from L_VAE only (not L_CFM)
        # - P2VAE decoder receives gradients from L_VAE
        # ------------------------------------------------------------------
        self.scaler.scale(total_loss).backward()

        # Unscale gradients before clipping (required by GradScaler API).
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping: prevents exploding gradients during fine-tuning.
        # Applied to all parameters (fmt + p2vae) jointly.
        all_params: List[nn.Parameter] = (
            list(self.fmt.parameters()) + list(self.p2vae.parameters())
        )
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.grad_clip)

        # Optimizer step (skipped if gradients contain inf/nan after unscaling).
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # LR scheduler step: called every training step (not every epoch).
        self.scheduler.step()

        return {
            "total_loss": total_loss.item(),
            "cfm_loss": l_cfm.item(),
            "vae_loss": l_vae.item(),
            "recon_loss": vae_loss_dict["recon_loss"].item(),
            "kl_loss": vae_loss_dict["kl_loss"].item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_step: int = 0,
    ) -> None:
        """Run the full fine-tuning loop for 5k steps on Kolmogorov turbulence.

        Cycles through the small training set (200 trajectories) as many
        times as needed to complete 5k steps. Logs metrics every 100 steps,
        validates every 1000 steps, and saves checkpoints at configured
        intervals and at the end of training.

        Args:
            train_loader: DataLoader over the Kolmogorov training set
                (200 trajectories). Will be cycled indefinitely.
            val_loader: DataLoader over the Kolmogorov validation set.
                Used for periodic L2RE and VRMSE evaluation.
            start_step: Step to resume from. Pass 0 for fresh fine-tuning
                or the step loaded from a checkpoint for resumption.
        """
        finetune_cfg: Dict = self.config.get("finetune", {})
        total_steps: int = int(finetune_cfg.get("total_steps", 5000))

        # Set both models to training mode.
        self.fmt.train()
        self.p2vae.train()

        # Create cycling iterator for the small training set.
        # With 200 trajectories and batch_size=16, each epoch has ~12 batches.
        # 5000 steps / 12 batches/epoch ≈ 416 epochs of cycling.
        train_iter: Iterator[Dict[str, object]] = _cycling_loader(train_loader)

        # Skip steps already completed when resuming from checkpoint.
        if start_step > 0:
            logger.info(
                "Resuming fine-tuning from step %d. Advancing data iterator...",
                start_step,
            )
            for _ in range(start_step):
                next(train_iter)

        logger.info(
            "Starting fine-tuning: steps %d → %d, "
            "train_size=%d, lambda_vae=%.2f",
            start_step,
            total_steps,
            len(train_loader.dataset) if hasattr(train_loader, "dataset") else -1,
            self.lambda_vae,
        )

        # Running loss accumulators for periodic averaged logging.
        running_total: float = 0.0
        running_cfm: float = 0.0
        running_vae: float = 0.0
        running_recon: float = 0.0
        running_kl: float = 0.0
        log_count: int = 0

        for step in range(start_step, total_steps):
            # Set models to training mode at the start of each step.
            # (validate() sets them to eval mode; we restore here)
            self.fmt.train()
            self.p2vae.train()

            # Fetch next batch from the cycling iterator.
            batch: Dict[str, object] = next(train_iter)

            # Execute one training step.
            loss_dict: Dict[str, float] = self.train_step(batch)

            # Accumulate losses for averaged logging.
            if not (
                loss_dict["total_loss"] != loss_dict["total_loss"]
            ):  # NaN check
                running_total += loss_dict["total_loss"]
                running_cfm += loss_dict["cfm_loss"]
                running_vae += loss_dict["vae_loss"]
                running_recon += loss_dict["recon_loss"]
                running_kl += loss_dict["kl_loss"]
                log_count += 1

            # ------------------------------------------------------------------
            # Periodic logging (every log_every=100 steps).
            # ------------------------------------------------------------------
            if (step + 1) % self.log_every == 0:
                if log_count > 0:
                    avg_total: float = running_total / log_count
                    avg_cfm: float = running_cfm / log_count
                    avg_vae: float = running_vae / log_count
                    avg_recon: float = running_recon / log_count
                    avg_kl: float = running_kl / log_count
                else:
                    avg_total = avg_cfm = avg_vae = avg_recon = avg_kl = float("nan")

                current_lr: float = loss_dict["lr"]

                logger.info(
                    "Finetune step %5d/%d | total=%.4f | cfm=%.4f | "
                    "vae=%.4f | recon=%.4f | kl=%.6f | lr=%.2e",
                    step + 1,
                    total_steps,
                    avg_total,
                    avg_cfm,
                    avg_vae,
                    avg_recon,
                    avg_kl,
                    current_lr,
                )

                if self.use_wandb:
                    try:
                        import wandb  # type: ignore[import]

                        wandb.log(
                            {
                                "finetune/total_loss": avg_total,
                                "finetune/cfm_loss": avg_cfm,
                                "finetune/vae_loss": avg_vae,
                                "finetune/recon_loss": avg_recon,
                                "finetune/kl_loss": avg_kl,
                                "finetune/lr": current_lr,
                                "step": step + 1,
                            },
                            step=step + 1,
                        )
                    except Exception as exc:
                        logger.warning(
                            "WandB log failed at step %d: %s", step + 1, exc
                        )

                # Reset accumulators.
                running_total = 0.0
                running_cfm = 0.0
                running_vae = 0.0
                running_recon = 0.0
                running_kl = 0.0
                log_count = 0

            # ------------------------------------------------------------------
            # Periodic validation (every val_every=1000 steps).
            # ------------------------------------------------------------------
            if (step + 1) % self.val_every == 0:
                val_metrics: Dict[str, float] = self.validate(val_loader)

                logger.info(
                    "Finetune step %5d/%d | val_l2re=%.4f | val_vrmse=%.4f",
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
                                "finetune/val_l2re": val_metrics["val_l2re"],
                                "finetune/val_vrmse": val_metrics["val_vrmse"],
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
                    best_path: str = os.path.join(
                        self.save_dir, "finetune_best.pt"
                    )
                    self.save_checkpoint(best_path, step + 1)
                    logger.info(
                        "New best val_l2re=%.4f. Saved to %s",
                        self.best_val_l2re,
                        best_path,
                    )

            # ------------------------------------------------------------------
            # Periodic checkpoint saving (every save_every steps).
            # ------------------------------------------------------------------
            if (step + 1) % self.save_every == 0:
                step_ckpt_path: str = os.path.join(
                    self.save_dir, f"finetune_step{step + 1}.pt"
                )
                self.save_checkpoint(step_ckpt_path, step + 1)

                # Always keep a 'latest' checkpoint for easy resumption.
                latest_path: str = os.path.join(
                    self.save_dir, "finetune_latest.pt"
                )
                self.save_checkpoint(latest_path, step + 1)

                logger.info(
                    "Checkpoint saved at step %d: %s", step + 1, step_ckpt_path
                )

        # ------------------------------------------------------------------
        # Final checkpoint after fine-tuning completes.
        # ------------------------------------------------------------------
        final_path: str = os.path.join(self.save_dir, "finetune_final.pt")
        self.save_checkpoint(final_path, total_steps)
        logger.info(
            "Fine-tuning complete. Final checkpoint: %s (best val_l2re=%.4f)",
            final_path,
            self.best_val_l2re,
        )

        if self.use_wandb:
            try:
                import wandb  # type: ignore[import]

                wandb.finish()
            except Exception:
                pass

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Compute reconstruction L2RE and VRMSE on the validation set.

        Evaluates the fine-tuned model's reconstruction quality by encoding
        each frame to its posterior mean (deterministic latent, no sampling)
        and decoding back to pixel space. This tests whether the VAE has
        maintained reconstruction quality during joint fine-tuning.

        Also evaluates one-step prediction quality by running the FMT
        forward pass on clean latents (k=1, deterministic prediction).

        Args:
            val_loader: DataLoader over the Kolmogorov validation set
                (500 trajectories from config: finetune.test_trajectories=500).

        Returns:
            Dictionary with keys:
                'val_l2re': Mean L2 relative error over the validation set.
                'val_vrmse': Mean variance-normalized RMSE over the validation set.
        """
        # Switch both models to evaluation mode.
        self.fmt.eval()
        self.p2vae.eval()

        all_l2re: List[float] = []
        all_vrmse: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                frames: Tensor = batch["frames"]  # type: ignore[assignment]

                # Move to device.
                frames = frames.to(self.device, non_blocking=True)

                b: int = frames.shape[0]
                t_frames: int = frames.shape[1]  # 4 frames

                # Reshape (B, 4, 3, 128, 128) → (B*4, 3, 128, 128).
                x_flat: Tensor = frames.view(b * t_frames, 3, 128, 128)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    # Encode to posterior mean (deterministic latent).
                    # get_latent() returns mu without sampling.
                    z: Tensor = self.p2vae.get_latent(x_flat)  # (B*4, 16, 16, 16)

                    # Decode latent back to pixel space.
                    x_hat: Tensor = self.p2vae.decode(z)  # (B*4, 3, 128, 128)

                # Compute reconstruction metrics in float32.
                x_f32: Tensor = x_flat.float()
                x_hat_f32: Tensor = x_hat.float()

                batch_l2re: Tensor = self.metrics.l2_relative_error(
                    x