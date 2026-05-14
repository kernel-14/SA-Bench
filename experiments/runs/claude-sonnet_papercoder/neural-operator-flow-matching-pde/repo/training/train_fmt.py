## Code: training/train_fmt.py

```python
## training/train_fmt.py
"""Stage 2 training loop for FMT (Flow Marching Transformer).

Trains the Flow Marching Transformer on frozen P2VAE latents using the
conditional flow marching (CFM) objective with diffusion forcing and
latent temporal pyramids.

Training configuration (from config.yaml, fmt.training):
    - AdamW optimizer: beta1=0.9, beta2=0.95, weight_decay=0.01
    - Cosine LR schedule with 10% linear warmup over 100k steps
    - Base LR = 1e-4 at batch_size=256 (linearly scaled for other sizes)
    - P2VAE weights are frozen (requires_grad=False, eval mode)
    - AMP float16 mixed precision on H-100 GPUs
    - DDP across 4 GPUs (config: fmt.hardware.num_gpus = 4)

The training objective (paper Section 3.2, Eq. 11):
    L_CFM = 0.5 * E[ Σ_{s=0}^{3} ||(1-t_s)*g_θ(x_{s,t_s}^{k_s}, t_s, h_{s-1})
                     - (x_{s+1} - x_{s,t_s}^{k_s})||^2 ]

Key design decisions:
    - 5 consecutive frames per training sample: (x_0, x_1, x_2, x_3, x_4)
      where (x_0..x_3) are inputs and (x_1..x_4) are targets
    - k-free objective: network receives (x_t^k, t) only, not k
    - GRU hidden state propagates sequentially through 4 timesteps
    - Temporal pyramid: 340 tokens (4+16+64+256) for 15× efficiency gain
"""

import logging
import os
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from evaluation.metrics import Metrics
from models.fmt import FMT, FlowMarchingKernel
from models.p2vae import P2VAE
from utils.distributed import is_main_process, reduce_tensor
from utils.lr_scheduler import CosineWarmupScheduler, get_lr_scale

logger = logging.getLogger(__name__)


def _infinite_loader(loader: DataLoader) -> Iterator[Dict[str, object]]:
    """Yield batches from a DataLoader indefinitely.

    Wraps the DataLoader in an infinite loop so the fixed-step training
    loop (100k steps) does not need to track epochs explicitly. When the
    DataLoader is exhausted, it restarts from the beginning.

    For DDP training with DistributedSampler, calls sampler.set_epoch(epoch)
    to ensure proper shuffling across epochs.

    Args:
        loader: PyTorch DataLoader to iterate over indefinitely.

    Yields:
        Batches from the DataLoader, cycling indefinitely.
    """
    epoch: int = 0
    while True:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)  # type: ignore[union-attr]
        for batch in loader:
            yield batch
        epoch += 1


class FMTTrainer:
    """Trainer for Stage 2: Flow Marching Transformer training on frozen P2VAE latents.

    Handles DDP multi-GPU training, AMP mixed precision, cosine LR scheduling
    with linear warmup, gradient clipping, periodic validation, and checkpoint
    saving/loading.

    The trainer processes trajectory batches of shape (B, 5, 3, 128, 128):
    5 consecutive frames are needed because the CFM objective predicts
    y_{s+1} from y_{s,t_s}^{k_s} for s=0,1,2,3, requiring frames 0..4.

    Attributes:
        rank: DDP rank of this process (0 to world_size - 1).
        world_size: Total number of DDP processes.
        config: Full configuration dictionary loaded from config.yaml.
        device: CUDA device assigned to this rank.
        fmt: Unwrapped FMT model (used for checkpoint saving and direct calls).
        p2vae: Frozen P2VAE model (eval mode, no gradients).
        fmt_ddp: DDP-wrapped FMT (or same as fmt when world_size=1).
        optimizer: AdamW optimizer with paper-specified hyperparameters.
        scheduler: CosineWarmupScheduler for LR decay.
        scaler: GradScaler for AMP float16 training.
        kernel: FlowMarchingKernel for interpolation and loss computation.
        metrics: Metrics instance for L2RE and VRMSE computation.
        grad_clip: Maximum gradient norm for clipping.
        log_every: Log training metrics every this many steps.
        val_every: Run validation every this many steps.
        save_every: Save checkpoint every this many steps.
        save_dir: Directory for checkpoint files.
        use_wandb: Whether to log to Weights & Biases.
        best_val_l2re: Best validation L2RE seen so far (for save_best).
        current_step: Current training step counter.
    """

    def __init__(
        self,
        fmt: FMT,
        p2vae: P2VAE,
        config: Dict,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Initialize FMTTrainer.

        Sets up device assignment, P2VAE freezing, DDP wrapping of FMT,
        AdamW optimizer with paper-specified hyperparameters, cosine LR
        scheduler, AMP GradScaler, and optional WandB logging.

        Args:
            fmt: Instantiated FMT model (not yet moved to device).
            p2vae: Pretrained P2VAE model (not yet moved to device).
                Will be frozen (requires_grad=False) and set to eval mode.
            config: Full configuration dictionary from config.yaml. Must
                contain 'fmt', 'logging', and 'checkpointing' top-level keys.
            rank: DDP rank of this process. 0 for single-GPU or main process.
                From config: fmt.hardware.num_gpus = 4 → ranks 0-3.
            world_size: Total number of DDP processes. 1 for single-GPU.
                From config: fmt.hardware.num_gpus = 4.
        """
        self.rank: int = rank
        self.world_size: int = world_size
        self.config: Dict = config
        self.current_step: int = 0

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

        # ------------------------------------------------------------------
        # Move both models to device before any wrapping or freezing.
        # ------------------------------------------------------------------
        fmt = fmt.to(self.device)
        p2vae = p2vae.to(self.device)

        # ------------------------------------------------------------------
        # Freeze P2VAE: disable all gradients and set to eval mode.
        # From config.yaml: fmt.training.freeze_p2vae = true.
        # The paper explicitly states P2VAE weights are frozen during FMT
        # training (Section 4.1): "Based on the 16M P2VAE (with frozen
        # weights), we train 3 FMTs..."
        # ------------------------------------------------------------------
        for param in p2vae.parameters():
            param.requires_grad = False
        p2vae.eval()
        self.p2vae: P2VAE = p2vae

        # ------------------------------------------------------------------
        # Store unwrapped FMT reference for checkpoint saving and direct
        # method calls (e.g., gru_forcing.init_hidden, temporal_pyramid).
        # ------------------------------------------------------------------
        self.fmt: FMT = fmt

        # ------------------------------------------------------------------
        # DDP wrapping of FMT (only when world_size > 1).
        # find_unused_parameters=False: all FMT parameters participate in
        # every forward pass (no conditional computation paths).
        # ------------------------------------------------------------------
        if world_size > 1:
            self.fmt_ddp: nn.Module = nn.parallel.DistributedDataParallel(
                fmt,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
            )
        else:
            self.fmt_ddp = fmt

        # ------------------------------------------------------------------
        # FlowMarchingKernel: stateless utility for interpolation and loss.
        # Instantiated once and reused across all training steps.
        # ------------------------------------------------------------------
        self.kernel: FlowMarchingKernel = FlowMarchingKernel()

        # ------------------------------------------------------------------
        # Optimizer: AdamW with paper-specified hyperparameters.
        # From config.yaml: fmt.training.beta1=0.9, beta2=0.95,
        # weight_decay=0.01, base_lr=1e-4 at base_batch_size=256.
        # Note: beta2=0.95 differs from P2VAE's 0.995 (paper Section 4.1).
        # Only optimize FMT parameters — P2VAE is frozen.
        # ------------------------------------------------------------------
        training_cfg: Dict = config["fmt"]["training"]
        base_lr: float = float(training_cfg["base_lr"])  # 1e-4
        base_batch_size: int = int(training_cfg.get("base_batch_size", 256))
        beta1: float = float(training_cfg["beta1"])  # 0.9
        beta2: float = float(training_cfg["beta2"])  # 0.95
        weight_decay: float = float(training_cfg["weight_decay"])  # 0.01

        # Linear LR scaling: LR ∝ batch_size / base_batch_size.
        # From config: base_lr=1e-4 calibrated to base_batch_size=256.
        actual_batch_size: int = int(training_cfg.get("batch_size", base_batch_size))
        scaled_lr: float = get_lr_scale(base_lr, actual_batch_size, base_batch_size)

        logger.info(
            "FMTTrainer: base_lr=%.2e, actual_batch_size=%d, "
            "scaled_lr=%.2e (linear scaling)",
            base_lr,
            actual_batch_size,
            scaled_lr,
        )

        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.fmt.parameters(),
            lr=scaled_lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=1e-8,
        )

        # ------------------------------------------------------------------
        # LR scheduler: cosine decay with 10% linear warmup.
        # From config.yaml: fmt.training.total_steps=100000,
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
        # From config.yaml: fmt.training.gradient_clip = 1.0 [ESTIMATED].
        # ------------------------------------------------------------------
        self.grad_clip: float = float(training_cfg.get("gradient_clip", 1.0))

        # ------------------------------------------------------------------
        # Logging and checkpointing cadence from config.yaml.
        # ------------------------------------------------------------------
        logging_cfg: Dict = config.get("logging", {})
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
                    name="fmt_training",
                    config=config,
                    resume="allow",
                )
                logger.info("WandB initialized for FMT training.")
            except ImportError:
                logger.warning(
                    "wandb not installed. Disabling WandB logging."
                )
                self.use_wandb = False
            except Exception as exc:
                logger.warning("WandB initialization failed: %s. Disabling.", exc)
                self.use_wandb = False

        logger.info(
            "FMTTrainer initialized: rank=%d, world_size=%d, device=%s, "
            "total_steps=%d, scaled_lr=%.2e",
            rank,
            world_size,
            self.device,
            total_steps,
            scaled_lr,
        )

    @property
    def raw_fmt(self) -> FMT:
        """Return the unwrapped FMT model (without DDP wrapper).

        Used for checkpoint saving (saves raw state_dict, not DDP state_dict)
        and for direct method calls during validation.

        Returns:
            The underlying FMT instance, regardless of DDP wrapping.
        """
        if self.world_size > 1:
            return self.fmt_ddp.module  # type: ignore[return-value]
        return self.fmt

    def train_step(self, batch: Dict[str, object]) -> Dict[str, float]:
        """Execute one forward + backward pass on a single batch.

        Encodes 5 consecutive frames through frozen P2VAE, constructs noisy
        interpolated latents for each of the 4 physical timesteps using the
        flow marching kernel, runs the FMT forward pass with temporal pyramid
        and GRU diffusion forcing, and computes the preconditioned CFM loss.

        The training objective (paper Eq. 11):
            L_CFM = 0.5 * E[ Σ_{s=0}^{3} ||(1-t_s)*g_θ(x_{s,t_s}^{k_s}, t_s, h_{s-1})
                             - (x_{s+1} - x_{s,t_s}^{k_s})||^2 ]

        Args:
            batch: Dictionary from PDEUnifiedDataset.__getitem__ with keys:
                'frames': Tensor of shape (B, 5, 3, 128, 128), dtype float32.
                    5 consecutive frames: (x_0, x_1, x_2, x_3, x_4).
                    Input window: (x_0..x_3), target window: (x_1..x_4).
                'dataset_name': List of str (not used in training step).

        Returns:
            Dictionary with scalar loss values and current LR:
                'total_loss': Mean CFM loss over all 4 timesteps.
                'loss_s0': CFM loss for timestep s=0 (predicting x_1 from x_0).
                'loss_s1': CFM loss for timestep s=1 (predicting x_2 from x_1).
                'loss_s2': CFM loss for timestep s=2 (predicting x_3 from x_2).
                'loss_s3': CFM loss for timestep s=3 (predicting x_4 from x_3).
                'lr': Current learning rate from scheduler.
        """
        # Extract frames tensor from batch dict.
        frames: Tensor = batch["frames"]  # type: ignore[assignment]

        # Move to device with non-blocking transfer for overlap with compute.
        frames = frames.to(self.device, non_blocking=True)

        # Validate that we have 5 frames (required for 4-step CFM objective).
        b: int = frames.shape[0]
        n_frames: int = frames.shape[1]
        if n_frames < 5:
            raise ValueError(
                f"FMT training requires 5 consecutive frames per sample "
                f"(seq_len=5), but got {n_frames}. "
                "Ensure PDEUnifiedDataset is initialized with seq_len=5."
            )

        # ------------------------------------------------------------------
        # Step 1: Encode all 5 frames through frozen P2VAE.
        # Use torch.no_grad() to prevent gradients flowing through P2VAE.
        # get_latent() returns mu (posterior mean, no sampling) — this is
        # the deterministic latent used for FMT training (shared knowledge #3).
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Reshape (B, 5, 3, 128, 128) → (B*5, 3, 128, 128) for batch encoding.
            x_flat: Tensor = frames.view(b * n_frames, 3, 128, 128)

            # Encode: (B*5, 3, 128, 128) → (B*5, 16, 16, 16)
            y_flat: Tensor = self.p2vae.get_latent(x_flat)

            # Reshape back: (B*5, 16, 16, 16) → (B, 5, 16, 16, 16)
            y: Tensor = y_flat.view(b, n_frames, 16, 16, 16)

        # Extract individual latent frames.
        # Input window: y0..y3 (frames 0-3)
        # Target window: y1..y4 (frames 1-4)
        y0: Tensor = y[:, 0]  # (B, 16, 16, 16)
        y1: Tensor = y[:, 1]  # (B, 16, 16, 16)
        y2: Tensor = y[:, 2]  # (B, 16, 16, 16)
        y3: Tensor = y[:, 3]  # (B, 16, 16, 16)
        y4: Tensor = y[:, 4]  # (B, 16, 16, 16)

        # ------------------------------------------------------------------
        # Step 2: Sample independent t_s and k_s for each physical timestep.
        # From paper Section 3.2: "t_s, k_s are independently sampled at
        # each physical timestep s."
        # Shape (B, 1, 1, 1) enables broadcasting over (B, C, H, W) latents.
        # ------------------------------------------------------------------
        t0: Tensor = torch.rand(b, 1, 1, 1, device=self.device)
        k0: Tensor = torch.rand(b, 1, 1, 1, device=self.device)

        t1: Tensor = torch.rand(b, 1, 1, 1, device=self.device)
        k1: Tensor = torch.rand(b, 1, 1, 1, device=self.device)

        t2: Tensor = torch.rand(b, 1, 1, 1, device=self.device)
        k2: Tensor = torch.rand(b, 1, 1, 1, device=self.device)

        t3: Tensor = torch.rand(b, 1, 1, 1, device=self.device)
        k3: Tensor = torch.rand(b, 1, 1, 1, device=self.device)

        # ------------------------------------------------------------------
        # Step 3: Construct noisy interpolated latents x_{s,t_s}^{k_s}.
        # Formula (paper Eq. 1-3):
        #   x_t^k = t*x1 + k*(1-t)*x0 + (1-t)*(1-k)*z,  z ~ N(0,I)
        # k is used ONLY here to build the noisy state — NOT passed to network
        # (k-free objective, paper Section 3.1 and shared knowledge #7).
        # ------------------------------------------------------------------
        y0_tk: Tensor = self.kernel.sample_interpolation(
            x0=y0, x1=y1, t=t0, k=k0
        )  # (B, 16, 16, 16)
        y1_tk: Tensor = self.kernel.sample_interpolation(
            x0=y1, x1=y2, t=t1, k=k1
        )  # (B, 16, 16, 16)
        y2_tk: Tensor = self.kernel.sample_interpolation(
            x0=y2, x1=y3, t=t2, k=k2
        )  # (B, 16, 16, 16)
        y3_tk: Tensor = self.kernel.sample_interpolation(
            x0=y3, x1=y4, t=t3, k=k3
        )  # (B, 16, 16, 16)

        # ------------------------------------------------------------------
        # Step 4: Initialize GRU hidden state to zeros.
        # h_0 = zeros represents no prior history at the start of each
        # training sequence. The GRU updates h sequentially through s=0..3.
        # ------------------------------------------------------------------
        h_init: Tensor = self.raw_fmt.gru_forcing.init_hidden(
            batch_size=b, device=self.device
        )  # (B, embed_dim)

        # ------------------------------------------------------------------
        # Step 5: Forward pass through FMT with AMP mixed precision.
        # FMT.forward internally:
        #   1. Builds temporal pyramid from 4 noisy latents
        #   2. Embeds tokens via PatchEmbed + positional embeddings
        #   3. Runs SiT Transformer blocks conditioned on t + h
        #   4. Updates GRU hidden state sequentially
        #   5. Returns predicted velocities for all 4 frames + final h
        # ------------------------------------------------------------------
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # FMT forward pass: predict velocities for all 4 timesteps.
            # ts[3] is the primary conditioning timestep (newest frame).
            # The GRU processes frames 0→1→2→3 sequentially.
            velocities: List[Tensor]
            h_final: Tensor
            velocities, h_final = self.fmt_ddp(
                latents=[y0_tk, y1_tk, y2_tk, y3_tk],
                ts=[t0.squeeze(-1).squeeze(-1).squeeze(-1),
                    t1.squeeze(-1).squeeze(-1).squeeze(-1),
                    t2.squeeze(-1).squeeze(-1).squeeze(-1),
                    t3.squeeze(-1).squeeze(-1).squeeze(-1)],
                h_prev=h_init,
            )
            # velocities: list of 4 tensors, each (B, 16, 16, 16)
            # h_final: (B, embed_dim) — updated GRU state after all 4 frames

            # ------------------------------------------------------------------
            # Step 6: Compute preconditioned CFM loss for each timestep.
            # Loss (paper Eq. 9):
            #   L_FM = 0.5 * E[ ||(1-t)*g_θ(x_t^k, t) - (x_1 - x_t^k)||^2 ]
            # The (1-t) preconditioning prevents stiffness near t→1.
            # ------------------------------------------------------------------
            # Targets: y_{s+1} for each s
            targets: List[Tensor] = [y1, y2, y3, y4]
            noisy_latents: List[Tensor] = [y0_tk, y1_tk, y2_tk, y3_tk]
            ts_list: List[Tensor] = [t0, t1, t2, t3]

            step_losses: List[Tensor] = []
            for s in range(4):
                loss_s: Tensor = self.kernel.compute_loss(
                    g=velocities[s],
                    x_tk=noisy_latents[s],
                    x1=targets[s],
                    t=ts_list[s],
                )
                step_losses.append(loss_s)

            # Total loss: mean over all 4 physical timesteps.
            total_loss: Tensor = torch.stack(step_losses).mean()

        # ------------------------------------------------------------------
        # Step 7: Backward pass with AMP gradient scaling and clipping.
        # ------------------------------------------------------------------
        self.scaler.scale(total_loss).backward()

        # Unscale gradients before clipping (required by GradScaler API).
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping: prevents exploding gradients during early training.
        # From config: fmt.training.gradient_clip = 1.0 [ESTIMATED].
        torch.nn.utils.clip_grad_norm_(
            self.fmt.parameters(),
            max_norm=self.grad_clip,
        )

        # Optimizer step (skipped if gradients contain inf/nan after unscaling).
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # LR scheduler step: called every training step (not every epoch).
        self.scheduler.step()

        # ------------------------------------------------------------------
        # Step 8: Return scalar losses for logging.
        # Per-step losses help diagnose whether the model learns earlier vs.
        # later frames differently (e.g., s=0 is harder due to more noise).
        # ------------------------------------------------------------------
        return {
            "total_loss": total_loss.item(),
            "loss_s0": step_losses[0].item(),
            "loss_s1": step_losses[1].item(),
            "loss_s2": step_losses[2].item(),
            "loss_s3": step_losses[3].item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_step: int = 0,
    ) -> None:
        """Run the full FMT training loop for 100k steps.

        Iterates over the training DataLoader indefinitely (cycling through
        epochs as needed), logging metrics every 100 steps, validating every
        1000 steps, and saving checkpoints every 10000 steps.

        The P2VAE is kept in eval mode throughout training. The FMT is set
        to train mode for training steps and eval mode for validation.

        Args:
            train_loader: DataLoader over PDEUnifiedDataset (train split,
                seq_len=5). Should use DistributedSampler when world_size > 1.
            val_loader: DataLoader over PDEUnifiedDataset (val split, seq_len=5).
                Used for periodic L2RE and VRMSE evaluation.
            start_step: Step to resume from. Pass 0 for fresh training or
                the step loaded from a checkpoint for resumption.
        """
        total_steps: int = int(
            self.config["fmt"]["training"]["total_steps"]
        )  # 100000

        self.current_step = start_step

        # Ensure P2VAE stays in eval mode throughout training.
        self.p2vae.eval()

        # Create infinite data iterator that handles epoch-based shuffling.
        data_iter: Iterator[Dict[str, object]] = _infinite_loader(train_loader)

        # Skip steps already completed when resuming from checkpoint.
        if start_step > 0:
            logger.info(
                "Resuming FMT training from step %d. Advancing data iterator...",
                start_step,
            )
            for _ in range(start_step):
                next(data_iter)

        logger.info(
            "Starting FMT training: steps %d → %d",
            start_step,
            total_steps,
        )

        # Running loss accumulators for periodic averaged logging.
        running_total: float = 0.0
        running_s0: float = 0.0
        running_s1: float = 0.0
        running_s2: float = 0.0
        running_s3: float = 0.0
        log_count: int = 0

        while self.current_step < total_steps:
            # Set FMT to training mode for each step.
            self.fmt.train()
            # P2VAE must always remain in eval mode (frozen).
            self.p2vae.eval()

            # Fetch next batch from the infinite iterator.
            batch: Dict[str, object] = next(data_iter)

            # Execute one training step.
            loss_dict: Dict[str, float] = self.train_step(batch)
            self.current_step += 1

            # Accumulate losses for averaged logging.
            running_total += loss_dict["total_loss"]
            running_s0 += loss_dict["loss_s0"]
            running_s1 += loss_dict["loss_s1"]
            running_s2 += loss_dict["loss_s2"]
            running_s3 += loss_dict["loss_s3"]
            log_count += 1

            # ------------------------------------------------------------------
            # Periodic logging (rank 0 only, every log_every=100 steps).
            # ------------------------------------------------------------------
            if self.current_step % self.log_every == 0 and is_main_process():
                avg_total: float = running_total / log_count
                avg_s0: float = running_s0 / log_count
                avg_s1: float = running_s1 / log_count
                avg_s2: float = running_s2 / log_count
                avg_s3: float = running_s3 / log_count
                current_lr: float = loss_dict["lr"]

                logger.info(
                    "Step %6d/%d | total=%.4f | s0=%.4f | s1=%.4f | "
                    "s2=%.4f | s3=%.4f | lr=%.2e",
                    self.current_step,
                    total_steps,
                    avg_total,
                    avg_s0,
                    avg_s1,
                    avg_s2,
                    avg_s3,
                    current_lr,
                )

                if self.use_wandb:
                    try:
                        import wandb  # type: ignore[import