```python
## training/trainer.py
"""Core pretraining trainer for OLMoE-1B-7B.

Orchestrates the full pretraining loop: forward pass, auxiliary loss computation,
backward pass, gradient clipping, optimizer step, LR scheduling, checkpointing,
and in-loop evaluation.

Key design decisions (from paper and config.yaml):
  - BF16 mixed precision via torch.autocast (no GradScaler — BF16 doesn't need loss scaling)
  - FP32 gradient reduction handled by FSDP MixedPrecision config (Table 10)
  - Three-phase LR schedule: warmup → cosine → linear annealing (Appendix B)
  - Auxiliary losses (LB + router z-loss) computed from OLMoEOutput routing metadata
  - Dataset reshuffled at annealing phase start (Section 2, Appendix B)
  - All metrics all-reduced across ranks before logging (rank 0 logs only)
  - Checkpoints saved every 5,000 steps (config.yaml: pretraining.save_every_steps)

Configuration values used (from config.yaml):
  pretraining.total_tokens: 5_133_000_000_000
  pretraining.annealing_tokens: 100_000_000_000
  pretraining.batch_size_tokens: 4_194_304
  pretraining.batch_size_samples: 1024
  pretraining.seq_len: 4096
  pretraining.grad_clip: 1.0
  pretraining.log_every_steps: 1
  pretraining.eval_every_steps: 1000
  pretraining.save_every_steps: 5000
  pretraining.bf16: true
  pretraining.fp32_reduce: true
  model.lb_loss_weight: 0.01
  model.router_z_loss_weight: 0.001
"""

import itertools
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config import OLMoEConfig, TrainingConfig
from model.olmoe_model import OLMoEModel, OLMoEOutput
from training.losses import AuxiliaryLosses
from training.lr_scheduler import LRScheduler
from utils.checkpoint import CheckpointManager
from utils.distributed import DistributedUtils
from utils.logging_utils import WandbLogger, get_logger

logger: logging.Logger = get_logger("olmoe.trainer")


class Trainer:
    """Core pretraining trainer for OLMoE-1B-7B.

    Implements the full pretraining loop described in Section 2 and Appendix B
    of the paper. Handles BF16 mixed precision, FSDP distributed training,
    three-phase LR scheduling, auxiliary loss computation, and checkpointing.

    The trainer is designed to be instantiated once and then have train() called.
    It supports resuming from a checkpoint via load_checkpoint().

    Attributes:
        model: The OLMoEModel (possibly FSDP-wrapped).
        train_loader: DataLoader for pretraining data.
        config: TrainingConfig with all pretraining hyperparameters.
        aux_losses: AuxiliaryLosses for load balancing and router z-loss.
        scheduler: LRScheduler implementing warmup → cosine → annealing.
        optimizer: AdamW optimizer with all parameters and weight_decay=0.1.
        wandb_logger: WandbLogger for experiment tracking (rank 0 only).
        checkpoint_manager: CheckpointManager for saving/loading checkpoints.
        global_step: Current training step (0-indexed, incremented after each step).
        max_steps: Total training steps derived from total_tokens / batch_size_tokens.
        annealing_start_step: Step at which linear annealing begins.
        device: CUDA device for this process.
        world_size: Total number of processes in the distributed group.

    Example:
        >>> config = TrainingConfig()
        >>> model = OLMoEModel(OLMoEConfig())
        >>> model = DistributedUtils.setup_fsdp(model, config)
        >>> optimizer = create_pretrain_optimizer(model, config)
        >>> scheduler = LRScheduler(optimizer, config)
        >>> aux_losses = AuxiliaryLosses(OLMoEConfig())
        >>> wandb_logger = WandbLogger(project="olmoe", run_name="olmoe-1b-7b")
        >>> ckpt_manager = CheckpointManager(output_dir="outputs")
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     config=config,
        ...     aux_losses=aux_losses,
        ...     scheduler=scheduler,
        ...     optimizer=optimizer,
        ...     wandb_logger=wandb_logger,
        ...     checkpoint_manager=ckpt_manager,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: OLMoEModel,
        train_loader: DataLoader,
        config: TrainingConfig,
        aux_losses: AuxiliaryLosses,
        scheduler: LRScheduler,
        optimizer: Optimizer,
        wandb_logger: WandbLogger,
        checkpoint_manager: CheckpointManager,
        evaluator: Optional[Any] = None,
    ) -> None:
        """Initialize Trainer.

        Args:
            model: The OLMoEModel (or FSDP-wrapped OLMoEModel) to train.
                   Must already have weights initialized and be on the correct device.
            train_loader: DataLoader yielding batches of shape (batch_size, seq_len).
                          Each batch must contain "input_ids" and "labels" tensors.
                          Should use DistributedSampler for multi-GPU training.
            config: TrainingConfig instance with all pretraining hyperparameters.
                    Key fields: total_tokens, annealing_tokens, batch_size_tokens,
                    batch_size_samples, seq_len, grad_clip, log_every_steps,
                    eval_every_steps, save_every_steps, bf16.
            aux_losses: AuxiliaryLosses instance for computing load balancing loss
                        (weight=0.01) and router z-loss (weight=0.001) from routing
                        metadata in OLMoEOutput. Used only during pretraining.
            scheduler: LRScheduler implementing the three-phase schedule:
                       warmup (0→4e-4) → cosine (4e-4→4e-5) → annealing (4e-5→0).
                       The scheduler is stateless — call scheduler.step(global_step)
                       on every training step.
            optimizer: AdamW optimizer with all model parameters in a single group
                       with weight_decay=0.1. Created by create_pretrain_optimizer().
            wandb_logger: WandbLogger for experiment tracking. Only active on rank 0.
                          All other ranks have logger._enabled=False and all methods
                          are no-ops.
            checkpoint_manager: CheckpointManager for saving checkpoints every
                                 save_every_steps=5000 steps and loading for resume.
            evaluator: Optional Evaluator instance for in-loop downstream task
                       evaluation every eval_every_steps=1000 steps. If None,
                       evaluation is skipped. Default: None.
        """
        self.model: OLMoEModel = model
        self.train_loader: DataLoader = train_loader
        self.config: TrainingConfig = config
        self.aux_losses: AuxiliaryLosses = aux_losses
        self.scheduler: LRScheduler = scheduler
        self.optimizer: Optimizer = optimizer
        self.wandb_logger: WandbLogger = wandb_logger
        self.checkpoint_manager: CheckpointManager = checkpoint_manager
        self.evaluator: Optional[Any] = evaluator

        # -----------------------------------------------------------------------
        # Training step counters.
        # global_step is 0-indexed and incremented AFTER each completed step.
        # It is restored from checkpoint when load_checkpoint() is called.
        # -----------------------------------------------------------------------
        self.global_step: int = 0
        """Current global training step (0-indexed). Incremented after each step."""

        # -----------------------------------------------------------------------
        # Derived step counts from config.
        # These match the values computed in TrainingConfig.__post_init__:
        #   max_steps = total_tokens // batch_size_tokens ≈ 1,223,958
        #   annealing_steps = annealing_tokens // batch_size_tokens ≈ 23,842
        #   annealing_start_step = max_steps - annealing_steps ≈ 1,200,116
        # -----------------------------------------------------------------------
        self.max_steps: int = config.max_steps
        """Total training steps: total_tokens // batch_size_tokens ≈ 1,223,958."""

        self.annealing_steps: int = config.annealing_steps
        """Steps in the annealing phase: annealing_tokens // batch_size_tokens ≈ 23,842."""

        self.annealing_start_step: int = self.max_steps - self.annealing_steps
        """Step at which linear annealing begins ≈ 1,200,116 (paper's step 1,200,000)."""

        # -----------------------------------------------------------------------
        # Device and distributed setup.
        # -----------------------------------------------------------------------
        self.device: torch.device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        """CUDA device for this process."""

        self.world_size: int = DistributedUtils.get_world_size()
        """Total number of processes in the distributed group."""

        # -----------------------------------------------------------------------
        # BF16 autocast context.
        # For BF16 training (config.bf16=True), we use torch.autocast to run
        # the forward pass in BF16. No GradScaler is needed because:
        #   - BF16 has the same dynamic range as FP32 (~3.4e38)
        #   - Loss scaling is only needed for FP16 (range ~6.5e4)
        #   - FSDP's MixedPrecision(reduce_dtype=float32) handles FP32 gradient reduction
        # -----------------------------------------------------------------------
        self.use_bf16: bool = config.bf16
        """Whether to use BF16 autocast for the forward pass."""

        # -----------------------------------------------------------------------
        # Last metrics dict for checkpoint metadata.
        # Updated after each train_step() call.
        # -----------------------------------------------------------------------
        self._last_metrics: Dict[str, float] = {}
        """Most recent training metrics. Stored in checkpoint metadata."""

        # -----------------------------------------------------------------------
        # Annealing reshuffle flag.
        # Set to True once the dataset has been reshuffled at annealing start.
        # Prevents repeated reshuffling if the step boundary is hit multiple times.
        # -----------------------------------------------------------------------
        self._annealing_reshuffled: bool = False
        """Whether the dataset has been reshuffled for the annealing phase."""

        logger.info(
            f"Trainer initialized: "
            f"max_steps={self.max_steps:,}, "
            f"annealing_start_step={self.annealing_start_step:,}, "
            f"annealing_steps={self.annealing_steps:,}, "
            f"device={self.device}, "
            f"world_size={self.world_size}, "
            f"use_bf16={self.use_bf16}, "
            f"lb_loss_weight={aux_losses.lb_loss_weight}, "
            f"router_z_loss_weight={aux_losses.router_z_loss_weight}"
        )

    def train(self) -> None:
        """Run the full pretraining loop from global_step to max_steps.

        Implements the training loop described in Section 2 and Appendix B:
          1. Iterate over the DataLoader, cycling through epochs as needed
          2. Detect annealing phase start and reshuffle dataset
          3. Run train_step() for each batch
          4. Log metrics every log_every_steps=1 steps
          5. Save checkpoints every save_every_steps=5000 steps
          6. Run evaluation every eval_every_steps=1000 steps
          7. Save final checkpoint and run final evaluation

        Epoch handling:
          The dataset is ~4T tokens trained for 1.3 epochs = 5.133T tokens.
          When the DataLoader is exhausted (end of epoch), the iterator is
          re-created to continue training. The dataset's shuffle state is
          managed by the DataLoader's DistributedSampler.

        Annealing phase:
          At step annealing_start_step (~1,200,116), the dataset is reshuffled
          and the LR begins linear decay from min_lr=4e-5 to 0. The LRScheduler
          handles the LR transition automatically via scheduler.step(global_step).

        Returns:
            None. Training runs until global_step reaches max_steps.
        """
        logger.info(
            f"Starting pretraining: "
            f"global_step={self.global_step}, "
            f"max_steps={self.max_steps:,}, "
            f"rank={DistributedUtils.get_rank()}"
        )

        # Set model to training mode.
        self.model.train()

        # Create the initial data iterator.
        # We use a manual iterator rather than itertools.cycle to allow
        # reshuffling at epoch boundaries and annealing phase start.
        train_iter: Iterator = iter(self.train_loader)

        # -----------------------------------------------------------------------
        # Main training loop.
        # -----------------------------------------------------------------------
        while self.global_step < self.max_steps:

            # -------------------------------------------------------------------
            # Handle annealing phase start: reshuffle dataset.
            # Per paper Section 2 and Appendix B:
            # "During our annealing phase (final 100B tokens) we first reshuffle
            # the entire dataset and then linearly decay the learning rate to 0."
            # -------------------------------------------------------------------
            if (
                self.global_step == self.annealing_start_step
                and not self._annealing_reshuffled
            ):
                logger.info(
                    f"Entering annealing phase at step {self.global_step:,}. "
                    f"Reshuffling dataset and switching to linear LR decay."
                )
                train_iter = self._reshuffle_and_restart(train_iter)
                self._annealing_reshuffled = True

            # -------------------------------------------------------------------
            # Get next batch, handling epoch boundaries.
            # -------------------------------------------------------------------
            batch: Optional[Dict[str, Tensor]] = None
            try:
                batch = next(train_iter)
            except StopIteration:
                # Epoch complete — reshuffle and restart the iterator.
                logger.info(
                    f"Epoch complete at step {self.global_step:,}. "
                    f"Reshuffling and restarting data iterator."
                )
                train_iter = self._reshuffle_and_restart(None)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    logger.error(
                        "DataLoader is empty after reshuffling. "
                        "Cannot continue training. Check dataset configuration."
                    )
                    break

            if batch is None:
                logger.error(
                    f"Received None batch at step {self.global_step:,}. Skipping."
                )
                continue

            # -------------------------------------------------------------------
            # Move batch tensors to the current device.
            # -------------------------------------------------------------------
            batch = self._move_batch_to_device(batch)

            # -------------------------------------------------------------------
            # Time the training step for throughput measurement.
            # -------------------------------------------------------------------
            step_start_time: float = time.time()

            # -------------------------------------------------------------------
            # Core training step: forward + loss + backward + optimizer.
            # -------------------------------------------------------------------
            step_metrics: Dict[str, float] = self.train_step(batch)

            # -------------------------------------------------------------------
            # Compute throughput (tokens/sec/GPU).
            # -------------------------------------------------------------------
            step_elapsed: float = time.time() - step_start_time
            tokens_per_step: int = self.config.batch_size_samples * self.config.seq_len
            # Total throughput across all GPUs:
            total_throughput: float = tokens_per_step / max(step_elapsed, 1e-9)
            # Per-GPU throughput:
            per_gpu_throughput: float = total_throughput / max(self.world_size, 1)
            step_metrics["throughput_tokens_per_sec_per_gpu"] = per_gpu_throughput
            step_metrics["throughput_tokens_per_sec_total"] = total_throughput

            # -------------------------------------------------------------------
            # Add step and token count to metrics.
            # -------------------------------------------------------------------
            step_metrics["step"] = float(self.global_step)
            step_metrics["tokens_seen"] = float(
                self.global_step * self.config.batch_size_tokens
            )

            # -------------------------------------------------------------------
            # All-reduce metrics across ranks for accurate global averages.
            # This ensures logged values represent the full batch, not just
            # rank 0's shard.
            # -------------------------------------------------------------------
            reduced_metrics: Dict[str, float] = DistributedUtils.all_reduce_dict(
                step_metrics
            )

            # Store for checkpoint metadata.
            self._last_metrics = reduced_metrics

            # -------------------------------------------------------------------
            # Log metrics (every log_every_steps=1 steps, rank 0 only).
            # -------------------------------------------------------------------
            if self.global_step % self.config.log_every_steps == 0:
                self._log_metrics(reduced_metrics)

            # -------------------------------------------------------------------
            # Save checkpoint (every save_every_steps=5000 steps).
            # -------------------------------------------------------------------
            if (
                self.global_step > 0
                and self.global_step % self.config.save_every_steps == 0
            ):
                self.save_checkpoint(self.global_step)

            # -------------------------------------------------------------------
            # Run in-loop evaluation (every eval_every_steps=1000 steps).
            # -------------------------------------------------------------------
            if (
                self.global_step > 0
                and self.global_step % self.config.eval_every_steps == 0
                and self.evaluator is not None
            ):
                self._run_evaluation()

            # -------------------------------------------------------------------
            # Increment global step.
            # -------------------------------------------------------------------
            self.global_step += 1

        # -----------------------------------------------------------------------
        # Training complete: save final checkpoint and run final evaluation.
        # -----------------------------------------------------------------------
        logger.info(
            f"Training complete at step {self.global_step:,}. "
            f"Saving final checkpoint."
        )
        self.save_checkpoint(self.global_step)

        if self.evaluator is not None:
            self._run_evaluation()

        if DistributedUtils.is_main_process():
            self.wandb_logger.finish()

        logger.info("Pretraining finished.")

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Execute a single training step.

        Performs the forward pass, auxiliary loss computation, backward pass,
        gradient clipping, and optimizer step. Returns a metrics dict for logging.

        The total training loss is:
            L = L_CE + α·L_LB + β·L_RZ
        where α=0.01 (lb_loss_weight) and β=0.001 (router_z_loss_weight).

        Args:
            batch: Dict containing:
                   - "input_ids": (batch_size, seq_len) token IDs
                   - "labels": (batch_size, seq_len) target token IDs
                   - "attention_mask": (batch_size, seq_len) optional mask
                   All tensors must already be on the correct device.

        Returns:
            Dict with scalar metric values (Python floats, not tensors):
                - "total_loss": L = L_CE + α·L_LB + β·L_RZ
                - "ce_loss": cross-entropy component
                - "lb_loss": load balancing loss (unweighted, for monitoring)
                - "rz_loss": router z-loss (unweighted, for monitoring)
                - "grad_norm": gradient norm before clipping
                - "lr": current learning rate from scheduler
        """
        # -----------------------------------------------------------------------
        # Step 1: Forward pass with BF16 autocast.
        #
        # torch.autocast runs the forward pass in BF16 for efficiency.
        # The model's internal computations (attention, MoE dispatch, etc.)
        # run in BF16. The cross-entropy loss is computed in float32 by
        # PyTorch's F.cross_entropy (which upcasts internally for stability).
        #
        # No GradScaler is used because:
        #   - BF16 has the same dynamic range as FP32 (~3.4e38)
        #   - Loss scaling is only needed for FP16 (range ~6.5e4)
        #   - FSDP's MixedPrecision(reduce_dtype=float32) handles FP32 reduction
        # -----------------------------------------------------------------------
        output: OLMoEOutput

        autocast_device_type: str = "cuda" if torch.cuda.is_available() else "cpu"
        autocast_dtype: torch.dtype = (
            torch.bfloat16 if self.use_bf16 else torch.float32
        )

        with torch.autocast(
            device_type=autocast_device_type,
            dtype=autocast_dtype,
            enabled=self.use_bf16,
        ):
            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                labels=batch["labels"],
            )

        # -----------------------------------------------------------------------
        # Step 2: Compute total loss with auxiliary losses.
        #
        # AuxiliaryLosses.total_loss() computes:
        #   L = L_CE + 0.01 * mean(L_LB per layer) + 0.001 * mean(L_RZ per layer)
        #
        # The CE loss is already computed inside model.forward() when labels
        # are provided. We pass it along with the routing metadata.
        # -----------------------------------------------------------------------
        if output.ce_loss is None:
            raise RuntimeError(
                "model.forward() returned None ce_loss. "
                "Ensure 'labels' is provided in the batch for pretraining."
            )

        total_loss: Tensor
        mean_lb_loss: Tensor
        mean_rz_loss: Tensor

        total_loss, mean_lb_loss, mean_rz_loss = self.aux_losses.total_loss(
            ce_loss=output.ce_loss,
            all_router_logits=output.router_logits,
            all_top_k_indices=output.top_k_indices,
            use_lb_loss=True,       # Always True during pretraining
            use_router_z_loss=True, # Always True during pretraining
        )

        # -----------------------------------------------------------------------
        # Step 3: Backward pass, gradient clipping, and optimizer step.
        # -----------------------------------------------------------------------
        grad_norm: float = self._backward_and_step(total_loss)

        # -----------------------------------------------------------------------
        # Step 4: Update learning rate for the next step.
        #
        # The scheduler is called AFTER the optimizer step so that the LR
        # used for the current step's update is the one computed for global_step.
        # The next call to train_step will use the updated LR.
        # -----------------------------------------------------------------------
        self.scheduler.step(self.global_step)

        # -----------------------------------------------------------------------
        # Step 5: Build and return metrics dict.
        #
        # All values are Python floats (via .item()) for safe logging and
        # all-reduce operations in the calling train() method.
        # -----------------------------------------------------------------------
        current_lr: float = self.scheduler.get_lr(self.global_step)

        metrics: Dict[str, float] = {
            "total_loss": total_loss.item(),
            "ce_loss": output.ce_loss.item(),
            "lb_loss": mean_lb_loss.item(),
            "rz_loss": mean_rz_loss.item(),
            "grad_norm": grad_norm,
            "lr": current_lr,
        }

        return metrics

    def _backward_and_step(self, loss: Tensor) -> float:
        """Execute backward pass, gradient clipping, and optimizer step.

        Handles the backward pass for both FSDP-wrapped and plain nn.Module
        models. For FSDP models, uses model.clip_grad_norm_() which correctly
        handles sharded gradients across all ranks.

        No GradScaler is used because BF16 training does not require loss
        scaling (unlike FP16). The FSDP MixedPrecision configuration handles
        FP32 gradient reduction automatically.

        Args:
            loss: The total training loss scalar tensor (L = L_CE + α·L_LB + β·L_RZ).
                  Must be a scalar (0-dimensional) tensor with requires_grad=True.

        Returns:
            grad_norm: The global gradient norm before clipping (float).
                       Used for logging and monitoring training stability.
                       Returns 0.0 if gradient norm computation fails.
        """
        # -----------------------------------------------------------------------
        # Step 1: Zero gradients before backward pass.
        #
        # set_to_none=True is more memory efficient than zeroing:
        #   - Avoids allocating zero tensors for gradients
        #   - Allows PyTorch to skip gradient accumulation for frozen params
        #   - Slightly faster than zero-filling
        # -----------------------------------------------------------------------
        self.optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------------------------------------
        # Step 2: Backward pass.
        #
        # For BF16 with FSDP:
        #   - Gradients are computed in BF16 during backward
        #   - FSDP's MixedPrecision(reduce_dtype=float32) all-reduces in FP32
        #   - Optimizer states are maintained in FP32 (optimizer_state_dtype: fp32)
        #
        # No loss scaling needed (BF16 has same dynamic range as FP32).
        # -----------------------------------------------------------------------
        loss.backward()

        # -----------------------------------------------------------------------
        # Step 3: Gradient clipping.
        #
        # Clips the global gradient norm to config.grad_clip=1.0.
        # This prevents large gradient updates that could destabilize training,
        # especially important for MoE models where router gradients can spike.
        #
        # FSDP vs non-FSDP handling:
        #   - FSDP-wrapped: use model.clip_grad_norm_() which gathers sharded
        #     gradients across all ranks to compute the true global norm
        #   - Non-FSDP: use torch.nn.utils.clip_grad_norm_() on model.parameters()
        #
        # The returned grad_norm is the norm BEFORE clipping (for logging).
        # -----------------------------------------------------------------------
        grad_norm: float = 0.0

        try:
            # Check if model is FSDP-wrapped.
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel
                is_fsdp: bool = isinstance(self.model, FullyShardedDataParallel)
            except ImportError:
                is_fsdp = False

            if is_fsdp:
                # FSDP provides clip_grad_norm_() that handles sharded gradients.
                # It gathers gradient norms from all ranks, computes the global norm,
                # and clips all shards consistently.
                grad_norm_tensor: Tensor = self.model.clip_grad_norm_(
                    self.config.grad_clip
                )
                grad_norm = grad_norm_tensor.item()
            else:
                # Non-FSDP: standard gradient clipping.
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
                grad_norm = grad_norm_tensor.item()

        except Exception as exc:
            # Log warning but don't crash training — grad clipping failure
            # is recoverable (training continues with unclipped gradients).
            logger.warning(
                f"Gradient clipping failed at step {self.global_step}: "
                f"{type(exc).__name__}: {exc}. "
                f"Proceeding without gradient clipping for this step."
            )
            grad_norm = 0.0

        # -----------------------------------------------------------------------
        # Step 4: Optimizer step.
        #
        # Updates model parameters using the computed (and clipped) gradients.
        # For FSDP, the optimizer operates on the local parameter shards.
        # -----------------------------------------------------------------------
        self.optimizer.step()

        return grad_norm

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics to wandb and console (rank 0 only).

        Formats metrics with "train/" prefix for wandb namespacing and logs
        them at the current global_step. Also prints a summary line to the
        console for real-time monitoring.

        Args:
            metrics: Dict of metric name → float value. Expected keys:
                     "total_loss", "ce_loss", "lb_loss", "rz_loss",
                     "grad_norm", "lr", "throughput_tokens_per_sec_per_gpu",
                     "step", "tokens_seen".
                     Additional keys are logged as-is.
        """
        # Only rank 0 logs to prevent duplicate output.
        if not DistributedUtils.is_main_process():
            return

        # -----------------------------------------------------------------------
        # Build wandb metrics dict with "train/" prefix for namespacing.
        # This organizes metrics in the wandb UI under the "train" section.
        # -----------------------------------------------------------------------
        wandb_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            # Add "train/" prefix to training metrics.
            # "step" and "tokens_seen" are logged without prefix as global counters.
            if key in ("step", "tokens_seen"):
                wandb_metrics[key] = value
            else:
                wandb_metrics[f"train/{key}"] = value

        # Log to wandb.
        self.wandb_logger.log(wandb_metrics, step=self.global_step)

        # -----------------------------------------------------------------------
        # Console logging: print a summary line every log_every_steps.
        # Format: "Step 1000/1223958 | loss=2.3456 | ce=2.3400 | lb=0.1234 |
        #          rz=0.0123 | gnorm=0.89 | lr=3.99e-04 | tok/s=23600"
        # -----------------------------------------------------------------------
        total_loss: float = metrics.get("total_loss", 0.0)
        ce_loss: float = metrics.get("ce_loss", 0.0)
        lb_loss: float = metrics.get("lb_loss", 0.0)
        rz_loss: float = metrics.get("rz_loss", 0.0)
        grad_norm: float = metrics.get("grad_norm", 0.0)
        lr: float = metrics.get("lr", 0.0)
        throughput: float = metrics.get("throughput_tokens_per_sec_per_gpu", 0.0)
        tokens_seen: float = metrics.get("tokens_seen", 0.0)

        logger.info(
            f"Step {self.global_step:,}/{self.max_steps:,} | "
            f"loss={total_loss:.4f} | "
            f"ce={ce_loss:.4f} | "
            f"lb={lb_loss:.4f} | "
            f"rz={rz_loss:.5f} | "
            f"gnorm={grad_norm:.3f} | "
            f"lr={lr:.3e} | "
            f"tok/s/gpu={throughput:.0f} | "
            f"tokens={tokens_seen / 1e9:.2f}B"
        