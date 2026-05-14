## Code: adaptation/sft_trainer.py

```python
## adaptation/sft_trainer.py
"""Supervised Fine-Tuning (SFT) trainer for OLMoE-1B-7B instruction tuning.

Implements the instruction tuning stage described in Section 4.3 and Appendix B
of the OLMoE paper. Key design decisions:

  1. NO auxiliary losses during SFT (Section 4.3, Table 7):
     - use_lb_loss=False improves avg score from 52.8 to 54.0
     - Routing patterns are already established during pretraining (Section 5.1)
     - Table 6 confirms load balancing loss decreases slightly during SFT

  2. Token-level loss aggregation (Appendix B, Muennighoff et al. [122]):
     - Aggregate loss at token level: sum(response_token_losses) / count(response_tokens)
     - Improves performance on long generative tasks like AlpacaEval
     - Different from per-sample averaging used in standard cross-entropy

  3. Constant learning rate of 2e-5 for 2 epochs (Appendix B):
     - No warmup, no decay — model is already well-initialized from pretraining
     - Post-annealing checkpoint is the starting point (Section 4.3)

  4. BF16 mixed precision without GradScaler (Appendix B):
     - BF16 has wider dynamic range than FP16, no loss scaling needed
     - FSDP MixedPrecision handles FP32 gradient reduction

Configuration values used (from config.yaml sft section):
  sft.learning_rate: 2.0e-05
  sft.lr_schedule: "constant"
  sft.adam_beta1: 0.9
  sft.adam_beta2: 0.95
  sft.adam_eps: 1.0e-08
  sft.weight_decay: 0.1
  sft.num_epochs: 2
  sft.global_batch_size: 128
  sft.per_device_batch_size: 2
  sft.gradient_accumulation_steps: 2
  sft.max_seq_len: 4096
  sft.bf16: true
  sft.use_lb_loss: false
  sft.use_router_z_loss: false
  sft.loss_aggregation: "token_level"
  sft.use_post_annealing_checkpoint: true
  sft.grad_clip: 1.0
"""

import logging
import math
import os
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from config import OLMoEConfig, SFTConfig, TrainingConfig
from model.olmoe_model import OLMoEModel, OLMoEOutput
from utils.checkpoint import CheckpointManager
from utils.distributed import DistributedUtils
from utils.logging_utils import WandbLogger, get_logger

logger: logging.Logger = get_logger("olmoe.sft")

# ---------------------------------------------------------------------------
# Optional FSDP import for no_sync() context manager.
# ---------------------------------------------------------------------------
try:
    from torch.distributed.fsdp import FullyShardedDataParallel
    FSDP_AVAILABLE: bool = True
except ImportError:
    FSDP_AVAILABLE = False
    FullyShardedDataParallel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Optional AuxiliaryLosses import for ablation experiments (Table 7).
# Only imported when use_lb_loss=True to avoid unnecessary dependency.
# ---------------------------------------------------------------------------
_AuxiliaryLosses: Optional[Any] = None


def _get_auxiliary_losses_class() -> Any:
    """Lazily import AuxiliaryLosses for ablation experiments.

    This function is only called when use_lb_loss=True, which is the ablation
    path from Table 7. The default path (use_lb_loss=False) never imports this.

    Returns:
        The AuxiliaryLosses class from training/losses.py.

    Raises:
        ImportError: If training/losses.py cannot be imported.
    """
    global _AuxiliaryLosses
    if _AuxiliaryLosses is None:
        from training.losses import AuxiliaryLosses  # noqa: PLC0415
        _AuxiliaryLosses = AuxiliaryLosses
    return _AuxiliaryLosses


class SFTTrainer:
    """Supervised Fine-Tuning trainer for OLMoE-1B-7B instruction tuning.

    Implements the SFT stage from Section 4.3 and Appendix B of the paper.
    Trains the model on instruction-following data using token-level cross-entropy
    loss without auxiliary losses (load balancing or router z-loss).

    The trainer supports:
      - Token-level loss aggregation (sum/count over response tokens)
      - BF16 mixed precision via torch.autocast
      - Gradient accumulation for large effective batch sizes
      - FSDP-compatible gradient synchronization
      - Checkpoint saving at each epoch and final step
      - Wandb logging of training metrics

    Ablation support (Table 7):
      - use_lb_loss=True enables load balancing loss for comparison
      - Expected result: avg drops from 54.0 to 52.8 with LBL enabled

    Attributes:
        model: The OLMoEModel to fine-tune (possibly FSDP-wrapped).
        train_loader: DataLoader yielding SFT batches with input_ids, labels, attention_mask.
        config: SFTConfig with all SFT hyperparameters.
        use_lb_loss: Whether to use load balancing loss (False for paper's setup).
        optimizer: AdamW optimizer with all parameters and weight_decay=0.1.
        checkpoint_manager: CheckpointManager for saving SFT checkpoints.
        wandb_logger: WandbLogger for experiment tracking (rank 0 only).
        global_step: Current optimizer step count (incremented after each optimizer.step()).
        current_epoch: Current training epoch (0-indexed).
        total_steps: Total optimizer steps across all epochs.
        device: CUDA device for this process.
        world_size: Total number of processes in the distributed group.
        use_bf16: Whether to use BF16 autocast.
        gradient_accumulation_steps: Number of micro-steps per optimizer step.
        model_config: OLMoEConfig for auxiliary loss computation (ablation only).
        aux_losses: AuxiliaryLosses instance (only created when use_lb_loss=True).

    Example:
        >>> sft_config = SFTConfig()
        >>> model = OLMoEModel(OLMoEConfig())
        >>> # Load post-annealing checkpoint into model
        >>> trainer = SFTTrainer(
        ...     model=model,
        ...     train_loader=sft_dataloader,
        ...     config=sft_config,
        ...     use_lb_loss=False,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: OLMoEModel,
        train_loader: DataLoader,
        config: SFTConfig,
        use_lb_loss: bool = False,
        model_config: Optional[OLMoEConfig] = None,
        wandb_logger: Optional[WandbLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        """Initialize SFTTrainer.

        Args:
            model: The OLMoEModel (or FSDP-wrapped OLMoEModel) to fine-tune.
                   Must be loaded from the post-annealing pretraining checkpoint
                   (config.yaml: sft.use_post_annealing_checkpoint=true).
                   All parameters should have requires_grad=True.
            train_loader: DataLoader yielding SFT batches. Each batch must contain:
                          - "input_ids": (batch_size, seq_len) token IDs
                          - "labels": (batch_size, seq_len) with -100 for prompt tokens
                          - "attention_mask": (batch_size, seq_len) all-ones for packed seqs
                          Samples must be filtered to max_seq_len=4096 tokens.
            config: SFTConfig instance with all SFT hyperparameters.
                    Key fields: learning_rate=2e-5, num_epochs=2,
                    gradient_accumulation_steps=2, use_lb_loss=False.
            use_lb_loss: Whether to use load balancing loss during SFT.
                         Default: False (paper's recommended setting, Section 4.3).
                         Set to True only for the ablation experiment in Table 7.
                         WARNING: Setting to True is expected to HURT performance
                         (avg drops from 54.0 to 52.8 per Table 7).
            model_config: OLMoEConfig instance for auxiliary loss computation.
                          Required only when use_lb_loss=True. If None and
                          use_lb_loss=True, will attempt to read from model.config.
            wandb_logger: Optional WandbLogger for experiment tracking.
                          If None, a new logger is created on rank 0.
            checkpoint_manager: Optional CheckpointManager for saving checkpoints.
                                 If None, a new manager is created using config.output_dir.

        Raises:
            ValueError: If use_lb_loss=True but model_config cannot be determined.
            RuntimeError: If the model has no trainable parameters.
        """
        # -----------------------------------------------------------------------
        # Warn if load balancing loss is enabled — this is the ablation path.
        # The paper explicitly shows this hurts SFT performance (Table 7).
        # -----------------------------------------------------------------------
        if use_lb_loss:
            logger.warning(
                "use_lb_loss=True for SFT. "
                "Per paper Section 4.3 and Table 7, this is expected to HURT performance "
                "(avg drops from 54.0 to 52.8). "
                "This setting is only for reproducing the ablation experiment in Table 7. "
                "For the paper's recommended setup, use use_lb_loss=False."
            )

        # Also check config's use_lb_loss field for consistency.
        if config.use_lb_loss and not use_lb_loss:
            logger.warning(
                "config.use_lb_loss=True but use_lb_loss=False was passed to SFTTrainer. "
                "Using use_lb_loss=False (the constructor argument takes precedence). "
                "This matches the paper's recommended setup."
            )

        self.model: OLMoEModel = model
        """The OLMoEModel to fine-tune (possibly FSDP-wrapped)."""

        self.train_loader: DataLoader = train_loader
        """DataLoader yielding SFT batches."""

        self.config: SFTConfig = config
        """SFTConfig with all SFT hyperparameters."""

        self.use_lb_loss: bool = use_lb_loss
        """Whether to use load balancing loss (False for paper's setup)."""

        # -----------------------------------------------------------------------
        # Determine model config for auxiliary losses (ablation path only).
        # -----------------------------------------------------------------------
        self.model_config: Optional[OLMoEConfig] = None
        if use_lb_loss:
            if model_config is not None:
                self.model_config = model_config
            elif hasattr(model, "config") and isinstance(model.config, OLMoEConfig):
                self.model_config = model.config
            else:
                raise ValueError(
                    "use_lb_loss=True requires model_config to be provided or "
                    "model.config to be an OLMoEConfig instance. "
                    "Cannot compute load balancing loss without architecture config."
                )

        # -----------------------------------------------------------------------
        # Create auxiliary losses instance (ablation path only).
        # -----------------------------------------------------------------------
        self.aux_losses: Optional[Any] = None
        if use_lb_loss and self.model_config is not None:
            AuxiliaryLossesClass = _get_auxiliary_losses_class()
            self.aux_losses = AuxiliaryLossesClass(self.model_config)
            logger.info(
                f"AuxiliaryLosses created for SFT ablation: "
                f"lb_loss_weight={self.model_config.lb_loss_weight}"
            )

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
        # BF16 mixed precision configuration.
        # config.yaml: sft.bf16=true
        # BF16 does not need GradScaler (wider dynamic range than FP16).
        # -----------------------------------------------------------------------
        self.use_bf16: bool = config.bf16
        """Whether to use BF16 autocast for the forward pass."""

        # -----------------------------------------------------------------------
        # Gradient accumulation configuration.
        # config.yaml: sft.gradient_accumulation_steps=2
        # Effective global batch = per_device_batch * num_gpus * grad_accum_steps
        # = 2 * 32 * 2 = 128 (matches sft.global_batch_size=128)
        # -----------------------------------------------------------------------
        self.gradient_accumulation_steps: int = config.gradient_accumulation_steps
        """Number of micro-steps per optimizer step = 2 (config.yaml: sft.gradient_accumulation_steps)."""

        # -----------------------------------------------------------------------
        # Create optimizer: AdamW with ALL parameters and weight_decay=0.1.
        # Uses create_optimizer from training/optimizer.py which enforces
        # universal weight decay (no no-decay groups).
        # config.yaml: sft.learning_rate=2e-5, sft.adam_beta1=0.9,
        #              sft.adam_beta2=0.95, sft.adam_eps=1e-8, sft.weight_decay=0.1
        # -----------------------------------------------------------------------
        self.optimizer: Optimizer = self._create_optimizer()
        """AdamW optimizer with all parameters and weight_decay=0.1."""

        # -----------------------------------------------------------------------
        # Training step counters.
        # global_step: optimizer steps (incremented after each optimizer.step())
        # current_epoch: current epoch (0-indexed)
        # -----------------------------------------------------------------------
        self.global_step: int = 0
        """Current optimizer step count (0-indexed)."""

        self.current_epoch: int = 0
        """Current training epoch (0-indexed)."""

        # -----------------------------------------------------------------------
        # Compute total optimizer steps across all epochs.
        # total_steps = num_epochs * (num_batches // gradient_accumulation_steps)
        # -----------------------------------------------------------------------
        num_batches_per_epoch: int = len(train_loader)
        self.total_steps: int = (
            config.num_epochs
            * (num_batches_per_epoch // self.gradient_accumulation_steps)
        )
        """Total optimizer steps across all epochs."""

        # -----------------------------------------------------------------------
        # Checkpoint manager for saving SFT checkpoints.
        # -----------------------------------------------------------------------
        if checkpoint_manager is not None:
            self.checkpoint_manager: CheckpointManager = checkpoint_manager
        else:
            self.checkpoint_manager = CheckpointManager(
                output_dir=config.output_dir,
                max_checkpoints=3,  # Keep last 3 SFT checkpoints
            )
        """CheckpointManager for saving SFT checkpoints."""

        # -----------------------------------------------------------------------
        # Wandb logger for experiment tracking (rank 0 only).
        # -----------------------------------------------------------------------
        if wandb_logger is not None:
            self.wandb_logger: WandbLogger = wandb_logger
        else:
            self.wandb_logger = WandbLogger(
                project=config.wandb_project,
                run_name=config.run_name,
                config_dict=config.to_dict(),
            )
        """WandbLogger for experiment tracking (rank 0 only)."""

        # -----------------------------------------------------------------------
        # Last metrics dict for checkpoint metadata.
        # -----------------------------------------------------------------------
        self._last_metrics: Dict[str, float] = {}
        """Most recent training metrics. Stored in checkpoint metadata."""

        # -----------------------------------------------------------------------
        # Log initialization summary.
        # -----------------------------------------------------------------------
        logger.info(
            f"SFTTrainer initialized: "
            f"use_lb_loss={use_lb_loss}, "
            f"loss_aggregation='{config.loss_aggregation}', "
            f"learning_rate={config.learning_rate:.2e}, "
            f"num_epochs={config.num_epochs}, "
            f"gradient_accumulation_steps={self.gradient_accumulation_steps}, "
            f"total_steps={self.total_steps:,}, "
            f"global_batch_size={config.global_batch_size}, "
            f"max_seq_len={config.max_seq_len}, "
            f"use_bf16={self.use_bf16}, "
            f"device={self.device}, "
            f"world_size={self.world_size}"
        )

        # Verify effective global batch size matches config.
        effective_global_batch: int = (
            config.per_device_batch_size
            * self.world_size
            * self.gradient_accumulation_steps
        )
        if effective_global_batch != config.global_batch_size:
            logger.warning(
                f"Effective global batch size mismatch: "
                f"per_device_batch_size={config.per_device_batch_size} × "
                f"world_size={self.world_size} × "
                f"gradient_accumulation_steps={self.gradient_accumulation_steps} = "
                f"{effective_global_batch}, "
                f"but config.global_batch_size={config.global_batch_size}. "
                f"This may indicate a misconfiguration. "
                f"Expected: 2 × 32 × 2 = 128 for the paper's setup."
            )

    def _create_optimizer(self) -> Optimizer:
        """Create AdamW optimizer for SFT with universal weight decay.

        Uses the paper's SFT hyperparameters from config.yaml (sft section):
          - lr=2e-5 (constant throughout training)
          - betas=(0.9, 0.95)
          - eps=1e-8
          - weight_decay=0.1 applied to ALL parameters

        Collects all trainable parameters directly to avoid importing
        create_optimizer (which would create a circular dependency risk).
        Applies weight decay to ALL parameters including embeddings and
        RMSNorm weights, matching the paper's non-standard design decision.

        Returns:
            AdamW optimizer with all trainable parameters in a single group.

        Raises:
            RuntimeError: If the model has no trainable parameters.
        """
        # Collect all trainable parameters — no filtering, no grouping.
        # This enforces universal weight decay (Sections 4.2.3, 4.2.4).
        trainable_params: List[nn.Parameter] = [
            p for p in self.model.parameters() if p.requires_grad
        ]

        if len(trainable_params) == 0:
            raise RuntimeError(
                "No trainable parameters found in model for SFT. "
                "Ensure the model has parameters with requires_grad=True. "
                "Check that the model was not accidentally frozen before SFTTrainer init."
            )

        # Create AdamW with a single parameter group containing ALL trainable params.
        # CRITICAL: Do NOT split into decay/no-decay groups.
        # The paper explicitly applies weight_decay=0.1 to all parameters.
        optimizer: AdamW = AdamW(
            params=trainable_params,
            lr=self.config.learning_rate,          # 2e-5 (config.yaml: sft.learning_rate)
            betas=(self.config.adam_beta1, self.config.adam_beta2),  # (0.9, 0.95)
            eps=self.config.adam_eps,              # 1e-8 (config.yaml: sft.adam_eps)
            weight_decay=self.config.weight_decay, # 0.1 (config.yaml: sft.weight_decay)
            fused=False,  # Disable fused kernel for FSDP compatibility
        )

        # Verify all parameters are included.
        total_optimizer_params: int = sum(
            len(group["params"]) for group in optimizer.param_groups
        )
        assert total_optimizer_params == len(trainable_params), (
            f"Optimizer parameter count mismatch: "
            f"optimizer has {total_optimizer_params} parameters but "
            f"model has {len(trainable_params)} trainable parameters. "
            f"All parameters must be included for SFT."
        )

        # Verify weight decay is applied to all groups.
        for i, group in enumerate(optimizer.param_groups):
            group_wd: float = group.get("weight_decay", 0.0)
            if abs(group_wd - self.config.weight_decay) > 1e-9:
                logger.warning(
                    f"SFT optimizer param group {i} has weight_decay={group_wd}, "
                    f"expected {self.config.weight_decay}. "
                    f"The paper applies weight_decay=0.1 to ALL parameters."
                )

        # Count total parameter elements for logging.
        total_elements: int = sum(p.numel() for p in trainable_params)
        param_size_mb: float = total_elements * 4 / (1024 ** 2)  # float32 equivalent

        logger.info(
            f"SFT optimizer created: AdamW("
            f"lr={self.config.learning_rate:.2e}, "
            f"betas=({self.config.adam_beta1}, {self.config.adam_beta2}), "
            f"eps={self.config.adam_eps:.2e}, "
            f"weight_decay={self.config.weight_decay}), "
            f"num_param_tensors={len(trainable_params):,}, "
            f"total_elements={total_elements:,}, "
            f"param_size_fp32={param_size_mb:.1f}MB"
        )

        return optimizer

    def train(self) -> None:
        """Run the full SFT training loop for num_epochs=2 epochs.

        Implements the SFT training procedure from Appendix B:
          - 2 epochs with constant LR=2e-5
          - Token-level loss aggregation
          - No auxiliary losses
          - BF16 mixed precision
          - Gradient accumulation (2 steps)
          - Checkpoint saving at end of each epoch

        The training loop handles:
          1. Gradient accumulation with FSDP no_sync() for efficiency
          2. Gradient clipping at 1.0 global norm
          3. Metric logging every optimizer step
          4. Checkpoint saving at end of each epoch and final step
          5. Throughput measurement (tokens/second)

        Returns:
            None. Training runs for config.num_epochs=2 epochs.
        """
        logger.info(
            f"Starting SFT training: "
            f"num_epochs={self.config.num_epochs}, "
            f"total_steps={self.total_steps:,}, "
            f"rank={DistributedUtils.get_rank()}"
        )

        # Validate that the training loader is non-empty.
        if len(self.train_loader) == 0:
            raise ValueError(
                "SFT train_loader is empty. "
                "Ensure the SFT dataset is loaded correctly and has samples."
            )

        # -----------------------------------------------------------------------
        # Main training loop: iterate over epochs.
        # -----------------------------------------------------------------------
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            logger.info(
                f"Starting SFT epoch {epoch + 1}/{self.config.num_epochs} "
                f"(global_step={self.global_step:,}, "
                f"rank={DistributedUtils.get_rank()})"
            )

            # Set model to training mode at the start of each epoch.
            self.model.train()

            # Track epoch-level statistics.
            epoch_loss_sum: float = 0.0
            epoch_steps: int = 0
            epoch_start_time: float = time.time()

            # -----------------------------------------------------------------------
            # Inner loop: iterate over batches with gradient accumulation.
            # -----------------------------------------------------------------------
            micro_step: int = 0  # Counts micro-steps within the current optimizer step

            for batch_idx, batch in enumerate(self.train_loader):
                # Determine if this is an accumulation step (no optimizer update yet)
                # or a full step (optimizer update happens).
                is_last_micro_step: bool = (
                    (batch_idx + 1) % self.gradient_accumulation_steps == 0
                    or (batch_idx + 1) == len(self.train_loader)
                )

                # -------------------------------------------------------------------
                # Move batch to device.
                # -------------------------------------------------------------------
                batch = self._move_batch_to_device(batch)

                # -------------------------------------------------------------------
                # Validate batch sequence length.
                # -------------------------------------------------------------------
                if "input_ids" in batch:
                    seq_len: int = batch["input_ids"].shape[1]
                    if seq_len > self.config.max_seq_len:
                        logger.warning(
                            f"Batch {batch_idx} has seq_len={seq_len} > "
                            f"max_seq_len={self.config.max_seq_len}. "
                            f"This should have been filtered by the dataset loader. "
                            f"Truncating to max_seq_len."
                        )
                        batch["input_ids"] = batch["input_ids"][:, :self.config.max_seq_len]
                        if "labels" in batch:
                            batch["labels"] = batch["labels"][:, :self.config.max_seq_len]
                        if "attention_mask" in batch:
                            batch["attention_mask"] = batch["attention_mask"][:, :self.config.max_seq_len]

                # -------------------------------------------------------------------
                # Time the micro-step for throughput measurement.
                # -------------------------------------------------------------------
                step_start_time: float = time.time()

                # -------------------------------------------------------------------
                # Forward pass and loss computation.
                # Use FSDP no_sync() during accumulation steps to avoid redundant
                # gradient synchronization across ranks.
                # -------------------------------------------------------------------
                sync_context = self._get_sync_context(is_last_micro_step)

                with sync_context:
                    step_metrics: Dict[str, float] = self.train_step(batch)

                # -------------------------------------------------------------------
                # Optimizer step: only on the last micro-step of each accumulation.
                # -------------------------------------------------------------------
                if is_last_micro_step:
                    # Compute gradient norm before clipping (for logging).
                    grad_norm: float = self._clip_gradients()

                    # Optimizer step: update model parameters.
                    self.optimizer.step()

                    # Zero gradients for the next accumulation cycle.
                    # set_to_none=True is more memory efficient.
                    self.optimizer.zero_grad(set_to_none=True)

                    # -----------------------------------------------------------
                    # Compute throughput.
                    # -----------------------------------------------------------
                    step_elapsed: float = time.time() - step_start_time
                    # Tokens per step = batch_size * seq_len * gradient_accumulation_steps
                    # (accumulated over gradient_accumulation_steps micro-steps)
                    tokens_per_step: int = (
                        batch["input_ids"].shape[0]
                        * batch["input_ids"].shape[1]
                        * self.gradient_accumulation_steps
                    )
                    total_throughput: float = tokens_per_step / max(step_elapsed, 1e-9)
                    per_gpu_throughput: float = total_throughput / max(self.world_size, 1)

                    # -----------------------------------------------------------
                    # Build complete metrics dict for this optimizer step.
                    # -----------------------------------------------------------
                    full_metrics: Dict[str, float] = {
                        "loss": step_metrics.get("loss", 0.0),
                        "grad_norm": grad_norm,
                        "lr": self.config.learning_rate,  # Constant LR
                        "epoch": epoch + (batch_idx + 1) / len(self.train_loader),
                        "global_step": float(self.global_step),
                        "tokens_per_second_per_gpu": per_gpu_throughput,
                        "tokens_per_second_total": total_throughput,
                    }

                    # Add lb_loss to metrics if ablation mode is active.
                    if self.use_lb_loss and "lb_loss" in step_metrics:
                        full_metrics["lb_loss"] = step_metrics["lb_loss"]

                    # -----------------------------------------------------------
                    # All-reduce metrics across ranks for accurate global averages.
                    # -----------------------------------------------------------
                    reduced_metrics: Dict[str, float] = DistributedUtils.all_reduce_dict(
                        full_metrics
                    )

                    # Store for checkpoint metadata.
                    self._last_metrics = reduced_metrics

                    # -----------------------------------------------------------
                    # Log metrics (every optimizer step, rank 0 only).
                    # -----------------------------------------------------------
                    self._log_metrics(reduced_metrics)

                    # Accumulate epoch statistics.
                    epoch_loss_sum += reduced_metrics.get("loss", 0.0)
                    epoch_steps += 1

                    # Increment global step counter.
                    self.global_step += 1

                micro_step += 1

            # -----------------------------------------------------------------------
            # End of epoch: log epoch summary and save checkpoint.
            # -----------------------------------------------------------------------
            epoch_elapsed: float = time.time() - epoch_start_time
            epoch_avg_loss: float = epoch_loss_sum / max(epoch_steps, 1)

            logger.info(
                f"SFT epoch {epoch + 1}/{self.config.num_epochs} complete: "
                f"avg_loss={epoch_avg_loss:.4f}, "
                f"steps={epoch_steps:,}, "
                f"elapsed={epoch_elapsed:.1f}s, "
                f"global_step={self.global_step:,}"
            )

            # Save checkpoint at the end of each epoch.
            self.save_checkpoint(
                step=self.global_step,
                metrics={
                    "epoch": epoch + 1,
                    "avg_loss": epoch_avg_loss,
                    **self._last_metrics,
                },
            )

        # -----------------------------------------------------------------------
        # Training complete: save final checkpoint.
        # -----------------------------------------------------------------------
        logger.info(
            f"SFT training complete: "
            f"total_steps={self.global_step:,}, "
            f"num_epochs={self.config.num_epochs}"
        )

        # Save final checkpoint (may be same as last epoch checkpoint).
        self.save_checkpoint(
            step=self.global_step,
            metrics={"final": True, **self._last_metrics},
        )

        # Finish wandb run.
        if DistributedUtils.is_main_process():
            self.wandb_logger.finish()

        logger.info("SFT training finished.")

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Execute a single SFT micro-step (forward pass + loss computation + backward).

        Computes token-level cross-entropy loss on response tokens only (prompt
        tokens are masked with -100 in labels). Does NOT apply auxiliary losses
        (load balancing or router