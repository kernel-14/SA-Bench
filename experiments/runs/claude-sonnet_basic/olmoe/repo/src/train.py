"""
OLMoE Pretraining Script

Implements the pretraining setup from the paper:
- AdamW optimizer with epsilon=1e-8
- Cosine LR schedule with warmup
- Linear LR decay to 0 during annealing (final 100B tokens)
- Gradient clipping at 1.0
- Mixed precision (BF16)
- Weight decay applied to ALL parameters including embeddings and RMSNorm
- ZeRO via PyTorch FSDP

Training configuration (from Table 10):
- Batch size: ~4M tokens (1024 samples x 4096 seq len)
- Peak LR: 4e-4
- Minimum LR: 4e-5
- Warmup steps: 2500
- Total tokens: 5.133T (1.3 epochs)
- Annealing: final 100B tokens, linear decay to 0
"""

import os
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters for OLMoE-1B-7B."""

    # Optimizer
    learning_rate: float = 4e-4
    min_learning_rate: float = 4e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8  # Key: reduced from OLMo's 1e-5 to 1e-8
    grad_clip: float = 1.0

    # LR schedule
    warmup_steps: int = 2500
    total_steps: int = 1_250_000  # ~5T tokens / 4M tokens per step
    annealing_steps: int = 25_000  # ~100B tokens / 4M tokens per step

    # Batch
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    seq_len: int = 4096

    # Training
    num_epochs: float = 1.3  # Train for 1.3 epochs following Muennighoff et al.
    bf16: bool = True
    gradient_checkpointing: bool = False

    # Logging
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 5000  # Save checkpoint every 5000 steps

    # Paths
    output_dir: str = "checkpoints"
    data_path: str = "data"

    # Distributed
    use_fsdp: bool = True


def get_cosine_schedule_with_warmup_and_annealing(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    annealing_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Cosine LR schedule with warmup and linear annealing.

    From the paper:
    - Linear warmup for warmup_steps
    - Cosine decay from peak LR to min LR
    - During annealing (final annealing_steps): linear decay to 0

    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps (before annealing)
        annealing_steps: Number of annealing steps (linear decay to 0)
        min_lr_ratio: Ratio of min LR to peak LR (default 0.1 = 4e-5/4e-4)
    """
    def lr_lambda(current_step: int) -> float:
        # Annealing phase: linear decay to 0
        if current_step >= total_steps:
            annealing_progress = (current_step - total_steps) / annealing_steps
            annealing_progress = min(1.0, annealing_progress)
            return max(0.0, 1.0 - annealing_progress) * min_lr_ratio

        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale from 1.0 to min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def create_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    """Create AdamW optimizer.

    Key: weight decay is applied to ALL parameters including embeddings and RMSNorm.
    This differs from common practice but the paper finds it slightly better.
    """
    # All parameters get weight decay (including embeddings and RMSNorm)
    # This is a key finding from Section 4.2.3 and 4.2.4
    param_groups = [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "weight_decay": config.weight_decay,
        }
    ]

    optimizer = AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_epsilon,  # 1e-8, not the OLMo default of 1e-5
    )
    return optimizer


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    config: TrainingConfig,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Single training step."""
    model.train()

    input_ids = batch["input_ids"]
    labels = batch.get("labels", input_ids.clone())

    # Forward pass with mixed precision
    with torch.cuda.amp.autocast(enabled=config.bf16, dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        aux_loss = outputs["aux_loss"]

    # Scale loss for gradient accumulation
    loss = loss / config.gradient_accumulation_steps

    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    metrics = {
        "loss": loss.item() * config.gradient_accumulation_steps,
        "aux_loss": aux_loss.item() if aux_loss is not None else 0.0,
    }

    return metrics


def clip_and_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    config: TrainingConfig,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """Clip gradients and take optimizer step."""
    if scaler is not None:
        scaler.unscale_(optimizer)

    # Gradient clipping at 1.0 (global norm)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.grad_clip
    )

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    scheduler.step()
    optimizer.zero_grad()

    return grad_norm.item()


class OLMoETrainer:
    """Trainer for OLMoE pretraining."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader,
        eval_dataloader=None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.optimizer = create_optimizer(model, config)
        self.scheduler = get_cosine_schedule_with_warmup_and_annealing(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
            annealing_steps=config.annealing_steps,
            min_lr_ratio=config.min_learning_rate / config.learning_rate,
        )

        self.scaler = None
        if config.bf16 and torch.cuda.is_available():
            # BF16 doesn't need gradient scaling
            pass

        self.global_step = 0
        self.total_tokens = 0

    def train(self):
        """Main training loop."""
        logger.info("Starting OLMoE pretraining...")
        logger.info(f"Total steps: {self.config.total_steps}")
        logger.info(f"Annealing steps: {self.config.annealing_steps}")

        accumulation_step = 0
        running_loss = 0.0
        running_aux_loss = 0.0

        for epoch in range(math.ceil(self.config.num_epochs)):
            for batch in self.train_dataloader:
                # Move batch to device
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Training step
                metrics = train_step(
                    self.model, batch, self.optimizer,
                    self.scheduler, self.config, self.scaler
                )

                running_loss += metrics["loss"]
                running_aux_loss += metrics["aux_loss"]
                accumulation_step += 1

                # Update parameters after accumulation
                if accumulation_step % self.config.gradient_accumulation_steps == 0:
                    grad_norm = clip_and_step(
                        self.model, self.optimizer, self.scheduler,
                        self.config, self.scaler
                    )
                    self.global_step += 1
                    self.total_tokens += (
                        batch["input_ids"].numel() *
                        self.config.gradient_accumulation_steps
                    )

                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        avg_loss = running_loss / self.config.log_interval
                        avg_aux_loss = running_aux_loss / self.config.log_interval
                        lr = self.scheduler.get_last_lr()[0]

                        logger.info(
                            f"Step {self.global_step} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Aux Loss: {avg_aux_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Grad Norm: {grad_norm:.4f} | "
                            f"Tokens: {self.total_tokens/1e9:.2f}B"
                        )
                        running_loss = 0.0
                        running_aux_loss = 0.0

                    # Save checkpoint
                    if self.global_step % self.config.save_interval == 0:
                        self.save_checkpoint()

                    # Check if done
                    total_steps = self.config.total_steps + self.config.annealing_steps
                    if self.global_step >= total_steps:
                        logger.info("Training complete!")
                        self.save_checkpoint(final=True)
                        return

    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        suffix = "final" if final else f"step_{self.global_step}"
        path = os.path.join(self.config.output_dir, f"checkpoint_{suffix}.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "total_tokens": self.total_tokens,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
