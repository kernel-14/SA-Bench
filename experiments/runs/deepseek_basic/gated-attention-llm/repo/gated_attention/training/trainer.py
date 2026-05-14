"""
Training framework for gated LLMs.

Implements the training setup described in Sec 3.1:
  - AdamW optimizer with default hyperparameters
  - Cosine learning rate scheduler with warmup
  - Loss computation with optional Z-loss for MoE
  - Gradient accumulation and BF16 mixed precision support
  - Training stability monitoring (loss spikes, massive activations)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class TrainingConfig:
    """Training hyperparameters matching the paper's experiments.

    Key settings from Sec 3.1-3.2:
      - MoE 15A2B: max_lr=2e-3, warmup=1000 steps, cosine decay to 3e-5,
                    global bsz=1024, 100k steps on 400B tokens
      - Dense 1.7B (400B): max_lr=4e-3, bsz=1024
      - Dense 1.7B (3.5T): max_lr=4.5e-3, bsz=2048
      - Increased LR experiments: max_lr=8e-3
    """
    # Optimization
    max_lr: float = 2e-3
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    total_steps: int = 100000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Data
    global_batch_size: int = 1024
    micro_batch_size: int = 1
    seq_len: int = 4096

    # Mixed precision
    use_bf16: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 5000

    # MoE specific
    z_loss_coef: float = 0.001
    router_aux_coef: float = 0.0  # Additional load balancing if needed

    # Stability
    grad_clip: float = 1.0


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with linear warmup.

    Used in the paper: warmup to max_lr in warmup_steps,
    then cosine decay to min_lr.
    """

    def __init__(self, optimizer, config: TrainingConfig):
        self.optimizer = optimizer
        self.config = config

    def get_lr(self, step: int) -> float:
        cfg = self.config
        if step < cfg.warmup_steps:
            # Linear warmup
            return cfg.max_lr * (step / cfg.warmup_steps)
        elif step > cfg.total_steps:
            return cfg.min_lr
        else:
            # Cosine decay
            progress = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
            return cfg.min_lr + 0.5 * (cfg.max_lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))

    def step(self, step: int):
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class GatedLLMTrainer:
    """Trainer for Gated LLM models.

    Handles:
      - Training loop with gradient accumulation
      - BF16 mixed precision
      - Z-loss for MoE load balancing
      - Stability monitoring (loss spikes, gradient norms)
      - Periodic evaluation and checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.max_lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = CosineWarmupScheduler(self.optimizer, config)

        # Gradient accumulation steps
        self.grad_accum_steps = config.global_batch_size // config.micro_batch_size

        # Training state
        self.global_step = 0
        self.total_tokens = 0
        self.best_loss = float("inf")
        self.loss_history: List[float] = []
        self.lr_history: List[float] = []

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_bf16)

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute language modeling loss.

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) - shifted by 1 position
            attention_mask: optional attention mask

        Returns:
            loss: total loss (LM loss + auxiliary losses)
            metrics: dict with loss breakdown
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
        )

        logits = outputs["logits"]
        aux_losses = outputs.get("aux_losses", {})

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Language modeling loss
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Combine with auxiliary losses
        total_loss = lm_loss
        for loss_name, loss_val in aux_losses.items():
            total_loss = total_loss + loss_val

        metrics = {
            "lm_loss": lm_loss.item(),
            "total_loss": total_loss.item(),
        }
        metrics.update({k: v.item() if torch.is_tensor(v) else v for k, v in aux_losses.items()})

        return total_loss, metrics

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Single training step with gradient accumulation."""
        self.model.train()

        # Micro-batch processing for gradient accumulation
        micro_bsz = self.config.micro_batch_size
        total_bsz = input_ids.size(0)
        accumulated_loss = 0.0
        total_metrics = {}

        for i in range(0, total_bsz, micro_bsz):
            micro_input = input_ids[i:i + micro_bsz]
            micro_labels = labels[i:i + micro_bsz]
            micro_mask = attention_mask[i:i + micro_bsz] if attention_mask is not None else None

            # Forward with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.use_bf16, dtype=torch.bfloat16):
                loss, metrics = self.compute_loss(micro_input, micro_labels, micro_mask)
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps

            # Backward
            self.scaler.scale(loss).backward()

            accumulated_loss += metrics["total_loss"]
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

        # Average metrics
        num_micro = total_bsz // micro_bsz
        accumulated_loss /= num_micro
        for k in total_metrics:
            total_metrics[k] /= num_micro

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # Update LR
        lr = self.scheduler.step(self.global_step)

        self.global_step += 1
        self.total_tokens += total_bsz * self.config.seq_len

        total_metrics.update({
            "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            "lr": lr,
            "step": self.global_step,
            "total_tokens": self.total_tokens,
        })

        # Track history
        self.loss_history.append(accumulated_loss)
        self.lr_history.append(lr)

        return total_metrics

    def detect_loss_spike(
        self,
        window: int = 50,
        threshold: float = 3.0,
    ) -> bool:
        """Detect loss spikes during training (Fig. 1, right).

        Following Sec 3.2.2: gating reduces loss spike occurrence.
        """
        if len(self.loss_history) < window:
            return False

        recent = self.loss_history[-window:]
        mean_loss = sum(recent) / len(recent)
        current = self.loss_history[-1]

        return current > mean_loss * threshold

    def evaluate(
        self,
        eval_dataloader,
        max_eval_steps: int = 100,
    ) -> Dict:
        """Evaluate model on a validation set."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                if step >= max_eval_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")

                with torch.cuda.amp.autocast(enabled=self.config.use_bf16, dtype=torch.bfloat16):
                    _, metrics = self.compute_loss(input_ids, labels, attention_mask)

                total_loss += metrics["lm_loss"] * input_ids.size(0)
                total_tokens += input_ids.size(0)

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(avg_loss)

        return {
            "eval_loss": avg_loss,
            "eval_ppl": perplexity,
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "total_tokens": self.total_tokens,
            "config": self.config,
            "loss_history": self.loss_history,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.total_tokens = checkpoint["total_tokens"]
        self.loss_history = checkpoint.get("loss_history", [])
