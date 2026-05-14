"""Training loop for OLMoE pretraining and adaptation.

Implements the training recipe from Section 2 and Appendix B:
- AdamW optimizer with eps=1e-8
- Cosine learning rate schedule
- Load balancing loss and router z-loss
- Gradient clipping at 1.0
- Mixed precision training (BF16)
- Annealing phase with linear LR decay to 0
"""

import math
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..models.configuration import OLMoEConfig
from ..models.transformer import OLMoEModel


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine learning rate schedule with linear warmup.

    As used in OLMoE pretraining (Table 10):
    - Warmup for 2500 steps
    - Cosine decay from peak_lr to min_lr
    - During annealing: linear decay to 0
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(
                min_lr_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

    return LambdaLR(optimizer, lr_lambda)


def get_linear_annealing_schedule(
    optimizer: torch.optim.Optimizer,
    start_step: int,
    total_steps: int,
):
    """Linear learning rate decay to 0 during annealing phase.

    Used for the final 100B tokens of pretraining.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < start_step:
            return 1.0
        else:
            progress = float(current_step - start_step) / float(
                max(1, total_steps - start_step)
            )
            return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


class OLMoETrainer:
    """Trainer for OLMoE pretraining and adaptation.

    Handles the training loop with:
    - Auxiliary loss tracking (L_LB, L_RZ)
    - Gradient clipping
    - Mixed precision
    - Checkpoint saving
    """

    def __init__(
        self,
        model: OLMoEModel,
        config: OLMoEConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.dtype = dtype

        # Set up optimizer with paper's exact hyperparameters
        # Weight decay all parameters including embeddings and RMSNorm (§4.2.3, §4.2.4)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.peak_lr,
            betas=(config.beta1, config.beta2),
            eps=config.adamw_eps,
            weight_decay=config.weight_decay,
        )

        self.scheduler = None
        self.global_step = 0

        # For tracking metrics
        self.metrics: Dict[str, list] = {
            "ce_loss": [],
            "lb_loss": [],
            "rz_loss": [],
            "total_loss": [],
            "learning_rate": [],
            "grad_norm": [],
            "tokens_per_second": [],
        }

    def compute_grad_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)

    def train_step(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            labels: Labels for CE loss (same as input_ids for LM)
            attention_mask: Optional attention mask

        Returns:
            Dict of loss values
        """
        self.model.train()

        input_ids = input_ids.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        else:
            labels = input_ids.clone()

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass with automatic mixed precision
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

        loss = outputs["total_loss"]
        if loss is None:
            raise ValueError("No loss computed")

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (global, value=1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clipping
        )

        # Optimizer step
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        # Log metrics
        metrics = {
            "ce_loss": outputs["ce_loss"].item() if outputs["ce_loss"] is not None else 0.0,
            "lb_loss": (
                sum(layer.moe.lb_loss.item() for layer in self.model.layers if layer.moe is not None)
                if any(layer.moe is not None for layer in self.model.layers)
                else 0.0
            ),
            "rz_loss": (
                sum(layer.moe.rz_loss.item() for layer in self.model.layers if layer.moe is not None)
                if any(layer.moe is not None for layer in self.model.layers)
                else 0.0
            ),
            "total_loss": loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        for k, v in metrics.items():
            self.metrics[k].append(v)

        return metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
            "metrics": self.metrics,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.metrics = checkpoint.get("metrics", {})


class SFTTrainer:
    """Supervised Fine-Tuning trainer for OLMoE.

    Follows the adaptation recipe from Section 2 and Appendix B:
    - No load balancing loss during SFT (§4.3)
    - Learning rate: 2e-5
    - 2 epochs
    - Batch size: 128
    - BF16 precision
    """

    def __init__(
        self,
        model: OLMoEModel,
        config: OLMoEConfig,
        use_load_balancing: bool = False,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_load_balancing = use_load_balancing

        # Optionally disable auxiliary losses for SFT
        if not use_load_balancing:
            for layer in self.model.layers:
                if layer.moe is not None:
                    layer.moe.lb_loss_weight = 0.0
                    layer.moe.rz_loss_weight = 0.0

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.sft_learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.adamw_eps,
            weight_decay=0.0,  # No weight decay during SFT typically
        )

    def train_step(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single SFT training step."""
        self.model.train()

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

        loss = outputs["total_loss"]
        if loss is None:
            # Use CE loss only
            loss = outputs["ce_loss"]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "ce_loss": outputs["ce_loss"].item() if outputs["ce_loss"] is not None else 0.0,
            "total_loss": loss.item(),
        }


class DPOTrainer:
    """Direct Preference Optimization trainer for OLMoE.

    Follows the DPO recipe from Appendix B:
    - Learning rate: 5e-7
    - 3 epochs
    - Batch size: 32
    - DPO beta: 0.1
    - No load balancing loss
    """

    def __init__(
        self,
        model: OLMoEModel,
        ref_model: OLMoEModel,
        config: OLMoEConfig,
        beta: float = 0.1,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.config = config
        self.beta = beta
        self.device = device

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Disable auxiliary losses
        for layer in self.model.layers:
            if layer.moe is not None:
                layer.moe.lb_loss_weight = 0.0
                layer.moe.rz_loss_weight = 0.0

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.dpo_learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.adamw_eps,
            weight_decay=0.0,
        )

    def dpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DPO loss (Rafailov et al., 2023).

        L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                           - β * (log π_θ(y_l|x) - log π_ref(y_l|x)))]
        """
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)

        loss = -torch.nn.functional.logsigmoid(
            chosen_rewards - rejected_rewards
        ).mean()

        return loss

    def train_step(
        self,
        chosen_input_ids: torch.LongTensor,
        chosen_labels: torch.LongTensor,
        rejected_input_ids: torch.LongTensor,
        rejected_labels: torch.LongTensor,
    ) -> Dict[str, float]:
        """Single DPO training step."""
        self.model.train()

        chosen_input_ids = chosen_input_ids.to(self.device)
        chosen_labels = chosen_labels.to(self.device)
        rejected_input_ids = rejected_input_ids.to(self.device)
        rejected_labels = rejected_labels.to(self.device)

        # Get log probabilities from policy model
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            chosen_out = self.model(input_ids=chosen_input_ids, labels=chosen_labels)
            rejected_out = self.model(input_ids=rejected_input_ids, labels=rejected_labels)

            # With reference model
            with torch.no_grad():
                ref_chosen_out = self.ref_model(input_ids=chosen_input_ids, labels=chosen_labels)
                ref_rejected_out = self.ref_model(input_ids=rejected_input_ids, labels=rejected_labels)

        # Log probabilities (negative CE loss)
        chosen_logps = -chosen_out["ce_loss"]
        rejected_logps = -rejected_out["ce_loss"]
        ref_chosen_logps = -ref_chosen_out["ce_loss"]
        ref_rejected_logps = -ref_rejected_out["ce_loss"]

        loss = self.dpo_loss(
            chosen_logps, rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "dpo_loss": loss.item(),
            "chosen_logps": chosen_logps.item(),
            "rejected_logps": rejected_logps.item(),
        }
