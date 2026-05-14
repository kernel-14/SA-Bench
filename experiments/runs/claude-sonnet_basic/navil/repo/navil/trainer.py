"""
Training utilities for NaViL.

Implements the two-stage training recipe from the paper:

Stage 1: Multi-modal Generative Pre-training
  - Sub-stage 1a: Train on 300M web-scale image-text pairs
    * Frozen: text parameters (embed_tokens, LLM text experts, lm_head)
    * Trainable: visual encoder, MLP connector, MoE visual experts
    * Global batch size: 7000
  - Sub-stage 1b: Train on 185M high-quality multimodal + language data
    * Additionally unfreeze: text attention parameters (q/k/v/o projections)
    * Global batch size: 7000

Stage 2: Supervised Fine-tuning
  - Train on 68M high-quality multimodal data
  - All parameters unfrozen
  - Global batch size: 4614

Training objective: Next-Token-Prediction (NTP) with image captioning task.
"""

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for NaViL."""

    # Stage
    stage: str = "pretrain_1a"  # pretrain_1a, pretrain_1b, sft

    # Data
    data_path: str = ""
    output_dir: str = "./output"
    max_seq_len: int = 4096

    # Optimization
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    num_epochs: int = 1

    # Batch sizes (from paper)
    # Stage 1: global batch size 7000
    # Stage 2: global batch size 4614
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 1

    # Data sizes (from paper)
    # Stage 1a: 300M web-scale + 200M synthetic = 500M total
    # Stage 1b: 185M high-quality
    # Stage 2: 68M SFT
    stage_1a_samples: int = 500_000_000
    stage_1b_samples: int = 185_000_000
    stage_2_samples: int = 68_000_000

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Logging
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000

    # Distributed
    local_rank: int = -1
    world_size: int = 1

    # Image processing
    image_size: int = 448
    patch_size: int = 16

    # Multi-scale packing
    use_multiscale: bool = True
    multiscale_tau: float = 0.5 * math.sqrt(2)


class NaViLTrainer:
    """
    Trainer for NaViL following the paper's training recipe.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Setup training stage
        self._setup_stage()

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None

    def _setup_stage(self):
        """Configure model parameters based on training stage."""
        stage = self.config.stage

        if stage == "pretrain_1a":
            logger.info("Stage 1a: Freezing text parameters, training visual components")
            # Freeze all text parameters
            # Trainable: visual encoder, MLP connector, MoE visual experts
            self._freeze_for_stage_1a()

        elif stage == "pretrain_1b":
            logger.info("Stage 1b: Unfreezing text attention parameters")
            # Additionally unfreeze text attention
            self._freeze_for_stage_1b()

        elif stage == "sft":
            logger.info("Stage 2: SFT - all parameters trainable")
            self.model.unfreeze_all()

        else:
            raise ValueError(f"Unknown training stage: {stage}")

    def _freeze_for_stage_1a(self):
        """
        Stage 1a freezing strategy:
        - Frozen: text parameters (embed_tokens, LLM text experts, lm_head)
        - Trainable: visual encoder, MLP connector, MoE visual experts (index 0)
        """
        # First freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze visual encoder
        for param in self.model.visual_encoder.parameters():
            param.requires_grad = True

        # Unfreeze MoE visual experts (modality 0 = visual)
        for name, param in self.model.named_parameters():
            if "moe" in name.lower():
                # Visual expert parameters have index 0 in ModuleList
                if any(f"_projs.{VISUAL_MODALITY}" in name or
                       f"projs.{VISUAL_MODALITY}" in name
                       for VISUAL_MODALITY in [0]):
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Stage 1a: {trainable:,} / {total:,} parameters trainable")

    def _freeze_for_stage_1b(self):
        """
        Stage 1b: Additionally unfreeze text attention parameters.
        """
        self._freeze_for_stage_1a()

        # Unfreeze text attention (q/k/v/o projections in LLM layers)
        for name, param in self.model.named_parameters():
            if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Stage 1b: {trainable:,} / {total:,} parameters trainable")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "norm", "embedding"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def create_scheduler(self, num_training_steps: int):
        """Create cosine learning rate scheduler with warmup."""
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return max(
                self.config.min_lr / self.config.learning_rate,
                0.5 * (1.0 + math.cos(math.pi * progress))
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return self.scheduler

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Single training step."""
        self.model.train()

        outputs = self.model(
            input_ids=batch.get("input_ids"),
            images=batch.get("images"),
            visual_token_mask=batch.get("visual_token_mask"),
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

        loss = outputs["loss"]
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        return loss

    def train(self):
        """Full training loop."""
        if self.train_dataloader is None:
            raise ValueError("train_dataloader must be provided")

        num_training_steps = (
            len(self.train_dataloader) * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        self.create_scheduler(num_training_steps)

        global_step = 0
        total_loss = 0.0

        for epoch in range(self.config.num_epochs):
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                loss = self.training_step(batch)
                loss.backward()
                total_loss += loss.item()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )

                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / self.config.logging_steps
                        lr = self.optimizer.param_groups[0]["lr"]
                        logger.info(
                            f"Step {global_step}: loss={avg_loss:.4f}, lr={lr:.2e}"
                        )
                        total_loss = 0.0

                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(global_step)

        self.save_checkpoint(global_step, final=True)

    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if final else f"step_{step}"
        save_path = os.path.join(self.config.output_dir, f"checkpoint_{suffix}")
        os.makedirs(save_path, exist_ok=True)

        # Save model state dict
        torch.save(
            self.model.state_dict(),
            os.path.join(save_path, "model.pt")
        )

        # Save config
        import json
        config_dict = {k: v for k, v in vars(self.config).items()}
        with open(os.path.join(save_path, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Saved checkpoint to {save_path}")
