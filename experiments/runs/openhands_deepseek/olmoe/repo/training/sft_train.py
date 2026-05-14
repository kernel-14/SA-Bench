import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional
import wandb

from model.olmoe_model import OLMoEModel
from training.losses import compute_total_loss


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for OLMoE-1B-7B.

    Hyperparameters (Appendix B, Section 4.3):
        - BF16 mixed precision
        - Global batch size 128 (4 H100 nodes * 8 GPUs * 2 per_device * 2 grad_accum)
        - 2 epochs
        - Constant learning rate 2e-5
        - Sequence length 4096
        - No load balancing loss during SFT (experiment in Table 7)
        - Loss aggregated at token level (Muennighoff et al., 2024)
        - AdamW optimizer
        - Starting from annealed checkpoint
    """
    def __init__(
        self,
        model: OLMoEModel,
        config: dict,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config["adaptation"]["sft"]
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,  # typically no weight decay for SFT
        )

        mixed_precision = self.config.get("mixed_precision", "bf16")
        self.grad_scaler = GradScaler(enabled=(mixed_precision == "fp16"))
        self.mixed_precision_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float32

        moe_cfg = config["model"]["moe"]
        use_lb = self.config.get("load_balancing_loss", False)
        self.lb_weight = moe_cfg["load_balancing_loss_weight"] if use_lb else 0.0
        self.rz_weight = 0.0  # No router z-loss during SFT
        self.num_experts = moe_cfg["num_experts"]

        self.device = next(model.parameters()).device

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with autocast(
            device_type="cuda",
            enabled=self.mixed_precision_dtype != torch.float32,
            dtype=self.mixed_precision_dtype,
        ):
            logits, router_logits_list, router_probs_list = self.model(input_ids, attention_mask)
            loss_dict = compute_total_loss(
                logits=logits,
                labels=labels,
                router_logits_list=router_logits_list,
                router_probs_list=router_probs_list,
                num_experts=self.num_experts,
                load_balancing_weight=self.lb_weight,
                router_z_weight=self.rz_weight,
                ignore_index=-100,
            )

        loss = loss_dict["loss"]
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {k: v.item() for k, v in loss_dict.items()}

    def train(
        self,
        log_interval: int = 10,
        eval_interval: int = 500,
        save_dir: str = "./sft_checkpoints",
        wandb_project: Optional[str] = None,
    ):
        os.makedirs(save_dir, exist_ok=True)
        num_epochs = self.config["epochs"]
        grad_accum = self.config.get("gradient_accumulation_steps", 1)

        if wandb_project:
            wandb.init(project=wandb_project)

        self.optimizer.zero_grad()
        global_step = 0

        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = self.train_step(batch)
                global_step += 1

                if global_step % log_interval == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss_dict['loss']:.4f}")
                    if wandb_project:
                        wandb.log({"sft/loss": loss_dict["loss"]}, step=global_step)

            # Save per epoch
            self.save_checkpoint(os.path.join(save_dir, f"epoch_{epoch}"))

        self.save_checkpoint(os.path.join(save_dir, "final"))
        if wandb_project:
            wandb.finish()

    def save_checkpoint(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, os.path.join(path, "checkpoint.pt"))
