import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional
import wandb

from model.olmoe_model import OLMoEModel
from training.losses import dpo_loss


class DPOTrainer:
    """
    Direct Preference Optimization trainer for OLMoE-1B-7B-INSTRUCT.

    Hyperparameters (Appendix B, Section 4.3):
        - DPO beta = 0.1
        - Global batch size 32 (4 H100 nodes * 8 GPUs * 1 per_device * 1 grad_accum)
        - 3 epochs
        - Learning rate 5e-7
        - No load balancing loss
        - Starting from SFT checkpoint
    """
    def __init__(
        self,
        model: OLMoEModel,
        reference_model: OLMoEModel,
        config: dict,
        train_dataloader: DataLoader,
    ):
        self.model = model
        self.reference_model = reference_model
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.config = config["adaptation"]["dpo"]
        self.train_dataloader = train_dataloader

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        mixed_precision = self.config.get("mixed_precision", "bf16")
        self.grad_scaler = GradScaler(enabled=(mixed_precision == "fp16"))
        self.mixed_precision_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float32

        self.beta = self.config["beta"]
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def get_reference_logits(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits, _, _ = self.reference_model(input_ids, attention_mask)
        return logits.detach()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        # Get reference logits
        with torch.no_grad():
            ref_chosen_logits = self.get_reference_logits(chosen_input_ids)
            ref_rejected_logits = self.get_reference_logits(rejected_input_ids)

        with autocast(
            device_type="cuda",
            enabled=self.mixed_precision_dtype != torch.float32,
            dtype=self.mixed_precision_dtype,
        ):
            policy_chosen_logits, _, _ = self.model(chosen_input_ids)
            policy_rejected_logits, _, _ = self.model(rejected_input_ids)

            loss_dict = dpo_loss(
                policy_chosen_logits=policy_chosen_logits,
                policy_rejected_logits=policy_rejected_logits,
                ref_chosen_logits=ref_chosen_logits,
                ref_rejected_logits=ref_rejected_logits,
                chosen_labels=chosen_labels,
                rejected_labels=rejected_labels,
                beta=self.beta,
            )

        loss = loss_dict["loss"]
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {"loss": loss.item(), "accuracy": loss_dict["accuracy"].item()}

    def train(
        self,
        log_interval: int = 10,
        save_dir: str = "./dpo_checkpoints",
        wandb_project: Optional[str] = None,
    ):
        os.makedirs(save_dir, exist_ok=True)
        num_epochs = self.config["epochs"]

        if wandb_project:
            wandb.init(project=wandb_project)

        global_step = 0

        for epoch in range(num_epochs):
            for batch in self.train_dataloader:
                loss_dict = self.train_step(batch)
                global_step += 1

                if global_step % log_interval == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss_dict['loss']:.4f} | Acc: {loss_dict['accuracy']:.4f}")
                    if wandb_project:
                        wandb.log({"dpo/loss": loss_dict["loss"], "dpo/accuracy": loss_dict["accuracy"]}, step=global_step)

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
