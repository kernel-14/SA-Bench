import os
import math
import time
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional
import wandb

from model.olmoe_model import OLMoEModel
from training.losses import compute_total_loss


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_decay_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    min_lr: float = 0.0,
):
    """Linear learning rate decay to min_lr (used during annealing)."""
    start_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda(current_step: int) -> float:
        progress = float(current_step) / float(max(1, total_steps))
        return 1.0 - progress * (1.0 - min_lr / max(start_lrs[0], 1e-10))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class PretrainingTrainer:
    """
    Pretraining trainer for OLMoE-1B-7B.

    Key hyperparameters from Appendix B:
        - AdamW: peak_lr=4e-4, min_lr=4e-5, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8
        - Cosine LR schedule with 2500 warmup steps
        - Gradient clipping at 1.0
        - BF16 mixed precision
        - Batch size ~4M tokens (1024 samples * 4096 seq_len)
        - Load balancing loss weight 0.01
        - Router z-loss weight 0.001
        - Training for 5.133T tokens with final 100B annealing
    """
    def __init__(
        self,
        model: OLMoEModel,
        config: dict,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        cfg = config["pretraining"]
        opt_cfg = cfg["optimizer"]

        # Weight decay ALL parameters (Sections 4.2.3, 4.2.4)
        # "for simplicity, we weight decay all parameters in OLMOE-1B-7B
        #  including embedding and RMSNorm."
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=opt_cfg["peak_lr"],
            betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
            eps=opt_cfg["eps"],
            weight_decay=opt_cfg["weight_decay"],
        )

        self.grad_scaler = GradScaler(enabled=(cfg.get("mixed_precision", "bf16") == "fp16"))
        self.mixed_precision_dtype = torch.bfloat16 if cfg.get("mixed_precision") == "bf16" else torch.float32
        self.grad_clip = cfg["gradient_clipping"]

        # Auxiliary loss weights
        moe_cfg = config["model"]["moe"]
        self.lb_weight = moe_cfg["load_balancing_loss_weight"]
        self.rz_weight = moe_cfg["router_z_loss_weight"]
        self.num_experts = moe_cfg["num_experts"]

        self.device = next(model.parameters()).device

    def train_step(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """Single training step with mixed precision."""
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        with autocast(
            device_type="cuda",
            enabled=self.mixed_precision_dtype != torch.float32,
            dtype=self.mixed_precision_dtype,
        ):
            logits, router_logits_list, router_probs_list = self.model(input_ids)
            loss_dict = compute_total_loss(
                logits=logits,
                labels=labels,
                router_logits_list=router_logits_list,
                router_probs_list=router_probs_list,
                num_experts=self.num_experts,
                load_balancing_weight=self.lb_weight,
                router_z_weight=self.rz_weight,
            )

        loss = loss_dict["loss"]
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        self.model.eval()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        logits, router_logits_list, router_probs_list = self.model(input_ids)
        loss_dict = compute_total_loss(
            logits=logits,
            labels=labels,
            router_logits_list=router_logits_list,
            router_probs_list=router_probs_list,
            num_experts=self.num_experts,
            load_balancing_weight=self.lb_weight,
            router_z_weight=self.rz_weight,
        )

        return {k: v.item() for k, v in loss_dict.items()}

    def train(
        self,
        total_steps: int,
        annealing_steps: int = 0,
        log_interval: int = 100,
        eval_interval: int = 500,
        save_interval: int = 5000,
        save_dir: str = "./checkpoints",
        wandb_project: Optional[str] = None,
    ):
        """
        Main training loop.

        Args:
            total_steps: Total pretraining steps
            annealing_steps: Steps at which annealing starts (linearly decays LR to 0)
            log_interval: Log every N steps
            eval_interval: Evaluate every N steps
            save_interval: Save checkpoint every N steps (paper: every 5000 steps)
            save_dir: Checkpoint directory
            wandb_project: WandB project name
        """
        os.makedirs(save_dir, exist_ok=True)

        cfg = self.config["pretraining"]
        warmup_steps = cfg["warmup_steps"]
        pre_annealing_steps = total_steps - annealing_steps

        # Phase 1: Cosine schedule
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=pre_annealing_steps,
            min_lr_ratio=cfg["optimizer"]["min_lr"] / cfg["optimizer"]["peak_lr"],
        )

        if wandb_project:
            wandb.init(project=wandb_project, config=self.config)

        global_step = 0
        start_time = time.time()
        tokens_processed = 0
        tokens_per_step = cfg["batch_size_tokens"]

        dataloader_iter = iter(self.train_dataloader)

        while global_step < total_steps:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # Reshuffle at epoch boundary (paper shuffles at start of each epoch)
                dataloader_iter = iter(self.train_dataloader)
                batch = next(dataloader_iter)

            # Check if we should switch to annealing phase
            if global_step == pre_annealing_steps and annealing_steps > 0:
                # Phase 2: Linear decay to 0
                scheduler = get_linear_decay_schedule(
                    self.optimizer,
                    total_steps=annealing_steps,
                    min_lr=cfg["annealing"]["min_lr"],
                )

            loss_dict = self.train_step(batch, global_step)
            scheduler.step()
            tokens_processed += tokens_per_step

            global_step += 1

            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                log_msg = (
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {loss_dict['loss']:.4f} | "
                    f"CE: {loss_dict['ce_loss']:.4f} | "
                    f"LB: {loss_dict['lb_loss']:.4f} | "
                    f"RZ: {loss_dict['rz_loss']:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Tokens/s: {tokens_per_sec:.0f}"
                )
                print(log_msg)
                if wandb_project:
                    wandb.log({
                        "train/loss": loss_dict["loss"],
                        "train/ce_loss": loss_dict["ce_loss"],
                        "train/lb_loss": loss_dict["lb_loss"],
                        "train/rz_loss": loss_dict["rz_loss"],
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                    }, step=global_step)

            if self.val_dataloader is not None and global_step % eval_interval == 0:
                val_losses = []
                for val_batch in self.val_dataloader:
                    val_loss = self.eval_step(val_batch)
                    val_losses.append(val_loss)
                avg_val = {k: sum(d[k] for d in val_losses) / len(val_losses) for k in val_losses[0]}
                print(f"Validation - Loss: {avg_val['loss']:.4f} | CE: {avg_val['ce_loss']:.4f}")
                if wandb_project:
                    wandb.log({f"val/{k}": v for k, v in avg_val.items()}, step=global_step)

            if save_interval > 0 and global_step % save_interval == 0:
                self.save_checkpoint(os.path.join(save_dir, f"step_{global_step}"))

        # Save final checkpoint
        self.save_checkpoint(os.path.join(save_dir, "final"))
        if wandb_project:
            wandb.finish()

    def save_checkpoint(self, path: str):
        """Save model checkpoint with optimizer state."""
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, os.path.join(path, "checkpoint.pt"))
        # Also save in HF-compatible format
        self.model.save_pretrained(path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(os.path.join(path, "checkpoint.pt"), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("config", self.config)
