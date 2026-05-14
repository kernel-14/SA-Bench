import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from typing import Optional, Callable, Dict, Any
import wandb

from config import ExperimentConfig, ModelConfig, TrainingConfig, DiffusionConfig
from models import MaskedDiffusionModel, get_num_params
from diffusion import (
    get_noise_schedule,
    forward_mask,
    score_entropy_loss,
    d_alpha_dt,
)
from data import (
    get_dataloader,
    sample_permutation,
    PermutationDataloader,
)


def get_optimizer(model: nn.Module, cfg: TrainingConfig) -> AdamW:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(optim_groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))


def get_lr_scheduler(optimizer, cfg: TrainingConfig):
    def cosine_schedule(step):
        if step < cfg.warmup_steps:
            return cfg.learning_rate * step / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_schedule)


def train_mdm(
    cfg: ExperimentConfig,
    dataloader: DataLoader,
    model: Optional[MaskedDiffusionModel] = None,
    device: torch.device = None,
):
    """Train a Masked Diffusion Model using the score-entropy loss."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = MaskedDiffusionModel(cfg.model).to(device)

    model.train()
    optimizer = get_optimizer(model, cfg.training)
    scheduler = get_lr_scheduler(optimizer, cfg.training)
    noise_schedule_fn = get_noise_schedule(cfg.diffusion)
    scaler = GradScaler() if cfg.training.dtype == "float16" else None

    mask_token = cfg.model.mask_token_id
    total_steps = cfg.training.max_steps
    step = 0
    epoch = 0
    best_loss = float("inf")

    # For iterable datasets like text
    iterator = iter(dataloader) if hasattr(dataloader, '__iter__') else None

    pbar = range(total_steps)
    for step in pbar:
        if iterator is not None:
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                iterator = iter(dataloader)
                batch = next(iterator)
        else:
            if step % len(dataloader) == 0:
                epoch += 1
            batch = next(iter(dataloader))

        if isinstance(batch, (tuple, list)):
            x_0 = batch[0].to(device)
        else:
            x_0 = batch.to(device)

        B, L = x_0.shape

        # Sample random noise levels t ∈ [0, 1]
        t_values = torch.rand(B, device=device)

        # Forward process: mask tokens
        x_t, mask_indicator, alpha_t = forward_mask(
            x_0, mask_token, t_values, noise_schedule_fn
        )

        # Model forward pass
        x_t_input = x_t.clone()
        # Replace masked tokens with mask_token (already done in forward_mask)

        with autocast(device_type=device.type, dtype=torch.bfloat16) if cfg.training.dtype == "bfloat16" else torch.no_grad():
            logits = model(x_t_input)

        # Compute score-entropy loss
        loss = score_entropy_loss(
            x_0=x_0,
            x_t=x_t,
            mask_indicator=mask_indicator,
            alpha_t=alpha_t,
            t_values=t_values,
            logits=logits,
            noise_schedule_fn=noise_schedule_fn,
            mask_token=mask_token,
        )

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()

        scheduler.step()

        if step % cfg.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"step {step:06d} | epoch {epoch:03d} | loss {loss.item():.4f} | lr {lr:.2e}")
            if cfg.wandb_project:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": step})

        if step > 0 and step % cfg.save_interval == 0:
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "cfg": cfg}
            os.makedirs(cfg.output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(cfg.output_dir, f"mdm_step_{step}.pt"))

        if step >= total_steps:
            break

    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "cfg": cfg}
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(cfg.output_dir, "mdm_final.pt"))
    return model


def train_pi_learner(
    cfg: ExperimentConfig,
    dataloader: DataLoader,
    permutation: torch.Tensor,
    model: Optional[MaskedDiffusionModel] = None,
    device: torch.device = None,
):
    """
    Train a π-learner (Section 3.2).
    The model is trained with causal attention on permuted sequences,
    predicting each token given all previous tokens in π-order.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model_cfg = cfg.model
        model_cfg.use_rope = False
        model_cfg.use_learned_pos_emb = True
        model = MaskedDiffusionModel(model_cfg).to(device)

    model.train()
    optimizer = get_optimizer(model, cfg.training)
    scheduler = get_lr_scheduler(optimizer, cfg.training)
    scaler = GradScaler() if cfg.training.dtype == "float16" else None

    mask_token = cfg.model.mask_token_id
    total_steps = cfg.training.max_steps
    step = 0

    pi = permutation.to(device)
    perm_dataloader = PermutationDataloader(dataloader, permutation)
    iterator = iter(perm_dataloader)

    for step in range(total_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(perm_dataloader)
            batch = next(iterator)

        x_pi = batch.to(device)  # (B, L) — already permuted
        B, L = x_pi.shape

        # Run model with causal attention on permuted sequence
        with autocast(device_type=device.type, dtype=torch.bfloat16) if cfg.training.dtype == "bfloat16" else torch.no_grad():
            logits = model(x_pi, causal=True)

        # Causal loss: predict token i from prefix [0..i-1]
        vocab_size = logits.shape[-1] - 1
        shift_logits = logits[:, :-1, :vocab_size].contiguous()
        shift_labels = x_pi[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()

        scheduler.step()

        if step % cfg.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            ppl = math.exp(loss.item()) if loss.item() < 100 else float("inf")
            print(f"step {step:06d} | loss {loss.item():.4f} | ppl {ppl:.2f} | lr {lr:.2e}")
            if cfg.wandb_project:
                wandb.log({"train/loss": loss.item(), "train/ppl": ppl, "train/lr": lr, "step": step})

        if step > 0 and step % cfg.save_interval == 0:
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
            os.makedirs(cfg.output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(cfg.output_dir, f"pi_learner_step_{step}.pt"))

    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(cfg.output_dir, "pi_learner_final.pt"))
    return model


def train_arm(
    cfg: ExperimentConfig,
    dataloader: DataLoader,
    model: Optional[MaskedDiffusionModel] = None,
    device: torch.device = None,
    order_info: bool = False,
):
    """
    Train an Autoregressive Model (ARM).
    - Without ordering: standard left-to-right training
    - With ordering: teacher-forced training with known correct token order
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model_cfg = cfg.model
        model_cfg.use_rope = True
        model_cfg.use_learned_pos_emb = False
        model = MaskedDiffusionModel(model_cfg).to(device)

    model.train()
    optimizer = get_optimizer(model, cfg.training)
    scheduler = get_lr_scheduler(optimizer, cfg.training)
    scaler = GradScaler() if cfg.training.dtype == "float16" else None

    mask_token = cfg.model.mask_token_id
    total_steps = cfg.training.max_steps
    step = 0

    iterator = iter(dataloader) if hasattr(dataloader, '__iter__') else None

    for step in range(total_steps):
        if iterator is not None:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
        else:
            if step % len(dataloader) == 0:
                pass
            batch = next(iter(dataloader))

        if isinstance(batch, (tuple, list)):
            x_0 = batch[0].to(device)
        else:
            x_0 = batch.to(device)

        if order_info:
            # With ordering info: use target (solution) as input
            x_input = batch[1].to(device) if isinstance(batch, (tuple, list)) and len(batch) > 1 else x_0
        else:
            x_input = x_0

        B, L = x_input.shape

        with autocast(device_type=device.type, dtype=torch.bfloat16) if cfg.training.dtype == "bfloat16" else torch.no_grad():
            logits = model(x_input, causal=True)

        vocab_size = logits.shape[-1] - 1
        shift_logits = logits[:, :-1, :vocab_size].contiguous()
        shift_labels = x_input[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()

        scheduler.step()

        if step % cfg.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"step {step:06d} | loss {loss.item():.4f} | lr {lr:.2e}")
            if cfg.wandb_project:
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": step})

        if step > 0 and step % cfg.save_interval == 0:
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
            os.makedirs(cfg.output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(cfg.output_dir, f"arm_step_{step}.pt"))

    ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(cfg.output_dir, "arm_final.pt"))
    return model


def compute_pi_learner_likelihood(
    model: MaskedDiffusionModel,
    x_0: torch.Tensor,
    pi: torch.Tensor,
) -> float:
    """
    Compute log-likelihood of sequence x_0 under π-order:
    log p_θ(x_0) = Σ_i log p_θ(x_0^{π(i)} | x_0[π{i,...,L-1}])
    """
    B, L = x_0.shape
    device = x_0.device
    x_pi = x_0[:, pi.to(device)]

    # Create input where positions i..L-1 are masked
    x_input = x_pi.clone()
    # For causal prediction we feed the full permuted sequence
    with torch.no_grad():
        logits = model(x_input, causal=True)  # (B, L, V+1)
    vocab_size = logits.shape[-1] - 1
    # logits at position i predict token at i
    # but for this to work with causal, we need to look at logits output
    # Standard causal: logits[:, i, :] predicts x_pi[:, i] given x_pi[:, :i]
    # So we use the full sequence:
    shift_logits = logits[:, :, :vocab_size]
    shift_labels = x_pi
    nll = F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_labels.reshape(-1),
        reduction='sum'
    )
    return -nll.item() / B
