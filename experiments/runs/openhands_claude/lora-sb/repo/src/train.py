"""Training loop for LoRA-SB and all baselines."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


@dataclass
class TrainingConfig:
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Scheduler
    lr_scheduler: str = "cosine"  # "cosine" or "linear"
    warmup_ratio: float = 0.02
    num_epochs: int = 1

    # Batch
    batch_size: int = 1
    gradient_accumulation_steps: int = 32

    # Regularization
    dropout: float = 0.0
    max_grad_norm: float = 1.0

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    output_dir: str = "outputs"

    # Misc
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"


def get_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    """Create AdamW optimizer with weight decay applied only to non-bias/norm params."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )


def get_scheduler(
    optimizer: AdamW,
    config: TrainingConfig,
    num_training_steps: int,
) -> Any:
    """Create learning rate scheduler."""
    num_warmup_steps = int(config.warmup_ratio * num_training_steps)

    if config.lr_scheduler == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif config.lr_scheduler == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    config: TrainingConfig,
    device: torch.device,
    epoch: int = 0,
) -> Dict[str, float]:
    """Run one training epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_steps = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(**{k: v for k, v in batch.items() if k in (
            "input_ids", "attention_mask", "labels", "token_type_ids"
        )})
        loss = outputs.loss / config.gradient_accumulation_steps
        loss.backward()

        total_loss += outputs.loss.item()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            num_steps += 1

            if num_steps % config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    # Handle remaining gradients
    remaining = len(dataloader) % config.gradient_accumulation_steps
    if remaining > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    return {"loss": avg_loss}


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    eval_fn: Optional[Any] = None,
) -> Dict[str, List[float]]:
    """Full training loop.

    Args:
        model: Model with PEFT adapters applied.
        train_dataloader: Training data.
        config: Training configuration.
        device: Compute device.
        eval_fn: Optional callable(model, device) -> dict of metrics.

    Returns:
        Training history dict.
    """
    model.to(device)
    optimizer = get_optimizer(model, config)

    num_update_steps = (
        len(train_dataloader) // config.gradient_accumulation_steps * config.num_epochs
    )
    scheduler = get_scheduler(optimizer, config, num_update_steps)

    history: Dict[str, List[float]] = {"loss": []}
    os.makedirs(config.output_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        epoch_metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            epoch=epoch,
        )
        history["loss"].append(epoch_metrics["loss"])
        print(f"Epoch {epoch + 1}/{config.num_epochs} — Loss: {epoch_metrics['loss']:.4f}")

        if eval_fn is not None:
            eval_metrics = eval_fn(model, device)
            for k, v in eval_metrics.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)
            print(f"  Eval: {eval_metrics}")

    # Save final model
    save_path = os.path.join(config.output_dir, "final_model.pt")
    torch.save(
        {k: v for k, v in model.state_dict().items() if "lora_R" in k or "lora_A" in k or "lora_B" in k},
        save_path,
    )
    print(f"Saved adapter weights to {save_path}")

    return history


def train_glue(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    task_name: str,
    compute_metrics_fn: Any,
) -> Dict[str, Any]:
    """Training loop for GLUE tasks (classification/regression).

    GLUE uses 30 epochs with batch size 30 per the paper.
    """
    model.to(device)
    optimizer = get_optimizer(model, config)

    num_update_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_scheduler(optimizer, config, num_update_steps)

    history: Dict[str, List[float]] = {"loss": [], "eval_metric": []}
    best_metric = -float("inf")
    best_state = None

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        history["loss"].append(avg_loss)

        # Evaluate
        eval_metrics = evaluate_glue(model, eval_dataloader, device, task_name, compute_metrics_fn)
        metric_value = list(eval_metrics.values())[0]
        history["eval_metric"].append(metric_value)

        print(f"Epoch {epoch + 1}/{config.num_epochs} — Loss: {avg_loss:.4f} — {eval_metrics}")

        if metric_value > best_metric:
            best_metric = metric_value
            trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
            best_state = {k: v.clone() for k, v in model.state_dict().items() if k in trainable_names}

    return {"history": history, "best_metric": best_metric}


def evaluate_glue(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task_name: str,
    compute_metrics_fn: Any,
) -> Dict[str, float]:
    """Evaluate model on a GLUE task."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)

            if task_name == "stsb":
                preds = outputs.logits.squeeze(-1)
            else:
                preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return compute_metrics_fn(all_preds, all_labels)
