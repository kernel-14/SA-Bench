"""
Training utilities for LoRA-SB.

Provides training loops and configuration for fine-tuning models with LoRA-SB.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import logging
from .lora_sb_layer import LoRA_SB_Layer
from .gradient_opt import LoRASBOptimizerWrapper

logger = logging.getLogger(__name__)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_lora_sb_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters specific to LoRA-SB layers.

    Returns:
        Dict with counts for:
        - 'total': total trainable parameters
        - 'R_params': parameters in R matrices
        - 'B_params': parameters in B matrices (frozen)
        - 'A_params': parameters in A matrices (frozen)
        - 'rank': the rank used
    """
    stats = {'R_params': 0, 'B_params': 0, 'A_params': 0}
    rank = None

    for module in model.modules():
        if isinstance(module, LoRA_SB_Layer):
            stats['R_params'] += module.R.numel()
            stats['B_params'] += module.B.numel()
            stats['A_params'] += module.A.numel()
            if rank is None:
                rank = module.rank

    stats['total'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats['rank'] = rank
    return stats


def get_lora_sb_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer for LoRA-SB training.

    Only the R matrices are trainable; B and A are frozen buffers.

    Args:
        model: Model with LoRA-SB layers.
        lr: Learning rate.
        weight_decay: Weight decay.
        betas: Adam betas.
        eps: Adam epsilon.

    Returns:
        AdamW optimizer.
    """
    # Only optimize R parameters (and any other trainable params like biases)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found. Make sure LoRA-SB layers "
            "are properly applied with R as nn.Parameter."
        )

    logger.info(f"Optimizing {sum(p.numel() for p in trainable_params)} parameters")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    return optimizer


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[Any] = None,
    gradient_opt_wrapper: Optional[LoRASBOptimizerWrapper] = None,
    max_steps: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
    log_interval: int = 10,
) -> Dict[str, float]:
    """
    Train for one epoch with LoRA-SB.

    Args:
        model: The model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        lr_scheduler: Learning rate scheduler.
        gradient_opt_wrapper: Optional wrapper for optimal gradient transformation.
        max_steps: Maximum number of training steps.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        clip_grad_norm: Max gradient norm for clipping.
        log_interval: Log every N steps.

    Returns:
        Dict with average loss and other metrics.
    """
    model.train()
    device = next(model.parameters()).device

    total_loss = 0.0
    num_steps = 0

    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Apply optimal gradient transformation if enabled
        if gradient_opt_wrapper is not None:
            gradient_opt_wrapper.apply_optimal_gradients()

        # Gradient clipping
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm
            )

        # Step optimizer
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        num_steps += 1

        if (step + 1) % log_interval == 0:
            logger.info(
                f"Step {step + 1}: loss = {total_loss / num_steps:.4f}"
            )

    avg_loss = total_loss / num_steps if num_steps > 0 else float('inf')
    return {'loss': avg_loss, 'steps': num_steps}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate the model.

    Args:
        model: The model.
        dataloader: Evaluation data loader.
        max_steps: Maximum evaluation steps.

    Returns:
        Dict with evaluation metrics.
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    num_steps = 0

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            total_loss += loss.item()
            num_steps += 1

    avg_loss = total_loss / num_steps if num_steps > 0 else float('inf')
    return {'loss': avg_loss, 'steps': num_steps}


def merge_and_save(model: nn.Module, path: str):
    """
    Merge LoRA-SB weights into the original weights and save.

    This converts LoRA-SB layers to regular nn.Linear layers by computing
    W_merged = W0 + s * B @ R @ A.

    Args:
        model: Model with LoRA-SB layers.
        path: Path to save the merged model.
    """
    import copy
    merged_model = copy.deepcopy(model)

    # Replace LoRA-SB layers with merged linear layers
    for name, module in list(merged_model.named_modules()):
        if isinstance(module, LoRA_SB_Layer):
            # Get parent and replace
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = merged_model.get_submodule(parent_name)
            else:
                parent = merged_model

            merged_linear = module.merge()
            setattr(parent, child_name, merged_linear)

    torch.save(merged_model.state_dict(), path)
    logger.info(f"Merged model saved to {path}")
