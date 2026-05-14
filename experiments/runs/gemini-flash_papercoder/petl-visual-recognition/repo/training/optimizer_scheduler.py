## training/optimizer_scheduler.py

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Iterator, Dict, Any, Tuple


def create_optimizer_scheduler(
    model_parameters: Iterator[nn.Parameter],
    total_steps: int,
    lr: float,
    weight_decay: float,
    config: Dict[str, Any]
) -> Tuple[Optimizer, LRScheduler]:
    """
    Creates and returns an optimizer and a learning rate scheduler based on the
    provided configuration and training parameters.

    Args:
        model_parameters (Iterator[nn.Parameter]): An iterator over the model's
                                                    trainable parameters.
        total_steps (int): The total number of training steps (epochs * steps_per_epoch).
                           Used as T_max for the CosineAnnealingLR scheduler.
        lr (float): The base learning rate for the optimizer.
        weight_decay (float): The weight decay value for the optimizer.
        config (Dict[str, Any]): The full configuration dictionary, used to extract
                                 optimizer and scheduler specific parameters.

    Returns:
        Tuple[Optimizer, LRScheduler]: A tuple containing the initialized optimizer
                                       and learning rate scheduler.

    Raises:
        ValueError: If an unsupported optimizer or scheduler type is specified
                    in the configuration.
    """

    # --- Optimizer Initialization ---
    optimizer_name: str = config['training']['optimizer']
    optimizer_params: Dict[str, Any] = config['training']['optimizer_params']

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params=model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(optimizer_params.get('betas', [0.9, 0.999])), # Ensure betas is a tuple
            eps=optimizer_params.get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_name}. Only 'AdamW' is supported.")

    # --- Scheduler Initialization ---
    scheduler_name: str = config['training']['scheduler']

    if scheduler_name == "cosine":
        # CosineAnnealingLR requires T_max (total number of iterations)
        # eta_min sets the minimum learning rate, defaulting to 0.0 for cosine decay to zero.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_steps,
            eta_min=0.0
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_name}. Only 'cosine' is supported.")

    return optimizer, scheduler

