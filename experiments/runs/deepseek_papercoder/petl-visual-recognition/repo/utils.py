## utils.py

import random
import copy
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def set_seed(seed: int) -> None:
    """
    Set seeds for reproducibility across Python, NumPy, and PyTorch.
    Also configures CUDA deterministic algorithms where possible.

    Args:
        seed: integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms for convolution operations.
        # This may impact performance but improves reproducibility.
        torch.backends.cudnn.deterministic = True
        # Keep benchmark enabled (default) to allow cuDNN to pick fastest algorithms;
        # this is acceptable as we fixed the seed.
        torch.backends.cudnn.benchmark = True
    # For CPU operations (if relevant, e.g., MKL), set the number of threads to 1
    # to avoid nondeterminism from parallelisation. Not strictly required by the paper,
    # but useful for exact reproducibility.
    # torch.set_num_threads(1)


def compute_trainable_params(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in the given model.

    Args:
        model: PyTorch Module.

    Returns:
        Total number of elements in parameters with requires_grad=True.
    """
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
    return total


def merge_dicts(*dicts: dict) -> dict:
    """
    Recursively merge an arbitrary number of dictionaries.
    Later dictionaries take precedence for overlapping keys.
    If both values for a key are dictionaries, they are merged recursively.

    Args:
        *dicts: positional arguments, each a dict to merge.

    Returns:
        A new merged dictionary.
    """
    if not dicts:
        return {}
    merged = copy.deepcopy(dicts[0])
    for d in dicts[1:]:
        for key, value in d.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = merge_dicts(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    epochs: int,
    **kwargs: Any
) -> Tuple[Optimizer, LRScheduler]:
    """
    Create an AdamW optimizer and cosine-annealing learning rate scheduler
    for all trainable parameters of the model.

    Args:
        model: PyTorch Module whose trainable parameters will be optimised.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay (L2 regularisation) coefficient.
        epochs: Total number of training epochs; used for the scheduler's T_max.
        **kwargs: Additional keyword arguments (reserved for future use).

    Returns:
        Tuple (optimizer, scheduler) where scheduler is stepped once per epoch.
    """
    # Collect all trainable parameters (requires_grad == True)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # If no trainable parameters, raise an error (should not happen in practice)
    # but handle gracefully by returning a dummy optimizer/scheduler? We'll raise.
    trainable_params = list(trainable_params)
    if not trainable_params:
        raise ValueError("No trainable parameters found in the model.")

    # Optimizer: AdamW with default betas=(0.9,0.999), eps=1e-8
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay
    )

    # Scheduler: cosine decay without restarts, T_max = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    return optimizer, scheduler
