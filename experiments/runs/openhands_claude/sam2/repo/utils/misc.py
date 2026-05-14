"""
Miscellaneous utilities for SAM 2 training.

Includes:
  - Layer-wise learning rate decay (LLRD) for AdamW optimizer
  - Reciprocal square-root learning rate schedule
  - Checkpoint saving/loading
  - Logging utilities
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Layer-wise learning rate decay
# ---------------------------------------------------------------------------

def get_layer_id_for_hiera(name: str, num_layers: int) -> int:
    """
    Assign a layer ID to each parameter in the Hiera image encoder
    for layer-wise learning rate decay.

    Layer 0: patch embedding + positional embedding
    Layer 1..num_layers-1: transformer blocks
    Layer num_layers: head / FPN
    """
    if "patch_embed" in name or "pos_embed" in name:
        return 0
    if "stages" in name:
        # Extract stage and block index
        parts = name.split(".")
        try:
            stage_idx = int(parts[parts.index("stages") + 1])
            block_idx = int(parts[parts.index("stages") + 2])
            # Assign layer IDs proportionally
            return 1 + stage_idx * 4 + block_idx
        except (ValueError, IndexError):
            return num_layers // 2
    return num_layers


def build_optimizer_with_layer_decay(
    model: nn.Module,
    base_lr: float = 4e-4,
    weight_decay: float = 0.1,
    layer_decay: float = 0.9,
    encoder_variant: str = "B+",
) -> torch.optim.AdamW:
    """
    Build AdamW optimizer with layer-wise learning rate decay on the image encoder.

    Layer decay values (Table 12):
      T, S: 0.8
      B+:   0.9
      L:    0.925
    """
    # Determine number of encoder layers
    num_encoder_layers = {
        "T": 24, "S": 24, "B+": 24, "L": 48
    }.get(encoder_variant, 24)

    param_groups: List[Dict] = []
    seen_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_params:
            continue
        seen_params.add(id(param))

        # No weight decay for bias, LayerNorm, and embedding parameters
        no_decay = (
            "bias" in name
            or "norm" in name.lower()
            or "embed" in name
            or param.ndim == 1
        )

        # Compute layer-wise LR scale for image encoder
        if "image_encoder" in name:
            layer_id = get_layer_id_for_hiera(name, num_encoder_layers)
            lr_scale = layer_decay ** (num_encoder_layers - layer_id)
        else:
            lr_scale = 1.0

        param_groups.append({
            "params": [param],
            "lr": base_lr * lr_scale,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "lr_scale": lr_scale,
        })

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


# ---------------------------------------------------------------------------
# Reciprocal square-root learning rate schedule
# ---------------------------------------------------------------------------

def reciprocal_sqrt_schedule(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    timescale: int = 1000,
    cooldown_steps: int = 5000,
    total_steps: int = 90000,
) -> LambdaLR:
    """
    Reciprocal square-root schedule with linear warmup and cooldown.
    (Zhai et al., 2022; Table 12)

    lr(t) = base_lr * sqrt(timescale / max(t, timescale))
    with linear warmup for warmup_steps and linear cooldown at the end.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        cooldown_start = total_steps - cooldown_steps
        if step >= cooldown_start:
            progress = float(step - cooldown_start) / float(max(1, cooldown_steps))
            # Compute the value at cooldown_start
            t = cooldown_start
            base = math.sqrt(timescale / max(t, timescale))
            return base * (1.0 - progress)

        return math.sqrt(timescale / max(step, timescale))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def clip_gradients(model: nn.Module, max_norm: float = 0.1) -> float:
    """Clip gradients by L2 norm. Returns the norm before clipping."""
    return nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    # Handle DataParallel / DistributedDataParallel prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
    if missing:
        logging.warning(f"Missing keys: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys: {unexpected}")

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = "sam2", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


class AverageMeter:
    """Tracks running average of a scalar metric."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class MetricLogger:
    """Tracks multiple AverageMeters."""

    def __init__(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k)
            self.meters[k].update(v)

    def __str__(self) -> str:
        return "  ".join(str(m) for m in self.meters.values())

    def get(self, key: str) -> float:
        return self.meters[key].avg if key in self.meters else 0.0
