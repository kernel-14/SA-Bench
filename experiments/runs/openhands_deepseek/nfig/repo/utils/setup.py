"""
Training utilities: checkpoint management, logging, and setup.
"""

import os
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional
import logging


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossTracker:
    """Track multiple losses during training."""

    def __init__(self):
        self.meters = {}

    def update(self, losses: Dict[str, float], n: int = 1):
        for name, value in losses.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter()
            self.meters[name].update(value, n)

    def get_avg(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.meters.items()}

    def reset(self):
        for meter in self.meters.values():
            meter.reset()


def setup_logging(output_dir: str, name: str = "nfig") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    if output_dir:
        fh = logging.FileHandler(os.path.join(output_dir, "training.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    return logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    path: str,
    extra_state: Optional[Dict[str, Any]] = None,
):
    """Save a training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if extra_state:
        checkpoint.update(extra_state)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Load a training checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def setup_training(
    config,
    seed: int = 42,
) -> Dict[str, Any]:
    """Set up training environment."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(config.output_dir)

    return {"device": device, "logger": logger}


class EMAModel:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
