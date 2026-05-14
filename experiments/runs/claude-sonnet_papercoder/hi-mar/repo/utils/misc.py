## utils/misc.py
"""Shared utility functions and classes for Hi-MAR.

This module provides foundational infrastructure consumed by training, inference,
and evaluation scripts. It has no project-internal dependencies and sits at the
bottom of the dependency graph.
"""

import logging
import math
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_device() -> torch.device:
    """Returns the best available compute device.

    Checks for CUDA availability first. The paper uses H100 GPUs exclusively,
    so CUDA is the expected runtime environment.

    Returns:
        torch.device: "cuda" if a CUDA-capable GPU is available, else "cpu".
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across all relevant RNGs for reproducibility.

    Configures Python's built-in random, NumPy, and PyTorch (CPU + CUDA) RNGs.
    Also enforces deterministic cuDNN behaviour at the cost of some throughput.

    Args:
        seed: Integer seed value. The config specifies seed=42 as the default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Required for multi-GPU distributed runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model.

    Useful for verifying that instantiated models match the parameter counts
    reported in Table 1 of the paper (244M / 529M / 1090M / ~100M).

    Note: The VAE is frozen (requires_grad=False) and lives in VAETokenizer,
    not inside HiMAR, so calling this on a HiMAR instance correctly counts
    only the trainable backbone and diffusion head parameters.

    Args:
        model: Any nn.Module instance.

    Returns:
        Total number of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Tracks a running average of a scalar metric without storing history.

    Typical usage in a training loop::

        meter = AverageMeter()
        for batch in dataloader:
            loss = compute_loss(batch)
            meter.update(loss.item(), n=batch_size)
        print(f"Epoch avg loss: {meter.avg:.4f}")
        meter.reset()

    Attributes:
        val: Most recently observed value.
        avg: Running average over all updates since last reset.
        sum: Cumulative weighted sum since last reset.
        count: Total number of samples accumulated since last reset.
    """

    def __init__(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        """Resets all accumulators to zero. Call at the start of each epoch."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Records a new observation.

        Args:
            val: The scalar value to record. For batch-averaged losses, pass
                the already-reduced mean; the weight ``n`` accounts for the
                actual number of samples that produced that mean.
            n: Number of samples represented by ``val``. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_yaml(data: dict[str, Any], path: str) -> None:
    """Persists a dictionary to a YAML file.

    Creates any missing parent directories automatically. Used by main.py to
    save a copy of the active config alongside each training run for
    reproducibility.

    Args:
        data: Dictionary to serialise. Must contain only YAML-safe types.
        path: Destination file path (e.g. "outputs/run_001/config.yaml").
    """
    dirname: str = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_yaml(path: str) -> dict[str, Any]:
    """Loads a YAML file and returns its contents as a plain dictionary.

    Uses yaml.safe_load to prevent arbitrary code execution from untrusted
    YAML files. The caller (Config.from_yaml) is responsible for validating
    and transforming the returned dict into typed config objects.

    Args:
        path: Path to the YAML file to load.

    Returns:
        Dictionary representation of the YAML file contents.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Configures and returns a named logger with console and file handlers.

    Idempotent: if the named logger already has handlers attached (e.g. from a
    previous call in the same process), the existing logger is returned without
    adding duplicate handlers.

    The config specifies log_dir="outputs/logs"; callers should construct
    ``log_file`` relative to that directory.

    Args:
        name: Logger name (e.g. "trainer", "evaluator"). Appears in each log
            line so different subsystems can be distinguished in shared logs.
        log_file: Absolute or relative path to the log file. Parent directories
            are created automatically.

    Returns:
        Configured logging.Logger instance.
    """
    logger: logging.Logger = logging.getLogger(name)

    # Guard against duplicate handlers when called multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — always present.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — create parent directories if needed.
    log_dirname: str = os.path.dirname(log_file)
    if log_dirname:
        os.makedirs(log_dirname, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def linear_warmup_cosine_decay(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Builds a LambdaLR scheduler with linear warmup and optional cosine decay.

    The paper specifies a "constant lr schedule with a 1e-4 learning rate and
    100-epoch linear warmup" for ImageNet, and a constant schedule with 8K-step
    linear warmup for MS-COCO. In both cases the LR is held constant after the
    warmup phase.

    Behaviour:
        - Steps 0 … warmup_steps-1: LR scales linearly from 0 → base_lr.
        - Steps warmup_steps … total_steps (if total_steps <= warmup_steps):
          LR stays at base_lr (constant schedule as specified in the paper).
        - Steps warmup_steps … total_steps (if total_steps > warmup_steps):
          LR decays from base_lr → 0 following a cosine curve. This branch
          is provided for completeness but is NOT used by the paper's default
          training configurations.

    To reproduce the paper's constant-LR-after-warmup behaviour, callers
    should pass ``total_steps == warmup_steps`` (or any value ≤ warmup_steps).

    Args:
        optimizer: The AdamW optimizer whose LR will be scheduled.
        warmup_steps: Number of linear warmup steps. For ImageNet this is
            ``warmup_epochs * steps_per_epoch``; for COCO it is 8000.
        total_steps: Total training steps. Pass ``warmup_steps`` (or 0) to
            obtain a constant LR after warmup, matching the paper's setting.

    Returns:
        A torch.optim.lr_scheduler.LambdaLR instance ready to be stepped once
        per optimiser step.
    """

    def lr_lambda(current_step: int) -> float:
        """Computes the LR multiplier for the given step."""
        if current_step < warmup_steps:
            # Linear ramp: 0 → 1 over warmup_steps.
            return float(current_step) / float(max(1, warmup_steps))

        if total_steps <= warmup_steps:
            # Constant LR after warmup — the paper's default for both tasks.
            return 1.0

        # Cosine decay from 1.0 → 0.0 over the remaining steps.
        decay_steps: int = total_steps - warmup_steps
        completed_steps: int = current_step - warmup_steps
        progress: float = float(completed_steps) / float(max(1, decay_steps))
        # Clamp progress to [0, 1] to handle steps beyond total_steps gracefully.
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
