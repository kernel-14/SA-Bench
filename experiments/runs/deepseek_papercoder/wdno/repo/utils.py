## utils.py
"""
General‑purpose utility functions for the WDNO reproduction project.
Provides seed setting, logging configuration, device selection, and tensor normalisation.
All utilities are self‑contained and never import from other project modules.
"""

import random
import logging
import os
import sys
from typing import Optional, Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch to encourage reproducibility.
    Also configures CuDNN to be deterministic (at a performance cost).

    Args:
        seed: Integer seed value (e.g., 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi‑GPU safe

    # Enable deterministic behaviour in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure the root logger with a consistent format and optional file output.
    Removes any previously attached handlers to avoid duplicate messages.

    Args:
        log_level: Logging level, e.g. logging.DEBUG, logging.INFO.
        log_file: If given, a file path where logs will be appended.

    Returns:
        The root logger (standard library). Global logging module is also configured.
    """
    logger = logging.getLogger()
    # Remove all existing handlers to ensure a clean state
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # File handler (optional)
    handlers: list[logging.Handler] = [console_handler]
    if log_file is not None:
        # Create parent directories if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(log_level)
    return logger


def get_device(device_str: str) -> torch.device:
    """
    Return a PyTorch device object, falling back to CPU if CUDA is requested but unavailable.

    Args:
        device_str: Device string, e.g. 'cuda', 'cuda:0', 'cpu'.

    Returns:
        Corresponding torch.device.

    Raises:
        RuntimeError: If a specific GPU (e.g. 'cuda:2') is requested but not available.
    """
    device_str = device_str.strip().lower()

    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            logging.warning(
                "CUDA requested but not available. Falling back to CPU."
            )
            return torch.device("cpu")

    if device_str.startswith("cuda:"):
        # For explicit GPU index
        index = int(device_str.split(":")[1])
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device index {index} but only "
                f"{torch.cuda.device_count()} GPU(s) are available."
            )
        return torch.device(device_str)

    # Default: treat any other string as is (could be cpu, cpu:0, etc.)
    return torch.device(device_str)


def normalize_tensor(
    x: torch.Tensor,
    dim: Optional[int] = None,
    eps: float = 1e-8,
    unbiased: bool = True,
) -> torch.Tensor:
    """
    Normalise a tensor to have zero mean and unit variance.
    Normalisation can be performed over the whole tensor or along a specific dimension.

    Args:
        x: Input tensor.
        dim: Dimension along which to compute mean and std (default: overall mean/std).
        eps: Small value to avoid division by zero.
        unbiased: If True, use Bessel's correction for std (n-1). The paper does not specify,
                  but True is the safer default for statistical normalisation.

    Returns:
        Normalised tensor with the same shape as x.
    """
    if dim is None:
        # Overall normalisation
        mean = x.mean()
        std = x.std(unbiased=unbiased)
    else:
        # Compute statistics along specified dimension, keeping dimensions for broadcasting
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True, unbiased=unbiased)

    return (x - mean) / (std + eps)
