"""
Utility functions for OLMoE model training and evaluation.

This module provides helper functions for device management, mixed precision
data types, model initialization, and gradient clipping, all essential
for reproducing the OLMoE experiments.
"""

import torch
import torch.nn.init as init
import torch.nn.utils as utils
from typing import Iterable, Union

def get_device(device_str: str = "cuda") -> torch.device:
    """
    Determines and returns the appropriate PyTorch device ('cuda' or 'cpu').

    Args:
        device_str: The desired device as a string (e.g., "cuda", "cpu").
                    Defaults to "cuda" if not specified.

    Returns:
        A torch.device object representing the selected device.
    """
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_mixed_precision_dtype(precision_str: str = "bf16") -> torch.dtype:
    """
    Maps a string representation of mixed precision to its corresponding torch.dtype.

    Args:
        precision_str: The desired precision as a string (e.g., "bf16", "fp16", "fp32").
                       Defaults to "bf16" if not specified.

    Returns:
        A torch.dtype object for the selected precision.
    """
    if precision_str.lower() == "bf16":
        return torch.bfloat16
    elif precision_str.lower() == "fp16":
        return torch.float16
    return torch.float32


def truncated_normal_init_(
    tensor: torch.Tensor,
    std: float = 0.02,
    a: float = -0.06,  # Corresponds to -3 * std (0.02)
    b: float = 0.06,   # Corresponds to +3 * std (0.02)
) -> None:
    """
    Initializes a tensor with values drawn from a truncated normal distribution.

    Values are drawn from a normal distribution N(0, std^2) and then truncated
    to be within the range [a, b].

    Args:
        tensor: The torch.Tensor to be initialized in-place.
        std: The standard deviation of the normal distribution.
        a: The lower bound for truncation.
        b: The upper bound for truncation.
    """
    # torch.nn.init.trunc_normal_ requires PyTorch >= 1.10.0
    init.trunc_normal_(tensor, mean=0.0, std=std, a=a, b=b)


def clip_grad_norm_(
    parameters: Union[Iterable[torch.Tensor], torch.Tensor],
    max_norm: float = 1.0,
) -> float:
    """
    Clips the gradients of model parameters by their global L2 norm.

    Args:
        parameters: An iterable of Tensors or a single Tensor whose gradients
                    are to be clipped.
        max_norm: The maximum allowed value of the gradients' total norm.
                  Defaults to 1.0.

    Returns:
        The total norm of the gradients (before clipping).
    """
    return utils.clip_grad_norm_(parameters, max_norm=max_norm)

