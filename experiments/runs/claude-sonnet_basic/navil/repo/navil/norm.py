"""
RMSNorm implementation for compatibility with PyTorch < 2.4.

nn.RMSNorm was added in PyTorch 2.4. This module provides a compatible
implementation that works with older versions.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Reference: https://arxiv.org/abs/1910.07467
    
    This is equivalent to nn.RMSNorm (added in PyTorch 2.4) but works
    with older PyTorch versions.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def get_rms_norm(normalized_shape: int, eps: float = 1e-6) -> nn.Module:
    """
    Get RMSNorm module, using nn.RMSNorm if available (PyTorch >= 2.4),
    otherwise falling back to custom implementation.
    """
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(normalized_shape, eps=eps)
    else:
        return RMSNorm(normalized_shape, eps=eps)
