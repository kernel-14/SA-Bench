## utils.py

"""
Utility functions for Masked Diffusion Models.

Provides noise schedule computations, masking primitives, and numerical
helpers. All functions operate on PyTorch tensors and are stateless.
"""

import math
from typing import Tuple

import torch


# Token IDs (must match configuration)
MASK_TOKEN_ID: int = 0      # The mask token
PAD_TOKEN_ID: int = 2       # Padding token for L&O‑NAE‑SAT sequences


# ----------------------------------------------------------------------
# Noise schedule: alpha(t) = cos(pi/2 * t),  t in [0,1]
# ----------------------------------------------------------------------

def alpha(t: torch.Tensor) -> torch.Tensor:
    """
    Compute alpha(t) = cos(pi/2 * t).
    Args:
        t: Tensor of any shape, values in [0,1].
    Returns:
        Tensor of same shape, alpha(t).
    """
    return torch.cos((math.pi / 2.0) * t)


def alpha_prime(t: torch.Tensor) -> torch.Tensor:
    """
    Derivative of alpha(t) w.r.t. t:
        alpha'(t) = -pi/2 * sin(pi/2 * t)
    Args:
        t: Tensor of any shape, values in [0,1].
    Returns:
        Tensor of same shape, alpha'(t).
    """
    return - (math.pi / 2.0) * torch.sin((math.pi / 2.0) * t)


def noise_weight(t: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Compute the score‑entropy loss weighting factor:
        w(t) = |alpha'(t)| / (1 - alpha(t))
    Numerator is positive because alpha'(t) is negative.
    The denominator is clamped to avoid division by zero near t=0.
    Args:
        t: Tensor of any shape.
        eps: Minimum value for (1 - alpha(t)). Default 1e-4.
    Returns:
        Tensor of same shape, positive weight.
    """
    one_minus_alpha = torch.clamp(1.0 - alpha(t), min=eps)
    weight = -alpha_prime(t) / one_minus_alpha   # -alpha' is positive
    return weight


def compute_K(num_masked: int, t: torch.Tensor, s: torch.Tensor, eps: float = 1e-4) -> int:
    """
    Deterministically compute how many tokens to unmask when going from
    noise level t to s (s < t).
    K = round( num_masked * (alpha(s) - alpha(t)) / (1 - alpha(t)) )
    Args:
        num_masked: integer number of currently masked tokens.
        t: scalar tensor, current time (larger).
        s: scalar tensor, next time (smaller).
        eps: clamp for denominator to avoid division by zero.
    Returns:
        int K.
    """
    with torch.no_grad():
        one_minus_alpha_t = torch.clamp(1.0 - alpha(t), min=eps)
        frac = (alpha(s) - alpha(t)) / one_minus_alpha_t
        K = int(torch.round(num_masked * frac).item())
    return max(0, K)   # Ensure non‑negative


# ----------------------------------------------------------------------
# Masking operations
# ----------------------------------------------------------------------

def mask_tokens(x: torch.Tensor, mask_prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Independently mask each token with probability `mask_prob`.
    Args:
        x: Long tensor of shape (B, L), token indices.
        mask_prob: float in [0,1], probability of masking a token.
    Returns:
        x_masked: same shape, tokens replaced by MASK_TOKEN_ID.
        masked_mask: boolean tensor (same shape), True where token was masked.
    """
    # Random mask
    rand = torch.rand(x.shape, device=x.device, dtype=torch.float32)
    mask = rand < mask_prob

    x_masked = x.clone()
    x_masked[mask] = MASK_TOKEN_ID
    return x_masked, mask


def create_mask(seq_len: int, num_mask: int) -> torch.Tensor:
    """
    Create a boolean mask of length `seq_len` with exactly `num_mask`
    randomly chosen entries set to True.
    Args:
        seq_len: total sequence length.
        num_mask: number of positions to mask.
    Returns:
        1‑D tensor of shape (seq_len,), dtype torch.bool.
    """
    if num_mask <= 0:
        return torch.zeros(seq_len, dtype=torch.bool)
    if num_mask >= seq_len:
        return torch.ones(seq_len, dtype=torch.bool)

    perm = torch.randperm(seq_len)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[perm[:num_mask]] = True
    return mask


# ----------------------------------------------------------------------
# Numerical helper
# ----------------------------------------------------------------------

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable computation of log(1 - exp(x)) for x <= 0.
    Args:
        x: Tensor with elements <= 0.
    Returns:
        Tensor of the same shape, log(1 - exp(x)).
    """
    # Use two regimes for precision
    # For x < log(2): use log(-expm1(x))
    # For x >= log(2): use log1p(-exp(x))
    log2 = math.log(2)
    mask = x < log2
    # Safe implementation using where
    return torch.where(
        mask,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x))
    )
