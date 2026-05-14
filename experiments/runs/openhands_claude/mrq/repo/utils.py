import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


# ── symexp / symlog ───────────────────────────────────────────────────────────

def symexp(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * (exp(|x|) - 1)  – inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(|x| + 1)  – compresses large values."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


# ── Reward bin construction ───────────────────────────────────────────────────

def build_reward_bins(num_bins: int, reward_range: float, device: torch.device) -> torch.Tensor:
    """
    Build bin centres spaced uniformly in symlog space over [-reward_range, reward_range].
    Applying symexp gives the actual reward values represented by each bin.
    """
    lin = torch.linspace(-reward_range, reward_range, num_bins, device=device)
    return symexp(lin)


# ── Two-hot encoding ──────────────────────────────────────────────────────────

def two_hot(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Encode scalar values as a two-hot distribution over the provided bin centres.

    Args:
        values: (B,) or (B, 1) tensor of scalar values.
        bins:   (K,) tensor of bin centres (must be sorted ascending).

    Returns:
        (B, K) two-hot encoded tensor.
    """
    values = values.view(-1)
    B, K = values.shape[0], bins.shape[0]

    # Clamp to bin range
    values = values.clamp(bins[0], bins[-1])

    # Find lower bin index for each value
    # searchsorted returns the index where value would be inserted to keep order
    lower_idx = torch.searchsorted(bins, values, right=True) - 1
    lower_idx = lower_idx.clamp(0, K - 2)
    upper_idx = lower_idx + 1

    lower_val = bins[lower_idx]
    upper_val = bins[upper_idx]

    # Linear interpolation weight for upper bin
    span = (upper_val - lower_val).clamp(min=1e-8)
    upper_weight = (values - lower_val) / span
    lower_weight = 1.0 - upper_weight

    target = torch.zeros(B, K, device=values.device, dtype=values.dtype)
    target.scatter_(1, lower_idx.unsqueeze(1), lower_weight.unsqueeze(1))
    target.scatter_(1, upper_idx.unsqueeze(1), upper_weight.unsqueeze(1))
    return target


def two_hot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Decode a categorical distribution over bins to a scalar expected value.

    Args:
        logits: (B, K) raw logits.
        bins:   (K,) bin centres.

    Returns:
        (B,) expected values.
    """
    probs = F.softmax(logits, dim=-1)
    return (probs * bins.unsqueeze(0)).sum(dim=-1)


# ── Reward cross-entropy loss ─────────────────────────────────────────────────

def reward_cross_entropy(
    pred_logits: torch.Tensor,
    target_rewards: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy between predicted reward distribution and two-hot target.

    Args:
        pred_logits:    (B, K) raw logits from the MDP predictor.
        target_rewards: (B,) scalar reward targets.
        bins:           (K,) bin centres.

    Returns:
        Scalar mean loss.
    """
    target = two_hot(target_rewards, bins)                  # (B, K)
    log_probs = F.log_softmax(pred_logits, dim=-1)          # (B, K)
    loss = -(target * log_probs).sum(dim=-1)                # (B,)
    return loss.mean()


# ── Huber loss ────────────────────────────────────────────────────────────────

def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta, reduction="none")


# ── Weight initialisation ─────────────────────────────────────────────────────

def xavier_uniform_init(module: torch.nn.Module) -> None:
    """Apply Xavier uniform initialisation and zero bias to all Linear layers."""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


# ── Normalisation helpers ─────────────────────────────────────────────────────

def compute_mean_abs_reward(rewards: np.ndarray) -> float:
    """Mean absolute reward over a buffer sample – used for reward scaling."""
    mean_abs = float(np.mean(np.abs(rewards)))
    return max(mean_abs, 1e-8)


# ── Action helpers ────────────────────────────────────────────────────────────

def scale_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Map action from [-1, 1] to [low, high]."""
    return low + (action + 1.0) * 0.5 * (high - low)


def unscale_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Map action from [low, high] to [-1, 1]."""
    return 2.0 * (action - low) / (high - low + 1e-8) - 1.0


# ── Seeding ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
