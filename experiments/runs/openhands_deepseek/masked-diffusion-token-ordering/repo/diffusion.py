import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Callable
from config import DiffusionConfig


def linear_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """α_t = 1 - t (linear mask probability)"""
    return 1.0 - t


def loglinear_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """α_t = exp(-β t) style schedule; use a smooth log-linear decay."""
    # α_0 ≈ 1, α_1 ≈ 0
    beta = 12.0
    return torch.exp(-beta * t)


def cosine_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """α_t = cos(πt/2) schedule."""
    return torch.cos(torch.pi * t / 2.0)


def get_noise_schedule(cfg: DiffusionConfig) -> Callable[[torch.Tensor], torch.Tensor]:
    if cfg.noise_schedule == "linear":
        base_schedule = linear_noise_schedule
    elif cfg.noise_schedule == "cosine":
        base_schedule = cosine_noise_schedule
    else:
        base_schedule = loglinear_noise_schedule

    # Rescale to satisfy α_0 ≈ cfg.alpha_0, α_1 ≈ cfg.alpha_1
    def schedule(t: torch.Tensor) -> torch.Tensor:
        base = base_schedule(t)
        # interpolate: base has range from base(0)≈1 to base(1)≈0
        return cfg.alpha_0 - (cfg.alpha_0 - cfg.alpha_1) * (1.0 - base)
    return schedule


def sample_mask_indices(
    batch_size: int,
    seq_len: int,
    mask_prob: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample which tokens to mask, given per-sequence masking probability."""
    # mask_prob shape: (B,)
    rand = torch.rand(batch_size, seq_len, device=device)
    threshold = mask_prob.view(-1, 1).expand(batch_size, seq_len)
    return rand < threshold


def forward_mask(
    x_0: torch.Tensor,
    mask_token: int,
    t_values: torch.Tensor,
    noise_schedule_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the forward masking process.
    x_0: clean data, shape (B, L)
    t_values: noise level t per sample, shape (B,)
    Returns: (x_t, mask_indicator, alpha_t)
        x_t: masked sequence
        mask_indicator: bool tensor, True where masked
        alpha_t: α_t per sample
    """
    B, L = x_0.shape
    alpha_t = noise_schedule_fn(t_values)
    mask_prob = 1.0 - alpha_t  # probability a token is masked
    mask = sample_mask_indices(B, L, mask_prob, x_0.device)
    x_t = x_0.clone()
    x_t[mask] = mask_token
    return x_t, mask, alpha_t


def d_alpha_dt(
    t: torch.Tensor,
    noise_schedule_fn: Callable,
) -> torch.Tensor:
    """Compute derivative α'_t using finite differences."""
    eps = 1e-4
    alpha_plus = noise_schedule_fn(t + eps)
    alpha_minus = noise_schedule_fn(t - eps)
    return (alpha_plus - alpha_minus) / (2 * eps)


def score_entropy_loss(
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    mask_indicator: torch.Tensor,
    alpha_t: torch.Tensor,
    t_values: torch.Tensor,
    logits: torch.Tensor,
    noise_schedule_fn: Callable,
    mask_token: int,
) -> torch.Tensor:
    """
    Compute the score-entropy loss from Equation (1):
    L_θ = ∫ α'_t / (1 - α_t) 𝔼_{x_t} Σ_{i: x_t^i=0} -log p_θ(x_0^i | x_t, t) dt

    Monte Carlo estimate: sample t ~ Uniform(0,1), independently mask each token
    with prob 1-α_t, then weight per-sample contribution by α'_t/(1-α_t).
    logits: output from model, shape (B, L, V+1), includes mask token logits
    """
    B, L = x_0.shape
    alpha_prime = d_alpha_dt(t_values, noise_schedule_fn)
    denom = (1.0 - alpha_t).clamp(min=1e-8)
    weight = alpha_prime / denom  # (B,)

    vocab_size = logits.shape[-1] - 1
    pred_logits = logits[:, :, :vocab_size]  # (B, L, V)

    num_masked = mask_indicator.sum().float().clamp(min=1)

    target = x_0[mask_indicator]
    pred_logits_masked = pred_logits[mask_indicator]

    ce = F.cross_entropy(pred_logits_masked, target.long(), reduction="none")

    # Expand per-sample weight to each masked position within that sample
    b_idx = torch.arange(B, device=x_0.device).unsqueeze(1).expand(B, L)[mask_indicator]
    ce_weighted = ce * weight[b_idx]

    # Average over batch dimension
    loss = ce_weighted.sum() / B
    return loss


def compute_pi_learner_loss(
    x_0: torch.Tensor,
    logits: torch.Tensor,
    pi: torch.Tensor,
    mask_token: int,
) -> torch.Tensor:
    """
    Compute the π-learner loss as in Equation (3):
    log p_θ(x_0) = Σ_i log p_θ(x_0^{π(i)} | x_0[π{i,...,L-1}])

    x_0: (B, L) original tokens
    logits: (B, L, V) output from model run on partially masked input
    pi: (L,) permutation
    Returns: per-batch averaged negative log-likelihood
    """
    B, L, V = logits.shape
    # Reorder both x_0 and logits according to π
    x_pi = x_0[:, pi]  # (B, L)
    logits_pi = logits[:, pi, :]  # (B, L, V)

    losses = []
    for i in range(L):
        # predict token at position i given positions i..L-1 are masked
        # But in practice, we already have logits from causal forward pass
        # on permuted sequence
        loss_i = F.cross_entropy(
            logits_pi[:, i, :].contiguous(), x_pi[:, i].long(), reduction="none"
        )
        losses.append(loss_i)

    losses = torch.stack(losses, dim=1)  # (B, L)
    total_loss = losses.sum(dim=1)  # (B,)
    return total_loss.mean()


def extract_logits_for_mask(
    model: nn.Module,
    x_t: torch.Tensor,
    mask_indicator: torch.Tensor,
) -> torch.Tensor:
    """
    Get model logits for the clean token predictions.
    Returns logits for all positions.
    """
    logits = model(x_t)  # (B, L, V+1)
    return logits
