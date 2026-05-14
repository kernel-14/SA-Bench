"""
Masked Diffusion Model (MDM) — model wrapper, noise schedule, and training loss.

Implements the framework from:
  Shi et al. (2024) "Simplified and Generalized Masked Diffusion for Discrete Data"
  Sahoo et al. (2025) "Simple and Effective Masked Diffusion Language Models"

The MDM loss (Eq. 1 in the paper) is:
  L_θ = ∫_0^1 [α_t' / (1 - α_t)] E_{x_t ~ p_data} Σ_{i: x_t^i=0} -log p_θ(x_0^i | x_t, t) dt

Under the time-embedding-free assumption (p_θ(·|x_t,t) = p_θ(·|x_t)), this is equivalent
to the any-order autoregressive loss (Proposition 2.1):
  L_θ = -Σ_{M⊆[L], i∈M} (1/|M|) * (1/C(L,|M|)) * E[log p_θ(x_0^i | x_0[M])]
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Transformer, build_mdm_transformer
from config import NoiseScheduleConfig, MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Noise schedule: α_t
# ---------------------------------------------------------------------------

class NoiseSchedule(nn.Module):
    """
    Predefined noise schedule α_t with α_0 ≈ 1 (clean) and α_1 ≈ 0 (fully masked).

    Linear schedule: α_t = 1 - t
    Cosine schedule: α_t = cos²(π/2 · t)
    """

    def __init__(self, schedule_type: str = "linear"):
        super().__init__()
        self.schedule_type = schedule_type

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute α_t for a batch of noise levels t ∈ [0, 1]."""
        if self.schedule_type == "linear":
            return 1.0 - t
        elif self.schedule_type == "cosine":
            return torch.cos(math.pi / 2.0 * t) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        """Compute dα_t/dt."""
        if self.schedule_type == "linear":
            return -torch.ones_like(t)
        elif self.schedule_type == "cosine":
            return -math.pi * torch.sin(math.pi * t) * torch.cos(math.pi / 2.0 * t)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the loss weight α_t' / (1 - α_t) from Eq. 1."""
        alpha_t = self.alpha(t)
        alpha_t_prime = self.alpha_prime(t)
        # α_t' is negative (α decreases), so |α_t'| / (1 - α_t)
        return -alpha_t_prime / (1.0 - alpha_t + 1e-8)


# ---------------------------------------------------------------------------
# Forward process: q_{t|0}(x_t | x_0)
# ---------------------------------------------------------------------------

def sample_masked_sequence(
    x0: torch.Tensor,
    alpha_t: torch.Tensor,
    mask_token_id: int = 0,
) -> torch.Tensor:
    """
    Sample x_t from the forward process q_{t|0}(x_t | x_0).

    Each token x_0^i is independently masked with probability (1 - α_t):
      x_t^i = mask_token  with prob (1 - α_t)
      x_t^i = x_0^i       with prob α_t

    Args:
        x0:           (B, L) clean token sequences
        alpha_t:      (B,) or scalar noise level α_t ∈ [0, 1]
        mask_token_id: integer id of the mask token (0 in the paper)

    Returns:
        x_t:          (B, L) partially masked sequences
    """
    B, L = x0.shape
    if alpha_t.dim() == 0:
        alpha_t = alpha_t.expand(B)
    alpha_t = alpha_t.view(B, 1).expand(B, L)

    keep_mask = torch.bernoulli(alpha_t).bool()
    x_t = torch.where(keep_mask, x0, torch.full_like(x0, mask_token_id))
    return x_t


# ---------------------------------------------------------------------------
# MDM training loss
# ---------------------------------------------------------------------------

def mdm_loss(
    model: Transformer,
    x0: torch.Tensor,
    noise_schedule: NoiseSchedule,
    mask_token_id: int = 0,
    num_time_samples: int = 1,
) -> torch.Tensor:
    """
    Compute the MDM training loss (Eq. 1 in the paper).

    In practice (time-embedding-free), we:
    1. Sample t ~ Uniform(0, 1)
    2. Compute α_t and mask x_0 → x_t
    3. Predict x_0^i for all masked positions i
    4. Weight by α_t' / (1 - α_t)

    This is equivalent to the any-order autoregressive loss (Proposition 2.1).

    Args:
        model:            the denoising network p_θ(·|x_t)
        x0:               (B, L) clean token sequences
        noise_schedule:   NoiseSchedule instance
        mask_token_id:    integer id of the mask token
        num_time_samples: number of t samples per batch element (default 1)

    Returns:
        scalar loss
    """
    B, L = x0.shape
    device = x0.device
    total_loss = torch.tensor(0.0, device=device)
    total_masked = torch.tensor(0.0, device=device)

    for _ in range(num_time_samples):
        t = torch.rand(B, device=device)
        alpha_t = noise_schedule.alpha(t)

        x_t = sample_masked_sequence(x0, alpha_t, mask_token_id)
        masked_positions = (x_t == mask_token_id)

        if not masked_positions.any():
            continue

        logits = model(x_t)  # (B, L, vocab_size)

        # Cross-entropy loss only at masked positions
        loss = F.cross_entropy(
            logits[masked_positions],
            x0[masked_positions],
            reduction="sum",
        )

        # Weight by α_t' / (1 - α_t) averaged over masked positions per sample
        weight = noise_schedule.loss_weight(t)  # (B,)
        # Normalize: sum over masked positions, weight by per-sample weight
        n_masked_per_sample = masked_positions.float().sum(dim=1)  # (B,)
        weighted_loss = (weight * n_masked_per_sample).sum()

        # Combine: loss is already summed over masked positions
        # We want E_t[weight * (1/n_masked) * sum_masked -log p]
        # = E_t[weight * mean_masked -log p]
        # Implemented as: sum_masked(-log p) / total_masked * mean_weight
        total_loss = total_loss + loss
        total_masked = total_masked + masked_positions.float().sum()

    if total_masked > 0:
        return total_loss / total_masked / num_time_samples
    return total_loss


def mdm_loss_simple(
    model: Transformer,
    x0: torch.Tensor,
    mask_token_id: int = 0,
    mask_prob: Optional[float] = None,
) -> torch.Tensor:
    """
    Simplified MDM loss: randomly mask each token independently with probability
    drawn from Uniform(0, 1), then predict masked tokens.

    This is the practical implementation used in most MDM papers (MDLM, SEDD).
    Equivalent to the full loss under the time-embedding-free assumption.

    Args:
        model:        the denoising network p_θ(·|x_t)
        x0:           (B, L) clean token sequences
        mask_token_id: integer id of the mask token
        mask_prob:    if provided, use fixed masking probability; otherwise sample per-batch

    Returns:
        scalar loss (mean cross-entropy over masked positions)
    """
    B, L = x0.shape
    device = x0.device

    if mask_prob is None:
        # Sample a single masking probability per sequence
        p = torch.rand(B, 1, device=device).expand(B, L)
    else:
        p = torch.full((B, L), mask_prob, device=device)

    mask = torch.bernoulli(p).bool()

    # Ensure at least one token is masked per sequence
    for i in range(B):
        if not mask[i].any():
            idx = torch.randint(L, (1,))
            mask[i, idx] = True

    x_t = x0.clone()
    x_t[mask] = mask_token_id

    logits = model(x_t)  # (B, L, vocab_size)

    loss = F.cross_entropy(
        logits[mask],
        x0[mask],
        reduction="mean",
    )
    return loss


# ---------------------------------------------------------------------------
# MDM model wrapper
# ---------------------------------------------------------------------------

class MDM(nn.Module):
    """
    Masked Diffusion Model wrapper.

    Wraps a bidirectional transformer with the MDM noise schedule and loss.
    The denoising network is time-embedding-free: p_θ(·|x_t) (no explicit t input).
    """

    MASK_TOKEN_ID = 0

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        model_config: dict,
        noise_schedule_type: str = "linear",
        use_rope: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.noise_schedule = NoiseSchedule(noise_schedule_type)

        self.transformer = build_mdm_transformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            model_config=model_config,
            use_rope=use_rope,
        )

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Run the denoising network.

        Args:
            x_t: (B, L) partially masked token sequences

        Returns:
            logits: (B, L, vocab_size) — predicted distribution over x_0
        """
        return self.transformer(x_t)

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Compute MDM training loss for a batch of clean sequences."""
        return mdm_loss_simple(self.transformer, x0, self.MASK_TOKEN_ID)

    def get_token_probs(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Get predicted token probabilities for all positions.

        Args:
            x_t: (B, L) partially masked sequences

        Returns:
            probs: (B, L, vocab_size) — softmax probabilities
        """
        logits = self.forward(x_t)
        return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        return self.transformer.count_parameters()

    @torch.no_grad()
    def sample_masked_sequence(
        self, x0: torch.Tensor, t: float
    ) -> torch.Tensor:
        """Sample x_t from the forward process at noise level t."""
        alpha_t = self.noise_schedule.alpha(
            torch.tensor(t, device=x0.device)
        )
        return sample_masked_sequence(x0, alpha_t, self.MASK_TOKEN_ID)
