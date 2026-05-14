## samplers.py
"""
Samplers for masked diffusion models.

Implements vanilla and adaptive inference strategies (top‑probability,
top‑probability margin) as described in "Train for the Worst, Plan for the Best".
The reverse diffusion loop is shared; only the position selection policy differs.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Gumbel

from configs import ExperimentConfig
from utils import MASK_TOKEN_ID, alpha


# ---------------------------------------------------------------------------
# Abstract base sampler
# ---------------------------------------------------------------------------

class Sampler(ABC):
    """
    Base class for MDM inference samplers.

    The :meth:`sample` method runs the discrete‑time reverse process,
    delegating the choice of **which** positions to unmask to
    :meth:`select_positions`.

    Args:
        model: A denoising network (``MDMTransformer`` or compatible)
               that returns logits over real tokens via ``get_logits(x)``.
        config: :class:`ExperimentConfig` instance holding diffusion and
                sampling hyperparameters.
    """

    def __init__(self, model: torch.nn.Module, config: ExperimentConfig) -> None:
        self.model = model
        self.config = config
        self.num_steps: int = config.diffusion.inference_steps
        self.mask_token_id: int = config.diffusion.mask_token_id
        self.temperature: float = config.diffusion.sampling_temperature

    # ------------------------------------------------------------------
    # Abstract position selection
    # ------------------------------------------------------------------

    @abstractmethod
    def select_positions(
        self,
        logits: torch.Tensor,       # (B, L, V)
        masked_mask: torch.Tensor,  # (B, L) bool, True = currently masked
        K: int,                     # total number of masked positions to select
    ) -> torch.Tensor:              # (B, L) boolean selection mask
        """
        Choose exactly ``K`` positions among those that are currently
        masked. The returned mask must have ``True`` only where
        ``masked_mask`` is ``True``, and exactly ``K`` ``True`` entries
        (fewer if ``K`` exceeds the number of available masked positions).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Full reverse diffusion loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, x_masked: torch.Tensor) -> torch.Tensor:
        """
        Perform the reverse diffusion for ``self.num_steps`` iterations,
        starting from a (part‑)masked sequence.

        Args:
            x_masked: Long tensor ``(B, L)``. Positions holding
                      ``self.mask_token_id`` are considered masked;
                      other positions are frozen (e.g., clue tokens).

        Returns:
            Long tensor ``(B, L)`` with no mask tokens remaining.
        """
        device = x_masked.device
        x = x_masked.clone()
        B, L = x.shape
        T = self.num_steps

        for step in range(1, T + 1):
            # Current noise levels
            t_val = 1.0 - (step - 1) / T
            s_val = 1.0 - step / T
            t_tensor = torch.tensor(t_val, device=device)
            s_tensor = torch.tensor(s_val, device=device)

            alpha_t = alpha(t_tensor)       # scalar
            alpha_s = alpha(s_tensor)       # scalar

            # Model prediction
            logits = self.model.get_logits(x)              # (B, L, V)

            # Apply sampling temperature
            if self.temperature != 1.0:
                logits = logits / self.temperature

            # Identify currently masked positions
            masked_mask = x == self.mask_token_id          # (B, L)

            # How many tokens to unmask in this step
            total_masked = masked_mask.long().sum().item()
            if total_masked == 0:
                break

            # Deterministic K as described in the paper:
            # K = round(total_masked * (α_s - α_t) / (1 - α_t))
            denom = 1.0 - alpha_t.item() + 1e-6
            frac = max(0.0, (alpha_s.item() - alpha_t.item())) / denom
            K = int(torch.round(torch.tensor(total_masked * frac)).item())
            K = min(K, total_masked)                       # safety clip

            # Select which positions to fill
            selected_mask = self.select_positions(logits, masked_mask, K)  # (B, L)

            # Sample token values for selected positions
            probs = torch.softmax(logits, dim=-1)          # (B, L, V)
            dist = Categorical(probs)
            sampled_tokens = dist.sample()                 # (B, L)

            # Update the sequence only at selected positions
            x = x.clone()
            x[selected_mask] = sampled_tokens[selected_mask]

        # ------------------------------------------------------------------
        # Safety: force‑fill any residual masks (should be none after T steps)
        # ------------------------------------------------------------------
        if (x == self.mask_token_id).any():
            residual_mask = x == self.mask_token_id
            logits = self.model.get_logits(x)
            probs = torch.softmax(logits, dim=-1)
            fill_tokens = probs.argmax(dim=-1)             # (B, L)
            x[residual_mask] = fill_tokens[residual_mask]

        return x


# =========================================================================
# Concrete position selection strategies
# =========================================================================

class VanillaSampler(Sampler):
    """
    Vanilla MDM inference: selects positions uniformly at random from
    among the currently masked tokens.
    """

    def select_positions(
        self,
        logits: torch.Tensor,
        masked_mask: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """
        Random selection of exactly ``K`` masked positions.

        Args:
            logits: Unnormalised model outputs ``(B, L, V)`` (unused).
            masked_mask: Boolean mask ``(B, L)``, ``True`` where tokens are masked.
            K: Total number of positions to select across the batch.

        Returns:
            ``(B, L)`` boolean mask with ``K`` ``True`` entries.
        """
        B, L = masked_mask.shape
        flat_mask = masked_mask.reshape(-1)                     # (B*L,)
        available = flat_mask.nonzero(as_tuple=False).view(-1)  # (num_masked,)

        if K == 0 or len(available) == 0:
            return torch.zeros_like(masked_mask)

        if K > len(available):
            K = len(available)

        # Random permutation of available indices, then pick first K
        perm = torch.randperm(len(available), device=masked_mask.device)
        chosen = available[perm[:K]]

        selection = torch.zeros_like(flat_mask, dtype=torch.bool)
        selection[chosen] = True
        return selection.reshape(B, L)


class TopProbSampler(Sampler):
    """
    Top‑probability adaptive sampler.

    Unmasks positions where the model's most confident prediction
    (largest ``max(softmax(logits))``) is highest.
    """

    def select_positions(
        self,
        logits: torch.Tensor,
        masked_mask: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """
        Select the ``K`` masked positions with highest maximum probability.

        Args:
            logits: ``(B, L, V)`` raw model outputs.
            masked_mask: ``(B, L)`` boolean mask for currently masked positions.
            K: Number of positions to select.

        Returns:
            ``(B, L)`` boolean selection mask.
        """
        # Compute per‑position maximum probability
        probs = torch.softmax(logits, dim=-1)                # (B, L, V)
        max_prob, _ = probs.max(dim=-1)                      # (B, L)

        # Mask out already unmasked positions
        max_prob[~masked_mask] = float("-inf")

        # Flatten and pick top‑K indices
        flat_max_prob = max_prob.reshape(-1)
        _, top_indices = torch.topk(flat_max_prob, K, dim=-1, sorted=False)

        selection = torch.zeros_like(flat_max_prob, dtype=torch.bool)
        selection[top_indices] = True
        return selection.reshape(max_prob.shape)


class TopMarginSampler(Sampler):
    """
    Top‑probability‑margin adaptive sampler.

    Unmasks positions where the difference between the two largest
    predicted probabilities is greatest. Optionally adds Gumbel noise
    to the margin (exploration) before sorting.
    """

    def __init__(self, model: torch.nn.Module, config: ExperimentConfig) -> None:
        super().__init__(model, config)
        self.gumbel_noise_coeff: float = config.diffusion.gumbel_noise_coeff

    def select_positions(
        self,
        logits: torch.Tensor,
        masked_mask: torch.Tensor,
        K: int,
    ) -> torch.Tensor:
        """
        Select the ``K`` masked positions with largest probability margin.

        Args:
            logits: ``(B, L, V)`` raw model outputs.
            masked_mask: ``(B, L)`` boolean mask for currently masked positions.
            K: Number of positions to select.

        Returns:
            ``(B, L)`` boolean selection mask.
        """
        # Top two probabilities
        probs = torch.softmax(logits, dim=-1)                    # (B, L, V)
        top2 = torch.topk(probs, 2, dim=-1).values               # (B, L, 2)

        margin = top2[..., 0] - top2[..., 1]                     # (B, L)

        # Optional Gumbel perturbation
        if self.gumbel_noise_coeff > 0:
            gumbel = Gumbel(0, 1)
            noise = gumbel.sample(margin.shape).to(margin.device)
            margin = margin + noise * self.gumbel_noise_coeff

        # Set margin of non‑masked positions to -inf
        margin[~masked_mask] = float("-inf")

        # Flatten and take top‑K
        flat_margin = margin.reshape(-1)
        _, top_indices = torch.topk(flat_margin, K, dim=-1, sorted=False)

        selection = torch.zeros_like(flat_margin, dtype=torch.bool)
        selection[top_indices] = True
        return selection.reshape(margin.shape)


# ---------------------------------------------------------------------------
# Convenience factory function
# ---------------------------------------------------------------------------

def get_sampler(model: torch.nn.Module, config: ExperimentConfig) -> Sampler:
    """
    Instantiate the appropriate sampler based on
    ``config.diffusion.adaptive_sampler``.

    Args:
        model: Denoising network.
        config: :class:`ExperimentConfig` instance.

    Returns:
        A concrete :class:`Sampler` object.

    Raises:
        ValueError: If the requested sampler type is unknown.
    """
    sampler_type = config.diffusion.adaptive_sampler.strip().lower()

    if sampler_type == "vanilla":
        return VanillaSampler(model, config)
    elif sampler_type in ("top_prob", "top_probability"):
        return TopProbSampler(model, config)
    elif sampler_type in ("top_margin", "top_probability_margin"):
        return TopMarginSampler(model, config)
    else:
        raise ValueError(
            f"Unknown sampler type '{sampler_type}'. "
            f"Available: vanilla, top_prob, top_margin."
        )

