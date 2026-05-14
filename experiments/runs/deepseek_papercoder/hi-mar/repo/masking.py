"""
masking.py

TokenMasker – implements all masking‑related operations for Hi‑MAR training and inference.

Responsibilities:
  - Sampling the masking ratio for each training step (phase‑ and dataset‑specific).
  - Replacing tokens with a learnable mask embedding according to a random mask.
  - Providing the cosine schedule for autoregressive inference.
  - Ranking masked tokens by reconstruction error (confidence) for unmasking.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
from torch import Tensor

from config import MaskConfig, PhaseMaskConfig


class TokenMasker:
    """
    Stateless (parameter‑free) utility that performs masking operations.

    All learned components (namely the mask token embedding) are stored in the
    Hi‑MAR Transformer and passed explicitly to the relevant methods.
    """

    def __init__(self, mask_config: MaskConfig, dataset: str) -> None:
        """
        Args:
            mask_config: Global masking settings from ``config.yaml``.
            dataset:     One of ``"imagenet"`` or ``"coco"``, used to select the
                         correct Phase‑2 ratio distribution.
        """
        if dataset not in ("imagenet", "coco"):
            raise ValueError(
                f"Unknown dataset '{dataset}'. Expected 'imagenet' or 'coco'."
            )

        self.dataset = dataset
        self.phase1_cfg: PhaseMaskConfig = mask_config.phase1

        # Phase‑2 configuration depends on the dataset
        if dataset == "imagenet":
            self.phase2_cfg: PhaseMaskConfig = mask_config.phase2_imagenet
        else:
            self.phase2_cfg: PhaseMaskConfig = mask_config.phase2_coco

    # ------------------------------------------------------------------ ratio sampling

    def sample_mask_ratio(self, phase: int) -> float:
        """
        Return a masking ratio for a training step.

        Phase 1 (always uniform):
            ``r ~ U(0.7, 1.0)``

        Phase 2:
            - ImageNet: ``r = 1 - cos(π/2 * u)`` with ``u ~ U(0, 1)``.
              This heavily favours large ratios (MaskGIT‑style).
            - COCO:     ``r ~ Beta(α=4, β=1)``.

        Args:
            phase: ``1`` for low‑resolution tokens, ``2`` for high‑resolution tokens.

        Returns:
            Sampling ratio as a Python ``float`` in [0, 1].
        """
        if phase == 1:
            ratio_min = self.phase1_cfg.ratio_min
            ratio_max = self.phase1_cfg.ratio_max
            if ratio_min >= ratio_max:
                return ratio_min
            u = torch.rand(1).item()
            return float(u * (ratio_max - ratio_min) + ratio_min)

        elif phase == 2:
            return self._sample_phase2_ratio()

        else:
            raise ValueError(f"Phase must be 1 or 2, got {phase}")

    def _sample_phase2_ratio(self) -> float:
        """Dispatch ratio sampling for Phase‑2 based on dataset."""
        if self.dataset == "imagenet":
            # Cosine distribution (inverse sine): r = 1 - cos(π/2 * u)
            u = torch.rand(1).item()
            return float(1.0 - math.cos((math.pi / 2.0) * u))
        else:   # coco
            # Beta(4, 1) distribution
            alpha = self.phase2_cfg.beta_alpha if self.phase2_cfg.beta_alpha is not None else 4.0
            beta  = self.phase2_cfg.beta_beta  if self.phase2_cfg.beta_beta  is not None else 1.0
            # torch.distributions.Beta is more numerically stable
            dist = torch.distributions.Beta(alpha, beta)
            return float(dist.sample().item())

    # ------------------------------------------------------------------ apply masks

    def apply_masks(
        self,
        tokens: Tensor,
        ratio: float,
        mask_token: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Replace a random subset of tokens with the learnable mask embedding.

        Args:
            tokens:     Clean continuous latent tokens of shape ``(B, N, D)``.
            ratio:      Fraction of tokens to mask, ``0 ≤ ratio ≤ 1``.
            mask_token: Learnable mask token embedding of shape ``(D,)`` (or ``(1, D)``).

        Returns:
            - **masked_tokens** – tokens with masked positions replaced by ``mask_token``,
              shape ``(B, N, D)``.
            - **mask**         – boolean mask indicating which positions are masked,
              shape ``(B, N)``.
        """
        B, N, D = tokens.shape
        device = tokens.device

        # Ensure mask_token is on same device and has shape (D,)
        mask_token = mask_token.to(device)
        if mask_token.ndim == 2 and mask_token.shape[0] == 1:
            mask_token = mask_token.squeeze(0)

        # Number of tokens to mask per sample (ceil to guarantee at least 1 if ratio>0)
        num_mask = max(1, int(ratio * N)) if ratio > 0 else 0

        # Generate random permutation for each batch element
        # We create a random score and then get the top-k indices
        scores = torch.rand(B, N, device=device)
        _, indices = torch.topk(scores, num_mask, dim=-1)   # (B, num_mask)

        # Create boolean mask
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, indices, True)

        # Build masked tokens
        masked_tokens = tokens.clone()
        # Use advanced indexing to set masked positions to mask_token
        masked_tokens[mask] = mask_token.expand_as(masked_tokens[mask])

        return masked_tokens, mask

    # ------------------------------------------------------------------ inference schedule

    @staticmethod
    def cosine_schedule_step(step: int, total_steps: int) -> float:
        """
        Cumulative fraction of tokens that should be **unmasked** after the given step.

        Implements the MaskGIT‑style cosine schedule:
            ``fraction = 1 - cos(π/2 * (step + 1) / total_steps)``

        Args:
            step:        Current step (0‑indexed).
            total_steps: Total number of autoregressive steps (e.g., 32 or 4).

        Returns:
            Fraction of the sequence to be unmasked at this step, ``0 < fraction ≤ 1``.
        """
        if step < 0 or step >= total_steps:
            raise ValueError(
                f"step ({step}) out of range [0, {total_steps})"
            )
        # Use (step+1) so that fraction > 0 at step 0
        frac = 1.0 - math.cos((math.pi / 2.0) * (step + 1) / total_steps)
        return float(frac)

    # ------------------------------------------------------------------ confidence ranking

    @staticmethod
    def confidence_ranking(error: Tensor, mask: Tensor) -> Tensor:
        """
        Order masked tokens by reconstruction error (ascending).

        Tokens with smaller error are considered more confident and should be
        unmasked first during autoregressive decoding.

        Args:
            error: Per‑token error values, shape ``(B, N)``.
            mask:  Boolean mask indicating currently masked positions, shape ``(B, N)``.

        Returns:
            A tensor of shape ``(B, max_num_masked)`` where each row contains the
            indices of the masked tokens sorted by ascending error.  Shorter rows
            are padded with ``-1``.
        """
        B, N = error.shape
        device = error.device

        # Set error of unmasked tokens to infinity so they sort last
        error_filled = error.clone()
        error_filled[~mask] = float('inf')

        # Full sort of all positions
        _, sorted_indices = torch.sort(error_filled, dim=1)   # (B, N)

        # Compute number of masked tokens per sample
        num_masked = mask.sum(dim=1)   # (B,)
        max_masked = num_masked.max().item()

        # Extract the first `num_masked` entries per sample
        # We build a pad mask to fill the rest with -1
        output = torch.full((B, max_masked), -1, dtype=torch.long, device=device)
        for b in range(B):
            k = num_masked[b].item()
            if k > 0:
                output[b, :k] = sorted_indices[b, :k]

        return output

    # ------------------------------------------------------------------ convenience initialiser

    @staticmethod
    def init_masked(
        batch_size: int,
        num_tokens: int,
        mask_token: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Create a fully masked input sequence, used at the start of inference.

        Args:
            batch_size:  Number of samples.
            num_tokens:  Number of tokens per sample (e.g., 64 or 256).
            mask_token:  Mask embedding of shape ``(D,)`` (or ``(1, D)``).

        Returns:
            - **masked_tokens**: tensor of ``mask_token`` repeated to shape
              ``(batch_size, num_tokens, D)``.
            - **mask**:         all‑``True`` boolean mask of shape ``(batch_size, num_tokens)``.
        """
        mask_token = mask_token.view(-1)[:1]  # handle (1,D) or (D,) via first element
        # Actually we need the full tensor; ensure shape (D,)
        if mask_token.ndim == 2:
            mask_token = mask_token.squeeze(0)
        D = mask_token.shape[0]

        masked = mask_token.expand(batch_size, num_tokens, D).clone()
        full_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=mask_token.device)
        return masked, full_mask

