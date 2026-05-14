## oracles/base_oracle.py
"""Abstract base class for adaptive inference oracles in Masked Diffusion Models.

Defines the BaseOracle interface used by AdaptiveSampler to select which
masked positions to unmask at each reverse diffusion step, implementing the
adaptive inference strategy from Section 4.1 of "Train for the Worst, Plan
for the Best: Understanding Token Ordering in Masked Diffusions".

The core idea: instead of randomly selecting positions to unmask (vanilla MDM
inference), the oracle selects the K positions where the model is most
"certain" about the correct token value. This allows the MDM to sidestep
hard masking subproblems by always unmasking the easiest positions first.

Two concrete oracle strategies are defined in sibling modules:
  - TopProbabilityOracle  (oracles/top_probability.py): certainty = max_j p(x^i=j|x_t)
  - TopMarginOracle       (oracles/top_margin.py):      certainty = |p(j1|x_t) - p(j2|x_t)|

Adaptive inference algorithm (Section 4.1):
    (a) S = F(θ, x_t) = TopK(certainty_score(i))  for i in masked positions
    (b) For each i in S: sample x_s^i ~ p_θ(x^i | x_t)

This module implements step (a). Step (b) is handled by AdaptiveSampler.

Config alignment (config.yaml):
  oracles.top_probability.gumbel_coeff = 0.5   (puzzle experiments)
  oracles.top_margin.gumbel_coeff      = 0.5   (puzzle experiments)
  oracles.top_probability.noise_sigma  = 0.0   (text: set > 0 for diversity)
  oracles.top_margin.noise_sigma       = 0.0   (text: set > 0 for diversity)
  text_generation.inference.noise_sigma = 0.1  (Gaussian noise for text)

Typical usage (inside AdaptiveSampler.sample_step)::

    probs = model.get_probs(x_t)                    # [B, L, V]
    masked_pos = (x_t == mask_token_id)             # [B, L] bool
    k = compute_k(n_masked, alpha_s, alpha_t)       # int
    selected = oracle.select_positions(probs, masked_pos, k)  # [B, k]
    # Then sample token values at selected positions from probs.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Small epsilon used in Gumbel noise sampling to prevent log(0).
_GUMBEL_EPS: float = 1e-10

#: Sentinel value used to mark non-masked positions as ineligible for selection.
#: Using float('-inf') ensures these positions are never selected by torch.topk.
_NEG_INF: float = float("-inf")


class BaseOracle(ABC):
    """Abstract base class for adaptive MDM inference oracles.

    An oracle implements the position-selection function F(θ, x_t) from
    Section 4.1 of the paper. Given the model's current probability
    distributions over all positions, it returns the indices of the K
    positions that should be unmasked next.

    Subclasses must implement :meth:`score`, which computes a scalar
    certainty score for each masked position. Higher score = more certain =
    should be unmasked sooner. Non-masked positions must receive score
    ``-inf`` so they are never selected by :meth:`select_positions`.

    The shared :meth:`select_positions` method calls :meth:`score` and then
    applies TopK selection, handling edge cases (k=0, fewer masked positions
    than k) uniformly for all oracle implementations.

    Noise utilities :meth:`add_gumbel_noise` and :meth:`add_gaussian_noise`
    are provided for subclasses to call within their :meth:`score`
    implementations. Subclasses store their noise parameters (``gumbel_coeff``,
    ``noise_sigma``) and apply noise before returning from :meth:`score`.

    Attributes:
        None at this level — subclasses define ``gumbel_coeff`` and
        ``noise_sigma`` as instance attributes.
    """

    def __init__(self) -> None:
        """Initialises the BaseOracle.

        Subclasses should call ``super().__init__()`` and then set their
        own noise parameters (``gumbel_coeff``, ``noise_sigma``).
        """
        pass

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def score(
        self,
        probs: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Computes a certainty score for each position in the sequence.

        This is the core method that distinguishes different oracle strategies.
        Subclasses implement their specific certainty metric here and are
        responsible for:
          1. Computing raw certainty scores for all positions.
          2. Setting scores to ``-inf`` for non-masked positions (using
             ``masked_fill`` or equivalent).
          3. Optionally adding noise (Gumbel or Gaussian) via the utility
             methods :meth:`add_gumbel_noise` and :meth:`add_gaussian_noise`
             before returning.

        **Contract:**
          - Masked positions (``masked_positions[b, i] == True``): score is
            a finite float representing certainty. Higher = more certain.
          - Non-masked positions (``masked_positions[b, i] == False``): score
            must be ``-inf`` (or a very large negative number) to ensure they
            are never selected by :meth:`select_positions`.

        Args:
            probs: Per-position probability distributions from the model.
                Shape ``[B, L, V]``, dtype ``torch.float32``.  Values are
                non-negative and sum to 1 along the last dimension (output
                of ``MDMTransformer.get_probs()``).
            masked_positions: Boolean tensor of shape ``[B, L]``.  ``True``
                at positions that are currently masked (eligible for
                unmasking), ``False`` at positions that already have a value.
                Derived from ``x_t == mask_token_id`` in the sampler.

        Returns:
            Certainty score tensor of shape ``[B, L]``, dtype
            ``torch.float32``.  Masked positions have finite scores;
            non-masked positions have score ``-inf``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Concrete shared logic
    # ------------------------------------------------------------------

    def select_positions(
        self,
        probs: torch.Tensor,
        masked_positions: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Selects the K most certain masked positions to unmask next.

        Calls :meth:`score` to get certainty scores, then applies TopK
        selection along the sequence dimension. Handles edge cases:
          - ``k == 0``: returns an empty tensor of shape ``[B, 0]``.
          - Fewer masked positions than ``k``: clamps ``k`` to the minimum
            number of masked positions across the batch to avoid requesting
            more positions than are available.

        This method is the primary entry point called by ``AdaptiveSampler``
        at each reverse diffusion step.

        Args:
            probs: Per-position probability distributions from the model.
                Shape ``[B, L, V]``, dtype ``torch.float32``.
            masked_positions: Boolean tensor of shape ``[B, L]``.  ``True``
                at currently masked positions.
            k: Number of positions to select for unmasking.  Computed by
                the sampler as:
                ``k = round(n_masked * (alpha_s - alpha_t) / (1 - alpha_t))``
                May be 0 when ``alpha_s ≈ alpha_t``.

        Returns:
            Index tensor of shape ``[B, k_actual]``, dtype ``torch.long``,
            where ``k_actual = min(k, min_masked_per_sample)``.  Each row
            contains the sequence-dimension indices of the positions selected
            for unmasking in the corresponding batch element.

            When ``k == 0`` or no positions are masked, returns an empty
            tensor of shape ``[B, 0]``.

        Note:
            The returned indices are positions in the sequence dimension
            (range ``[0, L)``), not values. The sampler uses these indices
            to look up token probabilities in ``probs`` and sample values.
        """
        batch_size: int = probs.shape[0]
        device: torch.device = probs.device

        # ------------------------------------------------------------------ #
        # Early exit: k == 0 means no tokens to unmask at this step           #
        # ------------------------------------------------------------------ #
        if k <= 0:
            return torch.zeros(
                batch_size, 0, dtype=torch.long, device=device
            )

        # ------------------------------------------------------------------ #
        # Compute certainty scores via the subclass implementation             #
        # ------------------------------------------------------------------ #
        # scores: [B, L], float32
        # Masked positions: finite scores (higher = more certain)
        # Non-masked positions: -inf (excluded from TopK)
        scores: torch.Tensor = self.score(probs, masked_positions)

        # ------------------------------------------------------------------ #
        # Clamp k to the minimum number of masked positions in the batch      #
        # ------------------------------------------------------------------ #
        # n_masked_per_sample: [B], number of masked positions per sample.
        n_masked_per_sample: torch.Tensor = masked_positions.sum(dim=-1)  # [B]
        min_masked: int = int(n_masked_per_sample.min().item())

        if min_masked == 0:
            # No masked positions remain in at least one sample.
            # Return empty selection for the whole batch.
            logger.debug(
                "select_positions: no masked positions remain in at least "
                "one batch element.  Returning empty selection."
            )
            return torch.zeros(
                batch_size, 0, dtype=torch.long, device=device
            )

        # Clamp k to the minimum available masked count across the batch.
        # This ensures torch.topk never requests more elements than exist.
        k_actual: int = min(k, min_masked)

        if k_actual < k:
            logger.debug(
                "select_positions: clamped k from %d to %d "
                "(min masked positions in batch = %d).",
                k,
                k_actual,
                min_masked,
            )

        # ------------------------------------------------------------------ #
        # TopK selection: pick the k_actual highest-scoring positions          #
        # ------------------------------------------------------------------ #
        # torch.topk returns (values, indices) along dim=-1 (sequence dim).
        # Since non-masked positions have score -inf, they are never selected.
        # Shape of indices: [B, k_actual]
        _, selected_indices = torch.topk(
            scores,
            k=k_actual,
            dim=-1,
            largest=True,
            sorted=False,  # order within the k positions doesn't matter
        )

        return selected_indices  # [B, k_actual], dtype torch.long

    # ------------------------------------------------------------------
    # Noise utility methods (called by subclasses within score())
    # ------------------------------------------------------------------

    def add_gumbel_noise(
        self,
        scores: torch.Tensor,
        coeff: float = 0.5,
    ) -> torch.Tensor:
        """Adds Gumbel noise to certainty scores for stochastic position selection.

        Implements the noise injection described in Appendix D.2 of the paper:
        "we add Gumbel noise with a coefficient of 0.5 to the MDM inference
        oracle F" for Sudoku and Zebra puzzle experiments.

        The Gumbel noise prevents the oracle from being fully deterministic,
        which could cause it to get stuck in degenerate patterns when multiple
        positions have identical certainty scores.

        Gumbel(0, 1) sampling via the inverse CDF method:
            U ~ Uniform(0, 1)
            G = -log(-log(U + eps) + eps)

        The noise is added to the raw scores before TopK selection:
            noisy_scores = scores + coeff * G

        Config alignment:
          - ``oracles.top_probability.gumbel_coeff = 0.5`` (puzzle experiments)
          - ``oracles.top_margin.gumbel_coeff = 0.5`` (puzzle experiments)
          - ``coeff = 0.0`` effectively disables noise (text/LLaDA experiments)

        Args:
            scores: Certainty score tensor of any shape, dtype
                ``torch.float32``.  Typically shape ``[B, L]``.
            coeff: Gumbel noise coefficient.  From config.yaml:
                ``oracles.top_probability.gumbel_coeff = 0.5`` and
                ``oracles.top_margin.gumbel_coeff = 0.5`` for puzzle
                experiments.  Set to ``0.0`` to disable noise.

        Returns:
            Noise-augmented score tensor of the same shape and dtype as
            ``scores``.  When ``coeff == 0.0``, returns ``scores`` unchanged
            (no allocation).
        """
        if coeff == 0.0:
            return scores

        # Sample U ~ Uniform(0, 1) with the same shape as scores.
        u: torch.Tensor = torch.rand_like(scores)

        # Gumbel(0, 1) via inverse CDF: G = -log(-log(U + eps) + eps)
        # The double eps guards prevent log(0) at both the inner and outer log.
        gumbel_noise: torch.Tensor = -torch.log(
            -torch.log(u + _GUMBEL_EPS) + _GUMBEL_EPS
        )

        return scores + coeff * gumbel_noise

    def add_gaussian_noise(
        self,
        scores: torch.Tensor,
        sigma: float = 0.0,
    ) -> torch.Tensor:
        """Adds Gaussian noise to certainty scores for text generation diversity.

        Implements the temperature-augmented oracle from Appendix D.1.2 of
        the paper:
            F(θ, x_t) = TopK(|p(x^i=j1|x_t) - p(x^i=j2|x_t)| + ε)
        where ε ~ N(0, σ²).

        This is used for unconditional text generation to preserve sample
        diversity, since the top probability margin oracle tends toward greedy
        selection which reduces entropy.

        Config alignment:
          - ``text_generation.inference.noise_sigma = 0.1`` (text experiments)
          - ``oracles.top_margin.noise_sigma = 0.0`` (puzzles — no Gaussian noise)
          - ``llada_eval.inference.noise_sigma = 0.0`` (LLaDA — no Gaussian noise)

        Args:
            scores: Certainty score tensor of any shape, dtype
                ``torch.float32``.  Typically shape ``[B, L]``.
            sigma: Standard deviation of the Gaussian noise.  From
                config.yaml: ``text_generation.inference.noise_sigma = 0.1``
                for text generation experiments.  Set to ``0.0`` to disable
                noise (default for puzzle and LLaDA experiments).

        Returns:
            Noise-augmented score tensor of the same shape and dtype as
            ``scores``.  When ``sigma == 0.0``, returns ``scores`` unchanged
            (no allocation).
        """
        if sigma == 0.0:
            return scores

        # Sample ε ~ N(0, σ²) with the same shape as scores.
        gaussian_noise: torch.Tensor = sigma * torch.randn_like(scores)

        return scores + gaussian_noise
