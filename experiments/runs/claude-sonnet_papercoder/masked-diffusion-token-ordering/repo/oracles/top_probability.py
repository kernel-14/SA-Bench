## oracles/top_probability.py
"""Top Probability oracle for adaptive MDM inference.

Implements the 'Top probability' certainty-based position selection strategy
from Section 4.1 of "Train for the Worst, Plan for the Best: Understanding
Token Ordering in Masked Diffusions".

The certainty at position i is estimated by the maximum probability assigned
to any value in the vocabulary:

    certainty(i) = max_{j ∈ {0,...,V-1}} p_θ(x^i = j | x_t)

The oracle selects the K masked positions with the highest certainty scores
for unmasking at each reverse diffusion step.

This strategy is equivalent to the one proposed in Zheng et al. (2023) and
works well in practice for many tasks. However, it can be misleading when
the model assigns nearly equal high probabilities to two competing values —
in that case, the position appears certain but is actually ambiguous. The
TopMarginOracle (oracles/top_margin.py) addresses this limitation.

Config alignment (config.yaml):
    oracles:
      top_probability:
        gumbel_coeff: 0.5   # puzzle experiments (Sudoku, Zebra, NAE-SAT)
        noise_sigma: 0.0    # text experiments: set to 0.1 for diversity

Typical usage::

    # Puzzle experiments (Sudoku, Zebra)
    oracle = TopProbabilityOracle(gumbel_coeff=0.5, noise_sigma=0.0)

    # Text generation (unconditional)
    oracle = TopProbabilityOracle(gumbel_coeff=0.0, noise_sigma=0.1)

    # Inside AdaptiveSampler.sample_step:
    probs = model.get_probs(x_t)                          # [B, L, V]
    masked_pos = (x_t == mask_token_id)                   # [B, L] bool
    k = compute_k(n_masked, alpha_s, alpha_t)             # int
    selected = oracle.select_positions(probs, masked_pos, k)  # [B, k]
"""

import logging
from typing import Optional

import torch

from oracles.base_oracle import BaseOracle, _NEG_INF

logger = logging.getLogger(__name__)


class TopProbabilityOracle(BaseOracle):
    """Adaptive inference oracle using maximum vocabulary probability as certainty.

    Implements the 'Top probability' strategy from Section 4.1 of the paper.
    For each masked position i, the certainty score is:

        score(i) = max_{j ∈ {0,...,V-1}} p_θ(x^i = j | x_t)

    Positions with higher maximum probability are considered more certain and
    are selected for unmasking first. Non-masked positions receive score -inf
    and are never selected.

    Optional noise injection (Gumbel for puzzles, Gaussian for text) prevents
    fully deterministic selection, which can cause degenerate patterns when
    multiple positions have identical or near-identical certainty scores.

    Attributes:
        gumbel_coeff: Coefficient for Gumbel noise added to certainty scores.
            From config.yaml: ``oracles.top_probability.gumbel_coeff = 0.5``
            for puzzle experiments. Set to 0.0 to disable Gumbel noise.
        noise_sigma: Standard deviation for Gaussian noise added to certainty
            scores. From config.yaml:
            ``oracles.top_probability.noise_sigma = 0.0`` (default).
            Set to a positive value (e.g., 0.1) for text generation diversity.
    """

    def __init__(
        self,
        gumbel_coeff: float = 0.5,
        noise_sigma: float = 0.0,
    ) -> None:
        """Initialises the TopProbabilityOracle.

        Args:
            gumbel_coeff: Coefficient for Gumbel noise added to certainty
                scores before TopK selection. Used in puzzle experiments
                (Sudoku, Zebra, NAE-SAT) to introduce controlled stochasticity
                and prevent the oracle from always selecting the same positions
                when scores are tied. From config.yaml:
                ``oracles.top_probability.gumbel_coeff = 0.5``.
                Set to 0.0 to disable Gumbel noise (text/LLaDA experiments).
            noise_sigma: Standard deviation of Gaussian noise added to
                certainty scores. Used in text generation experiments to
                preserve sample diversity (Appendix D.1.2). From config.yaml:
                ``oracles.top_probability.noise_sigma = 0.0`` (default).
                Set to a positive value (e.g., 0.1 from
                ``text_generation.inference.noise_sigma``) for text tasks.
        """
        super().__init__()

        self.gumbel_coeff: float = gumbel_coeff
        self.noise_sigma: float = noise_sigma

        logger.info(
            "TopProbabilityOracle initialised: gumbel_coeff=%.3f, "
            "noise_sigma=%.3f.",
            self.gumbel_coeff,
            self.noise_sigma,
        )

    # ------------------------------------------------------------------
    # Abstract method implementation
    # ------------------------------------------------------------------

    def score(
        self,
        probs: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Computes max-probability certainty scores for all sequence positions.

        For each masked position i in batch element b, computes:

            score[b, i] = max_{j ∈ {0,...,V-1}} probs[b, i, j]

        Non-masked positions receive score -inf to ensure they are never
        selected by the TopK operation in :meth:`select_positions`.

        Optionally adds Gumbel noise (for puzzle experiments) or Gaussian
        noise (for text generation) to the scores of masked positions before
        returning. The -inf values at non-masked positions are preserved
        through noise addition by applying noise only to finite-score
        positions.

        Args:
            probs: Per-position probability distributions from the model.
                Shape ``[B, L, V]``, dtype ``torch.float32``. Values are
                non-negative and sum to 1 along the last dimension (output
                of ``MDMTransformer.get_probs()``). V is the vocabulary size.
            masked_positions: Boolean tensor of shape ``[B, L]``. ``True``
                at positions that are currently masked (eligible for
                unmasking), ``False`` at positions that already have a value.
                Derived from ``x_t == mask_token_id`` in the sampler.

        Returns:
            Certainty score tensor of shape ``[B, L]``, dtype
            ``torch.float32``.

            - Masked positions (``masked_positions[b, i] == True``): finite
              score equal to ``max_j probs[b, i, j]``, optionally perturbed
              by Gumbel or Gaussian noise.
            - Non-masked positions (``masked_positions[b, i] == False``):
              score is ``-inf``, ensuring they are never selected by TopK.

        Note:
            The noise is applied only to positions with finite scores
            (masked positions) to preserve the -inf sentinel values at
            non-masked positions. This is implemented by computing noise
            for all positions but zeroing it out at non-masked positions
            before adding.
        """
        # ------------------------------------------------------------------ #
        # Step 1: Compute max probability over vocabulary for each position   #
        # ------------------------------------------------------------------ #
        # probs: [B, L, V] → max over V dim → [B, L]
        # max_scores[b, i] = max_{j} probs[b, i, j]
        max_scores: torch.Tensor = torch.max(probs, dim=-1).values  # [B, L]

        # ------------------------------------------------------------------ #
        # Step 2: Zero out non-masked positions with -inf sentinel            #
        # ------------------------------------------------------------------ #
        # Non-masked positions must never be selected by TopK.
        # masked_positions: [B, L] bool, True = masked (eligible)
        # ~masked_positions: True = already unmasked (ineligible)
        max_scores = max_scores.masked_fill(~masked_positions, _NEG_INF)

        # ------------------------------------------------------------------ #
        # Step 3: Optionally add Gumbel noise (puzzle experiments)            #
        # ------------------------------------------------------------------ #
        # From config.yaml: oracles.top_probability.gumbel_coeff = 0.5
        # Paper (Appendix D.2): "Gumbel noise with a coefficient of 0.5"
        # Applied only to masked positions to preserve -inf at unmasked ones.
        if self.gumbel_coeff > 0.0:
            # Compute Gumbel noise for all positions (same shape as max_scores).
            # add_gumbel_noise handles the -log(-log(U + eps)) computation.
            noisy_scores: torch.Tensor = self.add_gumbel_noise(
                max_scores, coeff=self.gumbel_coeff
            )
            # Restore -inf at non-masked positions that may have been perturbed.
            # (add_gumbel_noise adds noise to all positions including -inf ones,
            # which can produce NaN; we restore the sentinel values explicitly.)
            noisy_scores = noisy_scores.masked_fill(~masked_positions, _NEG_INF)
            max_scores = noisy_scores

        # ------------------------------------------------------------------ #
        # Step 4: Optionally add Gaussian noise (text generation experiments) #
        # ------------------------------------------------------------------ #
        # From config.yaml: oracles.top_probability.noise_sigma = 0.0
        # Paper (Appendix D.1.2): temperature-augmented oracle for text tasks.
        # Applied only to masked positions to preserve -inf at unmasked ones.
        if self.noise_sigma > 0.0:
            # Compute Gaussian noise for all positions.
            gaussian_scores: torch.Tensor = self.add_gaussian_noise(
                max_scores, sigma=self.noise_sigma
            )
            # Restore -inf at non-masked positions.
            gaussian_scores = gaussian_scores.masked_fill(
                ~masked_positions, _NEG_INF
            )
            max_scores = gaussian_scores

        return max_scores  # [B, L], float32

    # ------------------------------------------------------------------
    # select_positions is inherited from BaseOracle — not overridden.
    # BaseOracle.select_positions calls self.score() and applies TopK.
    # ------------------------------------------------------------------
