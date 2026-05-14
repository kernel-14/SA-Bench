## oracles/top_margin.py
"""Top Probability Margin oracle for adaptive MDM inference.

Implements the 'Top probability margin' certainty-based position selection
strategy from Section 4.1 of "Train for the Worst, Plan for the Best:
Understanding Token Ordering in Masked Diffusions".

The certainty at position i is estimated by the absolute difference between
the two most probable values at that position:

    certainty(i) = |p_θ(x^i = j1 | x_t) - p_θ(x^i = j2 | x_t)|

where j1 and j2 are the top-2 most probable vocabulary values at position i.

This oracle addresses a key limitation of TopProbabilityOracle: when the
model assigns nearly equal high probabilities to two competing values (e.g.,
[0.49, 0.48, 0.03, ...]), the max-probability oracle incorrectly treats the
position as high-certainty (score=0.49), while the margin oracle correctly
identifies it as uncertain (score=0.01) and defers it.

This distinction is critical for Sudoku, where cells often have two plausible
digit candidates. Table 2 of the paper shows Top Probability achieves 18.51%
accuracy vs Top Probability Margin at 89.49% on Sudoku puzzles.

Config alignment (config.yaml):
    oracles:
      top_margin:
        gumbel_coeff: 0.5   # puzzle experiments (Sudoku, Zebra, NAE-SAT)
        noise_sigma: 0.0    # text experiments: set to 0.1 for diversity

Typical usage::

    # Puzzle experiments (Sudoku, Zebra) — primary use case
    oracle = TopMarginOracle(gumbel_coeff=0.5, noise_sigma=0.0)

    # Text generation (unconditional) — temperature-augmented variant
    oracle = TopMarginOracle(gumbel_coeff=0.0, noise_sigma=0.1)

    # LLaDA-8B evaluation — deterministic oracle
    oracle = TopMarginOracle(gumbel_coeff=0.0, noise_sigma=0.0)

    # Inside AdaptiveSampler.sample_step:
    probs = model.get_probs(x_t)                          # [B, L, V]
    masked_pos = (x_t == mask_token_id)                   # [B, L] bool
    k = compute_k(n_masked, alpha_s, alpha_t)             # int
    selected = oracle.select_positions(probs, masked_pos, k)  # [B, k]
"""

import logging

import torch

from oracles.base_oracle import BaseOracle, _NEG_INF

logger = logging.getLogger(__name__)


class TopMarginOracle(BaseOracle):
    """Adaptive inference oracle using top-2 probability margin as certainty.

    Implements the 'Top probability margin' strategy from Section 4.1 of the
    paper. For each masked position i, the certainty score is:

        score(i) = p_θ(x^i = j1 | x_t) - p_θ(x^i = j2 | x_t)

    where j1 is the most probable value and j2 is the second most probable
    value at position i. Since j1 is the argmax, p(j1) >= p(j2) always holds,
    so the difference is non-negative and equals the absolute difference.

    A large margin indicates the model strongly prefers one value over all
    others — the position is certain and should be unmasked first. A small
    margin indicates the model is confused between two competing values —
    the position is uncertain and should be deferred.

    Non-masked positions receive score -inf and are never selected.

    Optional noise injection (Gumbel for puzzles, Gaussian for text) prevents
    fully deterministic selection, which can cause degenerate patterns when
    multiple positions have identical or near-identical margin scores.

    Attributes:
        gumbel_coeff: Coefficient for Gumbel noise added to certainty scores.
            From config.yaml: ``oracles.top_margin.gumbel_coeff = 0.5``
            for puzzle experiments. Set to 0.0 to disable Gumbel noise.
        noise_sigma: Standard deviation for Gaussian noise added to certainty
            scores. From config.yaml:
            ``oracles.top_margin.noise_sigma = 0.0`` (default).
            Set to a positive value (e.g., 0.1 from
            ``text_generation.inference.noise_sigma``) for text generation.
    """

    def __init__(
        self,
        gumbel_coeff: float = 0.5,
        noise_sigma: float = 0.0,
    ) -> None:
        """Initialises the TopMarginOracle.

        Args:
            gumbel_coeff: Coefficient for Gumbel noise added to certainty
                scores before TopK selection. Used in puzzle experiments
                (Sudoku, Zebra, NAE-SAT) to introduce controlled stochasticity
                and prevent the oracle from always selecting the same positions
                when scores are tied. From config.yaml:
                ``oracles.top_margin.gumbel_coeff = 0.5``.
                Set to 0.0 to disable Gumbel noise (text/LLaDA experiments).
            noise_sigma: Standard deviation of Gaussian noise added to
                certainty scores. Used in text generation experiments to
                preserve sample diversity (Appendix D.1.2 of the paper).
                From config.yaml: ``oracles.top_margin.noise_sigma = 0.0``
                (default). Set to a positive value (e.g., 0.1 from
                ``text_generation.inference.noise_sigma``) for text tasks.
        """
        super().__init__()

        self.gumbel_coeff: float = gumbel_coeff
        self.noise_sigma: float = noise_sigma

        logger.info(
            "TopMarginOracle initialised: gumbel_coeff=%.3f, noise_sigma=%.3f.",
            self.gumbel_coeff,
            self.noise_sigma,
        )

    # ------------------------------------------------------------------
    # Core margin computation
    # ------------------------------------------------------------------

    def compute_margin(self, probs: torch.Tensor) -> torch.Tensor:
        """Computes the top-2 probability margin for every sequence position.

        For each position i in batch element b, computes:

            margin[b, i] = probs[b, i, j1] - probs[b, i, j2]

        where j1 = argmax_j probs[b, i, j] and j2 = second-argmax_j.

        Since j1 is the argmax, probs[b, i, j1] >= probs[b, i, j2] always
        holds, so the difference is non-negative and equals the absolute
        difference |p(j1) - p(j2)|.

        Uses a single ``torch.topk(k=2)`` call for efficiency — this avoids
        the need to mask the argmax before finding the second-best value and
        is more numerically stable than double-argmax approaches.

        This method operates on ALL positions (masked and unmasked). The
        masking filter (setting non-masked positions to -inf) is applied in
        :meth:`score`, not here. This keeps ``compute_margin`` a pure
        mathematical operation that can be tested independently.

        Args:
            probs: Per-position probability distributions from the model.
                Shape ``[B, L, V]``, dtype ``torch.float32``. Values are
                non-negative and sum to 1 along the last dimension (output
                of ``MDMTransformer.get_probs()``). V is the vocabulary size.

        Returns:
            Margin tensor of shape ``[B, L]``, dtype ``torch.float32``.
            Values are non-negative floats in ``[0, 1]``.

            - A value close to 1.0 means the model is very certain (one
              value dominates with near-probability 1).
            - A value close to 0.0 means the model is confused between two
              nearly equally probable values.

        Note:
            When ``V == 1`` (degenerate single-token vocabulary), there is
            no second value to compare against. In this edge case, the margin
            is defined as 0.0 for all positions (maximum uncertainty). This
            should not occur in practice since all experiment configs have
            ``vocab_size >= 5``.
        """
        vocab_size: int = probs.shape[-1]

        # ------------------------------------------------------------------ #
        # Edge case: single-token vocabulary — no margin can be computed      #
        # ------------------------------------------------------------------ #
        if vocab_size < 2:
            logger.warning(
                "compute_margin: vocab_size=%d < 2. Returning zero margins.",
                vocab_size,
            )
            return torch.zeros(
                probs.shape[0], probs.shape[1],
                dtype=probs.dtype,
                device=probs.device,
            )

        # ------------------------------------------------------------------ #
        # Efficient top-2 extraction via torch.topk                           #
        # ------------------------------------------------------------------ #
        # torch.topk(probs, k=2, dim=-1) returns:
        #   top2_values:  [B, L, 2] — top-2 probabilities (sorted descending)
        #   top2_indices: [B, L, 2] — corresponding vocabulary indices
        #
        # top2_values[..., 0] = p(j1) = max probability at each position
        # top2_values[..., 1] = p(j2) = second-highest probability
        #
        # Since topk returns values in descending order:
        #   top2_values[..., 0] >= top2_values[..., 1] always holds.
        top2_values: torch.Tensor
        top2_values, _ = torch.topk(probs, k=2, dim=-1, largest=True, sorted=True)
        # top2_values shape: [B, L, 2]

        # ------------------------------------------------------------------ #
        # Compute margin: p(j1) - p(j2)                                       #
        # ------------------------------------------------------------------ #
        # Since p(j1) >= p(j2) by construction, the difference is always >= 0.
        # No absolute value is needed, but we clamp to [0, 1] for safety
        # against floating-point rounding errors that could produce tiny
        # negative values (e.g., -1e-7).
        p1: torch.Tensor = top2_values[..., 0]  # [B, L] — max probability
        p2: torch.Tensor = top2_values[..., 1]  # [B, L] — second probability

        margin: torch.Tensor = (p1 - p2).clamp(min=0.0)  # [B, L], in [0, 1]

        return margin

    # ------------------------------------------------------------------
    # Abstract method implementation
    # ------------------------------------------------------------------

    def score(
        self,
        probs: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Computes top-2 margin certainty scores for all sequence positions.

        For each masked position i in batch element b, computes:

            score[b, i] = p_θ(x^i = j1 | x_t) - p_θ(x^i = j2 | x_t)

        where j1 and j2 are the top-2 most probable vocabulary values.

        Non-masked positions receive score -inf to ensure they are never
        selected by the TopK operation in :meth:`select_positions`.

        Optionally adds Gumbel noise (for puzzle experiments, coeff=0.5) or
        Gaussian noise (for text generation, sigma=0.1) to the scores of
        masked positions. The -inf values at non-masked positions are
        preserved through noise addition since -inf + finite = -inf in
        IEEE 754 floating point.

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
              score equal to ``p(j1) - p(j2)`` at that position, optionally
              perturbed by Gumbel or Gaussian noise. Values in ``[0, 1]``
              before noise addition.
            - Non-masked positions (``masked_positions[b, i] == False``):
              score is ``-inf``, ensuring they are never selected by TopK.

        Note:
            Adding Gumbel or Gaussian noise to the full score tensor (including
            -inf positions) is safe because -inf + finite = -inf in IEEE 754
            float arithmetic. This avoids the need for a separate masking step
            after noise addition. We nonetheless re-apply the -inf mask after
            noise addition as a defensive measure against potential NaN
            propagation from -inf * 0 or similar edge cases.
        """
        # ------------------------------------------------------------------ #
        # Step 1: Compute top-2 margin for all positions                      #
        # ------------------------------------------------------------------ #
        # margin_scores: [B, L], non-negative floats in [0, 1]
        # Operates on all positions (masked and unmasked).
        margin_scores: torch.Tensor = self.compute_margin(probs)

        # ------------------------------------------------------------------ #
        # Step 2: Apply -inf sentinel to non-masked positions                 #
        # ------------------------------------------------------------------ #
        # Non-masked positions must never be selected by TopK in
        # select_positions(). Setting them to -inf ensures this.
        # masked_positions: [B, L] bool, True = masked (eligible for selection)
        # ~masked_positions: True = already unmasked (ineligible)
        scores: torch.Tensor = margin_scores.masked_fill(
            ~masked_positions, _NEG_INF
        )  # [B, L]

        # ------------------------------------------------------------------ #
        # Step 3: Optionally add Gumbel noise (puzzle experiments)            #
        # ------------------------------------------------------------------ #
        # From config.yaml: oracles.top_margin.gumbel_coeff = 0.5
        # Paper (Appendix D.2): "Gumbel noise with a coefficient of 0.5"
        # for Sudoku and Zebra puzzle inference.
        #
        # Adding noise to the full tensor is safe: -inf + finite = -inf.
        # We re-apply the mask after noise addition as a defensive measure
        # against potential NaN propagation (e.g., from -inf + 0 * nan).
        if self.gumbel_coeff > 0.0:
            scores = self.add_gumbel_noise(scores, coeff=self.gumbel_coeff)
            # Defensive re-masking: restore -inf at non-masked positions.
            scores = scores.masked_fill(~masked_positions, _NEG_INF)

        # ------------------------------------------------------------------ #
        # Step 4: Optionally add Gaussian noise (text generation experiments) #
        # ------------------------------------------------------------------ #
        # From config.yaml: text_generation.inference.noise_sigma = 0.1
        # Paper (Appendix D.1.2): temperature-augmented oracle for text tasks.
        # "adding a certain level of temperature to the oracle is useful
        #  because the top probability margin or the top probability often
        #  leads to greedy sampling, which harms the diversity (entropy)
        #  of the generated samples."
        if self.noise_sigma > 0.0:
            scores = self.add_gaussian_noise(scores, sigma=self.noise_sigma)
            # Defensive re-masking: restore -inf at non-masked positions.
            scores = scores.masked_fill(~masked_positions, _NEG_INF)

        return scores  # [B, L], float32

    # ------------------------------------------------------------------
    # select_positions is inherited from BaseOracle — not overridden.
    # BaseOracle.select_positions calls self.score() and applies TopK.
    # ------------------------------------------------------------------
