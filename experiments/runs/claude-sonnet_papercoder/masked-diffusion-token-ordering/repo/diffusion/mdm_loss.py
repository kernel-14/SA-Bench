## diffusion/mdm_loss.py
"""MDM training loss (continuous-time ELBO) for Masked Diffusion Models.

Implements the core training objective from Equation (1) of
"Train for the Worst, Plan for the Best: Understanding Token Ordering in
Masked Diffusions":

    L_theta = integral_0^1  (alpha_t' / (1 - alpha_t))
              * E_{x_t ~ p_data}  sum_{i: x_t^i = 0}  -log p_theta(x_0^i | x_t)  dt

The continuous-time ELBO is estimated via Monte Carlo:
  1. Sample t ~ Uniform(0, 1) per sequence in the batch.
  2. Apply independent Bernoulli masking with probability 1 - alpha_t.
  3. Run the time-embedding-free denoiser p_theta(· | x_t).
  4. Compute cross-entropy only over masked (non-pad) positions.
  5. Weight by |alpha_t'| / (1 - alpha_t) (the continuous-time ELBO weight).

The discrete reformulation (Proposition 2.1) shows this is equivalent to a
weighted sum over all possible infilling masks M, with weight 1/|M| * 1/C(L,|M|).
The 1/|M| normalization is implemented in cross_entropy_masked via division by
the number of masked tokens per sample.

Config alignment (config.yaml):
  noise_schedule.mask_token_id = 0   -> self.mask_token_id
  nae_sat.data.pad_token_id    = 4   -> self.pad_token_id (NAE-SAT only)
  noise_schedule.type          = linear -> NoiseSchedule.alpha(t) = 1 - t
  nae_sat.model.time_conditioned = false -> model receives only x_t, not t

Typical usage::

    schedule  = NoiseSchedule(schedule_type='linear', T=50)
    loss_fn   = MDMLoss(noise_schedule=schedule, mask_token_id=0,
                        pad_token_id=4)   # pad_token_id=None for Sudoku/Zebra

    # Inside MDMTrainer.train_step:
    loss = loss_fn.compute(model, batch['x0'])
    loss.backward()
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.noise_schedule import NoiseSchedule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Small epsilon added to (1 - alpha_t) denominator to prevent division by zero
#: when t is very close to 0 (alpha_t ≈ 1).
_EPS: float = 1e-8

#: Default mask token ID, aligned with config.yaml noise_schedule.mask_token_id
_DEFAULT_MASK_TOKEN_ID: int = 0


class MDMLoss(nn.Module):
    """Continuous-time ELBO training loss for Masked Diffusion Models.

    This module has no learnable parameters — it is a pure loss computation
    utility that wraps the noise schedule and implements the masking forward
    process and weighted cross-entropy objective.

    The loss is computed as:

        L = mean_over_batch(
                |alpha_t'| / (1 - alpha_t)
                * mean_over_masked_positions(-log p_theta(x_0^i | x_t))
            )

    where the sign of alpha_t' is handled by using its absolute value
    (alpha_t' is always negative since alpha_t is monotonically decreasing).

    Attributes:
        noise_schedule: The NoiseSchedule instance providing alpha(t),
            alpha_prime(t), and sample_t().
        mask_token_id: Token ID of the [MASK] token.  Masked positions in
            x_t are set to this value.  From config.yaml:
            ``noise_schedule.mask_token_id = 0``.
        pad_token_id: Token ID of the [PAD] token, or None if the dataset
            has no padding.  Pad positions are never masked and are excluded
            from the loss.  From config.yaml:
            ``nae_sat.data.pad_token_id = 4`` (None for Sudoku/Zebra).
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        mask_token_id: int = _DEFAULT_MASK_TOKEN_ID,
        pad_token_id: Optional[int] = None,
    ) -> None:
        """Initialises MDMLoss.

        Args:
            noise_schedule: Injected NoiseSchedule instance.  Must expose
                ``alpha(t)``, ``alpha_prime(t)``, and ``sample_t(batch_size,
                device)``.  Typically constructed as
                ``NoiseSchedule(schedule_type='linear', T=50)`` per
                config.yaml ``noise_schedule.type = linear``.
            mask_token_id: Integer ID of the [MASK] token.  Positions in
                x_t that are masked are set to this value.  Default: 0,
                aligned with config.yaml ``noise_schedule.mask_token_id``.
            pad_token_id: Integer ID of the [PAD] token, or None.  When
                set, pad positions are excluded from masking and from the
                loss computation.  For NAE-SAT experiments, set to 4 per
                config.yaml ``nae_sat.data.pad_token_id = 4``.  For
                Sudoku/Zebra experiments, leave as None (no padding in
                81-token or 25-token sequences).
        """
        super().__init__()

        self.noise_schedule: NoiseSchedule = noise_schedule
        self.mask_token_id: int = mask_token_id
        self.pad_token_id: Optional[int] = pad_token_id

        logger.info(
            "MDMLoss initialised: mask_token_id=%d, pad_token_id=%s, "
            "schedule_type='%s'.",
            self.mask_token_id,
            self.pad_token_id,
            self.noise_schedule.schedule_type,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        model: nn.Module,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the MDM ELBO training loss for a batch of clean sequences.

        Implements the full Monte Carlo estimator of Equation (1):

            L_theta ≈ mean_b(
                |alpha_t_b'| / (1 - alpha_t_b + eps)
                * mean_{i: x_t_b^i = MASK}(-log p_theta(x_0_b^i | x_t_b))
            )

        where t_b ~ Uniform(0, 1) is independently sampled for each sequence
        b in the batch.

        The model is time-embedding-free: it receives only x_t (not t),
        consistent with config.yaml ``nae_sat.model.time_conditioned = false``
        and the paper's observation that the number of masked tokens in x_t
        implicitly encodes t.

        Args:
            model: The MDMTransformer denoiser.  Must expose a ``forward``
                method accepting integer token tensors of shape ``[B, L]``
                and returning logits of shape ``[B, L, V]``.  The model
                should be in training mode (``model.train()``) when this
                method is called.
            x0: Clean token sequences of shape ``[B, L]``, dtype
                ``torch.long``.  Values are token IDs in
                ``[0, vocab_size)``.  Pad positions (if any) contain
                ``pad_token_id``.

        Returns:
            Scalar loss tensor with ``requires_grad=True``.  The gradient
            graph is intact for backpropagation via ``loss.backward()``.

        Note:
            The returned loss is the mean over the batch dimension.  For
            gradient accumulation, divide by the number of accumulation
            steps before calling ``backward()``.
        """
        batch_size: int = x0.shape[0]
        device: torch.device = x0.device

        # ------------------------------------------------------------------ #
        # Step 1: Sample noise levels t ~ Uniform(0, 1) per sequence          #
        # ------------------------------------------------------------------ #
        # Shape: [B], values in (0, 1).
        # Each sequence in the batch receives an independently sampled t,
        # implementing the Monte Carlo estimator of the integral over t.
        t: torch.Tensor = self.noise_schedule.sample_t(
            batch_size=batch_size,
            device=str(device),
        )

        # ------------------------------------------------------------------ #
        # Step 2: Apply forward masking process q_{t|0}(x_t | x_0)           #
        # ------------------------------------------------------------------ #
        # x_t: [B, L] — masked sequence (mask_token_id at masked positions)
        # is_masked: [B, L] bool — True at positions that were masked
        x_t: torch.Tensor
        is_masked: torch.Tensor
        x_t, is_masked = self.apply_masking(x0, t)

        # ------------------------------------------------------------------ #
        # Step 3: Forward pass through the time-embedding-free denoiser       #
        # ------------------------------------------------------------------ #
        # logits: [B, L, V] — raw logits for all positions.
        # The model receives only x_t (not t), consistent with the
        # time-embedding-free architecture described in Section 2.
        logits: torch.Tensor = model(x_t)

        # ------------------------------------------------------------------ #
        # Step 4: Per-sample cross-entropy over masked positions              #
        # ------------------------------------------------------------------ #
        # ce_loss: [B] — mean cross-entropy over masked positions per sample.
        # Implements the 1/|M| normalization from Proposition 2.1.
        ce_loss: torch.Tensor = self.cross_entropy_masked(
            logits=logits,
            targets=x0,
            mask=is_masked,
        )

        # ------------------------------------------------------------------ #
        # Step 5: Apply continuous-time ELBO weight |alpha_t'| / (1 - alpha_t)
        # ------------------------------------------------------------------ #
        # alpha_t: [B] — values in [0, 1], probability of token being unmasked
        alpha_t: torch.Tensor = self.noise_schedule.alpha(t)

        # alpha_prime_t: [B] — dα_t/dt, always ≤ 0 (alpha decreases with t)
        alpha_prime_t: torch.Tensor = self.noise_schedule.alpha_prime(t)

        # Use |alpha_prime_t| to get a non-negative weight.
        # For linear schedule: |alpha_prime_t| = 1 (constant).
        # For cosine schedule: |alpha_prime_t| = (pi/2) * sin(pi*t).
        abs_alpha_prime_t: torch.Tensor = torch.abs(alpha_prime_t)

        # Denominator: 1 - alpha_t, clamped away from zero.
        # When t ≈ 0, alpha_t ≈ 1, so 1 - alpha_t ≈ 0.
        # The eps guard prevents division by zero in this regime.
        denominator: torch.Tensor = (1.0 - alpha_t).clamp(min=_EPS)

        # ELBO weight: shape [B].
        weight: torch.Tensor = abs_alpha_prime_t / denominator

        # ------------------------------------------------------------------ #
        # Step 6: Weighted loss, averaged over the batch                      #
        # ------------------------------------------------------------------ #
        # weighted_loss: [B] — per-sample weighted cross-entropy.
        weighted_loss: torch.Tensor = weight * ce_loss

        # Scalar loss: mean over batch.
        loss: torch.Tensor = weighted_loss.mean()

        return loss

    def apply_masking(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the forward masking process q_{t|0}(x_t | x_0).

        Independently masks each token position with probability 1 - alpha_t,
        implementing the coordinate-independent forward process from Section 2:

            q_{t|0}(x_t^i | x_0^i) = Cat(alpha_t * e_{x_0^i} + (1-alpha_t) * e_0)

        Pad positions (if pad_token_id is set) are never masked and are
        excluded from the loss computation.

        Args:
            x0: Clean token sequences of shape ``[B, L]``, dtype
                ``torch.long``.
            t: Noise levels of shape ``[B]``, dtype ``torch.float32``,
                values in ``[0, 1]``.  One noise level per sequence.

        Returns:
            A tuple ``(x_t, is_masked)`` where:

            - ``x_t``: Masked sequence of shape ``[B, L]``, dtype
              ``torch.long``.  Masked positions contain ``mask_token_id``;
              unmasked positions retain their original values from ``x0``.

            - ``is_masked``: Boolean tensor of shape ``[B, L]``.  ``True``
              at positions that were masked (and are non-pad), ``False``
              elsewhere.  Used by ``cross_entropy_masked`` to restrict the
              loss to masked positions.
        """
        batch_size: int = x0.shape[0]
        seq_len: int = x0.shape[1]
        device: torch.device = x0.device

        # ------------------------------------------------------------------ #
        # Compute per-token masking probability: 1 - alpha_t                  #
        # ------------------------------------------------------------------ #
        # alpha_t: [B] → expand to [B, L] for per-token Bernoulli sampling.
        alpha_t: torch.Tensor = self.noise_schedule.alpha(t)  # [B]
        mask_prob: torch.Tensor = 1.0 - alpha_t               # [B]

        # Expand to [B, L]: each token in sequence b has the same mask_prob[b].
        mask_prob_expanded: torch.Tensor = mask_prob.unsqueeze(1).expand(
            batch_size, seq_len
        )  # [B, L]

        # ------------------------------------------------------------------ #
        # Sample Bernoulli mask: 1 = masked, 0 = unmasked                     #
        # ------------------------------------------------------------------ #
        # torch.bernoulli samples 1 with probability p and 0 with probability
        # 1-p.  Here p = mask_prob_expanded, so 1 = masked.
        is_masked: torch.Tensor = torch.bernoulli(mask_prob_expanded).bool()  # [B, L]

        # ------------------------------------------------------------------ #
        # Exclude pad positions from masking                                   #
        # ------------------------------------------------------------------ #
        # Pad tokens should never be masked — they are not real data tokens
        # and should not contribute to the loss.
        if self.pad_token_id is not None:
            is_pad: torch.Tensor = (x0 == self.pad_token_id)  # [B, L], bool
            # Force pad positions to be unmasked.
            is_masked = is_masked & (~is_pad)

        # ------------------------------------------------------------------ #
        # Construct x_t: replace masked positions with mask_token_id          #
        # ------------------------------------------------------------------ #
        x_t: torch.Tensor = x0.clone()
        x_t[is_masked] = self.mask_token_id

        return x_t, is_masked

    def cross_entropy_masked(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes per-sample cross-entropy loss restricted to masked positions.

        Implements the 1/|M| normalization from Proposition 2.1 of the paper:
        the loss for each sample is the mean cross-entropy over its masked
        positions (not the sum), so that samples with more masked tokens do
        not dominate the gradient.

        Args:
            logits: Raw (pre-softmax) logits of shape ``[B, L, V]``.
                Output of ``model.forward(x_t)``.
            targets: Clean token IDs of shape ``[B, L]``, dtype
                ``torch.long``.  The ground-truth tokens that the model
                should predict at masked positions.  This is ``x0``, not
                ``x_t``.
            mask: Boolean tensor of shape ``[B, L]``.  ``True`` at positions
                that were masked (and are non-pad).  Only these positions
                contribute to the loss.  Output of ``apply_masking``.

        Returns:
            Per-sample mean cross-entropy over masked positions, shape
            ``[B]``, dtype ``torch.float32``.  Samples with no masked
            positions (edge case when t ≈ 0) receive a loss of 0.0.

        Note:
            The normalization by ``n_masked.clamp(min=1)`` prevents division
            by zero when no tokens are masked in a sample.  This can occur
            when t is very close to 0 (alpha_t ≈ 1, mask_prob ≈ 0).
        """
        batch_size: int = logits.shape[0]
        seq_len: int = logits.shape[1]
        vocab_size: int = logits.shape[2]

        # ------------------------------------------------------------------ #
        # Compute element-wise cross-entropy for all positions                 #
        # ------------------------------------------------------------------ #
        # Flatten to [B*L, V] and [B*L] for F.cross_entropy.
        logits_flat: torch.Tensor = logits.view(-1, vocab_size)   # [B*L, V]
        targets_flat: torch.Tensor = targets.view(-1)              # [B*L]

        # reduction='none' gives per-token loss of shape [B*L].
        per_token_loss_flat: torch.Tensor = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none",
        )  # [B*L]

        # Reshape back to [B, L].
        per_token_loss: torch.Tensor = per_token_loss_flat.view(
            batch_size, seq_len
        )  # [B, L]

        # ------------------------------------------------------------------ #
        # Zero out non-masked positions                                         #
        # ------------------------------------------------------------------ #
        # Only masked positions contribute to the loss.
        # mask: [B, L] bool — True at masked (non-pad) positions.
        active_loss: torch.Tensor = per_token_loss * mask.float()  # [B, L]

        # ------------------------------------------------------------------ #
        # Sum over sequence dimension and normalize by number of masked tokens #
        # ------------------------------------------------------------------ #
        # per_sample_sum: [B] — sum of cross-entropy over masked positions.
        per_sample_sum: torch.Tensor = active_loss.sum(dim=1)  # [B]

        # n_masked: [B] — number of masked positions per sample.
        # clamp(min=1) prevents division by zero when no tokens are masked.
        n_masked: torch.Tensor = mask.float().sum(dim=1).clamp(min=1.0)  # [B]

        # per_sample_mean: [B] — mean cross-entropy over masked positions.
        # This implements the 1/|M| normalization from Proposition 2.1.
        per_sample_mean: torch.Tensor = per_sample_sum / n_masked  # [B]

        return per_sample_mean
