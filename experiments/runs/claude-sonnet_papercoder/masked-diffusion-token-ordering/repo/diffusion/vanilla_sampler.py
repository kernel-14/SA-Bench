## diffusion/vanilla_sampler.py
"""Vanilla (random-order) sampler for Masked Diffusion Model inference.

Implements the standard MDM reverse process described in Section 2.1.2 of
"Train for the Worst, Plan for the Best: Understanding Token Ordering in
Masked Diffusions".

At each reverse diffusion step t → s, the vanilla sampler:
  (a) Randomly selects K masked positions S ⊆ {i | x_t^i = 0},
      where K ≈ n_masked * (α_s - α_t) / (1 - α_t)
  (b) For each i ∈ S, samples x_s^i ~ p_θ(x^i | x_t)

This is the baseline against which adaptive inference strategies
(TopProbabilityOracle, TopMarginOracle) are compared in Tables 2–5 of the
paper. The key difference from AdaptiveSampler is that position selection
in step (a) is purely random — no oracle guidance is used.

Config alignment (config.yaml):
    noise_schedule.mask_token_id: 0
    sudoku.inference.n_steps: 50
    zebra.inference.n_steps: 50
    nae_sat.inference.n_steps: 50
    text_generation.generation.n_steps: 256

Typical usage::

    schedule = NoiseSchedule(schedule_type='linear', T=50)
    sampler  = VanillaSampler(
        model=mdm_model,
        noise_schedule=schedule,
        n_steps=50,
        mask_token_id=0,
    )

    # Sudoku: x_masked has clue digits (1-9) at given cells, 0 elsewhere.
    predictions = sampler.sample(x_masked)   # [B, 81]

    # Unconditional text generation (fully masked start):
    blank = torch.zeros(B, seq_len, dtype=torch.long)
    generated = sampler.sample(blank)        # [B, seq_len]
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from diffusion.noise_schedule import NoiseSchedule
from models.mdm_transformer import MDMTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default mask token ID, aligned with config.yaml noise_schedule.mask_token_id
_DEFAULT_MASK_TOKEN_ID: int = 0

#: Default number of reverse diffusion steps (puzzle experiments)
_DEFAULT_N_STEPS: int = 50

#: Small epsilon for numerical stability in compute_k denominator
_EPS: float = 1e-8


class VanillaSampler:
    """Standard random-order MDM reverse process sampler.

    Implements vanilla MDM inference where masked positions are selected
    uniformly at random at each reverse diffusion step. This is the baseline
    described in Section 2.1.2 of the paper.

    The sampler is stateless between calls to :meth:`sample` — each call
    runs an independent reverse process from a fully masked (or partially
    known) starting state.

    Attributes:
        model: The MDMTransformer denoiser p_θ(· | x_t).
        noise_schedule: The NoiseSchedule providing α_t values and timesteps.
        n_steps: Number of reverse diffusion steps.
        mask_token_id: Token ID of the [MASK] token.
        device: Device inferred from model parameters.
    """

    def __init__(
        self,
        model: MDMTransformer,
        noise_schedule: NoiseSchedule,
        n_steps: int = _DEFAULT_N_STEPS,
        mask_token_id: int = _DEFAULT_MASK_TOKEN_ID,
    ) -> None:
        """Initialises the VanillaSampler.

        Args:
            model: The MDMTransformer denoiser.  Must expose
                ``get_probs(x_t: Tensor[B,L]) -> Tensor[B,L,V]``.
                Should be in eval mode for inference.
            noise_schedule: Injected NoiseSchedule instance.  Must expose
                ``alpha(t: Tensor) -> Tensor`` and
                ``get_timesteps(n_steps: int) -> Tensor[n_steps+1]``.
                Typically constructed as
                ``NoiseSchedule(schedule_type='linear', T=50)`` per
                config.yaml ``noise_schedule.type = linear``.
            n_steps: Number of reverse diffusion steps.  From config.yaml:
                ``sudoku.inference.n_steps = 50``,
                ``nae_sat.inference.n_steps = 50``,
                ``text_generation.generation.n_steps = 256``.
            mask_token_id: Token ID of the [MASK] token.  From config.yaml:
                ``noise_schedule.mask_token_id = 0``.

        Raises:
            ValueError: If ``n_steps`` is not a positive integer.
        """
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError(
                f"n_steps must be a positive integer, got n_steps={n_steps!r}."
            )

        self.model: MDMTransformer = model
        self.noise_schedule: NoiseSchedule = noise_schedule
        self.n_steps: int = n_steps
        self.mask_token_id: int = mask_token_id

        # Infer device from model parameters.  Falls back to CPU if the model
        # has no parameters (e.g., in unit tests with mock models).
        try:
            self.device: torch.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logger.warning(
                "VanillaSampler: model has no parameters; defaulting to CPU."
            )

        logger.info(
            "VanillaSampler initialised: n_steps=%d, mask_token_id=%d, "
            "device=%s, schedule_type='%s'.",
            self.n_steps,
            self.mask_token_id,
            self.device,
            self.noise_schedule.schedule_type,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(
        self,
        x_masked: torch.Tensor,
        fixed_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs the full vanilla MDM reverse process to generate a sequence.

        Starts from a fully masked sequence and iteratively unmasks tokens
        in random order over ``n_steps`` reverse diffusion steps.  Known
        tokens (puzzle clues or conditioning context) provided via
        ``x_masked`` are fixed throughout the process.

        The reverse process follows the paper's formulation (Section 2.1.2):
        at each step t → s, randomly select K masked positions and fill them
        by sampling from p_θ(x^i | x_t).

        Args:
            x_masked: Initial token tensor of shape ``[B, L]``, dtype
                ``torch.long``.  Positions with known values (puzzle clues,
                conditioning context) contain their token IDs (e.g., digits
                1–9 for Sudoku).  Positions to be predicted contain
                ``mask_token_id = 0``.

                For unconditional generation, pass a fully masked tensor:
                ``torch.zeros(B, L, dtype=torch.long)``.

                For puzzle solving, pass the puzzle encoding where given
                cells contain their digit values and empty cells contain 0.

            fixed_tokens: Optional boolean tensor of shape ``[B, L]``.
                ``True`` at positions that are fixed (known) and must never
                be overwritten during the reverse process.  When ``None``
                (default), the fixed mask is inferred from ``x_masked`` as
                ``x_masked != mask_token_id``.

                Providing this explicitly is useful when some positions
                should be fixed even if they currently contain
                ``mask_token_id`` (e.g., positions that are known to be
                masked in the ground truth).

        Returns:
            Completed token tensor of shape ``[B, L]``, dtype ``torch.long``.
            All non-fixed positions have been filled with sampled token
            values.  Fixed positions retain their original values from
            ``x_masked``.

        Note:
            This method sets the model to eval mode and wraps all forward
            passes in ``torch.no_grad()`` for efficiency.  The model's
            original training/eval state is restored after sampling.
        """
        # Move input to the model's device.
        x_masked = x_masked.to(self.device)

        batch_size: int = x_masked.shape[0]
        seq_len: int = x_masked.shape[1]

        # ------------------------------------------------------------------ #
        # Determine fixed (known) positions                                    #
        # ------------------------------------------------------------------ #
        # fixed_mask[b, i] = True means position i in sample b is a known
        # token that must never be overwritten.
        if fixed_tokens is not None:
            fixed_mask: torch.Tensor = fixed_tokens.to(self.device).bool()
        else:
            # Infer from x_masked: non-mask positions are fixed.
            fixed_mask = (x_masked != self.mask_token_id)  # [B, L], bool

        # ------------------------------------------------------------------ #
        # Initialise x_t as fully masked, then set known positions            #
        # ------------------------------------------------------------------ #
        # Start from the fully masked state x_1 = (0, 0, ..., 0).
        x_t: torch.Tensor = torch.full(
            (batch_size, seq_len),
            fill_value=self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )

        # Set known (fixed) positions to their given values.
        # These positions will never be touched by sample_step since they
        # are not masked (x_t[b, i] != mask_token_id for fixed positions).
        x_t[fixed_mask] = x_masked[fixed_mask]

        # ------------------------------------------------------------------ #
        # Save model training state and switch to eval mode                   #
        # ------------------------------------------------------------------ #
        was_training: bool = self.model.training
        self.model.eval()

        # ------------------------------------------------------------------ #
        # Reverse diffusion loop: t = 1 → 0 over n_steps steps               #
        # ------------------------------------------------------------------ #
        # timesteps: [n_steps + 1] tensor, values from 1.0 down to 0.0.
        # Consecutive pairs (timesteps[i], timesteps[i+1]) give (t, s) where
        # t > s (we are moving from higher noise to lower noise).
        timesteps: torch.Tensor = self.noise_schedule.get_timesteps(self.n_steps)

        with torch.no_grad():
            for step_idx in range(self.n_steps):
                t_val: float = float(timesteps[step_idx].item())
                s_val: float = float(timesteps[step_idx + 1].item())

                # Perform one reverse diffusion step.
                x_t = self.sample_step(x_t, t_val, s_val)

                # Defensive re-application of fixed tokens.
                # sample_step should never touch fixed positions (they are
                # not masked), but this guard handles any edge cases.
                x_t[fixed_mask] = x_masked[fixed_mask]

        # ------------------------------------------------------------------ #
        # Restore model training state                                         #
        # ------------------------------------------------------------------ #
        if was_training:
            self.model.train()

        return x_t

    def sample_step(
        self,
        x_t: torch.Tensor,
        t: float,
        s: float,
    ) -> torch.Tensor:
        """Performs one step of the vanilla MDM reverse process.

        Implements the two-step vanilla inference procedure from Section 2.1.2:
          (a) Randomly select K masked positions S ⊆ {i | x_t^i = 0}
          (b) For each i ∈ S, sample x_s^i ~ p_θ(x^i | x_t)

        The number of positions to unmask K is computed deterministically as:
            K = round(n_masked * (α_s - α_t) / (1 - α_t))
        with a minimum of 1 when any masked positions remain (to prevent
        stalling) and a maximum of n_masked (to prevent over-unmasking).

        This method is called by :meth:`sample` at each reverse diffusion
        step and can also be called directly for custom inference loops.

        Args:
            x_t: Current partially masked sequence of shape ``[B, L]``,
                dtype ``torch.long``.  Masked positions contain
                ``mask_token_id = 0``; unmasked positions contain their
                sampled token values.
            t: Current noise level (higher), a float in ``(0, 1]``.
                Corresponds to ``timesteps[step_idx]`` in the reverse loop.
            s: Target noise level (lower), a float in ``[0, 1)``.
                Corresponds to ``timesteps[step_idx + 1]``.
                Must satisfy ``s < t``.

        Returns:
            Updated sequence ``x_s`` of shape ``[B, L]``, dtype
            ``torch.long``.  A clone of ``x_t`` with K randomly selected
            masked positions filled with sampled token values.  Unmasked
            positions in ``x_t`` are unchanged.

        Note:
            This method assumes the model is already in eval mode and that
            it is called within a ``torch.no_grad()`` context.  The
            :meth:`sample` method ensures both conditions.

            For batches where different samples have different numbers of
            masked tokens, K is computed independently per sample.
        """
        batch_size: int = x_t.shape[0]
        device: torch.device = x_t.device

        # ------------------------------------------------------------------ #
        # Compute α_s and α_t from the noise schedule                         #
        # ------------------------------------------------------------------ #
        # alpha_t: probability that a token is unmasked at noise level t.
        # For linear schedule: alpha_t = 1 - t.
        alpha_t_tensor: torch.Tensor = self.noise_schedule.alpha(
            torch.tensor(t, dtype=torch.float32, device=device)
        )
        alpha_s_tensor: torch.Tensor = self.noise_schedule.alpha(
            torch.tensor(s, dtype=torch.float32, device=device)
        )
        alpha_t: float = float(alpha_t_tensor.item())
        alpha_s: float = float(alpha_s_tensor.item())

        # ------------------------------------------------------------------ #
        # Forward pass: get model probability distributions                   #
        # ------------------------------------------------------------------ #
        # probs: [B, L, V] — per-position probability distributions.
        # Computed once per step and reused for all samples in the batch.
        probs: torch.Tensor = self.model.get_probs(x_t)  # [B, L, V]

        # ------------------------------------------------------------------ #
        # Per-sample random position selection and token sampling             #
        # ------------------------------------------------------------------ #
        # Clone x_t to create x_s — we will fill in selected positions.
        x_s: torch.Tensor = x_t.clone()

        for b in range(batch_size):
            # Find all currently masked positions in sample b.
            # masked_pos: 1D tensor of position indices where x_t[b, i] == mask_token_id
            masked_pos: torch.Tensor = (
                x_t[b] == self.mask_token_id
            ).nonzero(as_tuple=True)[0]  # [n_masked]

            n_masked: int = int(masked_pos.shape[0])

            if n_masked == 0:
                # No masked positions remain in this sample — nothing to do.
                continue

            # Compute K: number of positions to unmask at this step.
            k: int = self.compute_k(n_masked, alpha_s, alpha_t)

            if k == 0:
                # No positions to unmask at this step for this sample.
                continue

            # Randomly select K positions from the masked positions
            # (sampling without replacement via random permutation).
            # This approximates the independent Bernoulli sampling from the
            # paper: P(i ∈ S) = (α_s - α_t) / (1 - α_t).
            perm: torch.Tensor = torch.randperm(n_masked, device=device)
            selected_indices: torch.Tensor = masked_pos[perm[:k]]  # [k]

            # Sample token values at selected positions from the model's
            # predicted probability distribution.
            # probs[b, selected_indices, :]: [k, V]
            position_probs: torch.Tensor = probs[b, selected_indices, :]  # [k, V]

            # torch.multinomial samples one token per row from the categorical
            # distribution defined by each row of position_probs.
            # sampled_tokens: [k, 1] → squeeze to [k]
            sampled_tokens: torch.Tensor = torch.multinomial(
                position_probs,
                num_samples=1,
                replacement=False,
            ).squeeze(-1)  # [k]

            # Write sampled tokens into x_s at the selected positions.
            x_s[b, selected_indices] = sampled_tokens

        return x_s

    def compute_k(
        self,
        n_masked: int,
        alpha_s: float,
        alpha_t: float,
    ) -> int:
        """Computes the number of tokens to unmask at a reverse diffusion step.

        Implements the deterministic K selection from Appendix D.1.2 of the
        paper:

            K = round(n_masked * (α_s - α_t) / (1 - α_t))

        This is the expected number of positions that would be unmasked if
        each masked position were independently included in S with probability
        (α_s - α_t) / (1 - α_t), as in the paper's formulation.

        The paper notes that both deterministic and stochastic (Binomial)
        choices of K give comparable generative perplexity.

        Boundary conditions:
          - Minimum K = 1 when n_masked > 0: prevents the sampler from
            stalling when the computed K rounds to 0 but masked tokens remain.
          - Maximum K = n_masked: prevents requesting more positions than
            are available.
          - When α_t ≈ 1 (denominator near zero): the numerator α_s - α_t
            is also near zero (since α_s ≤ α_t ≤ 1), so K → 0. The minimum
            K = 1 guard handles this gracefully.

        Args:
            n_masked: Number of currently masked positions in the sequence.
                Must be non-negative.
            alpha_s: α_s = α(s), the noise schedule value at the target
                (lower) noise level s.  For linear schedule: α_s = 1 - s.
            alpha_t: α_t = α(t), the noise schedule value at the current
                (higher) noise level t.  For linear schedule: α_t = 1 - t.
                Must satisfy α_s >= α_t (since s < t and α is decreasing).

        Returns:
            Integer K in ``[0, n_masked]``.  Returns 0 only when
            ``n_masked == 0``.

        Note:
            For the linear schedule (α_t = 1 - t), the formula simplifies to:
                K = round(n_masked * (t - s) / t)
            At the first step (t=1, s=1-1/n_steps):
                K = round(n_masked * (1/n_steps) / 1) = round(n_masked / n_steps)
            This distributes the unmasking evenly across all n_steps steps.
        """
        if n_masked <= 0:
            return 0

        # Denominator: 1 - α_t, clamped away from zero for numerical stability.
        # When α_t ≈ 1 (t ≈ 0), both numerator and denominator approach 0,
        # so K → 0. The max(1, ...) guard below handles this case.
        denominator: float = max(1.0 - alpha_t, _EPS)

        # Numerator: α_s - α_t.
        # Since s < t and α is monotonically decreasing, α_s >= α_t, so
        # the numerator is non-negative.
        numerator: float = alpha_s - alpha_t

        # Unmasking probability per position.
        unmask_prob: float = numerator / denominator

        # Expected number of positions to unmask.
        k_float: float = n_masked * unmask_prob

        # Round to nearest integer.
        k: int = int(round(k_float))

        # Enforce minimum K = 1 when masked tokens remain (prevents stalling).
        # This ensures progress is made at every step.
        k = max(1, k)

        # Enforce maximum K = n_masked (cannot unmask more than available).
        k = min(k, n_masked)

        return k
