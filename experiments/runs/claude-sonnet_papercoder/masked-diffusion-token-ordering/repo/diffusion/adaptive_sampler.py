## diffusion/adaptive_sampler.py
"""Adaptive (certainty-guided) sampler for Masked Diffusion Model inference.

Implements the adaptive MDM inference strategy from Section 4.1 of
"Train for the Worst, Plan for the Best: Understanding Token Ordering in
Masked Diffusions".

At each reverse diffusion step t → s, the adaptive sampler:
  (a) Uses an oracle F(θ, x_t) to select the K most certain masked positions
      S = oracle.select_positions(probs, masked_positions, K)
  (b) For each i ∈ S, samples x_s^i ~ p_θ(x^i | x_t)

This contrasts with VanillaSampler, which selects positions randomly in step
(a). The oracle uses certainty scores (max probability or probability margin)
to identify which tokens the model is most confident about, allowing the MDM
to sidestep hard masking subproblems without any retraining.

Key results from the paper (Tables 2–5):
  - Sudoku: vanilla 6.88% → Top Probability Margin 89.49%
  - Zebra:  vanilla 76.9% → Top Probability 98.5%, Top Margin 98.3%
  - Hard Sudoku: ARM-with-ordering 32.57% → MDM Top Margin 49.88%

Config alignment (config.yaml):
    noise_schedule.mask_token_id: 0
    sudoku.inference.n_steps: 50
    sudoku.inference.gumbel_coeff: 0.5   (passed to oracle at construction)
    zebra.inference.n_steps: 50
    nae_sat.inference.n_steps: 50
    llada_eval.inference.n_steps: 256
    text_generation.generation.n_steps: 256
    text_generation.inference.noise_sigma: 0.1  (passed to oracle)

Typical usage::

    # Puzzle experiments (Sudoku, Zebra)
    schedule = NoiseSchedule(schedule_type='linear', T=50)
    oracle   = TopMarginOracle(gumbel_coeff=0.5, noise_sigma=0.0)
    sampler  = AdaptiveSampler(
        model=mdm_model,
        noise_schedule=schedule,
        oracle=oracle,
        n_steps=50,
        mask_token_id=0,
    )
    predictions = sampler.sample(x_masked)   # [B, 81] for Sudoku

    # Text generation (unconditional)
    oracle  = TopMarginOracle(gumbel_coeff=0.0, noise_sigma=0.1)
    sampler = AdaptiveSampler(model=mdm_1b, noise_schedule=schedule,
                              oracle=oracle, n_steps=256, mask_token_id=0)
    blank   = torch.zeros(B, seq_len, dtype=torch.long)
    texts   = sampler.sample(blank)          # [B, seq_len]
"""

import logging
from typing import List, Optional

import torch

from diffusion.noise_schedule import NoiseSchedule
from models.mdm_transformer import MDMTransformer
from oracles.base_oracle import BaseOracle

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


class AdaptiveSampler:
    """Certainty-guided adaptive MDM reverse process sampler.

    Implements adaptive MDM inference where masked positions are selected
    by an oracle based on the model's certainty scores, rather than randomly.
    This is the key inference-time innovation of the paper (Section 4.1).

    The sampler is stateless between calls to :meth:`sample` — each call
    runs an independent reverse process from a fully masked (or partially
    known) starting state.

    The oracle encapsulates all position-selection logic and noise injection
    (Gumbel for puzzles, Gaussian for text). The sampler itself is agnostic
    to the specific oracle strategy — it works with any :class:`BaseOracle`
    subclass.

    Attributes:
        model: The MDMTransformer denoiser p_θ(· | x_t).
        noise_schedule: The NoiseSchedule providing α_t values and timesteps.
        oracle: The position-selection oracle F(θ, x_t).
        n_steps: Number of reverse diffusion steps.
        mask_token_id: Token ID of the [MASK] token.
        device: Device inferred from model parameters.
    """

    def __init__(
        self,
        model: MDMTransformer,
        noise_schedule: NoiseSchedule,
        oracle: BaseOracle,
        n_steps: int = _DEFAULT_N_STEPS,
        mask_token_id: int = _DEFAULT_MASK_TOKEN_ID,
    ) -> None:
        """Initialises the AdaptiveSampler.

        Args:
            model: The MDMTransformer denoiser. Must expose
                ``get_probs(x_t: Tensor[B,L]) -> Tensor[B,L,V]``.
                Should be in eval mode for inference.
            noise_schedule: Injected NoiseSchedule instance. Must expose
                ``alpha(t: Tensor) -> Tensor`` and
                ``get_timesteps(n_steps: int) -> Tensor[n_steps+1]``.
                Typically constructed as
                ``NoiseSchedule(schedule_type='linear', T=50)`` per
                config.yaml ``noise_schedule.type = linear``.
            oracle: The position-selection oracle. Must expose
                ``select_positions(probs, masked_positions, k) -> Tensor[B,k]``.
                Concrete implementations: ``TopProbabilityOracle`` or
                ``TopMarginOracle``. The oracle's noise parameters
                (``gumbel_coeff``, ``noise_sigma``) are set at oracle
                construction time, not here.
            n_steps: Number of reverse diffusion steps. From config.yaml:
                ``sudoku.inference.n_steps = 50``,
                ``nae_sat.inference.n_steps = 50``,
                ``llada_eval.inference.n_steps = 256``,
                ``text_generation.generation.n_steps = 256``.
            mask_token_id: Token ID of the [MASK] token. From config.yaml:
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
        self.oracle: BaseOracle = oracle
        self.n_steps: int = n_steps
        self.mask_token_id: int = mask_token_id

        # Infer device from model parameters. Falls back to CPU if the model
        # has no parameters (e.g., in unit tests with mock models).
        try:
            self.device: torch.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
            logger.warning(
                "AdaptiveSampler: model has no parameters; defaulting to CPU."
            )

        logger.info(
            "AdaptiveSampler initialised: n_steps=%d, mask_token_id=%d, "
            "device=%s, schedule_type='%s', oracle=%s.",
            self.n_steps,
            self.mask_token_id,
            self.device,
            self.noise_schedule.schedule_type,
            type(self.oracle).__name__,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample(
        self,
        x_masked: torch.Tensor,
        fixed_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs the full adaptive MDM reverse process to generate a sequence.

        Starts from a fully masked sequence and iteratively unmasks tokens
        in certainty-guided order over ``n_steps`` reverse diffusion steps.
        Known tokens (puzzle clues or conditioning context) provided via
        ``x_masked`` are fixed throughout the process.

        The reverse process follows the adaptive inference formulation from
        Section 4.1 of the paper:
          (a) S = F(θ, x_t) — oracle selects K most certain masked positions
          (b) x_s^i ~ p_θ(x^i | x_t) for each i ∈ S — sample token values

        Args:
            x_masked: Initial token tensor of shape ``[B, L]``, dtype
                ``torch.long``. Positions with known values (puzzle clues,
                conditioning context) contain their token IDs (e.g., digits
                1–9 for Sudoku). Positions to be predicted contain
                ``mask_token_id = 0``.

                For unconditional generation, pass a fully masked tensor:
                ``torch.zeros(B, L, dtype=torch.long)``.

                For puzzle solving, pass the puzzle encoding where given
                cells contain their digit values and empty cells contain 0.

            fixed_tokens: Optional boolean tensor of shape ``[B, L]``.
                ``True`` at positions that are fixed (known) and must never
                be overwritten during the reverse process. When ``None``
                (default), the fixed mask is inferred from ``x_masked`` as
                ``x_masked != mask_token_id``.

                Providing this explicitly is useful when some positions
                should be fixed even if they currently contain
                ``mask_token_id`` (e.g., positions that are known to be
                masked in the ground truth).

        Returns:
            Completed token tensor of shape ``[B, L]``, dtype ``torch.long``.
            All non-fixed positions have been filled with sampled token
            values. Fixed positions retain their original values from
            ``x_masked``.

        Note:
            This method sets the model to eval mode and wraps all forward
            passes in ``torch.no_grad()`` for efficiency. The model's
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
        # token that must never be overwritten by the oracle or sampler.
        if fixed_tokens is not None:
            fixed_mask: torch.Tensor = fixed_tokens.to(self.device).bool()
        else:
            # Infer from x_masked: non-mask positions are fixed.
            fixed_mask = (x_masked != self.mask_token_id)  # [B, L], bool

        # ------------------------------------------------------------------ #
        # Initialise x_t as fully masked, then set known positions            #
        # ------------------------------------------------------------------ #
        # Start from the fully masked state x_1 = (0, 0, ..., 0).
        # This represents the starting point of the reverse process where
        # alpha_1 ≈ 0 means all tokens are masked.
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
        # t > s (moving from higher noise to lower noise).
        timesteps: torch.Tensor = self.noise_schedule.get_timesteps(self.n_steps)

        with torch.no_grad():
            for step_idx in range(self.n_steps):
                # Early termination: if no masked positions remain in any
                # sample, there is nothing left to unmask.
                n_still_masked: int = int(
                    (x_t == self.mask_token_id).sum().item()
                )
                if n_still_masked == 0:
                    logger.debug(
                        "AdaptiveSampler: all positions unmasked after %d/%d "
                        "steps. Terminating early.",
                        step_idx,
                        self.n_steps,
                    )
                    break

                t_val: float = float(timesteps[step_idx].item())
                s_val: float = float(timesteps[step_idx + 1].item())

                # Perform one adaptive reverse diffusion step.
                x_t = self.sample_step(x_t, t_val, s_val)

                # Defensive re-application of fixed tokens after each step.
                # sample_step should never touch fixed positions (they are
                # not masked), but this guard handles any edge cases where
                # the oracle might inadvertently select a fixed position.
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
        """Performs one step of the adaptive MDM reverse process.

        Implements the two-step adaptive inference procedure from Section 4.1:
          (a) S = F(θ, x_t) = oracle.select_positions(probs, masked_pos, K)
          (b) For each i ∈ S, sample x_s^i ~ p_θ(x^i | x_t)

        The key difference from VanillaSampler.sample_step is step (a):
        positions are selected by the oracle based on certainty scores
        rather than uniformly at random.

        The number of positions to unmask K is computed per-sample as:
            K = round(n_masked * (α_s - α_t) / (1 - α_t))
        with a minimum of 1 when any masked positions remain (to prevent
        stalling) and a maximum of n_masked (to prevent over-unmasking).

        This method is called by :meth:`sample` at each reverse diffusion
        step and can also be called directly for custom inference loops.

        Args:
            x_t: Current partially masked sequence of shape ``[B, L]``,
                dtype ``torch.long``. Masked positions contain
                ``mask_token_id = 0``; unmasked positions contain their
                sampled token values.
            t: Current noise level (higher), a float in ``(0, 1]``.
                Corresponds to ``timesteps[step_idx]`` in the reverse loop.
            s: Target noise level (lower), a float in ``[0, 1)``.
                Corresponds to ``timesteps[step_idx + 1]``.
                Must satisfy ``s < t``.

        Returns:
            Updated sequence ``x_s`` of shape ``[B, L]``, dtype
            ``torch.long``. A clone of ``x_t`` with K oracle-selected
            masked positions filled with sampled token values. Unmasked
            positions in ``x_t`` are unchanged.

        Note:
            This method assumes the model is already in eval mode and that
            it is called within a ``torch.no_grad()`` context. The
            :meth:`sample` method ensures both conditions.

            K is computed independently per sample to handle variable numbers
            of masked tokens across a batch (e.g., different puzzle difficulties
            in Sudoku where some puzzles have more given cells than others).
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
        # Sub-step 1: Forward pass — get model probability distributions      #
        # ------------------------------------------------------------------ #
        # probs: [B, L, V] — per-position probability distributions.
        # p_theta(x^i = j | x_t) for all positions i and vocabulary values j.
        # Computed once per step and reused for oracle scoring and token sampling.
        probs: torch.Tensor = self.model.get_probs(x_t)  # [B, L, V]

        # ------------------------------------------------------------------ #
        # Sub-step 2: Find masked positions per sample                        #
        # ------------------------------------------------------------------ #
        # masked_positions_list: List of B tensors, each containing the
        # sequence-dimension indices of masked positions in that sample.
        # Variable-length per sample since different puzzles have different
        # numbers of empty cells.
        masked_positions_list: List[torch.Tensor] = self.get_masked_positions(x_t)

        # ------------------------------------------------------------------ #
        # Sub-step 3 & 4: Per-sample oracle selection and token sampling      #
        # ------------------------------------------------------------------ #
        # Clone x_t to create x_s — we will fill in oracle-selected positions.
        x_s: torch.Tensor = x_t.clone()

        # Build a batched boolean mask for oracle input: [B, L], True = masked.
        # This is used by the oracle's score() method to identify eligible positions.
        masked_positions_bool: torch.Tensor = (x_t == self.mask_token_id)  # [B, L]

        for b in range(batch_size):
            masked_pos: torch.Tensor = masked_positions_list[b]  # [n_masked]
            n_masked: int = int(masked_pos.shape[0])

            if n_masked == 0:
                # No masked positions remain in this sample — nothing to do.
                continue

            # ---------------------------------------------------------------- #
            # Sub-step 3: Compute K — number of positions to unmask            #
            # ---------------------------------------------------------------- #
            k: int = self.compute_k(n_masked, alpha_s, alpha_t)

            if k == 0:
                # No positions to unmask at this step for this sample.
                continue

            # ---------------------------------------------------------------- #
            # Sub-step 4: Oracle selects K most certain masked positions       #
            # ---------------------------------------------------------------- #
            # The oracle operates on a single-sample slice to handle per-sample K.
            # probs_b: [1, L, V] — single-sample probability distributions.
            # masked_positions_bool_b: [1, L] — single-sample masked positions.
            probs_b: torch.Tensor = probs[b:b+1, :, :]          # [1, L, V]
            masked_bool_b: torch.Tensor = masked_positions_bool[b:b+1, :]  # [1, L]

            # oracle.select_positions returns [1, k_actual] indices.
            # k_actual may be less than k if fewer than k positions are masked.
            selected_indices_2d: torch.Tensor = self.oracle.select_positions(
                probs=probs_b,
                masked_positions=masked_bool_b,
                k=k,
            )  # [1, k_actual]

            # Squeeze batch dimension: [k_actual]
            if selected_indices_2d.shape[1] == 0:
                # Oracle returned no positions (edge case: k_actual = 0).
                continue

            selected_indices: torch.Tensor = selected_indices_2d.squeeze(0)  # [k_actual]

            # ---------------------------------------------------------------- #
            # Sub-step 5: Sample token values at selected positions            #
            # ---------------------------------------------------------------- #
            # For each selected position i, sample x_s^i ~ p_θ(x^i | x_t).
            # probs[b, selected_indices, :]: [k_actual, V]
            position_probs: torch.Tensor = probs[b, selected_indices, :]  # [k_actual, V]

            # torch.multinomial samples one token per row from the categorical
            # distribution defined by each row of position_probs.
            # This is multinomial (not argmax) sampling to preserve stochasticity,
            # consistent with the paper's use of Gumbel noise in the oracle.
            # sampled_tokens: [k_actual, 1] → squeeze to [k_actual]
            sampled_tokens: torch.Tensor = torch.multinomial(
                position_probs,
                num_samples=1,
                replacement=False,
            ).squeeze(-1)  # [k_actual]

            # ---------------------------------------------------------------- #
            # Sub-step 6: Write sampled tokens into x_s                       #
            # ---------------------------------------------------------------- #
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

        This ensures the expected number of tokens unmasked at each step
        matches vanilla MDM inference, keeping the marginal distributions
        aligned with what the model saw during training.

        The paper notes: "we set the number of tokens to unmask K so that
        the number of unmasked tokens matches that of vanilla MDM inference
        in expectation."

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
                (lower) noise level s. For linear schedule: α_s = 1 - s.
            alpha_t: α_t = α(t), the noise schedule value at the current
                (higher) noise level t. For linear schedule: α_t = 1 - t.
                Must satisfy α_s >= α_t (since s < t and α is decreasing).

        Returns:
            Integer K in ``[0, n_masked]``. Returns 0 only when
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
        # This ensures progress is made at every step even when the computed
        # K rounds to 0 due to small step sizes or near-boundary alpha values.
        k = max(1, k)

        # Enforce maximum K = n_masked (cannot unmask more than available).
        k = min(k, n_masked)

        return k

    def get_masked_positions(
        self,
        x_t: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Returns the indices of masked positions for each sample in the batch.

        For each sample b in the batch, returns a 1D tensor of sequence-
        dimension indices where ``x_t[b, i] == mask_token_id``. These are
        the candidate positions eligible for unmasking at the current step.

        Returns a Python list (not a padded tensor) because different samples
        may have different numbers of masked positions — especially in puzzle
        settings where different puzzles have different numbers of given cells.

        Args:
            x_t: Current partially masked sequence of shape ``[B, L]``,
                dtype ``torch.long``. Masked positions contain
                ``mask_token_id``; unmasked positions contain token values.

        Returns:
            List of length ``B``. Each element is a 1D ``torch.LongTensor``
            of shape ``[n_masked_b]`` containing the sequence-dimension
            indices of masked positions in sample b. The list element for
            sample b is empty (shape ``[0]``) if no positions are masked.

        Example::

            x_t = torch.tensor([[0, 3, 0, 5], [2, 0, 0, 1]])
            # mask_token_id = 0
            masked = sampler.get_masked_positions(x_t)
            # masked[0] = tensor([0, 2])  (positions 0 and 2 are masked)
            # masked[1] = tensor([1, 2])  (positions 1 and 2 are masked)
        """
        batch_size: int = x_t.shape[0]

        masked_positions: List[torch.Tensor] = []
        for b in range(batch_size):
            # nonzero(as_tuple=True)[0] returns a 1D tensor of indices where
            # the condition is True. This is equivalent to torch.where but
            # returns only the indices (not values).
            indices: torch.Tensor = (
                x_t[b] == self.mask_token_id
            ).nonzero(as_tuple=True)[0]  # [n_masked_b], dtype torch.long
            masked_positions.append(indices)

        return masked_positions
