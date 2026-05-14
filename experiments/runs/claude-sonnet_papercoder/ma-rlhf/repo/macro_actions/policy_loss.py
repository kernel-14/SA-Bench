## macro_actions/policy_loss.py
"""PPO policy and critic loss functions at the macro action level for MA-RLHF.

This module implements MacroPolicyLoss, the core loss modification that
distinguishes MA-PPO from vanilla PPO. The fundamental change is that
advantages are computed and applied at the macro action level (one scalar
per macro action), while importance sampling ratios are still computed
per token. The same advantage scalar is broadcast across all tokens within
a macro action during loss computation.

The implementation follows the PyTorch pseudocode from Appendix E of the
paper exactly:

    for i in range(len(split_list)):
        ratio_i = split_ratio[i]           # per-token ratios within macro action
        advantages_i = advantages[:, i]    # scalar advantage for macro action
        pg_loss1 = -advantages_i * ratio_i
        pg_loss2 = -advantages_i * torch.clamp(ratio_i, 1-clip, 1+clip)
        pg_loss += torch.sum(torch.max(pg_loss1, pg_loss2) * mask_i)
        total_mask_sum += mask_i.sum()
    pg_loss = pg_loss / total_mask_sum

When n_gram=1, each macro action contains exactly one token and the loss
reduces to standard token-level PPO (vanilla PPO baseline). When n_gram=None
(infinity), the entire sequence is one macro action, approximating REINFORCE.

Paper alignments:
  - Section 3.2.2: MA-PPO objective function (Equation for L^MA-PPO).
  - Appendix E: PyTorch pseudocode for policy_loss_macro_action().
  - config.yaml ppo.clip_ratio: 0.2 (default ε for all model sizes).

Dependencies:
    External: torch
    Internal: none (leaf module — no internal project imports)
"""

import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


class MacroPolicyLoss:
    """Computes PPO policy and critic losses at the macro action level.

    This class is instantiated once at trainer initialization and reused
    across all training steps. It has no learnable parameters — it is a
    pure computation module that implements the MA-PPO loss functions.

    The key insight is that the same advantage scalar Â_τ is broadcast
    across all tokens within macro action ω_τ during loss computation,
    while the importance sampling ratio is computed per token. This is
    mathematically equivalent to using the joint probability ratio at the
    macro level (product of per-token ratios), as shown in Section 3.2.2.

    Attributes:
        clip_range: The clipping parameter ε in the PPO objective.
            Default 0.2 matches config.yaml ppo.clip_ratio for all model
            sizes (Table 5 of the paper).
    """

    def __init__(self, clip_range: float = 0.2) -> None:
        """Initialize MacroPolicyLoss with the PPO clipping parameter.

        Args:
            clip_range: The clipping parameter ε in the PPO clipped
                surrogate objective. Controls how far the new policy can
                deviate from the old policy in a single update step.
                Default 0.2 matches config.yaml ppo.clip_ratio.
                Must be positive.

        Raises:
            ValueError: If clip_range is not positive.
        """
        if clip_range <= 0.0:
            raise ValueError(
                f"clip_range must be positive, got {clip_range}. "
                f"The PPO clipping parameter ε must be > 0 to define a "
                f"valid trust region. Default value is 0.2 (config.yaml "
                f"ppo.clip_ratio)."
            )

        self.clip_range: float = clip_range

        logger.info(
            "MacroPolicyLoss initialized: clip_range=%.4f.", clip_range
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_policy_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        sequence: List[int],
    ) -> torch.Tensor:
        """Compute the MA-PPO clipped policy loss.

        Implements the macro-action-level PPO objective from Section 3.2.2:

            L^MA-PPO(θ) = E_τ [min(
                ratio_τ · Â_τ,
                clip(ratio_τ, 1-ε, 1+ε) · Â_τ
            )]

        where ratio_τ = π_θ(ω_τ|s_τ) / π_θ_old(ω_τ|s_τ) is computed
        per-token within each macro action, and Â_τ is the macro-action-
        level advantage from GAE.

        The per-token ratio approach (rather than computing the joint
        probability ratio once per macro action) matches the paper's
        reference implementation in Appendix E and is numerically more
        stable.

        Gradient flow: gradients flow through logprobs (current policy)
        but NOT through old_logprobs (treated as a constant from the
        rollout buffer). The caller must ensure old_logprobs is detached.

        Args:
            logprobs: Per-token log-probabilities under the current policy
                π_θ. Shape: [batch_size, seq_len]. Each element is
                log π_θ(a_t | a_{<t}), gathered at the actual token index.
                This is the response portion only (sliced from start).
                Gradients flow through this tensor.
            old_logprobs: Per-token log-probabilities under the old policy
                π_θ_old (from the rollout buffer). Shape: [batch_size, seq_len].
                Same indexing as logprobs. Must be detached (no gradients).
            advantages: Macro-action-level advantage estimates from GAE.
                Shape: [batch_size, num_macro_actions]. Each element Â_τ
                is the advantage for one macro action, computed by
                RewardUtils.compute_gae() at the macro level.
                Must be detached (no gradients through advantages).
            mask: Binary action mask for valid response tokens.
                Shape: [batch_size, seq_len]. 1 for valid (non-padding)
                tokens, 0 for padding tokens. Corresponds to
                action_mask[:, start:] in the PPO pseudocode (Appendix E).
            sequence: Macro action boundary indices produced by
                MacroActionTermination.get_positions(). Format:
                    [start_offset, b1, b2, ..., end_offset]
                where start_offset=0 (since logprobs is already sliced),
                and end_offset = seq_len. The i-th macro action spans
                tokens [sequence[i], sequence[i+1]).
                len(sequence) - 1 must equal advantages.shape[1].

        Returns:
            Scalar policy loss tensor. Positive values indicate the policy
            is being penalized (loss to minimize). The loss is normalized
            by the total number of valid tokens across all macro actions.

        Raises:
            ValueError: If the number of macro actions derived from
                sequence does not match advantages.shape[1].
            ValueError: If sequence has fewer than 2 elements.
        """
        if len(sequence) < 2:
            raise ValueError(
                f"sequence must have at least 2 elements (start and end), "
                f"got {len(sequence)}: {sequence}."
            )

        # Compute macro action lengths from boundary differences.
        # split_list[i] = length of the i-th macro action in tokens.
        split_list: List[int] = self._compute_split_list(sequence)
        num_macro_actions: int = len(split_list)

        # Validate that split_list is consistent with advantages shape.
        expected_num_macro: int = advantages.shape[1]
        if num_macro_actions != expected_num_macro:
            raise ValueError(
                f"Number of macro actions from sequence ({num_macro_actions}) "
                f"does not match advantages.shape[1] ({expected_num_macro}). "
                f"sequence={sequence}, split_list={split_list}."
            )

        # --- Step 1: Compute per-token importance sampling ratio ---
        # log_ratio = (log π_θ - log π_θ_old) * mask
        # Masking before exp ensures padding positions get ratio=1.0,
        # which is neutral and will be masked out during loss accumulation.
        # Shape: [batch_size, seq_len]
        log_ratio: torch.Tensor = (logprobs - old_logprobs) * mask

        # ratio = π_θ(a_t) / π_θ_old(a_t) at each token position.
        # Shape: [batch_size, seq_len]
        ratio: torch.Tensor = torch.exp(log_ratio)

        # --- Step 2: Split ratio and mask by macro action boundaries ---
        # split_ratio[i]: shape [batch_size, split_list[i]]
        # split_mask[i]:  shape [batch_size, split_list[i]]
        split_ratio: tuple = torch.split(ratio, split_list, dim=-1)
        split_mask: tuple = torch.split(mask, split_list, dim=-1)

        # --- Step 3: Accumulate loss over macro actions ---
        # Following the paper's pseudocode from Appendix E exactly.
        pg_loss: torch.Tensor = torch.tensor(
            0.0, dtype=logprobs.dtype, device=logprobs.device
        )
        total_mask_sum: torch.Tensor = torch.tensor(
            0.0, dtype=logprobs.dtype, device=logprobs.device
        )

        for i in range(num_macro_actions):
            ratio_i: torch.Tensor = split_ratio[i]   # [batch, len_i]
            mask_i: torch.Tensor = split_mask[i]      # [batch, len_i]

            # Scalar advantage for this macro action, per batch item.
            # advantages[:, i] has shape [batch_size].
            # Unsqueeze for broadcasting over the token dimension.
            # Shape: [batch_size, 1]
            advantages_i: torch.Tensor = advantages[:, i].unsqueeze(-1)

            # Unclipped surrogate loss: -Â_τ · ratio_τ
            # Shape: [batch_size, len_i]
            pg_loss1: torch.Tensor = -advantages_i * ratio_i

            # Clipped surrogate loss: -Â_τ · clip(ratio_τ, 1-ε, 1+ε)
            # Shape: [batch_size, len_i]
            pg_loss2: torch.Tensor = -advantages_i * torch.clamp(
                ratio_i,
                1.0 - self.clip_range,
                1.0 + self.clip_range,
            )

            # PPO objective: take the max (pessimistic bound).
            # max(pg_loss1, pg_loss2) selects the more conservative update.
            # Multiply by mask to zero out padding positions.
            # Shape: [batch_size, len_i]
            pg_loss_i: torch.Tensor = (
                torch.max(pg_loss1, pg_loss2) * mask_i
            )

            # Accumulate total loss and valid token count.
            pg_loss = pg_loss + pg_loss_i.sum()
            total_mask_sum = total_mask_sum + mask_i.sum()

        # --- Step 4: Normalize by total valid token count ---
        # Clamp denominator to avoid division by zero for empty batches.
        policy_loss: torch.Tensor = pg_loss / total_mask_sum.clamp(min=1.0)

        return policy_loss

    def compute_critic_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
        sequence: List[int],
    ) -> torch.Tensor:
        """Compute the MA-PPO clipped critic (value function) loss.

        Implements the clipped value loss at the macro action level,
        analogous to the policy clipping. The critic loss prevents large
        value function updates that deviate too far from the old estimates:

            L_value = E_τ [max(
                (V(s_τ) - returns_τ)²,
                (clip(V(s_τ), V_old(s_τ)-ε, V_old(s_τ)+ε) - returns_τ)²
            )]

        Token-level values are aggregated to macro-action-level scalars
        using equal weighting (mean over valid tokens), consistent with
        the default sigma_type='equal' in MacroActionValueEstimator.

        Gradient flow: gradients flow through values (current critic)
        but NOT through old_values or returns (both are constants from
        the rollout buffer). The caller must ensure old_values and returns
        are detached.

        Args:
            values: Current critic value estimates at token level.
                Shape: [batch_size, seq_len]. These are the critic's
                current predictions, which will be updated by this loss.
                Gradients flow through this tensor.
            old_values: Critic value estimates from the rollout phase
                (before the PPO update). Shape: [batch_size, seq_len].
                Used as the center of the clipping range. Must be detached.
            returns: Macro-action-level regression targets from GAE.
                Shape: [batch_size, num_macro_actions]. Computed as
                advantages + values in RewardUtils.compute_gae().
                Must be detached (no gradients through returns).
            mask: Binary action mask for valid response tokens.
                Shape: [batch_size, seq_len]. Same mask as used in
                compute_policy_loss().
            sequence: Macro action boundary indices. Same list as used
                in compute_policy_loss(). len(sequence) - 1 must equal
                returns.shape[1].

        Returns:
            Scalar critic loss tensor. Normalized by the number of macro
            actions times batch size (per-macro-action average loss).

        Raises:
            ValueError: If the number of macro actions derived from
                sequence does not match returns.shape[1].
            ValueError: If sequence has fewer than 2 elements.
        """
        if len(sequence) < 2:
            raise ValueError(
                f"sequence must have at least 2 elements (start and end), "
                f"got {len(sequence)}: {sequence}."
            )

        # Compute macro action lengths from boundary differences.
        split_list: List[int] = self._compute_split_list(sequence)
        num_macro_actions: int = len(split_list)

        # Validate consistency with returns shape.
        expected_num_macro: int = returns.shape[1]
        if num_macro_actions != expected_num_macro:
            raise ValueError(
                f"Number of macro actions from sequence ({num_macro_actions}) "
                f"does not match returns.shape[1] ({expected_num_macro}). "
                f"sequence={sequence}, split_list={split_list}."
            )

        # --- Step 1: Split token-level tensors by macro action boundaries ---
        # split_values[i]:     shape [batch_size, split_list[i]]
        # split_old_values[i]: shape [batch_size, split_list[i]]
        # split_mask[i]:       shape [batch_size, split_list[i]]
        split_values: tuple = torch.split(values, split_list, dim=-1)
        split_old_values: tuple = torch.split(old_values, split_list, dim=-1)
        split_mask: tuple = torch.split(mask, split_list, dim=-1)

        # --- Step 2: Accumulate critic loss over macro actions ---
        total_critic_loss: torch.Tensor = torch.tensor(
            0.0, dtype=values.dtype, device=values.device
        )
        valid_macro_count: int = 0

        for i in range(num_macro_actions):
            values_chunk: torch.Tensor = split_values[i]      # [batch, len_i]
            old_values_chunk: torch.Tensor = split_old_values[i]  # [batch, len_i]
            mask_chunk: torch.Tensor = split_mask[i]           # [batch, len_i]

            # Aggregate token-level values to macro-action-level scalars
            # using equal weighting (mean over valid tokens).
            # Shape: [batch_size]
            values_i: torch.Tensor = self._masked_mean(
                values_chunk, mask_chunk
            )
            old_values_i: torch.Tensor = self._masked_mean(
                old_values_chunk, mask_chunk
            )

            # Regression target for this macro action.
            # Shape: [batch_size]
            returns_i: torch.Tensor = returns[:, i]

            # Clipped value estimate: prevent large deviations from old values.
            # Shape: [batch_size]
            clipped_values_i: torch.Tensor = torch.clamp(
                values_i,
                old_values_i - self.clip_range,
                old_values_i + self.clip_range,
            )

            # Unclipped squared error: (V(s_τ) - returns_τ)²
            # Shape: [batch_size]
            loss_unclipped_i: torch.Tensor = (values_i - returns_i).pow(2)

            # Clipped squared error: (clip(V(s_τ), ...) - returns_τ)²
            # Shape: [batch_size]
            loss_clipped_i: torch.Tensor = (
                clipped_values_i - returns_i
            ).pow(2)

            # Take the max (pessimistic bound, prevents gaming the clip).
            # Shape: [batch_size]
            critic_loss_i: torch.Tensor = torch.max(
                loss_unclipped_i, loss_clipped_i
            )

            # Accumulate over batch dimension.
            total_critic_loss = total_critic_loss + critic_loss_i.sum()
            valid_macro_count += critic_loss_i.size(0)  # batch_size

        # --- Step 3: Normalize by total macro action slots ---
        # Normalize by (num_macro_actions * batch_size) for a per-macro-action
        # average loss. This is standard for value function losses.
        normalizer: float = float(max(valid_macro_count, 1))
        critic_loss: torch.Tensor = total_critic_loss / normalizer

        return critic_loss

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_split_list(sequence: List[int]) -> List[int]:
        """Compute macro action lengths from boundary indices.

        Converts a list of boundary indices into a list of segment lengths
        using consecutive differences. This is the split_list used by
        torch.split() to partition token-level tensors into per-macro-action
        chunks.

        Example:
            sequence = [0, 5, 10, 13]
            split_list = [5, 5, 3]  (lengths of 3 macro actions)

        Args:
            sequence: Sorted list of integer boundary indices. Must have
                at least 2 elements. sequence[0] is the start offset and
                sequence[-1] is the end offset in the sliced tensor.

        Returns:
            List of positive integers representing the length of each
            macro action. Length is len(sequence) - 1.

        Raises:
            ValueError: If any computed length is non-positive (which
                would indicate duplicate or out-of-order boundaries).
        """
        split_list: List[int] = []

        for idx in range(len(sequence) - 1):
            length: int = int(sequence[idx + 1]) - int(sequence[idx])
            if length <= 0:
                raise ValueError(
                    f"Non-positive macro action length {length} at index {idx}. "
                    f"sequence must be strictly increasing. "
                    f"sequence={sequence}."
                )
            split_list.append(length)

        return split_list

    @staticmethod
    def _masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean of valid (non-padding) elements per batch item.

        Used in compute_critic_loss() to aggregate token-level values to
        macro-action-level scalars with equal weighting (sigma_type='equal').

        This is the same computation as MacroActionValueEstimator._equal_weight()
        but implemented inline here to keep MacroPolicyLoss self-contained
        (no internal project dependencies).

        Args:
            tensor: Values to average. Shape: [batch_size, seq_len].
                May be in bf16; cast to float32 for numerical accuracy.
            mask: Binary validity mask. Shape: [batch_size, seq_len].
                1 for valid tokens, 0 for padding tokens.

        Returns:
            Mean value per batch item. Shape: [batch_size].
            Returns 0.0 for batch items where all tokens are padding
            (valid_count == 0), avoiding division by zero.
        """
        # Cast to float32 for numerical accuracy (bf16 has limited precision).
        tensor_f32: torch.Tensor = tensor.float()
        mask_f32: torch.Tensor = mask.float()

        # Zero out padding positions before summing.
        # Shape: [batch_size, seq_len]
        masked_tensor: torch.Tensor = tensor_f32 * mask_f32

        # Sum of valid values per batch item. Shape: [batch_size]
        valid_sum: torch.Tensor = masked_tensor.sum(dim=-1)

        # Count of valid tokens per batch item. Shape: [batch_size]
        valid_count: torch.Tensor = mask_f32.sum(dim=-1)

        # Mean: divide by valid count, clamping to avoid division by zero.
        # When valid_count == 0, both numerator and denominator are 0,
        # so the result is 0.0 (correct: empty macro action has value 0).
        result: torch.Tensor = valid_sum / valid_count.clamp(min=1.0)

        # Cast back to the original dtype for consistency with downstream ops.
        return result.to(dtype=tensor.dtype)
