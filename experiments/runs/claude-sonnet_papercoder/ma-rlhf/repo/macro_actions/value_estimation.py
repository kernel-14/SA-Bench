## macro_actions/value_estimation.py
"""Macro action value and reward aggregation for MA-RLHF.

This module implements MacroActionValueEstimator, which bridges token-level
critic outputs and macro-action-level optimization. The critic model produces
a value estimate at every token position, but MA-PPO needs a single scalar
value per macro action to compute advantages via GAE. This module performs
that aggregation using one of three weighting schemes (sigma types).

Three sigma types are supported, as described in Appendix D.1 of the paper:
  1. equal (default): Mean of token values within the macro action.
     Best RM scores per Figure 19 ablation.
  2. unit: Value of the last valid token only.
     Best consistency/fluency per GPT-4 evaluation (Figure 19 right).
  3. position_decayed: Harmonic-weighted sum giving more weight to later
     tokens within the macro action.

The same sigma_type weighting is applied to both values and rewards for
consistency in the GAE advantage computation.

Paper alignments:
  - Appendix D.1: Value function estimation formula and three sigma types.
  - Algorithm 1 (Appendix E): get_macro_action_values() pseudocode.
  - config.yaml macro_action.sigma_type: 'equal' (default).
  - config.yaml ppo.rho: 1.0 (intra-macro discount, simplifies reward sum).

Data flow:
    token_values: [batch, full_seq_len]
         │ slice [:, start:]
         ▼
    token_values_response: [batch, response_len]
         │ torch.split(split_list, dim=-1)
         ▼
    [chunk_0, chunk_1, ..., chunk_T]  each: [batch, n_i]
         │ _equal_weight / _unit_weight / _position_decayed_weight
         ▼
    [scalar_0, scalar_1, ..., scalar_T]  each: [batch]
         │ torch.stack(dim=-1)
         ▼
    macro_values: [batch, num_macro_actions]

Dependencies:
    External: torch
    Internal: config.py (MacroActionConfig)
"""

import logging
from typing import List

import torch

from config import MacroActionConfig

logger = logging.getLogger(__name__)


class MacroActionValueEstimator:
    """Aggregates token-level values and rewards into macro-action-level scalars.

    This class is instantiated once at trainer initialization and reused
    across all training steps. It is a pure computation module with no
    model parameters or learnable components.

    The sigma_type determines how token-level values within a macro action
    are combined into a single scalar. The default 'equal' (mean) achieves
    the best RM scores in the paper's ablation study (Figure 19).

    Attributes:
        config: MacroActionConfig with sigma_type and other hyperparameters.
        sigma_type: The active weighting scheme ('equal', 'unit', or
            'position_decayed'). Cached from config for fast access.
    """

    # Valid sigma type identifiers.
    _VALID_SIGMA_TYPES: tuple = ("equal", "unit", "position_decayed")

    def __init__(self, config: MacroActionConfig) -> None:
        """Initialize the value estimator with the given configuration.

        No heavy initialization is needed — this is a pure computation
        module. The sigma_type is validated and cached for fast dispatch.

        Args:
            config: MacroActionConfig instance. The sigma_type field
                determines which weighting scheme is used. Valid values:
                  - 'equal': mean of valid token values (default, best RM).
                  - 'unit': value of the last valid token only.
                  - 'position_decayed': harmonic-weighted sum.
                All values are sourced from config.yaml macro_action section.

        Raises:
            ValueError: If config.sigma_type is not a valid weighting scheme.
        """
        if config.sigma_type not in self._VALID_SIGMA_TYPES:
            raise ValueError(
                f"config.sigma_type must be one of {self._VALID_SIGMA_TYPES}, "
                f"got '{config.sigma_type}'."
            )

        self.config: MacroActionConfig = config
        self.sigma_type: str = config.sigma_type

        logger.info(
            "MacroActionValueEstimator initialized: sigma_type='%s'.",
            self.sigma_type,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_macro_values(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        start: int,
        sequence: List[int],
    ) -> torch.Tensor:
        """Aggregate token-level critic values into macro-action-level scalars.

        Implements the value function estimation from Appendix D.1:
            V^π(s_τ, ω_τ) = Σ_{i=0}^{|ω_τ|} σ_{t_τ+i} · V^π(s_{t_τ+i}, a_{t_τ+i})

        where σ_τ is determined by the configured sigma_type.

        The paper's pseudocode (Appendix E) for this operation:
            split_list = torch.diff(torch.tensor(sequence)).tolist()
            splited_values = torch.split(values[:, start:], split_list, dim=-1)
            splited_mask = torch.split(mask[:, start:], split_list, dim=-1)
            inplace_values = torch.zeros(1, len(split_list), ...)
            for idx, (value_i, mask_i) in enumerate(zip(splited_values, splited_mask)):
                masked_values = value_i[mask_i != 0]
                inplace_values[0, idx] = torch.mean(masked_values) if ... else 0.0

        Args:
            values: Token-level value estimates from the critic model.
                Shape: [batch_size, full_seq_len]. Each element V(s_t, a_t)
                is the critic's estimate at token position t. In bf16 during
                training; intermediate computations use float32.
            mask: Attention/action mask of shape [batch_size, full_seq_len].
                1 for valid (non-padding) tokens, 0 for padding tokens.
                Corresponds to action_mask = attention_mask[:, 1:] in the
                PPO pseudocode (Appendix E).
            start: Index where the response begins in the full sequence.
                Corresponds to prompts.size()[-1] - 1 in the PPO pseudocode.
                Values at positions [0, start) are prompt positions and are
                excluded from macro action computation.
            sequence: List of integer boundary indices produced by
                MacroActionTermination.get_positions(). Format:
                    [start, b1, b2, ..., end]
                where end = mask.size(1) - 1. The length of each macro
                action i is sequence[i+1] - sequence[i].

        Returns:
            Macro-action-level value tensor of shape
            [batch_size, num_macro_actions], where num_macro_actions =
            len(sequence) - 1. Each element is the aggregated value for
            one macro action, computed using the configured sigma_type.
            Returns shape [batch_size, 0] if sequence has fewer than 2
            elements (no macro actions).

        Raises:
            ValueError: If start is out of bounds for the given tensors.
        """
        return self._aggregate(values, mask, start, sequence)

    def get_macro_rewards(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        start: int,
        sequence: List[int],
    ) -> torch.Tensor:
        """Aggregate per-token rewards into macro-action-level scalars.

        Applies the same sigma_type weighting as get_macro_values() to
        per-token rewards. This is consistent with the paper's formulation
        where both values and rewards use the same aggregation scheme.

        The macro reward R_τ with ρ=1.0 (config.yaml ppo.rho) is:
            R_τ = Σ_{i=0}^{|ω_τ|-1} ρ^i · r_{t_τ+i} = Σ r_{t_τ+i}

        Since ρ=1.0, this is a plain sum. However, the sigma_type weighting
        is applied for consistency with value aggregation. The 'equal' sigma
        type computes the mean (not sum), which is equivalent to sum when
        combined with the GAE computation that accounts for macro action
        lengths.

        The RM score is placed at the last valid response token by
        RewardUtils.reshape_reward(). When aggregated, it appears in the
        terminal macro action's reward, while earlier macro actions contain
        only KL penalties. This is the correct RLHF behavior.

        Args:
            rewards: Per-token KL-penalized rewards from RewardUtils.reshape_reward().
                Shape: [batch_size, full_seq_len]. Contains the RM score at
                the terminal token and -β·KL_t at all other positions.
            mask: Attention/action mask of shape [batch_size, full_seq_len].
                Same mask used for get_macro_values().
            start: Response start index. Same value used for get_macro_values().
            sequence: Macro action boundary indices. Same list used for
                get_macro_values().

        Returns:
            Macro-action-level reward tensor of shape
            [batch_size, num_macro_actions]. Same shape as the output of
            get_macro_values(), enabling direct use in RewardUtils.compute_gae().
        """
        return self._aggregate(rewards, mask, start, sequence)

    # ------------------------------------------------------------------
    # Private aggregation dispatcher
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
        start: int,
        sequence: List[int],
    ) -> torch.Tensor:
        """Core aggregation logic shared by get_macro_values and get_macro_rewards.

        Slices the tensor to the response region, splits by macro action
        boundaries, applies sigma weighting to each segment, and stacks
        the results.

        Args:
            tensor: Token-level tensor of shape [batch_size, full_seq_len].
            mask: Attention mask of shape [batch_size, full_seq_len].
            start: Response start index.
            sequence: Macro action boundary indices [start, b1, ..., end].

        Returns:
            Aggregated tensor of shape [batch_size, num_macro_actions].
        """
        batch_size: int = tensor.size(0)

        # Validate start index.
        if start < 0 or start >= tensor.size(1):
            raise ValueError(
                f"start={start} is out of bounds for tensor with "
                f"seq_len={tensor.size(1)}."
            )

        # Handle degenerate case: fewer than 2 boundary points → no macro actions.
        if len(sequence) < 2:
            logger.debug(
                "sequence has fewer than 2 elements (%d); "
                "returning empty macro tensor.",
                len(sequence),
            )
            return torch.zeros(
                batch_size,
                0,
                dtype=tensor.dtype,
                device=tensor.device,
            )

        # Compute macro action lengths from boundary differences.
        # torch.diff([start, b1, b2, ..., end]) = [b1-start, b2-b1, ..., end-b_{T-1}]
        # Each element is the number of tokens in the corresponding macro action.
        boundary_tensor: torch.Tensor = torch.tensor(
            sequence, dtype=torch.long
        )
        split_list: List[int] = torch.diff(boundary_tensor).tolist()
        split_list = [int(s) for s in split_list]

        # Guard: all split lengths must be positive.
        if any(s <= 0 for s in split_list):
            logger.warning(
                "Non-positive split length detected in split_list=%s. "
                "Sequence may have duplicate or out-of-order boundaries. "
                "Falling back to single macro action.",
                split_list,
            )
            # Fallback: treat entire response as one macro action.
            response_tensor: torch.Tensor = tensor[:, start:]
            response_mask: torch.Tensor = mask[:, start:]
            result: torch.Tensor = self._dispatch_weight(
                response_tensor, response_mask
            )
            return result.unsqueeze(-1)

        # Slice to response region: [batch, response_len].
        # response_len = sum(split_list) = sequence[-1] - sequence[0]
        response_len: int = sum(split_list)
        tensor_response: torch.Tensor = tensor[:, start: start + response_len]
        mask_response: torch.Tensor = mask[:, start: start + response_len]

        # Guard: ensure sliced length matches expected response_len.
        actual_len: int = tensor_response.size(1)
        if actual_len < response_len:
            logger.warning(
                "Tensor response region (%d tokens) is shorter than "
                "expected (%d tokens from sequence). Adjusting split_list.",
                actual_len,
                response_len,
            )
            # Truncate split_list to fit actual tensor length.
            split_list = self._truncate_split_list(split_list, actual_len)
            if not split_list:
                return torch.zeros(
                    batch_size,
                    0,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            tensor_response = tensor[:, start: start + sum(split_list)]
            mask_response = mask[:, start: start + sum(split_list)]

        # Split tensor and mask into per-macro-action chunks.
        # Each chunk has shape [batch, macro_action_len_i].
        split_tensors: tuple = torch.split(tensor_response, split_list, dim=-1)
        split_masks: tuple = torch.split(mask_response, split_list, dim=-1)

        num_macro_actions: int = len(split_list)

        # Aggregate each macro action chunk into a scalar per batch item.
        # Results list: each element has shape [batch].
        results: List[torch.Tensor] = []
        for idx in range(num_macro_actions):
            chunk_tensor: torch.Tensor = split_tensors[idx]  # [batch, n_i]
            chunk_mask: torch.Tensor = split_masks[idx]       # [batch, n_i]

            # Apply sigma weighting to get [batch] scalar per macro action.
            macro_scalar: torch.Tensor = self._dispatch_weight(
                chunk_tensor, chunk_mask
            )
            results.append(macro_scalar)

        # Stack along the macro action dimension.
        # Shape: [batch, num_macro_actions]
        macro_tensor: torch.Tensor = torch.stack(results, dim=-1)

        return macro_tensor

    def _dispatch_weight(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch to the appropriate sigma weighting method.

        Args:
            values: Token-level values for one macro action chunk.
                Shape: [batch_size, macro_action_len].
            mask: Validity mask for the same chunk.
                Shape: [batch_size, macro_action_len].

        Returns:
            Aggregated scalar per batch item. Shape: [batch_size].
        """
        if self.sigma_type == "equal":
            return self._equal_weight(values, mask)
        elif self.sigma_type == "unit":
            return self._unit_weight(values, mask)
        elif self.sigma_type == "position_decayed":
            return self._position_decayed_weight(values, mask)
        else:
            # Should never reach here due to __init__ validation.
            logger.warning(
                "Unknown sigma_type '%s'; falling back to equal weighting.",
                self.sigma_type,
            )
            return self._equal_weight(values, mask)

    # ------------------------------------------------------------------
    # Private sigma weighting implementations
    # ------------------------------------------------------------------

    def _equal_weight(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean of valid token values within a macro action.

        Formula (Appendix D.1, equal assignment):
            σ_i = 1/|ω_τ| for all i in [0, |ω_τ|-1]
            V(s_τ, ω_τ) = (1/|ω_τ|) · Σ_{valid i} V(s_{t_τ+i}, a_{t_τ+i})

        This is the default sigma type and achieves the best RM scores
        in the paper's ablation study (Figure 19 left panel).

        Implementation matches the paper's pseudocode (Appendix E):
            masked_values = value_i[mask_i != 0]
            inplace_values[0, idx] = torch.mean(masked_values)
                                     if masked_values.numel() > 0 else 0.0

        Args:
            values: Token values for one macro action. Shape: [batch, n].
                May be in bf16; cast to float32 for numerical accuracy.
            mask: Validity mask. Shape: [batch, n]. 1 for valid, 0 for padding.

        Returns:
            Mean value per batch item. Shape: [batch].
            Returns 0.0 for batch items where all tokens are padding.
        """
        # Cast to float32 for numerical accuracy (bf16 has limited precision).
        values_f32: torch.Tensor = values.float()
        mask_f32: torch.Tensor = mask.float()

        # Zero out padding positions.
        # Shape: [batch, n]
        masked_values: torch.Tensor = values_f32 * mask_f32

        # Sum of valid values per batch item. Shape: [batch]
        valid_sum: torch.Tensor = masked_values.sum(dim=-1)

        # Count of valid tokens per batch item. Shape: [batch]
        valid_count: torch.Tensor = mask_f32.sum(dim=-1)

        # Mean: divide by valid count, clamping to avoid division by zero.
        # When valid_count == 0, both numerator and denominator are 0,
        # so the result is 0.0 (correct: empty macro action has value 0).
        result: torch.Tensor = valid_sum / valid_count.clamp(min=1.0)

        # Cast back to the original dtype for consistency with downstream ops.
        return result.to(dtype=values.dtype)

    def _unit_weight(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Use the value of the last valid token as the macro action value.

        Formula (Appendix D.1, unit assignment):
            σ_τ = {0, 0, ..., 0, 1}  (only the last token contributes)
            V(s_τ, ω_τ) = V(s_{t_{τ+1}-1}, a_{t_{τ+1}-1})

        This sigma type achieves the best consistency and fluency per
        GPT-4 evaluation (Figure 19 right panel).

        Args:
            values: Token values for one macro action. Shape: [batch, n].
            mask: Validity mask. Shape: [batch, n]. 1 for valid, 0 for padding.

        Returns:
            Last valid token value per batch item. Shape: [batch].
            Returns 0.0 for batch items where all tokens are padding.
        """
        batch_size: int = values.size(0)

        # Count valid tokens per batch item. Shape: [batch]
        valid_count: torch.Tensor = mask.sum(dim=-1).long()

        # Index of the last valid token (0-based). Shape: [batch]
        # Clamp to 0 to handle all-padding case (valid_count == 0 → idx = -1).
        last_valid_idx: torch.Tensor = (valid_count - 1).clamp(min=0)

        # Extract values at the last valid token position using advanced indexing.
        # torch.arange(batch_size) provides the batch dimension indices.
        # Shape: [batch]
        result: torch.Tensor = values[
            torch.arange(batch_size, device=values.device),
            last_valid_idx,
        ]

        # Zero out batch items where all tokens are padding (valid_count == 0).
        # For these items, last_valid_idx was clamped to 0, so we extracted
        # values[:, 0] which may be non-zero padding. Explicitly zero them.
        all_padding_mask: torch.Tensor = (valid_count == 0)
        if all_padding_mask.any():
            result = result.masked_fill(all_padding_mask, 0.0)

        return result

    def _position_decayed_weight(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute harmonic-weighted sum of token values within a macro action.

        Formula (Appendix D.1, position decayed assignment):
            H = Σ_{i=0}^{|ω_τ|-1} 1/(|ω_τ| - i)
            σ_i = 1/((|ω_τ| - i) · H)
            V(s_τ, ω_τ) = Σ_{i=0}^{|ω_τ|-1} σ_i · V(s_{t_τ+i}, a_{t_τ+i})

        Note: At i=0, denominator = |ω_τ| · H (smallest weight).
              At i=|ω_τ|-1, denominator = 1 · H (largest weight).
        This gives MORE weight to LATER tokens in the macro action.

        The normalizer H ensures Σ σ_i = 1 over valid tokens.

        Args:
            values: Token values for one macro action. Shape: [batch, n].
                May be in bf16; cast to float32 for numerical accuracy.
            mask: Validity mask. Shape: [batch, n]. 1 for valid, 0 for padding.

        Returns:
            Harmonic-weighted sum per batch item. Shape: [batch].
            Returns 0.0 for batch items where all tokens are padding.
        """
        batch_size: int = values.size(0)
        seq_len: int = values.size(1)

        # Cast to float32 for numerical accuracy.
        values_f32: torch.Tensor = values.float()
        mask_f32: torch.Tensor = mask.float()

        # Count valid tokens per batch item. Shape: [batch]
        valid_count: torch.Tensor = mask_f32.sum(dim=-1)  # [batch]

        # Position indices: i = 0, 1, ..., seq_len-1. Shape: [seq_len]
        i_indices: torch.Tensor = torch.arange(
            seq_len, dtype=torch.float32, device=values.device
        )

        # Compute (|ω_τ| - i) for each batch item and position.
        # valid_count: [batch] → [batch, 1]
        # i_indices: [seq_len] → [1, seq_len]
        # denominators: [batch, seq_len]
        denominators: torch.Tensor = (
            valid_count.unsqueeze(-1) - i_indices.unsqueeze(0)
        )

        # For padding positions (mask == 0) and positions beyond valid_count,
        # set denominator to a large value so the weight → 0.
        # Also clamp to avoid division by zero at valid positions.
        # We use mask_f32 to zero out weights at padding positions after division.
        denominators_safe: torch.Tensor = denominators.clamp(min=1.0)

        # Raw weights: 1 / (|ω_τ| - i). Shape: [batch, seq_len]
        raw_weights: torch.Tensor = 1.0 / denominators_safe

        # Apply mask: zero out padding positions.
        # Shape: [batch, seq_len]
        masked_weights: torch.Tensor = raw_weights * mask_f32

        # Normalizer H = Σ_{valid i} 1/(|ω_τ| - i). Shape: [batch]
        H: torch.Tensor = masked_weights.sum(dim=-1)

        # Normalize weights: σ_i = raw_weight_i / H.
        # Expand H for broadcasting: [batch] → [batch, 1]
        # Clamp H to avoid division by zero for all-padding macro actions.
        normalized_weights: torch.Tensor = (
            masked_weights / H.clamp(min=1e-9).unsqueeze(-1)
        )

        # Weighted sum: V(s_τ, ω_τ) = Σ σ_i · V(s_{t_τ+i}, a_{t_τ+i})
        # Shape: [batch]
        result: torch.Tensor = (values_f32 * normalized_weights).sum(dim=-1)

        # Zero out batch items where all tokens are padding.
        all_padding_mask: torch.Tensor = (valid_count == 0)
        if all_padding_mask.any():
            result = result.masked_fill(all_padding_mask, 0.0)

        # Cast back to the original dtype.
        return result.to(dtype=values.dtype)

    # ------------------------------------------------------------------
    # Private utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_split_list(
        split_list: List[int],
        max_total: int,
    ) -> List[int]:
        """Truncate split_list so that its sum does not exceed max_total.

        Used as a safety guard when the tensor's actual response length
        is shorter than expected from the sequence boundaries. This can
        happen due to padding or truncation edge cases.

        Args:
            split_list: List of positive integers representing macro action
                lengths. Modified by truncating the last element if needed.
            max_total: Maximum allowed sum of split_list elements.

        Returns:
            A new list with the same elements as split_list but truncated
            so that sum(result) <= max_total. The last element may be
            reduced to fit. Empty list if max_total <= 0.
        """
        if max_total <= 0:
            return []

        result: List[int] = []
        cumulative: int = 0

        for length in split_list:
            remaining: int = max_total - cumulative
            if remaining <= 0:
                break
            if length <= remaining:
                result.append(length)
                cumulative += length
            else:
                # Partial last segment.
                result.append(remaining)
                cumulative += remaining
                break

        return result
