## utils/reward_utils.py
"""Reward utility functions for the MA-RLHF pipeline.

This module provides RewardUtils, a collection of static methods that form
the mathematical backbone of the MA-PPO reward pipeline. The three methods
handle:
  1. Per-token KL divergence penalty computation.
  2. Reward shaping: placing the RM score at the terminal token and
     subtracting the KL penalty at every token.
  3. Generalized Advantage Estimation (GAE) at the macro-action level.

All methods are pure functions operating on tensors with no side effects.
They are called from training/ma_ppo_trainer.py in the following sequence:

    logprobs, ref_logprobs → compute_kl_penalty → kl_penalties
    rm_score, kl_penalties → reshape_reward     → per_token_rewards
    macro_values, macro_rewards → compute_gae   → advantages, returns

Paper alignments:
  - KL penalty: Section 2.2, R(x,y) = r_φ(x,y) - β·D_KL(π_θ||π_sft)
  - GAE: Appendix E, "apply GAE without modification at the macro level"
  - γ=1.0, λ=0.95: Table 5 (all model sizes)
  - β=0.05: config.yaml ppo.kl_coeff (0.01 for Gemma-7B on TL;DR)

Dependencies:
    External: torch
    Internal: none (leaf module — no internal project imports)
"""

import logging
from typing import Tuple, Union

import torch

logger = logging.getLogger(__name__)


class RewardUtils:
    """Static utility methods for reward computation in MA-PPO.

    This class is never instantiated. All methods are static and operate
    purely on tensors. Config values (kl_coeff, gamma, lambda) are passed
    as explicit arguments rather than read from a config object, keeping
    this module stateless and easily unit-testable.

    Usage:
        kl = RewardUtils.compute_kl_penalty(logprobs, ref_logprobs, 0.05)
        rewards = RewardUtils.reshape_reward(rm_score, kl, response_length)
        advantages, returns = RewardUtils.compute_gae(values, rewards, 1.0, 0.95)
    """

    @staticmethod
    def compute_kl_penalty(
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        kl_coeff: float = 0.05,
    ) -> torch.Tensor:
        """Compute the per-token KL divergence penalty.

        Uses the first-order approximation of the KL divergence between
        the current policy and the reference (SFT) policy:

            KL_t ≈ log π_θ(a_t | s_t) - log π_ref(a_t | s_t)

        This is the standard approximation used in DeepSpeed-Chat and TRL.
        It avoids summing over the full vocabulary and reuses the already-
        computed per-token log probabilities from the policy and reference
        model forward passes.

        The result is positive when the policy assigns higher probability
        than the reference (penalizing divergence) and negative when the
        policy is more conservative (acting as a reward bonus).

        Paper reference: Section 2.2
            R(x, y) = r_φ(x, y) - β · D_KL(π_θ(·|x) ‖ π_sft(·|x))

        Config reference: config.yaml ppo.kl_coeff = 0.05
            Special case: 0.01 for Gemma-7B on TL;DR (training instability).
            The caller (MAPPOTrainer) passes the correct value from config.

        Args:
            logprobs: Per-token log probabilities from the current policy
                π_θ. Shape: [batch_size, seq_len]. Each element is
                log π_θ(a_t | a_{<t}), gathered at the actual token index.
                Padding positions should be 0.0 (handled upstream by the
                caller using the action mask).
            ref_logprobs: Per-token log probabilities from the reference
                policy π_ref (frozen SFT model). Shape: [batch_size, seq_len].
                Same indexing convention as logprobs.
            kl_coeff: The KL penalty coefficient β. Default 0.05 matches
                config.yaml ppo.kl_coeff. Passed explicitly to keep this
                method stateless.

        Returns:
            Per-token KL penalty tensor of shape [batch_size, seq_len].
            Element [b, t] = kl_coeff * (logprobs[b, t] - ref_logprobs[b, t]).
            This is subtracted from the reward in reshape_reward().
        """
        # Validate shapes match.
        if logprobs.shape != ref_logprobs.shape:
            raise ValueError(
                f"logprobs and ref_logprobs must have the same shape. "
                f"Got logprobs={logprobs.shape}, "
                f"ref_logprobs={ref_logprobs.shape}."
            )

        if kl_coeff < 0.0:
            raise ValueError(
                f"kl_coeff must be >= 0.0, got {kl_coeff}."
            )

        # Per-token KL approximation: log π_θ - log π_ref.
        # Shape: [batch_size, seq_len]
        kl_per_token: torch.Tensor = logprobs - ref_logprobs

        # Scale by the KL coefficient β.
        # Shape: [batch_size, seq_len]
        kl_penalty: torch.Tensor = kl_coeff * kl_per_token

        return kl_penalty

    @staticmethod
    def reshape_reward(
        rm_score: Union[float, torch.Tensor],
        kl_penalties: torch.Tensor,
        response_length: int,
    ) -> torch.Tensor:
        """Shape per-token rewards by placing RM score at the terminal token.

        Constructs the per-token reward tensor used as input to GAE:

            r_t = -β · KL_t                    for t < T (non-terminal)
            r_t = r_φ(x, y) - β · KL_t        for t = T (terminal token)

        The RM score r_φ(x, y) is a scalar for the entire (prompt, response)
        pair. Standard RLHF practice (Stiennon et al. 2020, Ouyang et al. 2022)
        places it at the last valid response token. The KL penalty is applied
        at every token position.

        This per-token reward tensor is then aggregated into macro-action-level
        rewards by MacroActionValueEstimator.get_macro_rewards().

        Paper reference: Section 2.2
            R(x, y) = r_φ(x, y) - β · D_KL(π_θ(·|x) ‖ π_sft(·|x))

        Indexing convention:
            kl_penalties is indexed over the response tokens only (not the
            full prompt + response sequence). The caller (MAPPOTrainer)
            slices kl_penalties[:, start:] before passing it here, where
            start = prompt_length - 1 (matching the PPO pseudocode in
            Appendix E: start = prompts.size()[-1] - 1).

            Therefore, response_length - 1 is the correct 0-based index
            of the terminal token within kl_penalties.

        Args:
            rm_score: Scalar reward model score for the full (prompt, response)
                pair. Can be:
                  - A Python float (most common case).
                  - A torch.Tensor of shape [] (scalar tensor).
                  - A torch.Tensor of shape [1] (single-element tensor).
                  - A torch.Tensor of shape [batch_size] (per-sample scores).
                For APPS, this is the compiler signal from CodeEvaluator
                (a float in {-1.0, -0.6, [-0.3, 1.0]}).
            kl_penalties: Per-token KL penalties from compute_kl_penalty(),
                already scaled by kl_coeff. Shape: [batch_size, seq_len]
                where seq_len covers the response tokens only (not prompt).
            response_length: Number of actual (non-padding) response tokens.
                The RM score is placed at index response_length - 1.
                Must satisfy 1 <= response_length <= kl_penalties.size(1).

        Returns:
            Per-token reward tensor of shape [batch_size, seq_len].
            Initialized to -kl_penalties, with rm_score added at position
            response_length - 1 for each batch element.

        Raises:
            ValueError: If response_length is out of valid range.
        """
        seq_len: int = kl_penalties.size(1)

        if response_length < 1:
            raise ValueError(
                f"response_length must be >= 1, got {response_length}."
            )

        if response_length > seq_len:
            raise ValueError(
                f"response_length ({response_length}) exceeds the sequence "
                f"length of kl_penalties ({seq_len}). "
                f"Ensure kl_penalties covers only response tokens."
            )

        # Initialize rewards as the negative KL penalty at every token.
        # Shape: [batch_size, seq_len]
        # Clone to avoid modifying the input tensor in-place.
        rewards: torch.Tensor = -kl_penalties.clone()

        # Resolve rm_score to a scalar or batch-compatible tensor.
        # The terminal token index (0-based) within the response.
        terminal_idx: int = response_length - 1

        if isinstance(rm_score, torch.Tensor):
            rm_score_tensor: torch.Tensor = rm_score.to(
                dtype=rewards.dtype, device=rewards.device
            )

            if rm_score_tensor.dim() == 0:
                # Scalar tensor: broadcast to all batch elements.
                rewards[:, terminal_idx] += rm_score_tensor
            elif rm_score_tensor.dim() == 1:
                batch_size: int = rewards.size(0)
                if rm_score_tensor.size(0) == 1:
                    # Shape [1]: broadcast to all batch elements.
                    rewards[:, terminal_idx] += rm_score_tensor.squeeze(0)
                elif rm_score_tensor.size(0) == batch_size:
                    # Shape [batch_size]: per-sample scores.
                    # rewards[:, terminal_idx] has shape [batch_size].
                    rewards[:, terminal_idx] += rm_score_tensor
                else:
                    raise ValueError(
                        f"rm_score tensor has incompatible shape "
                        f"{rm_score_tensor.shape}. Expected scalar, [1], "
                        f"or [batch_size={batch_size}]."
                    )
            else:
                raise ValueError(
                    f"rm_score tensor must be 0D or 1D, "
                    f"got shape {rm_score_tensor.shape}."
                )
        else:
            # Python float or int: broadcast to all batch elements.
            rm_score_float: float = float(rm_score)
            rewards[:, terminal_idx] += rm_score_float

        return rewards

    @staticmethod
    def compute_gae(
        values: torch.Tensor,
        rewards: torch.Tensor,
        gamma: float = 1.0,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE) at macro-action level.

        Applies the standard GAE algorithm (Schulman et al. 2016) to
        macro-action-level values and rewards. The paper states in Appendix E:
        "we apply Generalized Advantage Estimation (GAE) without modification
        to derive advantage estimates."

        GAE formula (computed backwards for efficiency):
            δ_τ = R_τ + γ · V(s_{τ+1}) - V(s_τ)
            Â_τ = δ_τ + (γλ) · Â_{τ+1}

        With γ=1.0 (from Table 5), the delta simplifies to:
            δ_τ = R_τ + V(s_{τ+1}) - V(s_τ)

        This means no temporal discounting of future macro actions, which is
        appropriate for RLHF where episodes are short and the full trajectory
        reward matters equally regardless of when it occurs.

        The returns (critic regression targets) are:
            returns_τ = Â_τ + V(s_τ)

        Paper reference: Table 5
            λ (GAE) = 0.95 for all model sizes.
            γ (GAE) = 1.0 for all model sizes.

        Config reference: config.yaml
            ppo.gae_lambda: 0.95
            ppo.gae_gamma: 1.0

        Args:
            values: Macro-action-level value estimates from the critic model,
                aggregated by MacroActionValueEstimator.get_macro_values().
                Shape: [batch_size, num_macro_actions].
                V(s_τ) for τ = 0, 1, ..., T-1.
            rewards: Macro-action-level rewards aggregated by
                MacroActionValueEstimator.get_macro_rewards().
                Shape: [batch_size, num_macro_actions].
                R_τ for τ = 0, 1, ..., T-1.
            gamma: Discount factor γ for future rewards beyond the macro
                action. Default 1.0 per Table 5 (no temporal discounting).
                Note: this is the inter-macro discount, distinct from ρ=1.0
                (the intra-macro discount in config.yaml ppo.rho).
            lam: GAE lambda λ for the exponential moving average of TD
                errors. Default 0.95 per Table 5. Controls the bias-variance
                tradeoff: λ=0 gives TD(0) (low variance, high bias),
                λ=1 gives Monte Carlo (high variance, low bias).

        Returns:
            A tuple (advantages, returns):
                advantages: Shape [batch_size, num_macro_actions].
                    Â_τ = GAE advantage estimates. Used as the advantage
                    signal in MacroPolicyLoss.compute_policy_loss().
                returns: Shape [batch_size, num_macro_actions].
                    returns_τ = Â_τ + V(s_τ). Used as the regression
                    target in MacroPolicyLoss.compute_critic_loss().

        Raises:
            ValueError: If values and rewards have different shapes.
            ValueError: If gamma or lam are outside valid ranges.
        """
        if values.shape != rewards.shape:
            raise ValueError(
                f"values and rewards must have the same shape. "
                f"Got values={values.shape}, rewards={rewards.shape}."
            )

        if not (0.0 <= gamma <= 1.0):
            raise ValueError(
                f"gamma must be in [0.0, 1.0], got {gamma}."
            )

        if not (0.0 <= lam <= 1.0):
            raise ValueError(
                f"lam must be in [0.0, 1.0], got {lam}."
            )

        batch_size: int = values.size(0)
        num_macro_actions: int = values.size(1)

        # Initialize advantages tensor (same shape, dtype, device as values).
        advantages: torch.Tensor = torch.zeros_like(values)

        # Initialize last_advantage as a batch-wise zero tensor.
        # Shape: [batch_size] — carries batch-wise state through the loop.
        # Using a tensor (not scalar 0) ensures correct broadcasting when
        # batch_size > 1.
        last_advantage: torch.Tensor = torch.zeros(
            batch_size,
            dtype=values.dtype,
            device=values.device,
        )

        # Effective per-step discount combining gamma and lambda.
        # With gamma=1.0 and lam=0.95, this is 0.95 per macro action.
        gamma_lam: float = gamma * lam

        # Backward pass through macro actions.
        # τ = T-1, T-2, ..., 1, 0
        for t in range(num_macro_actions - 1, -1, -1):
            # V(s_{τ+1}): value of the next macro state.
            # For the last macro action (t == T-1), there is no next state,
            # so V(s_T) = 0 (episode terminates after the last macro action).
            if t == num_macro_actions - 1:
                # Terminal macro action: no future value.
                # Shape: [batch_size]
                next_value: torch.Tensor = torch.zeros(
                    batch_size,
                    dtype=values.dtype,
                    device=values.device,
                )
            else:
                # Non-terminal: use the critic's value estimate for s_{τ+1}.
                # Shape: [batch_size]
                next_value = values[:, t + 1]

            # TD error: δ_τ = R_τ + γ · V(s_{τ+1}) - V(s_τ)
            # Shape: [batch_size]
            delta: torch.Tensor = (
                rewards[:, t] + gamma * next_value - values[:, t]
            )

            # GAE advantage: Â_τ = δ_τ + γλ · Â_{τ+1}
            # Shape: [batch_size]
            current_advantage: torch.Tensor = delta + gamma_lam * last_advantage

            # Store in the advantages tensor.
            advantages[:, t] = current_advantage

            # Update last_advantage for the next (earlier) time step.
            last_advantage = current_advantage

        # Returns: regression targets for the critic.
        # returns_τ = Â_τ + V(s_τ)
        # Shape: [batch_size, num_macro_actions]
        returns: torch.Tensor = advantages + values

        return advantages, returns
