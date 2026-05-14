"""
MA-PPO Trainer: Proximal Policy Optimization with Macro Actions.

This module implements the MA-PPO training loop described in Algorithm 1 of:
  "MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions"

The key modification over standard PPO is that policy gradient updates and
value function updates are computed at the macro-action level rather than
the token level, reducing the temporal distance between actions and rewards.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from macro_actions import (
    get_macro_action_positions,
    get_macro_action_values,
    policy_loss_macro_action,
    critic_loss_macro_action,
    compute_perplexity_sequence,
)


@dataclass
class MAPPOConfig:
    """Configuration for MA-PPO training."""

    # Macro action settings
    termination: str = "ngram"          # 'ngram', 'randomized_ngram', 'ppl', 'parser'
    n_gram: int = 5                     # n-gram length (used when termination='ngram')
    parser_cutoff: int = 5              # cutoff for parsing-based termination (C=5)
    value_assignment: str = "equal"     # 'equal', 'unit', 'position_decayed'

    # PPO hyperparameters
    cliprange: float = 0.2              # epsilon for PPO clipping
    cliprange_value: float = 0.2        # epsilon for value function clipping
    gamma: float = 1.0                  # discount factor (set to 1 in experiments)
    lam: float = 0.95                   # lambda for GAE
    kl_coef: float = 0.05              # KL penalty coefficient (beta)

    # Training settings
    policy_lr: float = 1.5e-5
    critic_lr: float = 1.5e-5
    batch_size: int = 256
    ppo_epochs: int = 1
    rollout_batch_size: int = 1
    max_prompt_len: int = 512
    max_response_len: int = 512
    temperature: float = 0.8
    top_p: float = 1.0
    top_k: int = 50
    warmup_steps: int = 200

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    output_dir: str = "./output"


class MAPPOTrainer:
    """
    MA-PPO Trainer that integrates macro actions into the RLHF training loop.

    The training procedure follows Algorithm 1:
    1. Generate experience using the policy model.
    2. Compute token-level values using the critic model.
    3. Compute the reward score using the reward model.
    4. Determine macro action boundaries using the termination rule.
    5. Aggregate token-level values into macro-action-level values.
    6. Compute advantages and returns using GAE at the macro-action level.
    7. Optimize the policy and critic using the MA-PPO objective.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        critic_model: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        config: MAPPOConfig,
    ):
        self.policy = policy_model
        self.ref = ref_model
        self.critic = critic_model
        self.reward = reward_model
        self.tokenizer = tokenizer
        self.config = config

        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=config.policy_lr
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=config.critic_lr
        )

    @torch.no_grad()
    def generate_experience(
        self, prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate responses and collect experience for PPO training.

        Returns a dict with:
          - input_ids: (batch, prompt_len + response_len)
          - attention_mask: (batch, prompt_len + response_len)
          - action_mask: (batch, response_len) -- 1 for response tokens
          - old_log_probs: (batch, response_len)
          - old_values: (batch, response_len)
          - rewards: (batch,) -- scalar reward from reward model
          - ref_log_probs: (batch, response_len)
          - prompt_len: int
        """
        self.policy.eval()
        self.critic.eval()
        self.ref.eval()
        self.reward.eval()

        # Tokenize prompts
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_len,
        )
        input_ids = enc["input_ids"].to(next(self.policy.parameters()).device)
        attention_mask = enc["attention_mask"].to(input_ids.device)
        prompt_len = input_ids.size(1)

        # Generate responses
        gen_output = self.policy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_response_len,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Full sequence: prompt + response
        full_ids = gen_output  # (batch, prompt_len + response_len)
        full_attention_mask = (full_ids != self.tokenizer.pad_token_id).long()

        # Action mask: 1 for response tokens only
        action_mask = torch.zeros_like(full_attention_mask)
        action_mask[:, prompt_len:] = full_attention_mask[:, prompt_len:]

        # Compute log-probs under current policy
        policy_logits = self.policy(
            input_ids=full_ids, attention_mask=full_attention_mask
        ).logits  # (batch, seq_len, vocab)
        old_log_probs = self._compute_log_probs(policy_logits, full_ids)
        old_log_probs = old_log_probs[:, prompt_len - 1:]  # shift: action at t uses logit at t-1

        # Compute log-probs under reference model (for KL penalty)
        ref_logits = self.ref(
            input_ids=full_ids, attention_mask=full_attention_mask
        ).logits
        ref_log_probs = self._compute_log_probs(ref_logits, full_ids)
        ref_log_probs = ref_log_probs[:, prompt_len - 1:]

        # Compute token-level values from critic
        old_values = self.critic(
            input_ids=full_ids, attention_mask=full_attention_mask
        )  # (batch, seq_len) -- critic outputs scalar per token
        if hasattr(old_values, "logits"):
            old_values = old_values.logits.squeeze(-1)
        old_values = old_values[:, prompt_len - 1:]

        # Compute reward from reward model (scalar per sequence)
        reward_scores = self.reward(
            input_ids=full_ids, attention_mask=full_attention_mask
        )
        if hasattr(reward_scores, "logits"):
            reward_scores = reward_scores.logits.squeeze(-1)
        # Take the reward at the last non-padding token
        last_token_idx = full_attention_mask.sum(dim=1) - 1
        rewards = reward_scores[torch.arange(reward_scores.size(0)), last_token_idx]

        return {
            "input_ids": full_ids,
            "attention_mask": full_attention_mask,
            "action_mask": action_mask[:, prompt_len:],
            "old_log_probs": old_log_probs,
            "old_values": old_values,
            "rewards": rewards,
            "ref_log_probs": ref_log_probs,
            "prompt_len": prompt_len,
            "policy_logits": policy_logits,
        }

    def _compute_log_probs(
        self, logits: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token log-probabilities.

        Args:
            logits: (batch, seq_len, vocab_size)
            input_ids: (batch, seq_len)

        Returns:
            log_probs: (batch, seq_len - 1) -- log P(a_t | a_{<t})
        """
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs

    def _compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token KL divergence penalty: KL(pi_theta || pi_ref).

        KL(pi || pi_ref) = pi * (log pi - log pi_ref)
                         ≈ log pi - log pi_ref  (first-order approximation)
        """
        kl = log_probs - ref_log_probs
        return kl * action_mask

    def _compute_rewards_with_kl(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the reshaped reward incorporating KL penalty (Equation 2):
            R(x, y) = r_phi(x, y) - beta * KL(pi_theta || pi_sft)

        The KL penalty is applied per-token, and the reward model score is
        added only at the final token position.

        Args:
            rewards: Scalar reward from reward model, shape (batch,).
            log_probs: Policy log-probs, shape (batch, response_len).
            ref_log_probs: Reference log-probs, shape (batch, response_len).
            action_mask: Response token mask, shape (batch, response_len).

        Returns:
            Per-token rewards, shape (batch, response_len).
        """
        kl_penalty = self._compute_kl_penalty(log_probs, ref_log_probs, action_mask)
        per_token_rewards = -self.config.kl_coef * kl_penalty

        # Add the reward model score at the last response token
        last_response_idx = action_mask.sum(dim=1).long() - 1
        last_response_idx = last_response_idx.clamp(min=0)
        for i in range(rewards.size(0)):
            per_token_rewards[i, last_response_idx[i]] += rewards[i]

        return per_token_rewards

    def _compute_gae(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) at the macro-action level.

        GAE(lambda) advantage:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_t+l

        Args:
            values: Macro-action-level values, shape (batch, num_macro_actions).
            rewards: Macro-action-level rewards, shape (batch, num_macro_actions).
            action_mask: Macro-action mask, shape (batch, num_macro_actions).

        Returns:
            advantages: shape (batch, num_macro_actions)
            returns: shape (batch, num_macro_actions)
        """
        gamma = self.config.gamma
        lam = self.config.lam

        batch_size, T = values.shape
        advantages = torch.zeros_like(values)
        last_gae = torch.zeros(batch_size, device=values.device)

        # Bootstrap from zero (episode ends at last token)
        next_value = torch.zeros(batch_size, device=values.device)

        for t in reversed(range(T)):
            mask_t = action_mask[:, t]
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            last_gae = delta + gamma * lam * last_gae * mask_t
            advantages[:, t] = last_gae
            next_value = values[:, t] * mask_t

        returns = advantages + values
        return advantages, returns

    def train_step(
        self,
        experience: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Perform one MA-PPO update step.

        This implements the core of Algorithm 1:
        1. Compute macro action boundaries.
        2. Aggregate values and rewards to macro-action level.
        3. Compute GAE advantages and returns.
        4. Optimize policy and critic using MA-PPO objectives.

        Args:
            experience: Dict from generate_experience().

        Returns:
            Dict of training metrics.
        """
        self.policy.train()
        self.critic.train()

        input_ids = experience["input_ids"]
        attention_mask = experience["attention_mask"]
        action_mask = experience["action_mask"]
        old_log_probs = experience["old_log_probs"]
        old_values = experience["old_values"]
        rewards_scalar = experience["rewards"]
        ref_log_probs = experience["ref_log_probs"]
        prompt_len = experience["prompt_len"]

        batch_size = input_ids.size(0)
        device = input_ids.device

        total_policy_loss = 0.0
        total_critic_loss = 0.0

        for b in range(batch_size):
            # Get macro action positions for this sample
            start = prompt_len - 1
            sample_mask = attention_mask[b:b+1]

            # Compute perplexity if needed for PPL-based termination
            ppl = None
            if self.config.termination == "ppl":
                ppl = compute_perplexity_sequence(
                    experience["policy_logits"][b:b+1],
                    input_ids[b:b+1],
                    start,
                )

            sequence = get_macro_action_positions(
                start=start,
                mask=sample_mask,
                termination=self.config.termination,
                n_gram=self.config.n_gram,
                ppl=ppl,
                cutoff=self.config.parser_cutoff,
            )

            # Aggregate token-level values to macro-action level
            macro_old_values = get_macro_action_values(
                old_values[b:b+1], action_mask[b:b+1], 0, sequence
            )  # (1, num_macro_actions)

            # Compute per-token rewards with KL penalty
            per_token_rewards = self._compute_rewards_with_kl(
                rewards_scalar[b:b+1],
                old_log_probs[b:b+1],
                ref_log_probs[b:b+1],
                action_mask[b:b+1],
            )

            # Aggregate rewards to macro-action level
            macro_rewards = get_macro_action_values(
                per_token_rewards, action_mask[b:b+1], 0, sequence
            )  # (1, num_macro_actions)

            # Macro-action mask (1 for each macro action that has valid tokens)
            num_macro = macro_old_values.size(1)
            macro_mask = torch.ones(1, num_macro, device=device)

            # Compute GAE at macro-action level
            advantages, returns = self._compute_gae(
                macro_old_values, macro_rewards, macro_mask
            )

            # Normalize advantages
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # Forward pass through policy
            policy_output = self.policy(
                input_ids=input_ids[b:b+1],
                attention_mask=attention_mask[b:b+1],
            )
            new_logits = policy_output.logits
            new_log_probs = self._compute_log_probs(new_logits, input_ids[b:b+1])
            new_log_probs_response = new_log_probs[:, start:]

            # Policy loss (MA-PPO objective, Equation 3)
            policy_loss = policy_loss_macro_action(
                logprobs=new_log_probs_response,
                old_logprobs=old_log_probs[b:b+1],
                advantages=advantages,
                mask=action_mask[b:b+1],
                sequence=sequence,
                cliprange=self.config.cliprange,
            )

            # Forward pass through critic
            critic_output = self.critic(
                input_ids=input_ids[b:b+1],
                attention_mask=attention_mask[b:b+1],
            )
            if hasattr(critic_output, "logits"):
                new_values = critic_output.logits.squeeze(-1)
            else:
                new_values = critic_output
            new_values_response = new_values[:, start:]

            # Critic loss
            critic_loss = critic_loss_macro_action(
                values=new_values_response,
                old_values=old_values[b:b+1],
                returns=returns,
                mask=action_mask[b:b+1],
                sequence=sequence,
                cliprange_value=self.config.cliprange_value,
            )

            total_policy_loss += policy_loss
            total_critic_loss += critic_loss

        # Average over batch
        total_policy_loss = total_policy_loss / batch_size
        total_critic_loss = total_critic_loss / batch_size

        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return {
            "policy_loss": total_policy_loss.item(),
            "critic_loss": total_critic_loss.item(),
        }
