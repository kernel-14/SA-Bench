"""PPO and MA-PPO algorithm implementations.

Implements both vanilla PPO (token-level) and MA-PPO (macro-action level)
as described in §2.2 and §3.2 of the paper. Algorithm 1 provides the
pseudocode for the MA-RLHF framework.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Literal
from macro_actions import (
    get_macro_action_positions,
    get_macro_action_values,
    get_macro_action_rewards,
)


class GAE:
    """Generalized Advantage Estimation.

    Computes advantages and returns for policy optimization.
    Same for both vanilla PPO and MA-PPO.
    """
    def __init__(self, gamma: float = 1.0, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

    def compute(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using GAE.

        Args:
            values: Value estimates V(s_t), shape (batch, seq_len)
            rewards: Rewards r_t, shape (batch, seq_len)
            mask: Attention mask

        Returns:
            advantages, returns — both shape (batch, seq_len)
        """
        batch_size, seq_len = values.shape
        advantages = torch.zeros_like(values)
        gae = torch.zeros(batch_size, device=values.device)

        for t in reversed(range(seq_len)):
            next_val = values[:, t + 1] if t < seq_len - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_val - values[:, t]
            delta = delta * mask[:, t]
            gae = delta + self.gamma * self.lam * gae * mask[:, t]
            advantages[:, t] = gae

        returns = advantages + values
        return advantages, returns


def compute_kl_penalty(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token KL divergence penalty.

    D_KL(π_θ || π_sft) at each token position.
    """
    kl = policy_log_probs - ref_log_probs
    kl = kl * action_mask
    return kl


def compute_reshaped_rewards(
    rm_reward: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    kl_coefficient: float,
    response_start: int,
) -> torch.Tensor:
    """Compute reshaped rewards with KL penalty.

    R(x, y) = r_φ(x, y) - β D_KL(π_θ || π_sft)
    """
    batch_size, seq_len = policy_log_probs.shape
    rewards = torch.zeros_like(policy_log_probs)

    kl = compute_kl_penalty(policy_log_probs, ref_log_probs, action_mask)
    per_token_reward = -kl_coefficient * kl

    for b in range(batch_size):
        rewards[b, :] = per_token_reward[b]
        last_pos = action_mask[b].nonzero()[-1].item() if action_mask[b].any() else seq_len - 1
        rewards[b, last_pos] += rm_reward[b]

    return rewards


class VanillaPPO:
    """Token-level PPO as described in §2.

    Uses the clipped PPO objective from Schulman et al. (2017).
    """
    def __init__(
        self,
        policy_model: nn.Module,
        critic_model: nn.Module,
        reference_model: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        clip_ratio: float = 0.2,
        gae_gamma: float = 1.0,
        gae_lambda: float = 0.95,
        kl_coefficient: float = 0.05,
        max_prompt_length: int = 512,
        max_response_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 50,
    ):
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.clip_ratio = clip_ratio
        self.gae = GAE(gamma=gae_gamma, lam=gae_lambda)
        self.kl_coefficient = kl_coefficient
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def generate_responses(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Generate responses using current policy."""
        prompt_len = input_ids.size(1)

        with torch.no_grad():
            generated = self.policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_response_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = generated[:, prompt_len:]
        full_ids = generated

        # Build action mask (1 for response tokens, 0 for prompt)
        action_mask = torch.zeros_like(full_ids, dtype=torch.float32)
        action_mask[:, prompt_len:] = 1.0

        # Create full attention mask
        full_attention_mask = torch.ones_like(full_ids, dtype=torch.float32)

        return {
            "input_ids": full_ids,
            "attention_mask": full_attention_mask,
            "action_mask": action_mask,
            "prompt_len": prompt_len,
        }

    def compute_rewards(
        self,
        full_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward model scores for generated sequences."""
        with torch.no_grad():
            rm_rewards = self.reward_model(
                input_ids=full_ids,
                attention_mask=full_attention_mask,
            )
        return rm_rewards

    def compute_log_probs(
        self,
        model: nn.Module,
        full_ids: torch.Tensor,
        full_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities under given model."""
        logits = model(input_ids=full_ids, attention_mask=full_attention_mask)
        if isinstance(logits, tuple):
            logits = logits[0]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        labels = full_ids[:, 1:]
        log_probs = log_probs[:, :-1, :]
        return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    def step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Execute one PPO training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Generate responses
        gen_output = self.generate_responses(input_ids, attention_mask)
        full_ids = gen_output["input_ids"]
        full_attention_mask = gen_output["attention_mask"]
        action_mask = gen_output["action_mask"]
        prompt_len = gen_output["prompt_len"]

        batch_size = full_ids.size(0)

        # Compute RM rewards
        rm_rewards = self.compute_rewards(full_ids, full_attention_mask)

        # Compute policy log probs (current)
        policy_log_probs = self.compute_log_probs(
            self.policy_model, full_ids, full_attention_mask,
        )

        # Compute reference log probs (frozen SFT)
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(
                self.reference_model, full_ids, full_attention_mask,
            )

        # Compute reshaped rewards with KL penalty
        rewards = compute_reshaped_rewards(
            rm_rewards, policy_log_probs, ref_log_probs,
            action_mask[:, 1:], self.kl_coefficient, prompt_len,
        )

        # Compute values from critic
        values = self.critic_model(
            input_ids=full_ids, attention_mask=full_attention_mask,
        )
        values = values[:, :-1]  # Align with action positions

        # Compute advantages and returns using GAE
        advantages, returns = self.gae.compute(values, rewards, action_mask[:, 1:])

        # Compute old log probs (before update)
        with torch.no_grad():
            old_log_probs = self.compute_log_probs(
                self.policy_model, full_ids, full_attention_mask,
            )

        # Compute PPO clipped objective
        policy_loss = self._policy_loss(
            policy_log_probs, old_log_probs, advantages, action_mask[:, 1:],
        )

        # Compute value loss
        value_loss = self._value_loss(values, returns, action_mask[:, 1:])

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "rm_reward_mean": rm_rewards.mean().item(),
            "kl_mean": (policy_log_probs - ref_log_probs).mean().item(),
        }

    def _policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped PPO policy loss. Equation (1) from the paper."""
        ratio = torch.exp(log_probs - old_log_probs) * mask
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / (mask.sum() + 1e-8)
        return pg_loss

    def _value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE value loss."""
        loss = 0.5 * ((values - returns) ** 2) * mask
        return loss.sum() / (mask.sum() + 1e-8)

    def get_state_dict(self) -> Dict:
        return {
            "policy": self.policy_model.state_dict(),
            "critic": self.critic_model.state_dict(),
        }


class MAPPO(VanillaPPO):
    """Macro-Action PPO as described in §3.2 (Algorithm 1).

    Extends vanilla PPO by:
    1. Grouping tokens into macro actions
    2. Computing macro-action-level values and rewards
    3. Computing joint log-probabilities for PPO objective
    """
    def __init__(
        self,
        policy_model: nn.Module,
        critic_model: nn.Module,
        reference_model: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        clip_ratio: float = 0.2,
        gae_gamma: float = 1.0,
        gae_lambda: float = 0.95,
        kl_coefficient: float = 0.05,
        max_prompt_length: int = 512,
        max_response_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 50,
        termination: Literal["ngram", "randomized_ngram", "ppl", "parser"] = "ngram",
        n_gram: int = 5,
        n_gram_list: Optional[List[int]] = None,
        n_gram_repeat_times: int = 3,
        parsing_cutoff: int = 5,
        value_estimation: Literal["equal", "unit", "position_decayed"] = "equal",
    ):
        super().__init__(
            policy_model=policy_model,
            critic_model=critic_model,
            reference_model=reference_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            clip_ratio=clip_ratio,
            gae_gamma=gae_gamma,
            gae_lambda=gae_lambda,
            kl_coefficient=kl_coefficient,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        self.termination = termination
        self.n_gram = n_gram
        self.n_gram_list = n_gram_list or [2, 3, 5, 10]
        self.n_gram_repeat_times = n_gram_repeat_times
        self.parsing_cutoff = parsing_cutoff
        self.value_estimation = value_estimation

    def step(
        self,
        batch: Dict[str, torch.Tensor],
        ppl_values: Optional[List[float]] = None,
        parse_tree_root=None,
    ) -> Dict[str, float]:
        """Execute one MA-PPO training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Generate responses
        gen_output = self.generate_responses(input_ids, attention_mask)
        full_ids = gen_output["input_ids"]
        full_attention_mask = gen_output["attention_mask"]
        action_mask = gen_output["action_mask"]
        prompt_len = gen_output["prompt_len"]

        batch_size = full_ids.size(0)
        start = prompt_len - 1  # Start position for macro actions

        # Compute RM rewards
        rm_rewards = self.compute_rewards(full_ids, full_attention_mask)

        # Compute policy log probs (current)
        policy_log_probs = self.compute_log_probs(
            self.policy_model, full_ids, full_attention_mask,
        )

        # Compute reference log probs
        with torch.no_grad():
            ref_logits = self.reference_model(full_ids, full_attention_mask)
            ref_log_probs = self.compute_log_probs(
                self.reference_model, full_ids, full_attention_mask,
            )

        # Compute reshaped rewards with KL penalty
        token_rewards = compute_reshaped_rewards(
            rm_rewards, policy_log_probs, ref_log_probs,
            action_mask[:, 1:], self.kl_coefficient, prompt_len,
        )

        # Compute token-level values from critic
        token_values = self.critic_model(
            input_ids=full_ids, attention_mask=full_attention_mask,
        )
        token_values = token_values[:, :-1]

        # ====== Macro Action Processing ======
        # Determine macro action boundaries
        sequence = get_macro_action_positions(
            start=start,
            mask=action_mask[:, 1:],
            termination=self.termination,
            n_gram=self.n_gram,
            n_gram_list=self.n_gram_list,
            repeat_times=self.n_gram_repeat_times,
            ppl_values=ppl_values,
            cutoff=self.parsing_cutoff,
            parse_tree_root=parse_tree_root,
        )

        # Compute macro-action-level values
        macro_values = get_macro_action_values(
            values=token_values,
            mask=action_mask[:, 1:],
            start=start,
            sequence=sequence,
            value_estimation=self.value_estimation,
        )

        # Compute macro-action-level rewards
        macro_rewards = get_macro_action_rewards(
            rewards=token_rewards,
            mask=action_mask[:, 1:],
            start=start,
            sequence=sequence,
        )

        # Compute GAE advantages at macro level
        advantages, returns = self.gae.compute(
            macro_values, macro_rewards,
            torch.ones_like(macro_values),
        )

        # Compute old joint log probs for macro actions
        with torch.no_grad():
            old_logits = self.policy_model(full_ids, full_attention_mask)
            old_joint_log_probs = self._compute_joint_log_probs(
                old_logits, full_ids, sequence, start, action_mask[:, 1:],
            )

        # Compute current joint log probs
        current_logits = self.policy_model(full_ids, full_attention_mask)
        current_joint_log_probs = self._compute_joint_log_probs(
            current_logits, full_ids, sequence, start, action_mask[:, 1:],
        )

        # Compute MA-PPO policy loss (Equation 4)
        policy_loss = self._ma_policy_loss(
            current_joint_log_probs, old_joint_log_probs, advantages,
        )

        # Compute value loss
        value_loss = self._value_loss(
            macro_values, returns, torch.ones_like(macro_values),
        )

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "rm_reward_mean": rm_rewards.mean().item(),
            "kl_mean": (policy_log_probs - ref_log_probs).mean().item(),
            "num_macro_actions": len(sequence) - 1,
        }

    def _compute_joint_log_probs(
        self,
        logits: torch.Tensor,
        full_ids: torch.Tensor,
        sequence: List[int],
        start: int,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute joint log probability of macro actions.

        π_θ(ω_τ | s_τ) = ∏_{t=t_τ}^{t_{τ+1}} π_θ(a_t | a_<t)
        (see §3.2.2, Equation 3)
        """
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        labels = full_ids[:, 1:]
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        # Apply action mask
        token_log_probs = token_log_probs * action_mask

        # Split into macro actions
        split_list = torch.diff(torch.tensor(sequence)).tolist()
        batch_size = token_log_probs.size(0)

        if len(split_list) == 0:
            return torch.zeros(batch_size, 1, device=logits.device)

        splited_log_probs = torch.split(
            token_log_probs[:, start:], split_list, dim=-1,
        )

        joint_log_probs = torch.zeros(
            batch_size, len(split_list),
            dtype=logits.dtype, device=logits.device,
        )
        for idx, lp_i in enumerate(splited_log_probs):
            joint_log_probs[:, idx] = lp_i.sum(dim=-1)

        return joint_log_probs

    def _ma_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """MA-PPO clipped policy loss. Equation (4) from the paper.

        L^{MA-PPO}(θ) = E[ min(r_τ * Â_τ, clip(r_τ, 1-ε, 1+ε) * Â_τ) ]
        where r_τ = π_θ(ω_τ|s_τ) / π_{θ_old}(ω_τ|s_τ)
        """
        ratio = torch.exp(log_probs - old_log_probs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio,
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        return pg_loss
