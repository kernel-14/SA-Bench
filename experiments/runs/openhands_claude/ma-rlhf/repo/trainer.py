"""
PPO and MA-PPO trainers for MA-RLHF.

Implements:
  - Generalized Advantage Estimation (GAE) at token level and macro level
  - Standard PPO (token-level, baseline)
  - MA-PPO (macro-action level, Algorithm 1 in the paper)
  - KL-penalised reward shaping (Eq. 2)
  - Experience collection (rollout)

The MA-PPO trainer overrides the advantage/value computation and the
policy/critic loss to operate at the macro action level (§3.2.2, §E).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import MacroActionConfig, PPOConfig
from macro_actions import (
    get_macro_action_positions,
    get_macro_action_values,
    policy_loss_macro_action,
    critic_loss_macro_action,
    compute_token_perplexities,
)
from model import PolicyModel, CriticModel, RewardModel, ReferenceModel


# ---------------------------------------------------------------------------
# Experience buffer
# ---------------------------------------------------------------------------

@dataclass
class Experience:
    """Stores one rollout experience for a single prompt."""
    input_ids: torch.Tensor          # (1, prompt_len + response_len)
    attention_mask: torch.Tensor     # (1, prompt_len + response_len)
    action_mask: torch.Tensor        # (1, response_len)  — 1 for response tokens
    log_probs: torch.Tensor          # (1, response_len)  — log π_θ(a_t|s_t)
    ref_log_probs: torch.Tensor      # (1, response_len)  — log π_sft(a_t|s_t)
    values: torch.Tensor             # (1, response_len)  — V(s_t)
    reward: torch.Tensor             # scalar — r_φ(x, y)
    prompt_len: int


# ---------------------------------------------------------------------------
# KL-penalised reward (Eq. 2)
# ---------------------------------------------------------------------------

def compute_kl_penalised_rewards(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    reward: torch.Tensor,
    action_mask: torch.Tensor,
    kl_coef: float,
) -> torch.Tensor:
    """Compute per-token KL-penalised rewards.

    R(x, y) = r_φ(x, y) - β * KL(π_θ || π_sft)

    The RM reward is placed at the last response token; the KL penalty is
    distributed across all response tokens (standard practice in RLHF).

    Args:
        log_probs: (1, T) current policy log-probs.
        ref_log_probs: (1, T) reference policy log-probs.
        reward: scalar RM reward.
        action_mask: (1, T) 1 for valid response tokens.
        kl_coef: β.

    Returns:
        Per-token rewards, shape (1, T).
    """
    kl = log_probs - ref_log_probs  # (1, T)
    kl_penalty = -kl_coef * kl * action_mask

    # Place RM reward at the last valid token
    token_rewards = kl_penalty.clone()
    last_token_idx = action_mask.sum(dim=1).long() - 1  # (1,)
    for b in range(token_rewards.size(0)):
        token_rewards[b, last_token_idx[b]] += reward[b] if reward.dim() > 0 else reward

    return token_rewards


# ---------------------------------------------------------------------------
# GAE (Generalized Advantage Estimation)
# ---------------------------------------------------------------------------

def compute_gae(
    values: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Works for both token-level (standard PPO) and macro-action level (MA-PPO)
    by treating the input as a 1-D sequence of time steps.

    Args:
        values: (1, T) value estimates.
        rewards: (1, T) per-step rewards.
        action_mask: (1, T) valid step mask.
        gamma: discount factor.
        lam: GAE lambda.

    Returns:
        advantages: (1, T)
        returns: (1, T)
    """
    T = values.size(1)
    advantages = torch.zeros_like(values)
    last_gae = 0.0

    for t in reversed(range(T)):
        mask_t = action_mask[0, t].item()
        if mask_t == 0:
            last_gae = 0.0
            continue
        next_value = values[0, t + 1].item() if t + 1 < T else 0.0
        delta = rewards[0, t].item() + gamma * next_value - values[0, t].item()
        last_gae = delta + gamma * lam * last_gae
        advantages[0, t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Standard PPO Trainer (token-level baseline)
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Token-level PPO trainer (standard RLHF baseline, §2.2)."""

    def __init__(
        self,
        policy: PolicyModel,
        critic: CriticModel,
        reward_model: Optional[RewardModel],
        ref_model: ReferenceModel,
        config: PPOConfig,
        device: torch.device,
        reward_fn=None,
    ):
        self.policy = policy
        self.critic = critic
        self.reward_model = reward_model  # None for APPS (uses compiler signal)
        self.ref_model = ref_model
        self.config = config
        self.device = device
        self.reward_fn = reward_fn  # callable(input_ids, tokenizer) -> float, for APPS

        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self._get_policy_lr()
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=self._get_critic_lr()
        )

    def _get_policy_lr(self) -> float:
        model_name = self.config.model_name.lower()
        task = self.config.task
        if "27b" in model_name:
            return self.config.policy_lr_27b
        elif "7b" in model_name:
            return self.config.policy_lr_7b
        elif task in ("apps",):
            return self.config.policy_lr_apps_2b
        return self.config.policy_lr_2b

    def _get_critic_lr(self) -> float:
        model_name = self.config.model_name.lower()
        task = self.config.task
        if "27b" in model_name:
            return self.config.critic_lr_27b
        elif "7b" in model_name:
            return self.config.critic_lr_7b
        elif task in ("apps",):
            return self.config.critic_lr_apps_2b
        return self.config.critic_lr_2b

    def _get_kl_coef(self) -> float:
        model_name = self.config.model_name.lower()
        task = self.config.task
        if "27b" in model_name:
            return self.config.kl_coef_27b
        elif "7b" in model_name and task == "webgpt":
            return self.config.kl_coef_7b_webgpt
        elif task in ("apps",):
            return self.config.kl_coef_apps
        return self.config.kl_coef_default

    @torch.no_grad()
    def collect_experience(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> Experience:
        """Generate a response and collect all quantities needed for PPO update."""
        prompt_ids = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)
        prompt_len = prompt_ids.size(1)

        # Generate response
        generated = self.policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )  # (1, prompt_len + response_len)

        full_ids = generated
        response_len = full_ids.size(1) - prompt_len
        full_mask = torch.ones_like(full_ids)

        # Action mask: 1 for response tokens only
        action_mask = torch.zeros(1, full_ids.size(1) - 1, device=self.device)
        action_mask[0, prompt_len - 1 : prompt_len - 1 + response_len] = 1

        # Log-probs under current policy
        _, log_probs = self.policy(full_ids, full_mask)  # (1, T-1)

        # Log-probs under reference policy
        ref_log_probs = self.ref_model(full_ids, full_mask)  # (1, T-1)

        # Value estimates
        values = self.critic(full_ids, full_mask)[:, :-1]  # (1, T-1)

        # RM reward (scalar) — or compiler signal for APPS
        if self.reward_model is not None:
            reward = self.reward_model(full_ids, full_mask)  # (1,)
        elif self.reward_fn is not None:
            reward_val = self.reward_fn(full_ids)
            reward = torch.tensor([reward_val], device=self.device, dtype=torch.float32)
        else:
            reward = torch.zeros(1, device=self.device)

        return Experience(
            input_ids=full_ids,
            attention_mask=full_mask,
            action_mask=action_mask,
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            values=values,
            reward=reward,
            prompt_len=prompt_len,
        )

    def compute_advantages(self, exp: Experience) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute token-level GAE advantages and returns."""
        kl_coef = self._get_kl_coef()
        token_rewards = compute_kl_penalised_rewards(
            exp.log_probs, exp.ref_log_probs, exp.reward,
            exp.action_mask, kl_coef
        )
        # Only use response portion for GAE
        start = exp.prompt_len - 1
        resp_values = exp.values[:, start:]
        resp_rewards = token_rewards[:, start:]
        resp_mask = exp.action_mask[:, start:]

        advantages, returns = compute_gae(
            resp_values, resp_rewards, resp_mask,
            gamma=self.config.gae_gamma, lam=self.config.gae_lambda
        )
        return advantages, returns

    def policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Standard token-level clipped PPO loss (Eq. 1)."""
        log_ratio = (log_probs - old_log_probs) * action_mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
        )
        loss = torch.sum(torch.max(pg_loss1, pg_loss2) * action_mask)
        loss = loss / (action_mask.sum() + 1e-8)
        return loss

    def critic_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped value function loss."""
        vf_loss1 = (values - returns) ** 2
        clipped = old_values + torch.clamp(
            values - old_values, -self.config.clip_ratio, self.config.clip_ratio
        )
        vf_loss2 = (clipped - returns) ** 2
        loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * action_mask)
        loss = loss / (action_mask.sum() + 1e-8)
        return loss

    def train_step(self, exp: Experience) -> Dict[str, float]:
        """One PPO update step."""
        advantages, returns = self.compute_advantages(exp)
        start = exp.prompt_len - 1

        old_log_probs = exp.log_probs[:, start:].detach()
        old_values = exp.values[:, start:].detach()
        resp_mask = exp.action_mask[:, start:]

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            # Policy update
            _, new_log_probs = self.policy(exp.input_ids, exp.attention_mask)
            new_log_probs_resp = new_log_probs[:, start:]

            p_loss = self.policy_loss(
                new_log_probs_resp, old_log_probs, advantages, resp_mask
            )
            self.policy_optimizer.zero_grad()
            p_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()

            # Critic update
            new_values = self.critic(exp.input_ids, exp.attention_mask)[:, :-1]
            new_values_resp = new_values[:, start:]

            c_loss = self.critic_loss(
                new_values_resp, old_values, returns, resp_mask
            )
            self.critic_optimizer.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

        metrics["policy_loss"] = p_loss.item()
        metrics["critic_loss"] = c_loss.item()
        metrics["reward"] = exp.reward.mean().item()
        return metrics


# ---------------------------------------------------------------------------
# MA-PPO Trainer (macro-action level, Algorithm 1)
# ---------------------------------------------------------------------------

class MAPPOTrainer(PPOTrainer):
    """MA-PPO: PPO with macro action-level advantage estimation (§3.2.2, §E).

    Overrides compute_advantages and the loss functions to operate at the
    macro action level.  The policy still outputs token-level probabilities;
    the joint macro action probability is the product of token probabilities
    within the macro action (§3.2.2).
    """

    def __init__(
        self,
        policy: PolicyModel,
        critic: CriticModel,
        reward_model: Optional[RewardModel],
        ref_model: ReferenceModel,
        ppo_config: PPOConfig,
        macro_config: MacroActionConfig,
        device: torch.device,
        reward_fn=None,
    ):
        super().__init__(policy, critic, reward_model, ref_model, ppo_config, device, reward_fn)
        self.macro_config = macro_config

    def _get_macro_positions(
        self,
        exp: Experience,
        parse_tree=None,
    ) -> List[int]:
        """Compute macro action boundary positions for a given experience."""
        start = exp.prompt_len - 1
        seq_len = exp.input_ids.size(1)
        mask = exp.action_mask  # (1, T-1)

        cfg = self.macro_config

        # Perplexity-based: compute PPL from reference model logits
        ppl_values = None
        if cfg.termination == "ppl":
            with torch.no_grad():
                ref_logits = self.ref_model.model(
                    exp.input_ids, exp.attention_mask
                ).logits
            ppl_values = compute_token_perplexities(
                ref_logits, exp.input_ids, start + 1
            )

        # n=∞: treat entire response as one macro action
        if cfg.use_full_sequence or cfg.n_gram == float("inf"):
            return [start, seq_len - 1]

        return get_macro_action_positions(
            start=start,
            seq_len=seq_len - 1,  # action_mask is T-1 long
            mask=mask,
            termination=cfg.termination,
            n_gram=cfg.n_gram,
            randomized_lengths=cfg.randomized_ngram_lengths,
            repeat_times=cfg.randomized_ngram_repeat_times,
            parse_tree=parse_tree,
            parser_cutoff=cfg.parser_cutoff,
            ppl_values=ppl_values,
        )

    def compute_advantages(
        self,
        exp: Experience,
        parse_tree=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Compute macro-level GAE advantages and returns.

        Returns:
            macro_advantages: (1, num_macro_actions)
            macro_returns: (1, num_macro_actions)
            sequence: boundary list
        """
        kl_coef = self._get_kl_coef()
        token_rewards = compute_kl_penalised_rewards(
            exp.log_probs, exp.ref_log_probs, exp.reward,
            exp.action_mask, kl_coef
        )

        start = exp.prompt_len - 1
        resp_mask = exp.action_mask[:, start:]
        resp_values = exp.values[:, start:]
        resp_rewards = token_rewards[:, start:]

        sequence = self._get_macro_positions(exp, parse_tree)

        # Aggregate token values and rewards to macro level
        macro_values = get_macro_action_values(
            resp_values, resp_mask, 0,
            [s - start for s in sequence],
            sigma_assignment=self.macro_config.sigma_assignment,
        )  # (1, num_macro)

        macro_rewards = get_macro_action_values(
            resp_rewards, resp_mask, 0,
            [s - start for s in sequence],
            sigma_assignment="equal",
        )  # (1, num_macro)

        num_macro = macro_values.size(1)
        macro_mask = torch.ones(1, num_macro, device=self.device)

        macro_advantages, macro_returns = compute_gae(
            macro_values, macro_rewards, macro_mask,
            gamma=self.config.gae_gamma, lam=self.config.gae_lambda
        )
        return macro_advantages, macro_returns, sequence

    def train_step(self, exp: Experience, parse_tree=None) -> Dict[str, float]:
        """One MA-PPO update step (Algorithm 1)."""
        macro_advantages, macro_returns, sequence = self.compute_advantages(
            exp, parse_tree
        )

        start = exp.prompt_len - 1
        old_log_probs = exp.log_probs[:, start:].detach()
        old_values = exp.values[:, start:].detach()
        resp_mask = exp.action_mask[:, start:]

        # Relative sequence (offset by start)
        rel_sequence = [s - start for s in sequence]

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            # Policy update (MA-PPO loss, Eq. 4)
            _, new_log_probs = self.policy(exp.input_ids, exp.attention_mask)
            new_log_probs_resp = new_log_probs[:, start:]

            p_loss = policy_loss_macro_action(
                new_log_probs_resp.squeeze(0),
                old_log_probs.squeeze(0),
                macro_advantages,
                resp_mask.squeeze(0),
                rel_sequence,
                clip_ratio=self.config.clip_ratio,
            )
            self.policy_optimizer.zero_grad()
            p_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()

            # Critic update (macro-level value loss)
            new_values = self.critic(exp.input_ids, exp.attention_mask)[:, :-1]
            new_values_resp = new_values[:, start:]

            c_loss = critic_loss_macro_action(
                new_values_resp.squeeze(0),
                old_values.squeeze(0),
                macro_returns.squeeze(0),
                resp_mask.squeeze(0),
                rel_sequence,
                clip_ratio=self.config.clip_ratio,
            )
            self.critic_optimizer.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

        metrics["policy_loss"] = p_loss.item()
        metrics["critic_loss"] = c_loss.item()
        metrics["reward"] = exp.reward.mean().item()
        metrics["num_macro_actions"] = macro_advantages.size(1)
        return metrics


# ---------------------------------------------------------------------------
# Training loop helper
# ---------------------------------------------------------------------------

def run_ppo_epoch(
    trainer: PPOTrainer,
    dataloader: DataLoader,
    global_step: int = 0,
    log_interval: int = 10,
) -> Tuple[int, Dict[str, float]]:
    """Run one epoch of PPO/MA-PPO training.

    Returns updated global_step and aggregated metrics.
    """
    total_metrics: Dict[str, float] = {}
    count = 0

    for batch in dataloader:
        prompt_ids = batch["input_ids"].to(trainer.device)
        prompt_mask = batch["attention_mask"].to(trainer.device)

        # Collect experience for each item in the batch
        for i in range(prompt_ids.size(0)):
            exp = trainer.collect_experience(
                prompt_ids[i : i + 1], prompt_mask[i : i + 1]
            )

            if isinstance(trainer, MAPPOTrainer):
                step_metrics = trainer.train_step(exp)
            else:
                step_metrics = trainer.train_step(exp)

            for k, v in step_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            count += 1
            global_step += 1

            if global_step % log_interval == 0:
                avg = {k: v / count for k, v in total_metrics.items()}
                print(f"Step {global_step}: " + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    avg_metrics = {k: v / max(count, 1) for k, v in total_metrics.items()}
    return global_step, avg_metrics
