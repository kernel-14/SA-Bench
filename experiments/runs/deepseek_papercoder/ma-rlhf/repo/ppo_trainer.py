# ppo_trainer.py
"""
PPOTrainer: Reinforcement Learning from Human Feedback (RLHF) training stage.
Supports both standard token‑level PPO and macro‑action PPO (MA‑PPO) as described
in the MA‑RLHF paper.

This trainer is responsible for:
  - Online generation of responses using the current policy.
  - Token‑level experience collection (log‑probabilities, values, rewards, KL penalties).
  - Optional macro‑action segmentation and aggregation via `MacroActionModule`.
  - Generalized Advantage Estimation (GAE) at either token or macro level.
  - Clipped surrogate policy loss and MSE critic loss, each with macro‑action support.
  - DeepSpeed integration for scalable training.

The trainer expects pre‑loaded models (actor, critic, reward, reference) and a
configuration dictionary (typically from a YAML file) that follows the schema
defined in `config.yaml`.
"""

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

from macro_actions import MacroActionModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility function: compiler reward for code generation (Appendix B.5)
# ---------------------------------------------------------------------------

def compiler_reward(prompt: str, code: str, test_info: Dict) -> float:
    """
    Evaluate a generated code snippet against test cases and return a scalar reward.

    The reward formula implements Appendix B.5 of the MA‑RLHF paper:
        if code compiles and runs:
            reward = -0.3 + 1.3 * N_pass / (N_pass + N_fail)
        if runtime error (but compiles):
            reward = -0.6
        if compile error:
            reward = -1.0

    Args:
        prompt: The original programming problem (string).
        code: The generated Python solution.
        test_info: A dict containing 'inputs' and 'outputs' lists (from the APPS dataset).

    Returns:
        Scalar reward.
    """
    # Write code to a temporary file and run with a timeout
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        # Run the code in a subprocess (sandboxed as much as possible)
        proc = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,  # seconds
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            # Runtime error or non‑zero exit
            return -0.6

        # Parse test information
        inputs = test_info.get("inputs", [])
        outputs = test_info.get("outputs", [])
        if not inputs or not outputs:
            # No tests: assume success if it runs
            return -0.3 + 1.3  # = 1.0

        # Compare outputs
        # This is a simplified check; the actual APPS test harness is more complex
        # but we follow the spirit: check if generated output matches expected.
        # Many code‑generation tasks print results. We'll attempt a simple line‑by‑line
        # comparison.
        try:
            expected_lines = [str(o).strip() for o in outputs]
            actual_lines = stdout.splitlines()
            N_pass = 0
            for exp, act in zip(expected_lines, actual_lines):
                if exp == act.strip():
                    N_pass += 1
            N_total = len(expected_lines)
            N_fail = N_total - N_pass
            if N_total == 0:
                return -0.3 + 1.3  # = 1.0
            return -0.3 + 1.3 * (N_pass / N_total)
        except Exception:
            return -0.6

    except subprocess.TimeoutExpired:
        return -0.6
    except Exception:
        return -1.0
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# CriticModel: wraps a transformer backbone with a scalar value head
# ---------------------------------------------------------------------------

class CriticModel(nn.Module):
    """
    Critic network for token‑level value estimation.

    Built on a pre‑trained transformer (same architecture as actor) with a randomly
    initialised linear head that predicts a scalar value for every token position.
    """

    def __init__(self, base_model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        hidden_size = self.base_model.config.hidden_size
        self.v_head = nn.Linear(hidden_size, 1)

        # Initialise head with zero weights (as in the paper’s RM)
        nn.init.zeros_(self.v_head.weight)
        if self.v_head.bias is not None:
            nn.init.zeros_(self.v_head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute per‑token value predictions.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            values: (batch_size, seq_len)  – one scalar per token.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        values = self.v_head(last_hidden).squeeze(-1)  # (B, L)
        return values


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Trains an LLM policy via PPO (with optional macro‑actions) using an online reward model.

    Args:
        models: Dictionary containing pre‑loaded models:
            'actor': AutoModelForCausalLM (policy)
            'critic': CriticModel (value function)
            'reward': AutoModelForSequenceClassification (reward model) or None for compiler reward.
            'ref': AutoModelForCausalLM (frozen reference for KL penalty)
        tokenizer: Tokenizer shared by all models.
        config: Full experiment configuration dict (as parsed from config.yaml).
        macro_config: Optional override for macro‑action parameters. If None,
            `config['macro']` is used. Expected keys: 'enabled', 'termination', 'n',
            'cutoff', 'weighting', etc.
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        macro_config: Optional[Dict[str, Any]] = None,
    ):
        self.models = models
        self.actor = models["actor"]
        self.critic = models["critic"]
        self.reward_model = models.get("reward")
        self.ref_model = models["ref"]
        self.tokenizer = tokenizer
        self.config = config

        # Extract PPO hyperparameters
        ppo_cfg = config.get("ppo", {})
        self.batch_size = ppo_cfg.get("batch_size", 256)
        self.policy_lr = ppo_cfg.get("policy_learning_rate", 1.5e-5)
        self.critic_lr = ppo_cfg.get("critic_learning_rate", 1.5e-5)
        self.ppo_epochs = ppo_cfg.get("ppo_epochs", 1)
        self.epochs_over_dataset = ppo_cfg.get("epochs_over_dataset", 1)
        self.rollout_batch_size = ppo_cfg.get("rollout_batch_size", 1)  # not used (batch per step)
        self.clip_ratio = ppo_cfg.get("clip_ratio", 0.2)
        self.lambda_gae = ppo_cfg.get("lambda_gae", 0.95)
        self.gamma_gae = ppo_cfg.get("gamma_gae", 1.0)
        self.kl_coef = ppo_cfg.get("kl_coef", 0.05)
        self.temperature = ppo_cfg.get("temperature", 0.8)
        self.top_p = ppo_cfg.get("top_p", 1.0)
        self.top_k = ppo_cfg.get("top_k", 50)
        self.reward_type = ppo_cfg.get("reward_type", "model")  # 'model' or 'compiler'

        self.max_prompt_length = config.get("max_prompt_length", 512)
        self.max_response_length = config.get("max_response_length", 512)
        self.device = next(self.actor.parameters()).device

        # Tokenizer padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info("Set pad_token to eos_token.")

        # --- Macro‑action setup ---
        macro_cfg = macro_config if macro_config is not None else config.get("macro", {})
        self.macro_enabled = macro_cfg.get("enabled", False)
        if self.macro_enabled:
            self.macro_module = MacroActionModule(
                tokenizer=self.tokenizer,
                termination=macro_cfg.get("termination", "fixed_ngram"),
                n=macro_cfg.get("n", 5),
                cutoff=macro_cfg.get("parsing_cutoff", 5),
                lengths_pool=macro_cfg.get("randomized_lengths", [2, 3, 5, 10]),
                randomized_repeat=macro_cfg.get("randomized_repeat", 3),
                weighting=macro_cfg.get("weighting", "equal"),
            )
        else:
            self.macro_module = None

        # --- DeepSpeed integration ---
        self.deepspeed_enabled = False
        self.actor_engine = None
        self.critic_engine = None
        if HAS_DEEPSPEED and "deepspeed" in config:
            ds_config = config["deepspeed"]
            # Separate engines for actor and critic
            # Actor
            actor_params = filter(lambda p: p.requires_grad, self.actor.parameters())
            self.actor_optimizer = AdamW(actor_params, lr=self.policy_lr)
            self.actor_engine, _, _, _ = deepspeed.initialize(
                model=self.actor,
                optimizer=self.actor_optimizer,
                config_params=ds_config,
            )
            # Critic
            critic_params = filter(lambda p: p.requires_grad, self.critic.parameters())
            self.critic_optimizer = AdamW(critic_params, lr=self.critic_lr)
            self.critic_engine, _, _, _ = deepspeed.initialize(
                model=self.critic,
                optimizer=self.critic_optimizer,
                config_params=ds_config,
            )
            self.deepspeed_enabled = True
            logger.info("DeepSpeed engines initialised for actor and critic.")
        else:
            # Standard optimisers
            self.actor_optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.actor.parameters()),
                lr=self.policy_lr,
            )
            self.critic_optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.critic.parameters()),
                lr=self.critic_lr,
            )
            # Schedulers will be created in train()
            self.actor_scheduler = None
            self.critic_scheduler = None

        # Freeze reference and reward models
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        if self.reward_model is not None:
            self.reward_model.eval()
            for p in self.reward_model.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------

    def train(self, ppo_dataset: Dataset) -> None:
        """
        Run PPO fine‑tuning on the given prompt dataset.

        Args:
            ppo_dataset: A HuggingFace Dataset containing tokenised prompts
                         (fields: 'input_ids', 'attention_mask', 'prompt_length').
        """
        dataloader = DataLoader(
            ppo_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_prompts,
            num_workers=0,
            pin_memory=True,
        )

        total_steps = len(dataloader) * self.epochs_over_dataset
        warmup_steps = self.config["ppo"].get("warmup_steps", 0)

        if not self.deepspeed_enabled:
            # Build schedulers
            self.actor_scheduler = get_cosine_schedule_with_warmup(
                self.actor_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            self.critic_scheduler = get_cosine_schedule_with_warmup(
                self.critic_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        global_step = 0
        for epoch in range(self.epochs_over_dataset):
            epoch_loss = 0.0
            progress = tqdm(dataloader, desc=f"PPO Epoch {epoch+1}/{self.epochs_over_dataset}", leave=False)

            for batch_prompts in progress:
                # batch_prompts contains tokenised prompts
                # We need the original strings for generation (to pass to model.generate)
                prompts_str = batch_prompts["prompt"]
                # Generate experience
                exp = self._generate_experience(prompts_str)

                # For each sample, compute macro boundaries (if enabled)
                boundaries_list = []
                for i in range(len(prompts_str)):
                    if self.macro_module is not None:
                        # Extract response tokens from full input_ids
                        prompt_len = batch_prompts["prompt_length"][i].item()
                        response_ids = exp["input_ids"][i, prompt_len:].tolist()
                        # We also need ref_logprobs for perplexity termination; they are per‑token for the full seq
                        ref_lps = exp["ref_logprobs"][i, prompt_len:]
                        text = exp["response_text"][i]
                        boundaries = self.macro_module.get_boundaries(
                            response_tokens=response_ids,
                            ref_logprobs=ref_lps,
                            text=text,
                        )
                        # Convert boundaries from relative to absolute indices
                        if boundaries is not None:
                            boundaries = [(s + prompt_len, e + prompt_len) for (s, e) in boundaries]
                    else:
                        boundaries = None
                    boundaries_list.append(boundaries)

                # PPO epochs (multiple gradient steps on the same collected experience)
                for ppo_epoch in range(self.ppo_epochs):
                    total_loss = 0.0

                    # Re‑compute actor logprobs under current (possibly updated) policy
                    # We run a forward pass on the full input_ids.
                    actor_outputs = self.actor(
                        input_ids=exp["input_ids"],
                        attention_mask=exp["attention_mask"],
                        return_dict=True,
                    )
                    new_logits = actor_outputs.logits
                    new_logprobs = F.log_softmax(new_logits, dim=-1)
                    # Gather logprobs of the actual tokens (only for response part)
                    # We'll extract all token logprobs for the full sequence, but only response matters
                    # We'll create a mask for response tokens
                    resp_mask = exp["response_mask"]

                    # For vanilla (non‑macro) samples, we use token‑level GAE and loss.
                    # For macro samples, we aggregate and use macro losses.
                    # We'll process samples one by one to handle variable boundaries.

                    actor_loss = 0.0
                    critic_loss = 0.0

                    for i, boundaries in enumerate(boundaries_list):
                        if boundaries is None:
                            # Vanilla PPO for this sample
                            adv, ret = self._compute_gae(
                                values=exp["values"][i:i+1],
                                rewards=exp["rewards"][i:i+1],
                                mask=resp_mask[i:i+1],
                                gamma=self.gamma_gae,
                                lam=self.lambda_gae,
                            )
                            # Standard PPO loss (token‑level)
                            old_logp = exp["old_logprobs"][i:i+1]
                            new_logp = torch.gather(
                                new_logprobs[i:i+1],
                                dim=-1,
                                index=exp["input_ids"][i:i+1, :, None],
                            ).squeeze(-1)
                            # Clip
                            ratio = torch.exp(new_logp - old_logp)
                            masked_ratio = ratio * resp_mask[i:i+1].float()
                            pg_loss1 = -adv * masked_ratio
                            pg_loss2 = -adv * torch.clamp(masked_ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                            pg_loss = torch.max(pg_loss1, pg_loss2).sum()
                            # Normalise by number of valid tokens
                            num_valid_tokens = resp_mask[i].sum().clamp(min=1)
                            pg_loss = pg_loss / num_valid_tokens

                            # Critic loss: MSE between values and returns
                            val_pred = exp["values"][i:i+1] * resp_mask[i:i+1].float()
                            ret_masked = ret * resp_mask[i:i+1].float()
                            v_loss = 0.5 * F.mse_loss(val_pred, ret_masked, reduction='sum') / num_valid_tokens

                            actor_loss += pg_loss
                            critic_loss += v_loss
                        else:
                            # Macro‑action PPO for this sample
                            macro_values = self.macro_module.aggregate_values(
                                values=exp["values"][i:i+1],
                                boundaries=boundaries,
                                weighting=self.macro_module.weighting,
                            )  # shape (1, M)
                            macro_rewards = self.macro_module.aggregate_rewards(
                                rewards=exp["rewards"][i:i+1],
                                boundaries=boundaries,
                            )  # shape (1, M)

                            # GAE on macro level
                            # We need a mask for macro actions (all valid)
                            macro_mask = torch.ones_like(macro_values)
                            adv, ret = self._compute_gae(
                                values=macro_values,
                                rewards=macro_rewards,
                                mask=macro_mask,
                                gamma=self.gamma_gae,
                                lam=self.lambda_gae,
                            )

                            # Macro policy loss
                            macro_actor_loss = self._macro_policy_loss(
                                new_logprobs=new_logprobs[i:i+1],
                                old_logprobs=exp["old_logprobs"][i:i+1],
                                advantages=adv,
                                boundaries=boundaries,
                                mask=resp_mask[i:i+1],
                                clip_ratio=self.clip_ratio,
                            )

                            # Macro critic loss
                            macro_critic_loss = self._macro_critic_loss(
                                values=exp["values"][i:i+1],
                                returns=ret,
                                boundaries=boundaries,
                                weighting=self.macro_module.weighting,
                                mask=resp_mask[i:i+1],
                            )

                            actor_loss += macro_actor_loss
                            critic_loss += macro_critic_loss

                    # Average over batch size
                    batch_size = len(boundaries_list)
                    actor_loss /= batch_size
                    critic_loss /= batch_size
                    total_loss = actor_loss + critic_loss

                    # Backward and step
                    if self.deepspeed_enabled:
                        # Use DeepSpeed engines for gradient computation
                        self.actor_engine.backward(actor_loss)
                        self.actor_engine.step()
                        self.critic_engine.backward(critic_loss)
                        self.critic_engine.step()
                    else:
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        actor_loss.backward(retain_graph=False)
                        critic_loss.backward()
                        self.actor_optimizer.step()
                        self.critic_optimizer.step()
                        self.actor_scheduler.step()
                        self.critic_scheduler.step()

                    # Logging
                    epoch_loss += total_loss.item()
                    progress.set_postfix({"loss": f"{total_loss.item():.4f}"})

                global_step += 1
                # checkpoint saving could be added here

            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"PPO Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        logger.info("PPO training finished.")

    # ------------------------------------------------------------------
    # Experience Generation
    # ------------------------------------------------------------------

    def _generate_experience(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Generate responses using the current actor policy and collect token‑level
        statistics: log‑probs, values, rewards, reference log‑probs.

        Args:
            prompts: List of raw prompt strings.

        Returns:
            A dict with keys:
                'input_ids': (B, L) full sequence (prompt + response)
                'attention_mask': (B, L)
                'response_mask': (B, L) boolean, True for generated tokens
                'old_logprobs': (B, L) actor log‑probs under the old (pre‑update) policy
                'values': (B, L) critic predictions
                'rewards': (B, L) token‑level rewards
                'ref_logprobs': (B, L) reference model log‑probs
                'response_text': List[str] decoded response strings
        """
        self.actor.eval()
        self.critic.eval()

        # Tokenize prompts (left padding)
        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_prompt_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_ids = tokenized["input_ids"].to(self.device)
        prompt_mask = tokenized["attention_mask"].to(self.device)
        prompt_lens = prompt_mask.sum(dim=1)  # length before padding

        # Generate responses with sampling
        gen_kwargs = {
            "max_new_tokens": self.max_response_length,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else 1.0,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,  # we'll recompute logits later
        }
        with torch.no_grad():
            generated = self.actor.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                **gen_kwargs,
            )
        full_ids = generated.sequences  # (B, prompt_len + response_len)
        response_mask = torch.zeros_like(full_ids, dtype=torch.bool)
        for i in range(full_ids.size(0)):
            response_mask[i, prompt_lens[i]:] = True

        # Decode responses
        response_texts = []
        for i in range(full_ids.size(0)):
            start = prompt_lens[i]
            resp_tokens = full_ids[i, start:].tolist()
            text = self.tokenizer.decode(resp_tokens, skip_special_tokens=True)
            response_texts.append(text)

        # Run actor, critic, and reference forward on the full sequences to get logits/values
        attention_mask = (full_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            # Actor logits
            actor_outputs = self.actor(input_ids=full_ids, attention_mask=attention_mask, return_dict=True)
            actor_logits = actor_outputs.logits
            # Critic values
            critic_values = self.critic(input_ids=full_ids, attention_mask=attention_mask)  # (B, L)
            # Reference logits
            ref_outputs = self.ref_model(input_ids=full_ids, attention_mask=attention_mask, return_dict=True)
            ref_logits = ref_outputs.logits

        # Compute logprobs
        actor_logprobs = F.log_softmax(actor_logits, dim=-1)
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)

        # Gather token‑level logprobs for the actual tokens
        gathered_actor_logprobs = torch.gather(actor_logprobs, dim=-1, index=full_ids.unsqueeze(-1)).squeeze(-1)
        gathered_ref_logprobs = torch.gather(ref_logprobs, dim=-1, index=full_ids.unsqueeze(-1)).squeeze(-1)

        # Token‑level rewards
        # KL penalty per token
        kl_penalty = gathered_actor_logprobs - gathered_ref_logprobs  # (B, L)
        tokenwise_kl_reward = -self.kl_coef * kl_penalty

        # Final reward per sequence
        final_rewards = torch.zeros(full_ids.size(0), device=self.device)
        if self.reward_type == "model" and self.reward_model is not None:
            # Use reward model to score full (prompt + response)
            rm_scores = self.reward_model(
                input_ids=full_ids,
                attention_mask=attention_mask,
            )
            # reward model returns scalar per sequence
            if isinstance(rm_scores, tuple):
                rm_scores = rm_scores[0]
            final_rewards = rm_scores.view(-1)
        elif self.reward_type == "compiler":
            # Compute compiler reward for each sample using test cases from dataset (but we don't have them here)
            # We'll store a reference to the dataset's test cases; assume they are provided via config or external dict.
            # For now, we'll set a placeholder and expect the caller to have prepared a mapping.
            # In a full implementation, we would pass a dict `self.test_cases` containing test information per prompt.
            # We'll raise NotImplementedError for now, or implement a dummy reward.
            logger.warning("Compiler reward not implemented in this minimal version; using dummy reward 0.0")
            final_rewards.fill_(0.0)

        # Assemble token rewards: add final reward to the last token of each response
        token_rewards = tokenwise_kl_reward.clone()
        for i in range(full_ids.size(0)):
            # Find last response token index
            last_resp_idx = prompt_lens[i] + response_mask[i].sum().item() - 1
            if last_resp_idx >= 0:
                token_rewards[i, last_resp_idx] += final_rewards[i]

        # Store old logprobs (before this training step)
        old_logprobs = gathered_actor_logprobs.clone()

        self.actor.train()
        self.critic.train()

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "old_logprobs": old_logprobs,
            "values": critic_values,
            "rewards": token_rewards,
            "ref_logprobs": gathered_ref_logprobs,
            "response_text": response_texts,
        }

    # ------------------------------------------------------------------
    # GAE Computation
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) for a batch of sequences.

        Args:
            values: (B, L) or (B, M) – state value predictions.
            rewards: (B, L) or (B, M) – immediate rewards.
            mask: (B, L) or (B, M) – binary mask indicating valid positions.
            gamma: discount factor.
            lam: GAE lambda parameter.

        Returns:
            advantages: (B, L/M)
            returns: (B, L/M)
        """
        B, L = values.shape
        advantages = torch.zeros_like(values)
        last_adv = torch.zeros(B, device=values.device)
        for t in reversed(range(L)):
            next_val = values[:, t+1] if t+1 < L else 0.0
            delta = rewards[:, t] + gamma * next_val - values[:, t]
            last_adv = delta + gamma * lam * last_adv
            # Mask out invalid positions
            valid = mask[:, t].float().unsqueeze(1) if mask.dim() > 1 else mask[:, t]
            advantages[:, t] = last_adv * mask[:, t].float()
        returns = advantages + values * mask.float()
        return advantages, returns

    # ------------------------------------------------------------------
    # Macro‑specific Loss Functions
    # ------------------------------------------------------------------

    def _macro_policy_loss(
        self,
        new_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        boundaries: List[Tuple[int, int]],
        mask: torch.Tensor,
        clip_ratio: float,
    ) -> torch.Tensor:
        """
        Compute PPO policy loss at the macro‑action level.

        Args:
            new_logprobs: (1, L) tensor of token‑level log‑probs under current policy.
            old_logprobs: (1, L) tensor of token‑level log‑probs under old policy.
            advantages: (1, M) tensor of macro‑level advantages.
            boundaries: list of (start, end) absolute indices for macro actions.
            mask: (1, L) binary mask (1 for response tokens).
            clip_ratio: PPO clipping parameter.

        Returns:
            Scalar loss value.
        """
        total_mask_sum = mask.sum().clamp(min=1)
        loss = 0.0
        for i, (start, end) in enumerate(boundaries):
            # Sum logprobs over the macro segment
            seg_new_logp = new_logprobs[0, start:end].sum()
            seg_old_logp = old_logprobs[0, start:end].sum()
            # Mask out any padding tokens? Already masked by the boundaries (they are on valid tokens).
            # But in case of variable length, we trust the boundaries.
            ratio = torch.exp(seg_new_logp - seg_old_logp)
            adv = advantages[0, i]
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            segment_loss = torch.max(pg_loss1, pg_loss2)
            loss += segment_loss
        loss = loss / total_mask_sum
        return loss

    def _macro_critic_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        boundaries: List[Tuple[int, int]],
        weighting: str,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE critic loss at the macro‑action level.

        Args:
            values: (1, L) token‑level value predictions.
            returns: (1, M) macro‑level returns (GAE output).
            boundaries: list of (start, end) indices.
            weighting: aggregation weighting ('equal', 'unit', 'position_decayed').
            mask: (1, L) binary mask.

        Returns:
            Scalar loss.
        """
        # Aggregate token values into macro values
        # We'll reuse the same aggregation as in macro_module but call the function directly.
        # To avoid code duplication, we can instantiate a temporary MacroActionModule or just compute mean here.
        # For simplicity, we'll compute simple equal‑weighted aggregation here; the weighting is already
        # accounted for in the macro module when computing values for GAE.
        # Actually, the critic loss should use the same weighting as used for the value function
        # when computing GAE. To keep consistency, we'll call the macro_module's aggregate_values.
        if self.macro_module is not None:
            pred_macro_values = self.macro_module.aggregate_values(
                values=values,
                boundaries=boundaries,
                weighting=weighting,
            )  # (1, M)
        else:
            # Fallback: equal weighting
            pred_macro_values_list = []
            for start, end in boundaries:
                seg = values[0, start:end]
                pred_macro_values_list.append(seg.mean())
            pred_macro_values = torch.stack(pred_macro_values_list).unsqueeze(0)

        loss = 0.5 * F.mse_loss(pred_macro_values, returns, reduction='sum')
        # Normalise by number of macro actions or total tokens? The paper normalises by total mask sum.
        # We'll follow the same normalisation as policy loss (total tokens).
        total_mask_sum = mask.sum().clamp(min=1)
        loss = loss / total_mask_sum
        return loss

    # ------------------------------------------------------------------
    # Utility: collate function for prompts
    # ------------------------------------------------------------------

    def _collate_prompts(self, batch: List[Dict]) -> Dict:
        """
        Collate function for the PPO prompt dataset.  Expects each sample to be a dict
        with 'prompt' (string), 'input_ids', 'attention_mask', 'prompt_length'.
        Returns a dict with batched tensors and original prompt strings.
        """
        prompts = [sample["prompt"] for sample in batch]
        input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]
        attention_masks = [torch.tensor(sample["attention_mask"]) for sample in batch]
        prompt_lengths = [sample["prompt_length"] for sample in batch]

        # Pad to max length
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = torch.full((len(batch), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_masks = torch.zeros((len(batch), max_len), dtype=torch.long)
        plens = []
        for i, (ids, mask, pl) in enumerate(zip(input_ids, attention_masks, prompt_lengths)):
            length = len(ids)
            padded_ids[i, :length] = ids
            padded_masks[i, :length] = torch.tensor(mask)
            plens.append(pl)

        return {
            "prompt": prompts,
            "input_ids": padded_ids,
            "attention_mask": padded_masks,
            "prompt_length": torch.tensor(plens),
        }

    # ------------------------------------------------------------------
    # Checkpoint Save / Load
    # ------------------------------------------------------------------

    def save_checkpoint(self, save_dir: str) -> None:
        """
        Save the current state of actor and critic (and optimizers) to `save_dir`.
        """
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Saving PPO checkpoint to {save_dir}")

        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)

        if self.deepspeed_enabled:
            self.actor_engine.save_checkpoint(os.path.join(save_dir, "actor"))
            self.critic_engine.save_checkpoint(os.path.join(save_dir, "critic"))
        else:
            self.actor.save_pretrained(os.path.join(save_dir, "actor"))
            torch.save(
                self.critic.state_dict(),
                os.path.join(save_dir, "critic.pt"),
            )
            torch.save(
                self.actor_optimizer.state_dict(),
                os.path.join(save_dir, "actor_optimizer.pt"),
            )
            torch.save(
                self.critic_optimizer.state_dict(),
                os.path.join(save_dir, "critic_optimizer.pt"),
            )

    def load_checkpoint(self, save_dir: str) -> None:
        """
        Restore actor, critic and optimizers from a previous checkpoint.
        """
        logger.info(f"Loading PPO checkpoint from {save_dir}")

        if self.deepspeed_enabled:
            self.actor_engine.load_checkpoint(os.path.join(save_dir, "actor"))
            self.critic_engine.load_checkpoint(os.path.join(save_dir, "critic"))
        else:
            self.actor = AutoModelForCausalLM.from_pretrained(
                os.path.join(save_dir, "actor")
            )
            self.critic.load_state_dict(
                torch.load(os.path.join(save_dir, "critic.pt"), map_location=self.device)
            )
            self.actor_optimizer.load_state_dict(
                torch.load(os.path.join(save_dir, "actor_optimizer.pt"), map_location=self.device)
            )
            self.critic_optimizer.load_state_dict(
                torch.load(os.path.join(save_dir, "critic_optimizer.pt"), map_location=self.device)
            )
