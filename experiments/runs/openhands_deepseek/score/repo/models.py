"""LLM model wrapper for SCoRe.

Provides a policy π_θ that can:
1. Generate text given a prompt (greedy or sampling)
2. Compute log-probabilities of generated tokens (for REINFORCE)
3. Compute KL divergence against a reference policy

The paper uses Gemini 1.0 Pro and 1.5 Flash models internally.
This implementation provides a generic interface that works with
HuggingFace transformer models as a proxy.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def compute_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute log-probabilities of labels under the model.

    Args:
        model: The causal LM
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        labels: Target token IDs [batch, seq_len] (shifted input_ids)

    Returns:
        Per-token log-probabilities [batch, seq_len]
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    log_probs_all = F.log_softmax(logits, dim=-1)
    # labels are the tokens we want log-probs for (shifted by 1)
    # logits are [batch, seq_len, vocab], labels should be [batch, seq_len]
    log_probs = torch.gather(
        log_probs_all, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    return log_probs


def compute_kl_divergence(
    log_probs_policy: torch.Tensor,
    log_probs_ref: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence D_KL(π_θ || π_ref) per token.

    D_KL(π_θ || π_ref) = E_{y~π_θ}[log π_θ(y) - log π_ref(y)]
    This is approximated from samples.

    Args:
        log_probs_policy: Log-probs under current policy [batch, seq_len]
        log_probs_ref: Log-probs under reference policy [batch, seq_len]

    Returns:
        Per-token KL divergence (mean over sequence, per sample)
    """
    kl = log_probs_policy - log_probs_ref
    return kl


class LLMPolicy(nn.Module):
    """Wrapper around a CausalLM for multi-turn RL training.

    Supports:
    - Single-turn and multi-turn generation
    - REINFORCE policy gradient with KL penalty
    - Stage I: constrained first-turn (KL to base model)
    - Stage II: joint optimization with reward shaping
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )

        # Reference policy (base model, frozen) for KL penalty
        self.ref_model = None  # set when loading reference

    def set_reference_model(self, ref_model: "LLMPolicy") -> None:
        """Set the reference policy for KL penalty computation."""
        self.ref_model = ref_model

    def _prepare_inputs(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize prompts and prepare model inputs."""
        encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {
            "input_ids": encodings["input_ids"].to(self.device),
            "attention_mask": encodings["attention_mask"].to(self.device),
        }

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> List[str]:
        """Generate responses for given prompts.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            do_sample: Whether to sample or use greedy decoding

        Returns:
            List of generated response strings
        """
        inputs = self._prepare_inputs(prompts)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        generated_ids = outputs[:, input_len:]
        responses = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return responses

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for loss computation."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    def compute_reinforce_loss(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: torch.Tensor,
        ref_model: Optional["LLMPolicy"] = None,
        beta: float = 0.01,
    ) -> torch.Tensor:
        """Compute REINFORCE policy gradient loss with optional KL penalty.

        L = -E[reward * log π_θ(y|x)] + β * D_KL(π_θ || π_ref)

        Args:
            prompts: Input prompts
            responses: Generated responses (sampled from π_θ)
            rewards: Rewards for each response [batch]
            ref_model: Reference model for KL penalty (optional)
            beta: KL penalty coefficient β₁

        Returns:
            REINFORCE loss
        """
        device = rewards.device
        batch_size = len(prompts)

        # Combine prompt and response
        full_texts = [p + r for p, r in zip(prompts, responses)]
        prompt_encodings = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        full_encodings = self.tokenizer(
            full_texts, return_tensors="pt", padding=True, truncation=True
        )

        input_ids = full_encodings["input_ids"].to(device)
        attention_mask = full_encodings["attention_mask"].to(device)

        prompt_lens = prompt_encodings["input_ids"].shape[1]

        # Labels for log-prob computation: shift input_ids left
        labels = input_ids.clone()
        # Set prompt tokens to ignore index
        labels[:, :prompt_lens] = -100

        # Compute log-probs under current policy
        log_probs_policy = compute_log_probs(
            self.model, input_ids, attention_mask, labels
        )  # [batch, seq_len]

        # Sum log-probs over the response tokens (ignoring padded/prompt tokens)
        mask = (labels != -100).float().to(device)
        seq_log_probs = (log_probs_policy * mask).sum(dim=-1)  # [batch]

        # Policy gradient loss: -E[r * log π]
        # Center rewards for variance reduction
        advantages = rewards - rewards.mean()
        if rewards.std() > 1e-8:
            advantages = advantages / (rewards.std() + 1e-8)

        pg_loss = -(advantages * seq_log_probs).mean()

        # KL penalty
        kl_loss = torch.tensor(0.0, device=device)
        if ref_model is not None and beta > 0:
            with torch.no_grad():
                log_probs_ref = compute_log_probs(
                    ref_model.model, input_ids, attention_mask, labels
                )
            seq_kl = (log_probs_policy - log_probs_ref) * mask
            kl_per_sample = seq_kl.sum(dim=-1)
            kl_loss = beta * kl_per_sample.mean()

        total_loss = pg_loss + kl_loss

        return total_loss

    def compute_stage1_loss(
        self,
        prompts_turn1: List[str],
        responses_turn1: List[str],
        prompts_turn2: List[str],
        responses_turn2: List[str],
        rewards_turn2: torch.Tensor,
        ref_model: Optional["LLMPolicy"] = None,
        beta2: float = 0.1,
        beta1: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """Compute Stage I loss (Equation 3).

        Stage I maximizes second-attempt reward while constraining
        first-turn distribution to be close to the base model via KL.

        L_StageI = -E[r(y2, y*)] + β₂·D_KL(π_θ(·|x1) || π_ref(·|x1))

        The default KL penalty β₁ is also applied to both turns.

        Args:
            prompts_turn1: First-turn prompts
            responses_turn1: First-turn responses
            prompts_turn2: Second-turn prompts (includes first response)
            responses_turn2: Second-turn responses
            rewards_turn2: Rewards for second responses [batch]
            ref_model: Reference model (base model)
            beta2: Stage I KL penalty for first turn constraint
            beta1: Default KL penalty coefficient

        Returns:
            Dictionary with loss components
        """
        device = rewards_turn2.device

        # Combine prompts and responses for turn 1
        full_texts_t1 = [p + r for p, r in zip(prompts_turn1, responses_turn1)]
        full_texts_t2 = [p + r for p, r in zip(prompts_turn2, responses_turn2)]

        # Tokenize turn 1
        t1_prompt_enc = self.tokenizer(
            prompts_turn1, return_tensors="pt", padding=True, truncation=True
        )
        t1_full_enc = self.tokenizer(
            full_texts_t1, return_tensors="pt", padding=True, truncation=True
        )
        t1_prompt_len = t1_prompt_enc["input_ids"].shape[1]

        # Tokenize turn 2
        t2_prompt_enc = self.tokenizer(
            prompts_turn2, return_tensors="pt", padding=True, truncation=True
        )
        t2_full_enc = self.tokenizer(
            full_texts_t2, return_tensors="pt", padding=True, truncation=True
        )
        t2_prompt_len = t2_prompt_enc["input_ids"].shape[1]

        # Turn 2: Compute REINFORCE loss for second attempt
        t2_input_ids = t2_full_enc["input_ids"].to(device)
        t2_attention_mask = t2_full_enc["attention_mask"].to(device)
        t2_labels = t2_input_ids.clone()
        t2_labels[:, :t2_prompt_len] = -100

        log_probs_t2 = compute_log_probs(
            self.model, t2_input_ids, t2_attention_mask, t2_labels
        )
        mask_t2 = (t2_labels != -100).float().to(device)
        seq_log_probs_t2 = (log_probs_t2 * mask_t2).sum(dim=-1)

        # REINFORCE loss for turn 2
        advantages_t2 = rewards_turn2 - rewards_turn2.mean()
        if rewards_turn2.std() > 1e-8:
            advantages_t2 = advantages_t2 / (rewards_turn2.std() + 1e-8)
        pg_loss_t2 = -(advantages_t2 * seq_log_probs_t2).mean()

        # Turn 2 default KL penalty (β₁)
        kl_loss_t2 = torch.tensor(0.0, device=device)
        if ref_model is not None and beta1 > 0:
            with torch.no_grad():
                log_probs_ref_t2 = compute_log_probs(
                    ref_model.model, t2_input_ids, t2_attention_mask, t2_labels
                )
            seq_kl_t2 = (log_probs_t2 - log_probs_ref_t2) * mask_t2
            kl_loss_t2 = beta1 * seq_kl_t2.sum(dim=-1).mean()

        # Turn 1: Only KL penalty to constrain to base model (β₂)
        t1_input_ids = t1_full_enc["input_ids"].to(device)
        t1_attention_mask = t1_full_enc["attention_mask"].to(device)
        t1_labels = t1_input_ids.clone()
        t1_labels[:, :t1_prompt_len] = -100

        kl_loss_t1 = torch.tensor(0.0, device=device)
        if ref_model is not None and beta2 > 0:
            log_probs_t1 = compute_log_probs(
                self.model, t1_input_ids, t1_attention_mask, t1_labels
            )
            with torch.no_grad():
                log_probs_ref_t1 = compute_log_probs(
                    ref_model.model, t1_input_ids, t1_attention_mask, t1_labels
                )
            mask_t1 = (t1_labels != -100).float().to(device)
            seq_kl_t1 = (log_probs_t1 - log_probs_ref_t1) * mask_t1
            kl_loss_t1 = beta2 * seq_kl_t1.sum(dim=-1).mean()

        total_loss = pg_loss_t2 + kl_loss_t1 + kl_loss_t2

        return {
            "loss": total_loss,
            "pg_loss_t2": pg_loss_t2.detach(),
            "kl_loss_t1": kl_loss_t1.detach(),
            "kl_loss_t2": kl_loss_t2.detach(),
        }

    def compute_stage2_loss(
        self,
        prompts_turn1: List[str],
        responses_turn1: List[str],
        rewards_turn1: torch.Tensor,
        prompts_turn2: List[str],
        responses_turn2: List[str],
        rewards_turn2: torch.Tensor,
        ref_model: Optional["LLMPolicy"] = None,
        beta1: float = 0.01,
        alpha: float = 10.0,
        use_shaping: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute Stage II loss (Equation 4 + reward shaping).

        L_StageII = -E[Σ_i r(y_i, y*)] + β₁·D_KL(π_θ || π_ref)
        with shaped reward at turn 2: r'(y2) = r(y2) + α·(r(y2) - r(y1))

        Args:
            prompts_turn1: First-turn prompts
            responses_turn1: First-turn responses
            rewards_turn1: Rewards for first responses [batch]
            prompts_turn2: Second-turn prompts
            responses_turn2: Second-turn responses
            rewards_turn2: Rewards for second responses [batch]
            ref_model: Reference model
            beta1: KL penalty coefficient
            alpha: Reward shaping multiplier
            use_shaping: Whether to apply reward shaping

        Returns:
            Dictionary with loss components
        """
        device = rewards_turn1.device

        # Shaped reward for turn 2
        if use_shaping:
            shaped_rewards_t2 = rewards_turn2 + alpha * (rewards_turn2 - rewards_turn1)
        else:
            shaped_rewards_t2 = rewards_turn2

        # Combine prompts and responses
        full_texts_t1 = [p + r for p, r in zip(prompts_turn1, responses_turn1)]
        full_texts_t2 = [p + r for p, r in zip(prompts_turn2, responses_turn2)]

        # Tokenize turn 1
        t1_prompt_enc = self.tokenizer(
            prompts_turn1, return_tensors="pt", padding=True, truncation=True
        )
        t1_full_enc = self.tokenizer(
            full_texts_t1, return_tensors="pt", padding=True, truncation=True
        )
        t1_prompt_len = t1_prompt_enc["input_ids"].shape[1]

        # Tokenize turn 2
        t2_prompt_enc = self.tokenizer(
            prompts_turn2, return_tensors="pt", padding=True, truncation=True
        )
        t2_full_enc = self.tokenizer(
            full_texts_t2, return_tensors="pt", padding=True, truncation=True
        )
        t2_prompt_len = t2_prompt_enc["input_ids"].shape[1]

        # Turn 1: Compute REINFORCE loss
        t1_input_ids = t1_full_enc["input_ids"].to(device)
        t1_attention_mask = t1_full_enc["attention_mask"].to(device)
        t1_labels = t1_input_ids.clone()
        t1_labels[:, :t1_prompt_len] = -100

        log_probs_t1 = compute_log_probs(
            self.model, t1_input_ids, t1_attention_mask, t1_labels
        )
        mask_t1 = (t1_labels != -100).float().to(device)
        seq_log_probs_t1 = (log_probs_t1 * mask_t1).sum(dim=-1)

        advantages_t1 = rewards_turn1 - rewards_turn1.mean()
        if rewards_turn1.std() > 1e-8:
            advantages_t1 = advantages_t1 / (rewards_turn1.std() + 1e-8)
        pg_loss_t1 = -(advantages_t1 * seq_log_probs_t1).mean()

        # Turn 2: Compute REINFORCE loss with shaped rewards
        t2_input_ids = t2_full_enc["input_ids"].to(device)
        t2_attention_mask = t2_full_enc["attention_mask"].to(device)
        t2_labels = t2_input_ids.clone()
        t2_labels[:, :t2_prompt_len] = -100

        log_probs_t2 = compute_log_probs(
            self.model, t2_input_ids, t2_attention_mask, t2_labels
        )
        mask_t2 = (t2_labels != -100).float().to(device)
        seq_log_probs_t2 = (log_probs_t2 * mask_t2).sum(dim=-1)

        advantages_t2 = shaped_rewards_t2 - shaped_rewards_t2.mean()
        if shaped_rewards_t2.std() > 1e-8:
            advantages_t2 = advantages_t2 / (shaped_rewards_t2.std() + 1e-8)
        pg_loss_t2 = -(advantages_t2 * seq_log_probs_t2).mean()

        # KL penalty (both turns, β₁)
        kl_loss_t1 = torch.tensor(0.0, device=device)
        kl_loss_t2 = torch.tensor(0.0, device=device)
        if ref_model is not None and beta1 > 0:
            with torch.no_grad():
                log_probs_ref_t1 = compute_log_probs(
                    ref_model.model, t1_input_ids, t1_attention_mask, t1_labels
                )
                log_probs_ref_t2 = compute_log_probs(
                    ref_model.model, t2_input_ids, t2_attention_mask, t2_labels
                )
            seq_kl_t1 = (log_probs_t1 - log_probs_ref_t1) * mask_t1
            seq_kl_t2 = (log_probs_t2 - log_probs_ref_t2) * mask_t2
            kl_loss_t1 = beta1 * seq_kl_t1.sum(dim=-1).mean()
            kl_loss_t2 = beta1 * seq_kl_t2.sum(dim=-1).mean()

        total_loss = pg_loss_t1 + pg_loss_t2 + kl_loss_t1 + kl_loss_t2

        return {
            "loss": total_loss,
            "pg_loss_t1": pg_loss_t1.detach(),
            "pg_loss_t2": pg_loss_t2.detach(),
            "kl_loss_t1": kl_loss_t1.detach(),
            "kl_loss_t2": kl_loss_t2.detach(),
        }


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = "cuda",
) -> LLMPolicy:
    """Load an LLMPolicy from a model name or path."""
    return LLMPolicy(model_name_or_path, device)
