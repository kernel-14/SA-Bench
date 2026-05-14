"""
Model definitions for MA-RLHF.

Three model wrappers around a pretrained causal LM:
  - PolicyModel   : generates responses; outputs per-token log-probs.
  - CriticModel   : estimates state values; adds a scalar value head.
  - RewardModel   : scores (prompt, response) pairs; adds a scalar reward head.

All three share the same base architecture (e.g. Gemma-2B) and are
initialised from the SFT checkpoint as described in §B.2.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


# ---------------------------------------------------------------------------
# Shared scalar head
# ---------------------------------------------------------------------------

class ScalarHead(nn.Module):
    """Linear projection from hidden_size → 1, used for value and reward heads."""

    def __init__(self, hidden_size: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=bias)
        nn.init.zeros_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states).squeeze(-1)


# ---------------------------------------------------------------------------
# Policy model
# ---------------------------------------------------------------------------

class PolicyModel(nn.Module):
    """Wraps a causal LM for use as the RL policy.

    The policy π_θ(a_t | s_t) is the standard next-token distribution of the
    pretrained LM.  No additional head is needed.
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__()
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )

    @property
    def config(self):
        return self.model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, log_probs_of_input_tokens).

        log_probs_of_input_tokens[i] = log π_θ(input_ids[i] | input_ids[:i]).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = outputs.logits  # (B, T, V)
        log_probs = self._token_log_probs(logits, input_ids)
        return logits, log_probs

    @staticmethod
    def _token_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute log π(a_t | s_t) for each position t.

        logits[:, :-1] predicts input_ids[:, 1:].
        Returns shape (B, T-1).
        """
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        target = input_ids[:, 1:].unsqueeze(-1)
        return log_probs.gather(-1, target).squeeze(-1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.model.config.eos_token_id,
            **kwargs,
        )

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()


# ---------------------------------------------------------------------------
# Critic model
# ---------------------------------------------------------------------------

class CriticModel(nn.Module):
    """Value function V^π(s_t) for PPO.

    Initialised from the reward model checkpoint (§B.2).  The scalar head
    is placed on top of the last hidden state at each token position.
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__()
        # Load the base transformer (without the LM head)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )
        hidden_size = self.model.config.hidden_size
        self.value_head = ScalarHead(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Returns per-token value estimates, shape (B, T)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]  # (B, T, H)
        values = self.value_head(hidden)    # (B, T)
        return values

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """Reward model r_φ(x, y) trained with the Bradley-Terry ranking loss (§2.2).

    The reward is the scalar head output at the last non-padding token of the
    full (prompt + response) sequence.
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__()
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )
        hidden_size = self.model.config.hidden_size
        self.reward_head = ScalarHead(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Returns scalar reward for each sequence in the batch, shape (B,)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        hidden = outputs.hidden_states[-1]  # (B, T, H)

        # Use the last non-padding token's hidden state
        if attention_mask is not None:
            last_token_idx = attention_mask.sum(dim=1) - 1  # (B,)
        else:
            last_token_idx = torch.full(
                (hidden.size(0),), hidden.size(1) - 1, device=hidden.device
            )

        last_hidden = hidden[
            torch.arange(hidden.size(0), device=hidden.device), last_token_idx
        ]  # (B, H)
        reward = self.reward_head(last_hidden)  # (B,)
        return reward

    def ranking_loss(
        self,
        chosen_reward: torch.Tensor,
        rejected_reward: torch.Tensor,
    ) -> torch.Tensor:
        """Bradley-Terry ranking loss (§2.2):
        L_RM = -log σ(r_φ(x, y+) - r_φ(x, y-))
        """
        return -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()


# ---------------------------------------------------------------------------
# Reference model (frozen SFT model for KL penalty)
# ---------------------------------------------------------------------------

class ReferenceModel(nn.Module):
    """Frozen copy of the SFT policy used to compute the KL penalty (§2.2, Eq. 2)."""

    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__()
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Returns per-token log-probs under the reference policy, shape (B, T-1)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = outputs.logits  # (B, T, V)
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        target = input_ids[:, 1:].unsqueeze(-1)
        return log_probs.gather(-1, target).squeeze(-1)


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
