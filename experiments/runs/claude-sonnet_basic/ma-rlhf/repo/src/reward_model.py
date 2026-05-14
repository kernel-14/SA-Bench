"""
Reward Model for RLHF training.

Implements the reward model training stage described in Section 2.2 of:
  "MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions"

The reward model is trained using the Bradley-Terry ranking loss on preference pairs.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


class RewardModel(nn.Module):
    """
    Reward model that wraps a causal LM and adds a scalar value head.

    The model is initialized from the SFT model checkpoint and trained on
    preference pairs using the ranking loss:
        L_RM = -log sigma(r_phi(x, y+) - r_phi(x, y-))

    The reward is computed as the scalar output at the last non-padding token.
    """

    def __init__(self, base_model: PreTrainedModel, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        # Linear value head: maps hidden states to scalar reward
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        # Initialize value head with small weights
        nn.init.normal_(self.value_head.weight, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward scores for a batch of sequences.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            rewards: (batch,) -- scalar reward per sequence
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use the last hidden state
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # Get the reward at the last non-padding token
        if attention_mask is not None:
            last_token_idx = attention_mask.sum(dim=1) - 1
        else:
            last_token_idx = torch.full(
                (input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device
            )

        batch_size = input_ids.size(0)
        last_hidden = hidden_states[torch.arange(batch_size), last_token_idx]  # (batch, hidden)
        rewards = self.value_head(last_hidden).squeeze(-1)  # (batch,)
        return rewards

    def get_token_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-token value estimates (used as critic in PPO).

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            values: (batch, seq_len)
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        values = self.value_head(hidden_states).squeeze(-1)  # (batch, seq_len)
        return values


def reward_model_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """
    Bradley-Terry ranking loss for reward model training.

    L_RM = -log sigma(r_phi(x, y+) - r_phi(x, y-))

    Args:
        chosen_rewards: Reward scores for preferred responses, shape (batch,).
        rejected_rewards: Reward scores for rejected responses, shape (batch,).

    Returns:
        Scalar loss.
    """
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


class RewardModelTrainer:
    """
    Trainer for the reward model stage of RLHF.

    Trains the reward model on preference pairs (x, y+, y-) using the
    Bradley-Terry ranking loss.
    """

    def __init__(
        self,
        model: RewardModel,
        tokenizer,
        learning_rate: float = 1e-5,
        batch_size: int = 64,
        max_length: int = 1024,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.max_length = max_length

    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> dict:
        """
        Perform one reward model training step.

        Args:
            chosen_input_ids: (batch, seq_len) -- preferred responses
            chosen_attention_mask: (batch, seq_len)
            rejected_input_ids: (batch, seq_len) -- rejected responses
            rejected_attention_mask: (batch, seq_len)

        Returns:
            Dict with 'loss', 'chosen_reward', 'rejected_reward', 'accuracy'.
        """
        self.model.train()

        chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)

        loss = reward_model_loss(chosen_rewards, rejected_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "accuracy": accuracy.item(),
        }
