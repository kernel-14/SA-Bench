"""Model architectures for MA-RLHF.

Implements:
- SFT model (base LM with LM head)
- Reward model (base LM with value head)
- Policy model (used during RLHF)
- Critic model (value function for PPO)
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from typing import Optional, Tuple


class SFTModel(nn.Module):
    """Supervised Fine-Tuning model.

    Wraps a pretrained CausalLM for instruction following.
    """
    def __init__(self, model_name: str, use_flash_attn: bool = False):
        super().__init__()
        model_kwargs = {}
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, path: str):
        self.base_model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str):
        model = cls.__new__(cls)
        super(SFTModel, model).__init__()
        model.base_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16
        )
        return model

    @property
    def config(self):
        return self.base_model.config


class RewardModel(nn.Module):
    """Reward Model for RLHF.

    Initialized from SFT model, with a linear value head on top.
    Trained with ranking loss: L_RM = -log σ(r(x, y+) - r(x, y-))
    """
    def __init__(self, model_name: str, sft_checkpoint: Optional[str] = None):
        super().__init__()
        if sft_checkpoint is not None:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                sft_checkpoint, torch_dtype=torch.bfloat16,
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
            )

        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)

        # Initialize value head
        nn.init.normal_(self.value_head.weight, std=1.0 / (hidden_size ** 0.5))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward score for a complete sequence.

        Returns scalar reward per batch item.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]

        # Take hidden state at last non-padding position
        batch_size = input_ids.size(0)
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_hidden_states = last_hidden[range(batch_size), seq_lengths]

        reward = self.value_head(last_hidden_states).squeeze(-1)
        return reward

    def save_pretrained(self, path: str):
        self.base_model.save_pretrained(path)
        torch.save(self.value_head.state_dict(), f"{path}/value_head.pt")

    @classmethod
    def from_pretrained(cls, path: str):
        model = cls.__new__(cls)
        super(RewardModel, model).__init__()
        model.base_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16,
        )
        hidden_size = model.base_model.config.hidden_size
        model.value_head = nn.Linear(hidden_size, 1, bias=False)
        model.value_head.load_state_dict(torch.load(f"{path}/value_head.pt"))
        return model

    @property
    def config(self):
        return self.base_model.config


class PolicyModel(nn.Module):
    """Policy model for RLHF — the model being optimized with PPO.

    Same architecture as SFT, with additional methods for PPO.
    """
    def __init__(self, sft_checkpoint: str):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            sft_checkpoint, torch_dtype=torch.bfloat16,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass, returns logits."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def log_prob(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute log probability of labels under current policy."""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    def joint_log_prob(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sequence: list,
        start: int,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute joint log probability of macro actions.

        π_θ(ω_τ | s_τ) = ∏ π_θ(a_t | a_<t) for t in macro action
        """
        log_probs = self.log_prob(logits[:, start:], labels[:, start:])
        log_probs = log_probs * action_mask[:, start:]

        split_list = torch.diff(torch.tensor(sequence)).tolist()
        splited_log_probs = torch.split(log_probs, split_list, dim=-1)

        joint_log_probs = torch.zeros(
            log_probs.size(0), len(split_list),
            dtype=log_probs.dtype, device=log_probs.device,
        )
        for idx, lp_i in enumerate(splited_log_probs):
            joint_log_probs[:, idx] = lp_i.sum(dim=-1)

        return joint_log_probs

    def save_pretrained(self, path: str):
        self.base_model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str):
        model = cls.__new__(cls)
        super(PolicyModel, model).__init__()
        model.base_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16,
        )
        return model

    @property
    def config(self):
        return self.base_model.config


class CriticModel(nn.Module):
    """Critic (Value) model for PPO.

    Initialized from the reward model checkpoint.
    Predicts V(s_t) for states.
    """
    def __init__(self, rm_checkpoint: str):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            rm_checkpoint, torch_dtype=torch.bfloat16,
        )
        hidden_size = self.base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)

        try:
            self.value_head.load_state_dict(torch.load(f"{rm_checkpoint}/value_head.pt"))
        except Exception:
            nn.init.normal_(self.value_head.weight, std=1.0 / (hidden_size ** 0.5))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token value estimates.

        Returns:
            values: shape (batch, seq_len) — V(s_t) for each token position
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        values = self.value_head(hidden_states).squeeze(-1)  # (batch, seq_len)
        return values

    def save_pretrained(self, path: str):
        self.base_model.save_pretrained(path)
        torch.save(self.value_head.state_dict(), f"{path}/value_head.pt")

    @classmethod
    def from_pretrained(cls, path: str):
        model = cls.__new__(cls)
        super(CriticModel, model).__init__()
        model.base_model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16,
        )
        hidden_size = model.base_model.config.hidden_size
        model.value_head = nn.Linear(hidden_size, 1, bias=False)
        model.value_head.load_state_dict(torch.load(f"{path}/value_head.pt"))
        return model

    @property
    def config(self):
        return self.base_model.config


class ReferenceModel(nn.Module):
    """Reference (frozen) model for KL penalty computation.

    Holds the SFT policy for computing KL divergence.
    """
    def __init__(self, sft_checkpoint: str):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            sft_checkpoint, torch_dtype=torch.bfloat16,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def log_prob(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    @property
    def config(self):
        return self.base_model.config
