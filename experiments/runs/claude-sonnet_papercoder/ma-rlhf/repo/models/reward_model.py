## models/reward_model.py
"""Reward model for the MA-RLHF pipeline.

This module implements the RewardModel class used in Stage 2 (reward model
training) and Stage 3 (MA-PPO rollouts). The reward model wraps a causal LM
backbone initialized from an SFT checkpoint and adds a scalar head that
outputs a single reward score per (prompt, response) pair.

Architecture:
    backbone: AutoModelForCausalLM (full causal LM, LM head unused)
    scalar_head: nn.Linear(hidden_size, 1, bias=False)

The scalar reward is extracted from the hidden state at the last valid token
position, which encodes the full (prompt + response) context in a causal LM.

Dependencies:
    - config.py: RMConfig
    - torch, torch.nn
    - transformers: AutoModelForCausalLM
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from config import RMConfig

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """Scalar reward model built on top of a causal LM backbone.

    The backbone is initialized from an SFT checkpoint (not the base
    pretrained model). The LM head is retained in memory but its output
    is never used — instead, the last layer's hidden states are passed
    through a small scalar head to produce a reward score.

    Attributes:
        backbone: The full AutoModelForCausalLM (transformer body + LM head).
            Only the transformer body's hidden states are used.
        scalar_head: nn.Linear(hidden_size, 1, bias=False) that maps the
            last-token hidden state to a scalar reward.
        config: The RMConfig instance with training hyperparameters.
    """

    def __init__(
        self,
        base_model_path: str,
        config: RMConfig,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the RewardModel from an SFT checkpoint.

        The caller (main.py) is responsible for passing the SFT checkpoint
        path, not the base pretrained model path. This ensures the reward
        model and policy share the same initialization distribution.

        Args:
            base_model_path: Path to the SFT model checkpoint directory.
                Must be a valid HuggingFace model directory containing
                config.json and model weights.
            config: RMConfig instance with training hyperparameters.
            torch_dtype: Optional dtype override. If None, defaults to
                torch.bfloat16 (matching distributed.dtype: bf16 in
                config.yaml). Pass torch.float32 for debugging.

        Raises:
            OSError: If base_model_path does not exist or is not a valid
                HuggingFace model directory.
        """
        super().__init__()

        self.config: RMConfig = config

        # Resolve dtype: default to bfloat16 per config.yaml distributed.dtype.
        effective_dtype: torch.dtype = (
            torch_dtype if torch_dtype is not None else torch.bfloat16
        )

        logger.info(
            "Loading reward model backbone from '%s' with dtype=%s.",
            base_model_path,
            effective_dtype,
        )

        # Load the full causal LM. The LM head is retained but unused.
        # output_hidden_states is not set here — it is passed at forward time.
        self.backbone: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=effective_dtype,
            trust_remote_code=True,
        )

        # Determine hidden size from the backbone's config.
        hidden_size: int = self.backbone.config.hidden_size

        logger.info(
            "Backbone loaded: hidden_size=%d, vocab_size=%d.",
            hidden_size,
            self.backbone.config.vocab_size,
        )

        # Scalar head: maps last-token hidden state to a scalar reward.
        # bias=False matches the paper's description and standard RLHF practice.
        self.scalar_head: nn.Linear = nn.Linear(hidden_size, 1, bias=False)

        # Initialize scalar head with small weights to avoid large initial
        # reward scores that could destabilize early RM training.
        nn.init.normal_(self.scalar_head.weight, mean=0.0, std=0.01)

        # Cast scalar head to the same dtype as the backbone for consistency.
        self.scalar_head = self.scalar_head.to(effective_dtype)

        logger.info(
            "RewardModel initialized: backbone=%s, scalar_head=%s -> 1.",
            type(self.backbone).__name__,
            hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar reward scores for a batch of (prompt, response) pairs.

        Runs the backbone with output_hidden_states=True, extracts the last
        layer's hidden state at the last valid token position per batch
        element, and passes it through the scalar head.

        The last valid token position is determined by:
            last_token_idx = attention_mask.sum(dim=-1) - 1
        This formula is correct for both left-padded and right-padded
        sequences, since it counts the number of non-padding tokens.

        Args:
            input_ids: Integer token IDs of shape [batch_size, seq_len].
                Contains the concatenated (prompt + response) tokens.
            attention_mask: Binary mask of shape [batch_size, seq_len].
                1 for real tokens, 0 for padding tokens.

        Returns:
            A float tensor of shape [batch_size] containing the scalar
            reward score for each (prompt, response) pair in the batch.
            Higher values indicate higher predicted quality.
        """
        # Run backbone forward pass, requesting all hidden states.
        # We do not use outputs.logits (the LM head output).
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states is a tuple of tensors, one per layer plus the embedding.
        # hidden_states[-1] is the last transformer layer's output.
        # Shape: [batch_size, seq_len, hidden_size]
        hidden_states: torch.Tensor = outputs.hidden_states[-1]

        # Compute the index of the last valid (non-padding) token per sample.
        # attention_mask.sum(dim=-1) counts non-padding tokens per row.
        # Subtract 1 to convert count to 0-based index.
        # Shape: [batch_size]
        last_token_idx: torch.Tensor = (
            attention_mask.sum(dim=-1) - 1
        ).long()

        # Clamp to valid range to guard against all-padding edge cases.
        seq_len: int = hidden_states.size(1)
        last_token_idx = last_token_idx.clamp(min=0, max=seq_len - 1)

        batch_size: int = hidden_states.size(0)

        # Index into hidden_states to extract the last-token hidden state
        # for each batch element. Uses advanced indexing for efficiency.
        # Shape: [batch_size, hidden_size]
        last_hidden: torch.Tensor = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            last_token_idx,
        ]

        # Pass through scalar head to get reward scores.
        # squeeze(-1) removes the trailing dimension-1 from Linear output.
        # Shape: [batch_size]
        reward: torch.Tensor = self.scalar_head(last_hidden).squeeze(-1)

        return reward

    @torch.no_grad()
    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar reward scores without gradient tracking.

        This is the inference-time method used by MAPPOTrainer._rollout()
        to score generated responses. Using torch.no_grad() prevents
        gradient accumulation during the rollout phase, saving memory.

        The reward model parameters are NOT updated during PPO training.
        Only the policy and critic models are updated.

        Args:
            input_ids: Integer token IDs of shape [batch_size, seq_len].
            attention_mask: Binary mask of shape [batch_size, seq_len].

        Returns:
            A float tensor of shape [batch_size] with reward scores.
            Identical output to forward() but without gradient tracking.
        """
        return self.forward(input_ids, attention_mask)

    def save_pretrained(self, path: str) -> None:
        """Save the reward model to a directory.

        Saves the backbone in HuggingFace format (config.json + weights)
        and the scalar head as a separate PyTorch state dict file. This
        two-file approach is necessary because the scalar head is not part
        of the standard HuggingFace model format.

        The saved directory can be passed to load_pretrained() to fully
        reconstruct the reward model.

        Args:
            path: Directory path where the model will be saved. Created
                if it does not exist.
        """
        os.makedirs(path, exist_ok=True)

        logger.info("Saving reward model backbone to '%s'.", path)

        # Save backbone in HuggingFace format.
        # This saves config.json and model weights (safetensors or bin).
        self.backbone.save_pretrained(path)

        # Save scalar head as a standalone state dict.
        scalar_head_path: str = os.path.join(path, "scalar_head.pt")
        torch.save(self.scalar_head.state_dict(), scalar_head_path)

        logger.info(
            "Scalar head saved to '%s'.",
            scalar_head_path,
        )

        # Save a metadata marker to distinguish RM checkpoints from SFT.
        metadata_path: str = os.path.join(path, "reward_model_metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("checkpoint_type: reward_model\n")
            f.write(
                f"hidden_size: {self.backbone.config.hidden_size}\n"
            )
            f.write(
                f"scalar_head_bias: False\n"
            )

        logger.info("Reward model saved successfully to '%s'.", path)

    @classmethod
    def load_pretrained(
        cls,
        path: str,
        config: RMConfig,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "RewardModel":
        """Load a saved RewardModel from a directory.

        Reconstructs the full reward model by:
        1. Loading the backbone via AutoModelForCausalLM.from_pretrained()
        2. Overwriting the scalar head weights from scalar_head.pt

        The backbone is loaded from the same directory, so the path must
        contain both the HuggingFace backbone files (config.json, weights)
        and the scalar_head.pt file saved by save_pretrained().

        Args:
            path: Directory path containing the saved reward model.
                Must contain config.json, model weights, and scalar_head.pt.
            config: RMConfig instance. Used to configure the reconstructed
                model (same config as used during training).
            torch_dtype: Optional dtype override. Defaults to bfloat16.

        Returns:
            A fully reconstructed RewardModel instance with weights loaded
            from the checkpoint.

        Raises:
            FileNotFoundError: If path does not exist or scalar_head.pt
                is missing.
            RuntimeError: If the scalar head state dict is incompatible
                with the backbone's hidden size.
        """
        scalar_head_path: str = os.path.join(path, "scalar_head.pt")
        if not os.path.exists(scalar_head_path):
            raise FileNotFoundError(
                f"scalar_head.pt not found in '{path}'. "
                f"This directory may not be a valid RewardModel checkpoint. "
                f"Expected file: '{scalar_head_path}'."
            )

        logger.info("Loading reward model from '%s'.", path)

        # __init__ calls AutoModelForCausalLM.from_pretrained(path),
        # which loads the backbone from the checkpoint directory.
        # The scalar head is initialized with random weights here and
        # overwritten below.
        model: "RewardModel" = cls(
            base_model_path=path,
            config=config,
            torch_dtype=torch_dtype,
        )

        # Overwrite the randomly initialized scalar head with saved weights.
        scalar_head_state: dict = torch.load(
            scalar_head_path,
            map_location="cpu",
        )
        model.scalar_head.load_state_dict(scalar_head_state)

        # Cast scalar head to match backbone dtype after loading.
        effective_dtype: torch.dtype = (
            torch_dtype if torch_dtype is not None else torch.bfloat16
        )
        model.scalar_head = model.scalar_head.to(effective_dtype)

        logger.info(
            "Reward model loaded successfully from '%s'.",
            path,
        )

        return model
