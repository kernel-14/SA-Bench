# rm_trainer.py
"""
Reward Model (RM) training stage for the MA‑RLHF pipeline.

The reward model is initialised from the SFT checkpoint. Its backbone is the
transformer, and a new linear head is attached that outputs a scalar reward
given a (prompt + response) sequence.  Training uses pairwise preference data
with the Bradley–Terry model loss.

This module integrates with the rest of the pipeline:
- It receives a tokenized Dataset (prepared by data_utils) containing
  'input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected',
  'attention_mask_rejected'.
- It saves a checkpoint that is later used by `ppo_trainer.py` and `evaluate.py`.
"""

import logging
import math
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

logger = logging.getLogger(__name__)


# ---------------------------
# Custom Reward Model
# ---------------------------

class RewardModel(nn.Module):
    """
    Scalar reward model built on a transformer backbone.

    This model is initialised from an SFT checkpoint: the transformer weights are
    loaded from the given path, and a randomly initialised linear head replaces
    the original language‑modelling head.

    The forward pass returns a single logit per sequence (the reward).
    """

    def __init__(self, model_name_or_path: str, tokenizer: PreTrainedTokenizer):
        super().__init__()
        # Load the transformer backbone (without LM head)
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        # The reward head maps the last hidden state (of the last token) to a scalar
        hidden_size = self.transformer.config.hidden_size
        self.v_head = nn.Linear(hidden_size, 1, bias=False)
        # Random initialisation
        nn.init.zeros_(self.v_head.weight)
        # Store tokenizer for convenience (e.g., access to pad_token_id)
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scalar reward values for a batch of token sequences.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            rewards: (batch_size,)  Float tensor.
        """
        # Get last hidden states
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state   # (batch, seq_len, hidden)

        # Extract the hidden state of the last non‑padding token.
        # We use the attention mask to find the position of the last token per sequence.
        # Equivalent: take the hidden state at the token just before the first padding.
        batch_size, seq_len = input_ids.shape
        # The last token index per sample (0‑based)
        # If there is padding, the last valid token is before the first padding.
        # We assume the sequences are padded to the right.
        # Compute sequence lengths from attention_mask
        lengths = attention_mask.sum(dim=1)  # (batch,)
        # The index of the last valid token is length - 1
        # We need to gather the hidden states at those positions.
        # Since gather requires indices of shape (batch, 1, hidden), we can use:
        gather_idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, last_hidden_state.size(-1))
        # Gather along seq_len dimension (dim=1)
        last_token_hidden = torch.gather(last_hidden_state, 1, gather_idx)  # shape (batch, 1, hidden)
        last_token_hidden = last_token_hidden.squeeze(1)  # (batch, hidden)

        # Apply reward head
        reward = self.v_head(last_token_hidden).squeeze(-1)  # (batch,)
        return reward

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[dict] = None):
        """Enable gradient checkpointing on the backbone transformer."""
        if hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs)


# ---------------------------
# Collator for RM data
# ---------------------------

def reward_collator(batch: List[Dict[str, List[int]]], tokenizer: PreTrainedTokenizer):
    """
    Pads *chosen* and *rejected* token sequences to the maximum length in the batch.

    Args:
        batch: List of samples, each containing 'input_ids_chosen',
               'attention_mask_chosen', 'input_ids_rejected',
               'attention_mask_rejected' as lists of ints.

    Returns:
        A dict with tensors ready for the model.
    """
    chosen_inputs = [torch.tensor(s["input_ids_chosen"]) for s in batch]
    chosen_masks = [torch.tensor(s["attention_mask_chosen"]) for s in batch]
    rejected_inputs = [torch.tensor(s["input_ids_rejected"]) for s in batch]
    rejected_masks = [torch.tensor(s["attention_mask_rejected"]) for s in batch]

    # Pad each group separately
    pad_id = tokenizer.pad_token_id
    chosen_inputs = nn.utils.rnn.pad_sequence(chosen_inputs, batch_first=True, padding_value=pad_id)
    chosen_masks = nn.utils.rnn.pad_sequence(chosen_masks, batch_first=True, padding_value=0)
    rejected_inputs = nn.utils.rnn.pad_sequence(rejected_inputs, batch_first=True, padding_value=pad_id)
    rejected_masks = nn.utils.rnn.pad_sequence(rejected_masks, batch_first=True, padding_value=0)

    return {
        "input_ids_chosen": chosen_inputs,
        "attention_mask_chosen": chosen_masks,
        "input_ids_rejected": rejected_inputs,
        "attention_mask_rejected": rejected_masks,
    }


# ---------------------------
# RMTrainer class
# ---------------------------

class RMTrainer:
    """
    Trains a reward model on pairwise preference data.

    Args:
        sft_checkpoint: Path or HuggingFace model identifier of the SFT checkpoint.
        tokenizer: Pre‑trained tokenizer (same as used for SFT).
        config: Dictionary containing RM hyperparameters and other relevant keys.
                Expected structure:
                  - rm: dict with 'batch_size', 'epochs', 'learning_rate',
                         'lr_scheduler', 'warmup_ratio'.
                  - max_prompt_length: int
                  - max_response_length: int
                  - (optional) deepspeed: dict
                  - (optional) gradient_checkpointing: bool
    """

    def __init__(
        self,
        sft_checkpoint: str,
        tokenizer: PreTrainedTokenizer,
        config: dict,
    ):
        self.tokenizer = tokenizer
        self.config = config

        # Ensure pad token is set; fallback to eos
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info("Pad token set to eos token.")

        # Load the reward model from the SFT backbone
        logger.info(f"Loading SFT checkpoint from {sft_checkpoint}")
        self.model = RewardModel(sft_checkpoint, self.tokenizer)
        self.model.train()

        # RM configuration
        rm_config = config.get("rm", {})
        self.batch_size = rm_config.get("batch_size", 64)
        self.num_epochs = rm_config.get("epochs", 1)
        self.learning_rate = rm_config.get("learning_rate", 1e-5)
        self.lr_scheduler_type = rm_config.get("lr_scheduler", "cosine")
        self.warmup_ratio = rm_config.get("warmup_ratio", 0.0)

        # Optional gradient checkpointing
        if config.get("gradient_checkpointing", False):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for RM training.")

        # DeepSpeed integration (optional)
        self.deepspeed_enabled = False
        self.engine = None
        if HAS_DEEPSPEED and "deepspeed" in config:
            deepspeed_config = config["deepspeed"]
            # Create optimizer manually (DeepSpeed will replace it with its own fused version)
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            )
            self.engine, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                config_params=deepspeed_config,
            )
            self.deepspeed_enabled = True
            logger.info("DeepSpeed engine initialised for RM trainer.")
        else:
            self.model = self.model  # keep as standard nn.Module
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            )

    def _loss_fn(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bradley‑Terry loss: -log(sigmoid(r_chosen - r_rejected)).

        Args:
            chosen_logits: (batch,) logits of preferred responses.
            rejected_logits: (batch,) logits of dispreferred responses.

        Returns:
            Scalar loss (averaged over batch).
        """
        diff = chosen_logits - rejected_logits
        loss = -torch.log(torch.sigmoid(diff)).mean()
        return loss

    def train(self, train_dataset: Dataset) -> None:
        """
        Runs reward model training for the specified number of epochs.

        Args:
            train_dataset: HuggingFace Dataset containing tokenized preference pairs
                           (fields: input_ids_chosen, attention_mask_chosen,
                            input_ids_rejected, attention_mask_rejected).
        """
        device = next(self.model.parameters()).device
        logger.info(f"Training reward model on {len(train_dataset)} samples, "
                    f"batch_size={self.batch_size}, epochs={self.num_epochs}")

        # Create DataLoader with custom collator
        collate_fn = lambda batch: reward_collator(batch, self.tokenizer)
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # no multiprocessing with tokenized dataset
            pin_memory=True,
        )

        # Total training steps for scheduler
        total_steps = len(dataloader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        # If not using DeepSpeed, create a scheduler
        if not self.deepspeed_enabled:
            if self.lr_scheduler_type == "cosine":
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )
            else:
                raise ValueError(f"Unsupported LR scheduler type: {self.lr_scheduler_type}")

        global_step = 0
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            progress = tqdm(dataloader, desc=f"RM Epoch {epoch+1}/{self.num_epochs}", leave=False)

            for batch in progress:
                # Move to device
                input_ids_chosen = batch["input_ids_chosen"].to(device)
                attention_mask_chosen = batch["attention_mask_chosen"].to(device)
                input_ids_rejected = batch["input_ids_rejected"].to(device)
                attention_mask_rejected = batch["attention_mask_rejected"].to(device)

                if self.deepspeed_enabled:
                    # Forward for chosen and rejected separately through DeepSpeed engine.
                    # DeepSpeed engine expects the model to be called directly.
                    # We'll compute loss manually after obtaining logits.
                    logits_chosen = self.engine(
                        input_ids=input_ids_chosen,
                        attention_mask=attention_mask_chosen,
                    )
                    # For the rejected batch we need a second forward pass.
                    # Since DeepSpeed wraps the model, we can call the raw module
                    # (self.engine.module) if we are careful, but the engine also
                    # supports a second forward. To keep it simple, we can use the
                    # underlying model: self.engine.module(...). However, that
                    # would bypass DeepSpeed’s hooks for logging, but gradients
                    # will be correct because the parameters are the same.
                    # A cleaner approach: use a separate call, accepting that
                    # two forward passes occur.
                    # DeepSpeed can handle multiple forward calls in a row before
                    # backward; we just need to ensure both passes are done before
                    # calling engine.backward and engine.step.
                    logits_rejected = self.engine(
                        input_ids=input_ids_rejected,
                        attention_mask=attention_mask_rejected,
                    )
                    loss = self._loss_fn(logits_chosen, logits_rejected)
                    self.engine.backward(loss)
                    self.engine.step()
                else:
                    self.optimizer.zero_grad()
                    logits_chosen = self.model(
                        input_ids=input_ids_chosen,
                        attention_mask=attention_mask_chosen,
                    )
                    logits_rejected = self.model(
                        input_ids=input_ids_rejected,
                        attention_mask=attention_mask_rejected,
                    )
                    loss = self._loss_fn(logits_chosen, logits_rejected)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                progress.set_postfix({"loss": f"{loss_val:.4f}"})
                global_step += 1

            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"RM Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        logger.info("RM training completed.")

    def save_checkpoint(self, save_dir: str) -> None:
        """
        Save the trained reward model and tokenizer.

        If DeepSpeed was used, the model weights are extracted from the engine
        and saved in standard HuggingFace format.

        Args:
            save_dir: Directory to save the checkpoint.
        """
        logger.info(f"Saving RM checkpoint to {save_dir}")

        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)

        # Extract model to save
        if self.deepspeed_enabled:
            model_to_save = self.engine.module
        else:
            model_to_save = self.model

        # Save model (as a RewardModel subclass; need to implement .save_pretrained)
        # We'll manually save the state dict and model config.
        # RewardModel does not have .save_pretrained by default; we can implement
        # a helper that saves both transformer and head.
        # Since we want to resume the RM later as a RewardModel instance,
        # we can save the whole torch model with torch.save, but for compatibility
        # with the rest of the pipeline (which may load via HuggingFace methods),
        # saving the transformer backbone and the head separately is safer.
        # Alternatively, we can save the transformer using its save_pretrained,
        # and save the head separately, and later reconstruct using the class.
        # The design assumes the RM checkpoint is loadable by `AutoModelForSequenceClassification`
        # maybe? But we are using a custom head. To keep it simple:
        # We'll save the full model state dict and a small config.
        import os
        import json

        os.makedirs(save_dir, exist_ok=True)

        # Save transformer backbone separately (HuggingFace format)
        model_to_save.transformer.save_pretrained(save_dir)
        # Save the reward head weight
        head_path = os.path.join(save_dir, "v_head.pt")
        torch.save(model_to_save.v_head.state_dict(), head_path)
        # Save a config JSON to know this is a RewardModel
        config_dict = {
            "model_type": "reward_model",
            "hidden_size": model_to_save.transformer.config.hidden_size,
        }
        with open(os.path.join(save_dir, "reward_config.json"), "w") as f:
            json.dump(config_dict, f)

        logger.info(f"Reward model saved to {save_dir}")

