# sft_trainer.py
"""
SFTTrainer: Supervised fine‑tuning stage for the MA‑RLHF pipeline.
Trains a causal language model on human‑written demonstrations (prompt + response)
using standard cross‑entropy loss.  The resulting checkpoint initialises the actor
in the subsequent PPO stage, and (with a classification head) the reward model.
"""

import logging
import math
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Supervised Fine‑Tuning trainer for autoregressive language models.

    Args:
        model: Pre‑trained HuggingFace causal LM (e.g., Gemma, CodeGemma).
        tokenizer: Tokenizer compatible with the model. Its `pad_token` will be set
            to `eos_token` if not already defined.
        config: Dictionary of SFT hyperparameters. Expected keys:
            - batch_size (int)
            - epochs (int)
            - learning_rate (float)
            - lr_scheduler (str, e.g. "cosine")
            - warmup_ratio (float)
            - (optional) gradient_checkpointing (bool)
            - (optional) deepspeed (dict) – if present and deepspeed is installed,
              initialises the DeepSpeed engine.
        max_seq_length: Optional explicit maximum total sequence length.  If not
            provided, sequences will be padded to the longest sample in each batch
            (dynamic padding).  Passing this value forces truncation of longer
            sequences, but the dataset is expected to already fit within the
            `max_prompt_length + max_response_length` limit.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        max_seq_length: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_length = max_seq_length

        # --- Tokenizer padding setup ---
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Set pad_token to eos_token ({self.tokenizer.pad_token})")

        # --- Hyperparameters ---
        self.batch_size = config.get("batch_size", 1)
        self.num_epochs = config.get("epochs", 1)
        self.learning_rate = config.get("learning_rate", 5e-5)
        self.lr_scheduler_type = config.get("lr_scheduler", "cosine")
        self.warmup_ratio = config.get("warmup_ratio", 0.0)

        # Optional gradient checkpointing (save memory)
        if config.get("gradient_checkpointing", False) and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")

        # --- Optimizer ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        # --- DeepSpeed engine (optional) ---
        self.deepspeed_enabled = False
        self.engine = None
        if HAS_DEEPSPEED and "deepspeed" in config:
            deepspeed_config = config["deepspeed"]
            # DeepSpeed's `initialize` also handles the optimizer and scheduler internally
            self.engine, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                optimizer=self.optimizer,
                config_params=deepspeed_config,
            )
            self.deepspeed_enabled = True
            logger.info("DeepSpeed engine initialised.")
        else:
            # Standard PyTorch: the scheduled will be built after total steps are known
            self.scheduler = None

        # Placeholder for the LR scheduler (will be created in `train()` after total steps known)
        self.total_steps = None

    def _collate_fn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pads `input_ids`, `attention_mask`, and `labels` to the maximum length in the batch."""
        input_ids = [torch.tensor(ids) for ids in batch["input_ids"]]
        attention_masks = [torch.tensor(am) for am in batch["attention_mask"]]
        labels = [torch.tensor(lbls) for lbls in batch["labels"]]

        # Pad sequences
        padded_inputs = nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_attention = nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        padded_labels = nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100  # ignore index for loss
        )

        return {
            "input_ids": padded_inputs,
            "attention_mask": padded_attention,
            "labels": padded_labels,
        }

    def train(self, train_dataset: Dataset) -> None:
        """
        Run SFT training on the provided dataset.

        Args:
            train_dataset: A HuggingFace Dataset containing 'input_ids', 'attention_mask',
                and 'labels' (already tokenized and truncated according to task).
        """
        self.model.train()
        device = next(self.model.parameters()).device

        # Create DataLoader with dynamic padding collator
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,  # keep simple; tokenization is offline
            pin_memory=True,
        )

        # Total training steps for scheduler (if not using DeepSpeed)
        if not self.deepspeed_enabled:
            self.total_steps = len(dataloader) * self.num_epochs
            warmup_steps = int(self.total_steps * self.warmup_ratio)

            if self.lr_scheduler_type == "cosine":
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=self.total_steps,
                )
            else:
                raise ValueError(f"Unsupported LR scheduler type: {self.lr_scheduler_type}")

        logger.info(f"Starting SFT for {self.num_epochs} epoch(s) ({len(dataloader)} steps/epoch).")

        global_step = 0
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False)

            for batch in progress:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if self.deepspeed_enabled:
                    # Forward pass through DeepSpeed engine
                    outputs = self.engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    self.engine.backward(loss)
                    self.engine.step()
                else:
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                # Logging
                loss_val = loss.item()
                epoch_loss += loss_val
                progress.set_postfix({"loss": f"{loss_val:.4f}"})
                global_step += 1

            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        logger.info("SFT training finished.")

    def save_checkpoint(self, save_dir: str) -> None:
        """
        Save the SFT model and tokenizer to `save_dir`.
        If DeepSpeed was used, the weights are extracted from the engine and saved
        in a standard HuggingFace format so that downstream stages can load them.
        """
        logger.info(f"Saving SFT checkpoint to {save_dir}")

        # Always save tokenizer
        self.tokenizer.save_pretrained(save_dir)

        if self.deepspeed_enabled:
            # Save DeepSpeed‑specific checkpoint
            self.engine.save_checkpoint(save_dir)
            # For compatibility, also extract the raw model and save as HF
            # DeepSpeed engine wraps the model: engine.module gives the original
            model_to_save = self.engine.module
        else:
            model_to_save = self.model

        # Save model in HuggingFace format
        model_to_save.save_pretrained(save_dir)
        logger.info(f"Checkpoint saved to {save_dir}")
