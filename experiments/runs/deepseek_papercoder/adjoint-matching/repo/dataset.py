## dataset.py

import torch
from torch.utils.data import Dataset
from typing import Any


class PromptDataset(Dataset):
    """
    PyTorch Dataset that loads text prompts from a file and tokenises them
    for a CLIP‑compatible text encoder.

    Args:
        prompt_file: Path to a plain text file, one prompt per line.
        tokenizer:   A pre‑loaded Hugging Face tokenizer (e.g., CLIPTokenizer).
        config:      The application configuration object (Config).
                     May contain an optional ``max_token_length`` attribute.

    Returns (per item):
        A dictionary with ``input_ids`` and ``attention_mask``, both 1‑D PyTorch
        tensors of length ``max_length``.
    """

    def __init__(self, prompt_file: str, tokenizer: Any, config: Any) -> None:
        """
        Reads prompts, stores the tokenizer, and determines the maximum
        sequence length.

        Raises:
            FileNotFoundError: If ``prompt_file`` does not exist.
        """
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Keep only non‑empty lines (stripped of whitespace)
        self.prompts = [line.strip() for line in lines if line.strip()]

        self.tokenizer = tokenizer
        # Use config.max_token_length if present, otherwise fall back to the
        # tokenizer's own maximum.
        self.max_length = getattr(config, "max_token_length", None)
        if self.max_length is None or self.max_length <= 0:
            self.max_length = tokenizer.model_max_length

    def __len__(self) -> int:
        """Return the total number of prompts."""
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        """
        Tokenise the prompt at position ``idx``.

        Returns:
            dict::
                - "input_ids":        torch.LongTensor of shape (max_length,)
                - "attention_mask":   torch.LongTensor of shape (max_length,)
        """
        prompt = self.prompts[idx]
        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Squeeze the dummy batch dimension (tokenizer returns [1, max_length])
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

