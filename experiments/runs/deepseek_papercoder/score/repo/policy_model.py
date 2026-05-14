"""
policy_model.py

Provides a PolicyModel class that wraps a HuggingFace causal language model
and its tokenizer.  It supplies three key operations required by on‑policy
REINFORCE training in SCoRe:

    * generate()          – sample new tokens given a prefix (returns token IDs)
    * compute_logprobs()  – per‑token log‑probabilities of a generated sequence
    * compute_kl_penalty()– per‑token difference between two log‑prob tensors

The class is deliberately lightweight: it does **not** handle distributed
training orchestration (that is the responsibility of the RLTrainer) nor does
it contain any reward‑shaping logic.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Default maximum sequence length for tokenizer truncation if not specified.
DEFAULT_MAX_SEQ_LENGTH = 2048

# Default number of new tokens to generate when not overridden.
DEFAULT_MAX_NEW_TOKENS = 512


class PolicyModel:
    """A generative language model with log‑probability queries for RL.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``meta-llama/Llama-2-7b-hf``).
    device_map : dict, optional
        Optional device map for HuggingFace Accelerate (e.g. ``{"": "cuda:0"}``).
    max_seq_length : int, optional
        Maximum number of tokens the tokenizer will keep after truncation.
        Defaults to ``DEFAULT_MAX_SEQ_LENGTH``.
    use_gradient_checkpointing : bool, optional
        If True, enable gradient checkpointing on the model.  Usually beneficial
        for large models during RL.  Default is True.
    """

    def __init__(
        self,
        model_name: str,
        device_map: Optional[dict] = None,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.max_seq_length = max_seq_length

        # Load the underlying model and tokenizer.
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

        # Ensure the tokenizer’s padding side is left.  This is critical for
        # correct batch generation when the input sequences have different
        # lengths, as happens after concatenating the first‑turn response.
        self.tokenizer.padding_side = "left"

        # Set padding token if missing (common for e.g. LLaMA).
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Pad token set to EOS token (%d).", self.tokenizer.pad_token_id)

        # Optionally enable gradient checkpointing to save memory during RL.
        if use_gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled on the model.")
            else:
                logger.warning("Model does not support gradient checkpointing; skipping.")

        # Start in evaluation mode.  The trainer will call .train() on the
        # policy model when needed; the reference model always stays in eval().
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _load_model(self) -> PreTrainedModel:
        """Load the causal LM from HuggingFace."""
        dtype = torch.float32  # safer for reproducibility; use fp16 only if needed
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=dtype,
            # trust_remote_code=True might be needed for some models
        )
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer and set the model‑maximum length."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.model_max_length = self.max_seq_length
        return tokenizer

    # ------------------------------------------------------------------ #
    #  Public API – matching the design
    # ------------------------------------------------------------------ #

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        """Generate new tokens from a prefix, returning only the **added** tokens.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Batch of token‑ID sequences, shape ``(batch, prompt_len)``.
            Left‑padded by the caller (tokenizer will have done so).
        max_new_tokens : int, optional
            Maximum number of tokens to generate.
        temperature : float, optional
            Sampling temperature.  Set to ``0.0`` for greedy decoding.

        Returns
        -------
        torch.LongTensor
            Newly generated token IDs, shape ``(batch, max_new_tokens)``,
            right‑padded with the tokenizer’s pad token ID if necessary.
        """
        assert input_ids.dim() == 2, "input_ids must be a 2D tensor."

        # Move inputs to the model’s device.
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        # Derive attention mask: ignore pad tokens (which are left‑padded).
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        attention_mask = (input_ids != pad_token_id).long()
        # Record the true length of each prompt (number of non‑padding tokens).
        prompt_lengths = attention_mask.sum(dim=1)

        do_sample = temperature > 0.0

        # Generate.
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # top_p=1.0, top_k=0  (defaults are fine)
        )

        # Extract the newly generated part (skip the prompt prefix).
        generated_ids_list = []
        max_len = 0
        for i in range(output_ids.size(0)):
            # The generated part for sequence i starts after prompt_lengths[i].
            gen = output_ids[i, prompt_lengths[i] :]
            generated_ids_list.append(gen)
            if gen.size(0) > max_len:
                max_len = gen.size(0)

        # Pad all generated sequences to the same length for batching.
        padded_generated = torch.full(
            (len(generated_ids_list), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=device,
        )
        for i, gen in enumerate(generated_ids_list):
            n = gen.size(0)
            padded_generated[i, :n] = gen

        return padded_generated

    def compute_logprobs(
        self,
        input_ids: torch.LongTensor,
        generated_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute per‑token log‑probabilities of `generated_ids` conditioned on
        `input_ids`.

        Parameters
        ----------
        input_ids : torch.LongTensor, shape (batch, prompt_len)
            Left‑padded prompt token IDs.
        generated_ids : torch.LongTensor, shape (batch, gen_len)
            Right‑padded generated token IDs (as returned by `generate`).

        Returns
        -------
        torch.Tensor, shape (batch, gen_len)
            Log‑probability of each generated token under the current model.
            Positions corresponding to padding tokens have value ``0.0``.
        """
        assert input_ids.dim() == 2 and generated_ids.dim() == 2
        batch_size, gen_len = generated_ids.shape

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        generated_ids = generated_ids.to(device)

        # Concatenate prompt and generated tokens.
        full_ids = torch.cat([input_ids, generated_ids], dim=1)  # (B, T)

        # Attention mask: 1 for non‑padding tokens.
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        full_attention_mask = (full_ids != pad_token_id).long()

        # Forward pass.  Gradients will flow if the model is in training mode.
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
        )
        logits = outputs.logits  # (B, T, V)

        # Shift logits and labels to align for next‑token‑prediction.
        # logits at position t predict token at t+1.
        shift_logits = logits[:, :-1, :]  # (B, T-1, V)
        shift_labels = full_ids[:, 1:]    # (B, T-1)

        # Log‑softmax over vocabulary.
        log_probs = torch.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

        # Gather the log‑probabilities of the actual tokens.
        selected_log_probs = torch.gather(
            log_probs, dim=2, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)

        # Now we only need the part that corresponds to the generated tokens.
        # The prompt occupies the first `prompt_len` tokens; after shifting,
        # the generated region starts at position (prompt_len - 1) in the
        # shifted tensor (the first generated token corresponds to shift index
        # `prompt_len - 1` because we dropped the last logit).
        prompt_len = input_ids.size(1)
        generated_start = prompt_len - 1
        generated_end = generated_start + gen_len

        # Slice the generated region.  If gen_len exceeds available positions
        # (e.g., because generation stopped early), we take as many as possible,
        # the rest will be ignored by padding mask later.
        available = selected_log_probs.size(1)
        actual_end = min(generated_end, available)
        logprobs_generated = selected_log_probs[:, generated_start:actual_end]

        # If the generated tensor was padded, pad the logprobs to gen_len.
        if actual_end < generated_end:
            pad_count = generated_end - actual_end
            padding = torch.zeros(batch_size, pad_count, device=device, dtype=logprobs_generated.dtype)
            logprobs_generated = torch.cat([logprobs_generated, padding], dim=1)

        # Zero out positions that correspond to padding in generated_ids.
        # (The padding tokens are pad_token_id; their logprobs are not meaningful.)
        mask_generated_pad = (generated_ids == pad_token_id)
        logprobs_generated[mask_generated_pad] = 0.0

        return logprobs_generated

    @staticmethod
    def compute_kl_penalty(
        logprobs_policy: torch.Tensor,
        logprobs_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per‑token difference `(log π_θ - log π_ref)`.

        Parameters
        ----------
        logprobs_policy : torch.Tensor, shape (B, L)
            Per‑token log‑probabilities from the current policy.
        logprobs_ref : torch.Tensor, shape (B, L)
            Per‑token log‑probabilities from the frozen reference model.

        Returns
        -------
        torch.Tensor, shape (B, L)
            Element‑wise difference.
        """
        return logprobs_policy - logprobs_ref

    # ------------------------------------------------------------------ #
    #  Convenience wrappers (not part of the published interface but
    #  needed by the trainer to convert between strings and token IDs)
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        """Return the device on which the model’s parameters reside."""
        return next(self.model.parameters()).device

    def decode_tokens(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token‑ID sequences to strings.

        This is a thin wrapper around `self.tokenizer.batch_decode`.

        Parameters
        ----------
        token_ids : torch.Tensor, shape (batch, seq_len)
            The token IDs to decode.
        skip_special_tokens : bool, optional
            Whether to remove special tokens (e.g., EOS) from the output.

        Returns
        -------
        list of str
            Decoded strings.
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def tokenize_prompts(
        self,
        texts: List[str],
        **kwargs,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Tokenize a list of prompt strings with left‑padding.

        Useful for obtaining `input_ids` and `attention_mask` before calling
        `generate`.

        Parameters
        ----------
        texts : list of str
            The raw prompt strings.
        **kwargs
            Passed to the tokenizer (e.g., ``max_length``).

        Returns
        -------
        input_ids : torch.LongTensor
        attention_mask : torch.LongTensor
        """
        # The tokenizer is already set to padding_side='left'.
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=kwargs.pop("max_length", self.max_seq_length),
            **kwargs,
        )
        return tokenized["input_ids"], tokenized["attention_mask"]
