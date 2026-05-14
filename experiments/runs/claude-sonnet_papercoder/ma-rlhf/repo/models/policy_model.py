## models/policy_model.py
"""Policy model for the MA-RLHF pipeline.

This module implements PolicyModel, a foundational class used in three
distinct roles throughout the MA-RLHF pipeline:
  1. Policy model (π_θ): the model being trained via MA-PPO.
  2. Reference policy (π_ref): frozen copy of the SFT model, used for
     KL penalty computation.
  3. Critic model: same architecture, with value head applied externally
     by MAPPOTrainer using hidden_states from forward().

The class wraps transformers.AutoModelForCausalLM and provides the
specific interfaces needed by MAPPOTrainer, SFTTrainer, and the
evaluation pipeline.

Dependencies:
    - config.py: PPOConfig
    - torch, torch.nn.functional
    - transformers: AutoModelForCausalLM, AutoTokenizer, GenerationConfig
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import PPOConfig

logger = logging.getLogger(__name__)


class PolicyModel:
    """Causal LM wrapper for policy, reference policy, and critic roles.

    This class is the single model abstraction used across all three
    MA-PPO components that require a causal LM. The caller determines
    the role (policy, reference, critic) by controlling gradient flow
    and whether hidden_states are requested.

    Attributes:
        model: The underlying AutoModelForCausalLM instance.
        tokenizer: The associated PreTrainedTokenizer with left-padding
            configured for batched generation.
        config: The PPOConfig instance with generation hyperparameters.
        generation_config: A GenerationConfig built from config values,
            used as the default for generate() calls.
    """

    def __init__(
        self,
        model_path: str,
        config: PPOConfig,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Load the pretrained model and tokenizer, configure generation.

        Sets up left-padding (required for batched generation with
        decoder-only models), handles the missing pad_token for Gemma
        models, and builds a GenerationConfig from the PPOConfig values.

        Args:
            model_path: Path to a HuggingFace model directory or a
                HuggingFace Hub model identifier (e.g., 'google/gemma-2b'
                or a local SFT checkpoint path).
            config: PPOConfig instance. Provides temperature, top_p,
                top_k, max_response_length, and other generation params.
                All values are sourced from config.yaml (Table 5).
            torch_dtype: Optional dtype override. If None, defaults to
                torch.bfloat16 as specified in config.yaml
                (distributed.dtype: bf16). Pass torch.float32 for
                debugging on CPU.

        Raises:
            OSError: If model_path does not exist or is not a valid
                HuggingFace model directory.
        """
        self.config: PPOConfig = config

        # Resolve dtype: default to bfloat16 per config.yaml distributed.dtype.
        effective_dtype: torch.dtype = (
            torch_dtype if torch_dtype is not None else torch.bfloat16
        )

        logger.info(
            "Loading policy model from '%s' with dtype=%s.",
            model_path,
            effective_dtype,
        )

        # --- Tokenizer ---
        # Load tokenizer and configure left-padding for batched generation.
        # Left-padding ensures all sequences end at the same position,
        # which is required for correct batched autoregressive generation
        # with decoder-only models.
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"

        # Gemma models lack a dedicated pad token; use EOS as pad.
        # This is safe because the attention_mask distinguishes real tokens
        # from padding — never rely on token ID alone to detect padding.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(
                "pad_token not set; using eos_token '%s' (id=%d) as pad_token.",
                self.tokenizer.eos_token,
                self.tokenizer.eos_token_id,
            )

        # --- Model ---
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=effective_dtype,
            trust_remote_code=True,
        )

        logger.info(
            "Policy model loaded: %s, hidden_size=%d, vocab_size=%d.",
            type(self.model).__name__,
            self.model.config.hidden_size,
            self.model.config.vocab_size,
        )

        # --- Generation config ---
        # Build from PPOConfig values (Table 5 of the paper).
        # temperature=0.8 for Gemma, 1.0 for CodeGemma.
        # top_k=50 for Gemma, 5 for CodeGemma.
        # top_p=1.0 for all models.
        # max_new_tokens=512 for all models.
        self.generation_config: GenerationConfig = self._build_generation_config(
            config=config,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        logger.info(
            "Generation config: temperature=%.2f, top_p=%.2f, top_k=%d, "
            "max_new_tokens=%d, do_sample=%s.",
            self.generation_config.temperature,
            self.generation_config.top_p,
            self.generation_config.top_k,
            self.generation_config.max_new_tokens,
            self.generation_config.do_sample,
        )

    # ------------------------------------------------------------------
    # Core interface methods
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> CausalLMOutputWithPast:
        """Standard autoregressive forward pass.

        Returns logits over the vocabulary at every position. When
        output_hidden_states=True, also returns all layer hidden states,
        which MAPPOTrainer uses to compute value estimates for the critic.

        Gradient flow is controlled by the caller:
        - Policy update: called normally (gradients flow).
        - Reference policy KL: called under torch.no_grad().
        - Critic value estimation: called normally (gradients flow for
          critic update) or under torch.no_grad() (rollout phase).

        Args:
            input_ids: Integer token IDs of shape [batch_size, seq_len].
            attention_mask: Binary mask of shape [batch_size, seq_len].
                1 for real tokens, 0 for padding tokens.
            output_hidden_states: If True, include all layer hidden states
                in the returned output. Required when using this model as
                a critic (MAPPOTrainer applies a value head to the last
                hidden state). Defaults to False for efficiency.

        Returns:
            CausalLMOutputWithPast with fields:
                - logits: shape [batch_size, seq_len, vocab_size]
                - past_key_values: cached attention states
                - hidden_states: tuple of per-layer tensors if
                  output_hidden_states=True, else None
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate a response given a prompt.

        Returns the full sequence (prompt tokens + response tokens).
        The caller uses start = prompts.size()[-1] - 1 to index into
        the response portion, consistent with the PPO pseudocode in
        Appendix E of the paper.

        Generation parameters default to self.generation_config but can
        be overridden via kwargs. This allows MAPPOTrainer to sweep
        temperature during Best-of-N evaluation without modifying the
        stored config.

        Temperature = 0.0 edge case: when temperature is 0 (from the
        temperature_sweep: [0.0, 0.2, ...] in config.yaml), do_sample
        is set to False for greedy decoding.

        Args:
            input_ids: Prompt token IDs of shape [batch_size, prompt_len].
                Must be left-padded (tokenizer.padding_side='left').
            **kwargs: Optional overrides for generation parameters.
                Supported keys: temperature, top_p, top_k, max_new_tokens,
                do_sample, attention_mask.

        Returns:
            Full sequence tensor of shape [batch_size, prompt_len + response_len].
            Padded to the longest sequence in the batch using pad_token_id.
        """
        # Build attention mask if not provided in kwargs.
        # Left-padded inputs require an explicit mask.
        if "attention_mask" not in kwargs:
            attention_mask: torch.Tensor = (
                input_ids != self.tokenizer.pad_token_id
            ).long()
        else:
            attention_mask = kwargs.pop("attention_mask")

        # Build a merged generation config from defaults + overrides.
        gen_config: GenerationConfig = self._merge_generation_config(kwargs)

        with torch.no_grad():
            output_ids: torch.Tensor = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )

        return output_ids

    def get_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for a given sequence.

        Uses the standard "shifted" indexing for causal LMs: the log prob
        of token a_t is derived from the logits at position t-1 (the model
        predicts the next token from the current context).

        This method is used for:
        - Storing old log probs (π_θ_old) in the rollout buffer.
        - Computing current log probs (π_θ) during the PPO update step.
        - Computing reference log probs (π_ref) for KL penalty.

        Gradient flow is controlled by the caller:
        - Old log probs and reference log probs: call under torch.no_grad().
        - Current log probs during PPO update: call normally.

        The returned tensor aligns with the PPO pseudocode in Appendix E:
        - start = prompts.size()[-1] - 1
        - action_mask = attention_mask[:, 1:]
        - logprobs[:, start:] are the response token log probs.

        Args:
            input_ids: Integer token IDs of shape [batch_size, seq_len].
                Contains the full (prompt + response) sequence.
            attention_mask: Binary mask of shape [batch_size, seq_len].
                1 for real tokens, 0 for padding tokens.

        Returns:
            Log probability tensor of shape [batch_size, seq_len - 1].
            Element [b, t] is log π(input_ids[b, t+1] | input_ids[b, :t+1]),
            i.e., the log prob of the (t+1)-th token given all preceding
            tokens. The first token has no log prob (no preceding context).
        """
        # Forward pass to get logits: [batch_size, seq_len, vocab_size].
        outputs: CausalLMOutputWithPast = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        logits: torch.Tensor = outputs.logits  # [batch, seq_len, vocab_size]

        # Apply log_softmax over vocab dimension.
        # Shape: [batch_size, seq_len, vocab_size]
        log_probs_all: torch.Tensor = F.log_softmax(logits, dim=-1)

        # Shifted indexing:
        # - Predictions at positions [0, seq_len-2] predict tokens at [1, seq_len-1].
        # - log_probs_all[:, :-1, :] has shape [batch, seq_len-1, vocab_size].
        # - input_ids[:, 1:] has shape [batch, seq_len-1] (the target tokens).
        log_probs_shifted: torch.Tensor = log_probs_all[:, :-1, :]
        target_ids: torch.Tensor = input_ids[:, 1:].unsqueeze(-1)
        # [batch, seq_len-1, 1]

        # Gather the log prob at each actual token position.
        # Shape after gather: [batch, seq_len-1, 1]
        # Shape after squeeze: [batch, seq_len-1]
        gathered: torch.Tensor = torch.gather(
            log_probs_shifted, dim=2, index=target_ids
        ).squeeze(-1)

        return gathered

    def save_pretrained(self, path: str) -> None:
        """Save the model and tokenizer to a directory.

        Saves in standard HuggingFace format (config.json + model weights
        + tokenizer files). The saved directory can be passed to
        PolicyModel.__init__() or AutoModelForCausalLM.from_pretrained()
        in subsequent pipeline stages.

        Note: When using DeepSpeed ZeRO-3, weight consolidation must be
        handled by CheckpointUtils (utils/checkpoint_utils.py) before
        calling this method, since ZeRO-3 shards model parameters across
        devices.

        Args:
            path: Directory path where the model will be saved. Created
                if it does not exist.
        """
        os.makedirs(path, exist_ok=True)

        logger.info("Saving policy model to '%s'.", path)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        logger.info("Policy model saved successfully to '%s'.", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_generation_config(
        config: PPOConfig,
        pad_token_id: int,
        eos_token_id: int,
    ) -> GenerationConfig:
        """Build a GenerationConfig from PPOConfig values.

        Handles the temperature=0 edge case by disabling sampling and
        using greedy decoding when temperature is at or below zero.

        Args:
            config: PPOConfig with generation hyperparameters from
                config.yaml (Table 5).
            pad_token_id: Tokenizer's pad token ID.
            eos_token_id: Tokenizer's EOS token ID.

        Returns:
            A GenerationConfig instance ready for use in model.generate().
        """
        # temperature <= 0 means greedy decoding (no sampling).
        do_sample: bool = config.temperature > 0.0

        # When do_sample=False, temperature and top_k/top_p are ignored
        # by HuggingFace's generate(), but we set them anyway for clarity.
        effective_temperature: float = max(config.temperature, 1e-7)

        return GenerationConfig(
            temperature=effective_temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_new_tokens=config.max_response_length,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    def _merge_generation_config(
        self,
        overrides: Dict[str, Any],
    ) -> GenerationConfig:
        """Merge default generation config with per-call overrides.

        Creates a new GenerationConfig by starting from the stored
        defaults and applying any overrides. The stored config is never
        modified, so subsequent calls use the original defaults.

        Handles the temperature=0 edge case: if the override sets
        temperature to 0.0, do_sample is automatically set to False.

        Args:
            overrides: Dict of generation parameter overrides. Supported
                keys: temperature, top_p, top_k, max_new_tokens, do_sample.
                Unknown keys are silently ignored.

        Returns:
            A new GenerationConfig with overrides applied.
        """
        # Start from stored defaults.
        temperature: float = self.generation_config.temperature
        top_p: float = self.generation_config.top_p
        top_k: int = self.generation_config.top_k
        max_new_tokens: int = self.generation_config.max_new_tokens
        do_sample: bool = self.generation_config.do_sample
        pad_token_id: int = self.generation_config.pad_token_id
        eos_token_id: int = self.generation_config.eos_token_id

        # Apply overrides.
        if "temperature" in overrides:
            temperature = float(overrides["temperature"])
            # temperature=0 means greedy decoding.
            do_sample = temperature > 0.0
            temperature = max(temperature, 1e-7)

        if "top_p" in overrides:
            top_p = float(overrides["top_p"])

        if "top_k" in overrides:
            top_k = int(overrides["top_k"])

        if "max_new_tokens" in overrides:
            max_new_tokens = int(overrides["max_new_tokens"])

        if "do_sample" in overrides:
            do_sample = bool(overrides["do_sample"])

        return GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
