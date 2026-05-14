```python
## models/model_wrapper.py
"""Model wrapper for SCoRe: Self-Correction via Reinforcement Learning.

This module implements ModelWrapper, the central interface between the
training/evaluation logic and the underlying HuggingFace causal language
model. Every component that needs to generate text or compute log-
probabilities goes through this class.

The log-probability computation is the most critical method: the REINFORCE
policy gradient depends on accurate per-token log-probs over response tokens
only (not prompt tokens). Any error here silently corrupts the gradient.

Design invariants:
    - compute_log_probs() returns the SUM (not mean) of log-probs over
      response tokens per sample. This equals log P(response | prompt)
      under the autoregressive factorization.
    - freeze=True creates the reference model: no LoRA, all params frozen,
      eval mode. The reference model is never updated.
    - Left-padding is used for batch generation; right-padding is used for
      log-prob computation (both are handled correctly).
    - All generation is wrapped in torch.no_grad(); log-prob computation
      is NOT wrapped (gradients must flow for training).

Typical usage:
    from config import Config
    from models.model_wrapper import ModelWrapper

    # Policy model (trainable, with LoRA)
    policy = ModelWrapper(config, freeze=False)

    # Reference model (frozen base model, no LoRA)
    ref_model = ModelWrapper(config, freeze=True)

    # Generate responses
    responses = policy.generate(prompts, temperature=1.0, max_new_tokens=1024)

    # Compute log-probs for REINFORCE gradient
    log_probs = policy.compute_log_probs(prompts, responses)

    # Compute reference log-probs for KL penalty
    with torch.no_grad():
        ref_log_probs = ref_model.compute_log_probs(prompts, responses)
"""

import logging
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — guarded so the module can be imported even if these
# packages are not installed (tests, linting, CI without full deps).
# ---------------------------------------------------------------------------
try:
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        PeftModel,
    )

    _PEFT_AVAILABLE: bool = True
except ImportError:
    _PEFT_AVAILABLE = False
    logger.warning(
        "peft is not installed. LoRA fine-tuning will be disabled. "
        "Install peft==0.12.0 to enable parameter-efficient training."
    )

# ---------------------------------------------------------------------------
# Dtype mapping from config string to torch dtype
# ---------------------------------------------------------------------------
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "auto": "auto",
}

# Ignore index for cross-entropy / log-prob masking (PyTorch convention)
_IGNORE_INDEX: int = -100


class ModelWrapper:
    """Unified interface for HuggingFace causal LM generation and log-prob computation.

    Wraps a HuggingFace AutoModelForCausalLM with:
        - Optional LoRA adapters (PEFT) for parameter-efficient fine-tuning.
        - Batch text generation with correct prompt stripping.
        - Per-sample log-probability computation over response tokens only.
        - Checkpoint save/load with PEFT-aware serialization.

    The freeze=True path creates the reference model (π_ref): no LoRA,
    all parameters frozen, eval mode. The reference model shares the same
    interface but is never updated during training.

    Attributes:
        config: The global Config instance driving all decisions.
        freeze: Whether this instance is a frozen reference model.
        model: The underlying PreTrainedModel (possibly PEFT-wrapped).
        tokenizer: The PreTrainedTokenizerBase for this model.
        device: The primary device of the model (used for tensor placement).
    """

    def __init__(self, config: Config, freeze: bool = False) -> None:
        """Initialize ModelWrapper.

        Loads the model and tokenizer from HuggingFace hub, optionally
        applies LoRA adapters, and optionally freezes all parameters.

        Args:
            config: The global Config instance. Reads:
                config.task (to select math vs code model name),
                config.model_name (HuggingFace model identifier),
                config.torch_dtype (dtype string, e.g. "bfloat16"),
                config.device_map (e.g. "auto"),
                config.use_lora (whether to apply LoRA),
                config.lora_rank, config.lora_alpha, config.lora_dropout,
                config.lora_target_modules (LoRA configuration).
            freeze: If True, skip LoRA and freeze all parameters. This
                creates the reference model (π_ref) that is never updated.
                If False, apply LoRA (if config.use_lora) and set to
                training mode.

        Raises:
            RuntimeError: If the model cannot be loaded from HuggingFace.
            ImportError: If use_lora=True but peft is not installed.
        """
        self.config: Config = config
        self.freeze: bool = freeze

        # These are populated by _load_model_and_tokenizer()
        self.model: PreTrainedModel = None  # type: ignore[assignment]
        self.tokenizer: PreTrainedTokenizerBase = None  # type: ignore[assignment]
        self.device: torch.device = torch.device("cpu")

        # Step 1: Load model and tokenizer from HuggingFace hub
        self._load_model_and_tokenizer()

        # Step 2: Apply LoRA (only for trainable policy model, not reference)
        if not freeze and config.use_lora:
            if not _PEFT_AVAILABLE:
                raise ImportError(
                    "config.use_lora=True but peft is not installed. "
                    "Install peft==0.12.0: pip install peft==0.12.0"
                )
            self._apply_lora()
        elif freeze and config.use_lora:
            # Reference model: do NOT apply LoRA. The reference model must
            # be the raw base model weights to ensure KL divergence is
            # computed against the true base model distribution.
            logger.info(
                "ModelWrapper (freeze=True): Skipping LoRA for reference model. "
                "Reference model uses raw base weights."
            )

        # Step 3: Freeze or set to training mode
        if freeze:
            # Freeze all parameters — reference model is never updated
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            logger.info(
                "ModelWrapper (freeze=True): All parameters frozen. "
                "Reference model in eval mode."
            )
        else:
            self.model.train()
            logger.info(
                "ModelWrapper (freeze=False): Policy model in training mode."
            )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def generate(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
    ) -> List[str]:
        """Generate responses for a batch of prompts.

        Returns only the newly generated tokens (not the prompt). Uses
        greedy decoding when temperature=0.0, stochastic sampling otherwise.

        The paper uses:
            - temperature=1.0 for training rollouts (Table 5)
            - temperature=0.0 for evaluation (Section 6: "greedy decoding")
            - temperature=0.7 for inference-compute scaling (Section 6.2)

        Args:
            prompts: List of prompt strings. Tokenized with left-padding
                so all sequences in the batch have the same length.
            temperature: Sampling temperature. 0.0 = greedy decoding.
                Must be >= 0.0.
            max_new_tokens: Maximum number of new tokens to generate per
                prompt. Sourced from config.max_new_tokens (default 1024).

        Returns:
            List of generated response strings, one per prompt, in the
            same order as the input. Prompt tokens are stripped — only
            the newly generated content is returned.

        Raises:
            ValueError: If temperature < 0.0.
        """
        if temperature < 0.0:
            raise ValueError(
                f"temperature must be >= 0.0 (got {temperature}). "
                "Use 0.0 for greedy decoding."
            )

        if not prompts:
            return []

        # ------------------------------------------------------------------
        # Tokenize with left-padding for batch generation.
        # Left-padding ensures the last real token before generation is
        # always at the rightmost position, which is required for correct
        # autoregressive generation in batches.
        # ------------------------------------------------------------------
        # Temporarily set padding side to left for generation
        original_padding_side: str = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        try:
            encoding = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length
                if hasattr(self.tokenizer, "model_max_length")
                and self.tokenizer.model_max_length is not None
                and self.tokenizer.model_max_length < 1_000_000
                else 4096,
            )
        finally:
            # Restore original padding side
            self.tokenizer.padding_side = original_padding_side

        input_ids: torch.Tensor = encoding["input_ids"].to(self.device)
        attention_mask: torch.Tensor = encoding["attention_mask"].to(self.device)

        # Record the prompt length (same for all after left-padding)
        prompt_length: int = input_ids.shape[1]

        # ------------------------------------------------------------------
        # Build generation kwargs based on temperature
        # ------------------------------------------------------------------
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature == 0.0:
            # Greedy decoding — deterministic, used for evaluation
            generation_kwargs["do_sample"] = False
        else:
            # Stochastic sampling — used for training rollouts
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature

        # ------------------------------------------------------------------
        # Generate with no_grad (we don't need gradients for generation)
        # ------------------------------------------------------------------
        with torch.no_grad():
            output_ids: torch.Tensor = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        # ------------------------------------------------------------------
        # Strip prompt tokens: output_ids has shape (batch, prompt_len + new_len)
        # Slice off the prompt portion to get only the generated tokens.
        # This works correctly because all prompts are left-padded to the
        # same length (prompt_length), so the slice is uniform across the batch.
        # ------------------------------------------------------------------
        generated_ids: torch.Tensor = output_ids[:, prompt_length:]

        # ------------------------------------------------------------------
        # Decode generated token ids to strings
        # ------------------------------------------------------------------
        responses: List[str] = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        logger.debug(
            "generate(): batch_size=%d, temperature=%.2f, "
            "max_new_tokens=%d, prompt_length=%d.",
            len(prompts),
            temperature,
            max_new_tokens,
            prompt_length,
        )

        return responses

    def compute_log_probs(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """Compute per-sample log-probabilities of responses given prompts.

        Returns the SUM of per-token log-probabilities over response tokens
        for each sample. This equals log P(response | prompt) under the
        autoregressive factorization:

            log P(y | x) = sum_t log P(y_t | x, y_{<t})

        This is the quantity used in the REINFORCE policy gradient:
            ∇_θ log π_θ(y | x) * R

        and in the KL divergence computation:
            D_KL(π_θ || π_ref) ≈ sum_t [log π_θ(y_t | ...) - log π_ref(y_t | ...)]

        CRITICAL: Only response tokens contribute to the sum. Prompt tokens
        are masked out via labels=-100. This is enforced by careful alignment
        of prompt/response boundaries in the padded batch.

        Args:
            prompts: List of prompt strings. Length N.
            responses: List of response strings. Length N. Must be in the
                same order as prompts (prompts[i] corresponds to responses[i]).

        Returns:
            1D tensor of shape (N,) containing the summed log-probability
            of each response given its prompt. Values are negative (log-probs
            are always <= 0). Returned on the same device as the model.

        Raises:
            ValueError: If len(prompts) != len(responses).
        """
        if len(prompts) != len(responses):
            raise ValueError(
                f"compute_log_probs: len(prompts)={len(prompts)} != "
                f"len(responses)={len(responses)}. "
                "Prompts and responses must be paired."
            )

        if not prompts:
            return torch.zeros(0, device=self.device)

        batch_size: int = len(prompts)

        # ------------------------------------------------------------------
        # Step 1: Tokenize prompts and responses separately (no padding)
        # to get exact token counts per sample.
        # This is the key to correctly identifying prompt/response boundaries
        # in the padded full-sequence batch.
        # ------------------------------------------------------------------
        # Tokenize each prompt individually to get exact (unpadded) lengths
        prompt_encodings = self.tokenizer(
            prompts,
            add_special_tokens=True,
            return_tensors=None,  # Returns list of lists
            padding=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length
            if hasattr(self.tokenizer, "model_max_length")
            and self.tokenizer.model_max_length is not None
            and self.tokenizer.model_max_length < 1_000_000
            else 4096,
        )
        prompt_ids_list: List[List[int]] = prompt_encodings["input_ids"]

        # Tokenize each response individually to get exact (unpadded) lengths
        # add_special_tokens=False: the BOS token is already in the prompt;
        # we don't want a second BOS at the start of the response.
        response_encodings = self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=self.tokenizer.model_max_length
            if hasattr(self.tokenizer, "model_max_length")
            and self.tokenizer.model_max_length is not None
            and self.tokenizer.model_max_length < 1_000_000
            else 4096,
        )
        response_ids_list: List[List[int]] = response_encodings["input_ids"]

        # ------------------------------------------------------------------
        # Step 2: Concatenate prompt + response token ids for each sample.
        # Build labels: -100 for prompt positions, actual token ids for
        # response positions.
        # ------------------------------------------------------------------
        full_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        for i in range(batch_size):
            p_ids: List[int] = prompt_ids_list[i]
            r_ids: List[int] = response_ids_list[i]

            # Full sequence: [prompt_tokens, response_tokens]
            full_ids: List[int] = p_ids + r_ids
            full_ids_list.append(full_ids)

            # Labels: mask prompt with -100, keep response token ids
            labels: List[int] = (
                [_IGNORE_INDEX] * len(p_ids) + r_ids
            )
            labels_list.append(labels)

        # ------------------------------------------------------------------
        # Step 3: Pad the batch of full sequences to the same length.
        # Use right-padding for log-prob computation (unlike generation
        # which uses left-padding). Right-padding is simpler for log-prob
        # computation since we're not generating — we just need the
        # attention mask to be correct.
        # ------------------------------------------------------------------
        max_len: int = max(len(ids) for ids in full_ids_list)

        # Pad token id for padding positions
        pad_id: int = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        padded_input_ids: List[List[int]] = []
        padded_labels: List[List[int]] = []
        attention_masks: List[List[int]] = []

        for i in range(batch_size):
            seq_len: int = len(full_ids_list[i])
            pad_len: int = max_len - seq_len

            # Right-pad input_ids with pad_id
            padded_input_ids.append(
                full_ids_list[i] + [pad_id] * pad_len
            )
            # Right-pad labels with -100 (padding positions are ignored)
            padded_labels.append(
                labels_list[i] + [_IGNORE_INDEX] * pad_len
            )
            # Attention mask: 1 for real tokens, 0 for padding
            attention_masks.append(
                [1] * seq_len + [0] * pad_len
            )

        # Convert to tensors and move to model device
        input_ids_tensor: torch.Tensor = torch.tensor(
            padded_input_ids, dtype=torch.long, device=self.device
        )
        labels_tensor: torch.Tensor = torch.tensor(
            padded_labels, dtype=torch.long, device=self.device
        )
        attention_mask_tensor: torch.Tensor = torch.tensor(
            attention_masks, dtype=torch.long, device=self.device
        )

        # ------------------------------------------------------------------
        # Step 4: Compute per-token log-probs via forward pass
        # ------------------------------------------------------------------
        per_token_log_probs: torch.Tensor = self.get_token_log_probs(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            labels=labels_tensor,
        )
        # per_token_log_probs: shape (batch_size, seq_len - 1)
        # Values are log-probs at response positions, 0.0 at prompt/pad positions

        # ------------------------------------------------------------------
        # Step 5: Sum per-token log-probs over response tokens per sample.
        # The sum gives log P(response | prompt) for each sample.
        # ------------------------------------------------------------------
        # Sum over the sequence dimension (dim=1)
        summed_log_probs: torch.Tensor = per_token_log_probs.sum(dim=1)
        # Shape: (batch_size,)

        logger.debug(
            "compute_log_probs(): batch_size=%d, max_seq_len=%d, "
            "mean_log_prob=%.4f.",
            batch_size,
            max_len,
            summed_log_probs.mean().item(),
        )

        return summed_log_probs

    def get_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities for response positions.

        Runs a forward pass through the model and extracts the log-
        probability of each actual next token at positions where
        labels != -100 (i.e., response positions).

        The autoregressive shift: the logit at position t predicts the
        token at position t+1. So we shift logits and labels by 1:
            shift_logits = logits[:, :-1, :]   # predicts positions 1..T
            shift_labels = labels[:, 1:]        # actual tokens at 1..T

        Args:
            input_ids: Token ids of shape (batch_size, seq_len).
            attention_mask: Binary mask of shape (batch_size, seq_len).
                1 for real tokens, 0 for padding.
            labels: Token ids of shape (batch_size, seq_len) where prompt
                and padding positions are set to -100 (IGNORE_INDEX) and
                response positions contain the actual token ids.

        Returns:
            Tensor of shape (batch_size, seq_len - 1) containing:
                - The log-probability of the actual next token at response
                  positions (where shifted labels != -100).
                - 0.0 at prompt and padding positions (where shifted
                  labels == -100).
            Gradients flow through this tensor during training.
        """
        # ------------------------------------------------------------------
        # Forward pass through the model
        # ------------------------------------------------------------------
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # logits: shape (batch_size, seq_len, vocab_size)
        logits: torch.Tensor = outputs.logits

        # ------------------------------------------------------------------
        # Autoregressive shift:
        # logits[:, t, :] predicts the token at position t+1
        # So shift_logits[:, t, :] predicts shift_labels[:, t]
        # ------------------------------------------------------------------
        shift_logits: torch.Tensor = logits[:, :-1, :].contiguous()
        # shape: (batch_size, seq_len - 1, vocab_size)

        shift_labels: torch.Tensor = labels[:, 1:].contiguous()
        # shape: (batch_size, seq_len - 1)

        # ------------------------------------------------------------------
        # Compute log-softmax over vocabulary dimension
        # ------------------------------------------------------------------
        # log_probs: shape (batch_size, seq_len - 1, vocab_size)
        log_probs: torch.Tensor = F.log_softmax(shift_logits, dim=-1)

        # ------------------------------------------------------------------
        # Gather the log-prob of the actual next token at each position
        # ------------------------------------------------------------------
        # Create a mask for response positions (where labels != -100)
        response_mask: torch.Tensor = (shift_labels != _IGNORE_INDEX)
        # shape: (batch_size, seq_len - 1), dtype: bool

        # Replace -100 with 0 for safe indexing (we'll zero out these
        # positions after gathering)
        safe_labels: torch.Tensor = shift_labels.clone()
        safe_labels[~response_mask] = 0

        # Gather log-probs at the actual token positions
        # gathered: shape (batch_size, seq_len - 1)
        gathered_log_probs: torch.Tensor = log_probs.gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)

        # Zero out prompt and padding positions (where labels == -100)
        # These positions should not contribute to the sum in compute_log_probs
        per_token_log_probs: torch.Tensor = gathered_log_probs * response_mask.float()

        return per_token_log_probs

    def save_checkpoint(self, path: str) -> None:
        """Save the model and tokenizer to disk.

        For LoRA models: saves only the adapter weights (much smaller than
        the full model). The base model weights are not saved since they
        are unchanged from the HuggingFace hub version.

        For full fine-tuning (use_lora=False): saves the full model weights.

        Args:
            path: Directory path to save the checkpoint. Created if it
                does not exist.
        """
        os.makedirs(path, exist_ok=True)

        # Save model (PEFT-aware: saves adapter weights for LoRA models)
        self.model.save_pretrained(path)

        # Save tokenizer
        self.tokenizer.save_pretrained(path)

        logger.info(
            "save_checkpoint(): Saved model and tokenizer to '%s'.", path
        )

    def load_checkpoint(self, path: str) -> None:
        """Load a previously saved checkpoint.

        For LoRA models: loads adapter weights on top of the already-loaded
        base model using PeftModel.from_pretrained.

        For full fine-tuning: loads the full model state dict.

        Args:
            path: Directory path containing the saved checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint directory does not exist.
            ImportError: If the checkpoint is a PEFT checkpoint but peft
                is not installed.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"load_checkpoint(): Checkpoint directory '{path}' does not "
                "exist. Ensure the path is correct."
            )

        # Check if this is a PEFT/LoRA checkpoint by looking for adapter_config.json
        adapter_config_path: str = os.path.join(path, "adapter_config.json")
        is_peft_checkpoint: bool = os.path.isfile(adapter_config_path)

        if is_peft_checkpoint:
            if not _PEFT_AVAILABLE:
                raise ImportError(
                    f"Checkpoint at '{path}' is a PEFT checkpoint "
                    "(adapter_config.json found), but peft is not installed. "
                    "Install peft==0.12.0: pip install peft==0.12.0"
                )
            # Load LoRA adapter weights on top of the base model
            self.model = PeftModel.from_pretrained(self.model, path)
            logger.info(
                "load_checkpoint(): Loaded PEFT adapter weights from '%s'.",
                path,
            )
        else:
            # Full model checkpoint — load state dict
            # Try to find the model file
            model_file: str = os.path.join(path, "pytorch_model.bin")
            if not os.path.isfile(model_file):
                # Try safetensors format
                model_file = os.path.join(path, "model.safetensors")

            if os.path.isfile(model_file):
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(
                    "load_checkpoint(): Loaded full model weights from '%s'.",
                    model_file,
                )
            else:
                # Fall back to from_pretrained for full model checkpoints
                # saved via save_pretrained
                self.model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=_DTYPE_MAP.get(
                        self.config.torch_dtype, torch.bfloat16
                    ),
                    device_map=self.config.device_map,
                )
                logger.info(
                    "load_checkpoint(): Loaded full model from '%s' via "
                    "from_pretrained.",
                    path,
                )

        # Reload tokenizer from checkpoint path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self._setup_tokenizer_padding()

        logger.info(
            "load_checkpoint(): Checkpoint loaded successfully from '%s'.",
            path,
        )

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _load_model_and_tokenizer(self) -> None:
        """Load the model and tokenizer from HuggingFace hub.

        Reads model_name from config.model_name (already resolved to the
        task-appropriate model by Config.from_dict). Loads with bfloat16
        dtype and device_map='auto' for multi-GPU support.

        Sets up the tokenizer with a pad token (required for batch
        generation) and left-padding (required for correct batch generation).

        Raises:
            RuntimeError: If the model or tokenizer cannot be loaded.
        """
        model_name: str = self.config.model_name

        logger.info(
            "_load_model_and_tokenizer(): Loading model '%s' with "
            "dtype='%s', device_map='%s'.",
            model_name,
            self.config.torch_dtype,
            self.config.device_map,
        )

        # ------------------------------------------------------------------
        # Load tokenizer
        # ------------------------------------------------------------------
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer for model '{model_name}'. "
                "Ensure the model identifier is correct and you have "
                "internet access. Original error: " + str(exc)
            ) from exc

        # Set up padding token and padding side
        self._setup_tokenizer_padding()

        # ------------------------------------------------------------------
        # Resolve torch dtype
        # ------------------------------------------------------------------
        dtype_str: str = self.config.torch_dtype
        torch_dtype = _DTYPE_MAP.get(dtype_str, torch.bfloat16)

        # ------------------------------------------------------------------
        # Load model
        # ------------------------------------------------------------------
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=self.config.device_map,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{model_name}'. "
                "Ensure the model identifier is correct, you have internet "
                "access, and sufficient GPU memory. "
                "Original error: " + str(exc)
            ) from exc

        # ------------------------------------------------------------------
        # Resolve the primary device from the model's parameter placement
        # ------------------------------------------------------------------
        try:
            first_param: torch.nn.Parameter = next(self.model.parameters())
            self.device = first_param.device
        except StopIteration:
            # Model has no parameters (shouldn't happen, but be defensive)
            self.device = torch.device("cpu")
            logger.warning(
                "_load_model_and_tokenizer(): Model has no parameters. "
                "Defaulting device to CPU."
            )

        # ------------------------------------------------------------------
        # Resize token embeddings if vocabulary sizes differ
        # (rare but defensive — handles tokenizers with added special tokens)
        # ------------------------------------------------------------------
        model_vocab_size: int = self.model.get_input_embeddings().weight.shape[0]
        tokenizer_vocab_size: int = len(self.tokenizer)

        if model_vocab_size != tokenizer_vocab_size:
            logger.warning(
                "_load_model_and_tokenizer(): Model vocab size (%d) != "
                "tokenizer vocab size (%d). Resizing model embeddings.",
                model_vocab_size,
                tokenizer_vocab_size,
            )
            self.model.resize_token_embeddings(tokenizer_vocab_size)

        logger.info(
            "_load_model_and_tokenizer(): Successfully loaded model '%s'. "
            "Primary device: %s. Vocab size: %d.",
            model_name,
            self.device,
            tokenizer_vocab_size,
        )

    def _setup_tokenizer_padding(self) -> None:
        """Configure tokenizer padding token and padding side.

        Many causal LMs (Llama, Qwen, DeepSeek)