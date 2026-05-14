"""
LLM policy wrapper for SCoRe.

Provides:
- Text generation (sampling and greedy)
- Per-token log-probability computation for REINFORCE
- KL divergence computation against a reference policy
- Utilities for multi-turn rollout collection
"""

import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class LLMPolicy:
    """
    Wraps a HuggingFace causal LM to expose the interface needed by SCoRe:
    - generate(): sample or greedy decode a response
    - log_prob(): compute the sum of log-probabilities of a response given a prompt
    - kl_divergence(): compute KL(π_θ || π_ref) for a given (prompt, response) pair
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 1024,
        load_in_8bit: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {"torch_dtype": torch_dtype}
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        else:
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        if not load_in_8bit:
            self.model.eval()

    @contextmanager
    def _inference_mode(self):
        with torch.inference_mode():
            yield

    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """Generate a single response for a prompt."""
        if do_sample is None:
            do_sample = temperature > 0.0
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with self._inference_mode():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        do_sample = temperature > 0.0
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with self._inference_mode():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)

        prompt_lengths = inputs["input_ids"].shape[1]
        responses = []
        for i in range(len(prompts)):
            new_ids = output_ids[i][prompt_lengths:]
            responses.append(self.tokenizer.decode(new_ids, skip_special_tokens=True))
        return responses

    def _compute_log_prob_from_ids(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        reduction: str = "sum",
        no_grad: bool = False,
    ) -> torch.Tensor:
        """Shared implementation for log-prob computation."""
        response_len = full_ids.shape[1] - prompt_len
        if response_len <= 0:
            return torch.tensor(0.0, device=self.device)

        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            logits = self.model(full_ids).logits  # (1, seq_len, vocab)

        response_logits = logits[0, prompt_len - 1: -1, :]  # (response_len, vocab)
        response_token_ids = full_ids[0, prompt_len:]        # (response_len,)

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs[
            torch.arange(response_len, device=self.device), response_token_ids
        ]

        if reduction == "sum":
            return token_log_probs.sum()
        elif reduction == "mean":
            return token_log_probs.mean()
        else:
            return token_log_probs

    def log_prob(
        self,
        prompt: str,
        response: str,
        reduction: str = "sum",
        no_grad: bool = False,
    ) -> torch.Tensor:
        """
        Compute log P(response | prompt) under the current model.

        Args:
            prompt: the conditioning context
            response: the response whose log-prob to compute
            reduction: "sum" (default) or "mean" over response tokens
            no_grad: if True, disable gradient tracking (for reference policy)

        Returns:
            scalar tensor (with gradients unless no_grad=True)
        """
        full_text = prompt + response
        full_ids = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=4096
        ).input_ids.to(self.device)
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).input_ids.to(self.device)

        prompt_len = prompt_ids.shape[1]
        return self._compute_log_prob_from_ids(full_ids, prompt_len, reduction, no_grad=no_grad)

    def log_prob_batch(
        self,
        prompts: List[str],
        responses: List[str],
        reduction: str = "sum",
        no_grad: bool = False,
    ) -> torch.Tensor:
        """
        Compute log-probs for a batch of (prompt, response) pairs.

        Returns a tensor of shape (batch_size,).
        """
        results = []
        for prompt, response in zip(prompts, responses):
            lp = self.log_prob(prompt, response, reduction=reduction, no_grad=no_grad)
            results.append(lp)
        return torch.stack(results)

    def compute_token_log_probs(
        self,
        prompt: str,
        response: str,
        no_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token log-probs for a (prompt, response) pair.

        Returns:
            token_log_probs: (response_len,) tensor
            response_token_ids: (response_len,) tensor
        """
        full_text = prompt + response
        full_ids = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=4096
        ).input_ids.to(self.device)
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).input_ids.to(self.device)

        prompt_len = prompt_ids.shape[1]
        response_len = full_ids.shape[1] - prompt_len

        if response_len <= 0:
            empty = torch.zeros(0, device=self.device)
            return empty, empty.long()

        token_log_probs = self._compute_log_prob_from_ids(
            full_ids, prompt_len, reduction="none", no_grad=no_grad
        )
        response_token_ids = full_ids[0, prompt_len:]
        return token_log_probs, response_token_ids

    def kl_divergence_from_ref(
        self,
        prompt: str,
        response: str,
        ref_policy: "LLMPolicy",
    ) -> torch.Tensor:
        """
        Compute KL(π_θ(·|prompt) || π_ref(·|prompt)) approximated over the
        response tokens:

            KL ≈ Σ_t [log π_θ(y_t|context_t) - log π_ref(y_t|context_t)]

        This is the standard per-sample KL estimate used in RLHF (Equation 2).
        Gradients flow through the π_θ term only.
        """
        log_prob_theta = self.log_prob(prompt, response, reduction="sum", no_grad=False)
        log_prob_ref = ref_policy.log_prob(prompt, response, reduction="sum", no_grad=True)
        return log_prob_theta - log_prob_ref


class ReferencePolicy:
    """
    Frozen copy of the base model used as π_ref in KL penalty computation.

    Wraps an LLMPolicy but disables gradient computation and parameter updates.
    """

    def __init__(self, policy: LLMPolicy):
        self._policy = policy
        # Freeze all parameters
        for param in self._policy.model.parameters():
            param.requires_grad_(False)

    def log_prob(self, prompt: str, response: str, reduction: str = "sum", no_grad: bool = True) -> torch.Tensor:
        return self._policy.log_prob(prompt, response, reduction=reduction, no_grad=True)

    def compute_token_log_probs(
        self, prompt: str, response: str, no_grad: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._policy.compute_token_log_probs(prompt, response, no_grad=True)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ReferencePolicy":
        policy = LLMPolicy(
            model_name_or_path, device=device, torch_dtype=torch_dtype
        )
        return cls(policy)


def load_policy_and_ref(
    model_name_or_path: str,
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
) -> Tuple[LLMPolicy, ReferencePolicy]:
    """
    Load the trainable policy and a frozen reference policy from the same checkpoint.

    The reference policy shares the same weights at initialization but is never
    updated during training.
    """
    policy = LLMPolicy(
        model_name_or_path,
        device=device,
        torch_dtype=torch_dtype,
        load_in_8bit=load_in_8bit,
    )
    ref_policy = ReferencePolicy.from_pretrained(
        model_name_or_path, device=device, torch_dtype=torch_dtype
    )
    return policy, ref_policy
