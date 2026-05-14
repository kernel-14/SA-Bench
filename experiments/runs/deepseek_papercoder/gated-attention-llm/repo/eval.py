## eval.py
"""
Evaluation module for the Gated Attention LLM reproduction.

Implements:
  - Perplexity computation on multi‑domain held‑out test sets.
  - Integration with `lm‑evaluation‑harness` for downstream benchmark scoring.
"""

from __future__ import annotations

import logging
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# lm_eval v0.4 imports
from lm_eval.api.model import LM
from lm_eval import simple_evaluate

# Local imports for type hints (avoid circular dependency)
from model import GPTModel

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Custom LM adapter for lm_eval
# ------------------------------------------------------------------

class GatedAttentionLM(LM):
    """
    Adapter that wraps a GPTModel instance for use with the `lm_eval` library.

    Implements token encoding/decoding, log‑likelihood computation, and
    greedy text generation, following the interface expected by `simple_evaluate`.

    Args:
        model: The trained GPTModel to evaluate.
        tokenizer: The tokenizer associated with the model.
        batch_size: Maximum number of sequences to process in a single call
                    (currently not used; one request at a time for simplicity).
        device: The device on which the model resides (e.g., 'cuda').
        max_length: Maximum sequence length allowed; longer inputs are truncated.
    """

    def __init__(
        self,
        model: GPTModel,
        tokenizer: AutoTokenizer,
        batch_size: int = 1,
        device: str = "cuda",
        max_length: int = 4096,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size        # not used currently
        self._device = device
        self._max_length = max_length

        # Ensure the tokenizer has a pad token; use eos if missing
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            else:
                # Add dummy pad token (should rarely happen)
                self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self._model.resize_token_embeddings(len(self._tokenizer))

        self._model.eval()
        self._model.to(device)

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device

    def tok_encode(self, string: str) -> List[int]:
        """
        Encode a text string into a list of token IDs.

        Args:
            string: Text to encode.

        Returns:
            List of integer token IDs.
        """
        # Some tasks expect `add_special_tokens=False` for proper continuation
        return self._tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back into a text string.

        Args:
            tokens: List of token IDs.
            skip_special_tokens: Whether to remove special tokens from the output.

        Returns:
            Decoded string.
        """
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Pass a batch of input IDs through the model and return logits.

        Args:
            inp: Long tensor of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
        """
        attention_mask = (inp != self._tokenizer.pad_token_id).long()
        with torch.no_grad():
            logits, _ = self._model(input_ids=inp, attention_mask=attention_mask, labels=None)
        return logits

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[torch.Tensor, torch.Tensor]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """
        Compute log‑likelihood of continuation tokens given context tokens.

        This is used for tasks like MMLU, Hellaswag, C‑Eval, CMMLU.

        Args:
            requests: List of (context_tokens, continuation_tokens) tuples,
                      both are 1D LongTensors.
            disable_tqdm: Whether to disable progress bar.

        Returns:
            List of (total_log_prob, is_greedy) pairs. `is_greedy` is True if
            the model's greedy prediction perfectly matches the continuation.
        """
        res = []
        for ctx, cont in requests:
            # Ensure the total length does not exceed max_length
            ctx = ctx.to(self.device)
            cont = cont.to(self.device)

            total_len = ctx.size(0) + cont.size(0)
            if total_len > self._max_length:
                # Truncate context on the left, preserving the continuation
                new_ctx_len = max(0, self._max_length - cont.size(0))
                ctx = ctx[-new_ctx_len:]

            # Build full input
            full_seq = torch.cat([ctx, cont], dim=0)  # (L,)
            inp = full_seq.unsqueeze(0)                # (1, L)
            logits = self._model_call(inp)             # (1, L, vocab_size)
            logits = logits[0]                         # (L, vocab_size)
            logprobs = torch.log_softmax(logits, dim=-1)

            # Compute log‑prob of each continuation token
            ctx_len = ctx.size(0)
            cont_logprobs = torch.zeros(cont.size(0), device=self.device)
            is_greedy_all = True
            for j in range(cont.size(0)):
                pos = ctx_len + j - 1   # model predicts token at pos+1
                # pos < 0 would only happen with empty context (not used here)
                if pos < 0:
                    cont_logprobs[j] = math.log(1.0 / logits.size(-1))
                    is_greedy_all = False
                    continue
                token_logprobs = logprobs[pos]
                target_id = cont[j].item()
                cont_logprobs[j] = token_logprobs[target_id].item()
                pred = token_logprobs.argmax().item()
                if pred != target_id:
                    is_greedy_all = False

            total_logprob = cont_logprobs.sum().item()
            res.append((total_logprob, is_greedy_all))

        return res

    def generate_until(self, requests: List[Dict]) -> List[str]:
        """
        Generate text until a stopping condition is met. Used for tasks like GSM8k, HumanEval.

        Args:
            requests: List of dicts, each containing:
                "text": str, the context/prompt.
                "until": List[str], stop sequences.
                "do_sample": bool (default False).
                "temperature": float (default 1.0).
                "max_new_tokens": int (default 256).
                ... other optional keys.

        Returns:
            List of generated strings (excluding the original context).
        """
        res = []
        for req in requests:
            context = req["text"]
            until = req.get("until", [])
            do_sample = req.get("do_sample", False)
            temperature = req.get("temperature", 1.0)
            max_new_tokens = req.get("max_new_tokens", 256)

            # Encode context
            input_ids = torch.tensor(
                self._tokenizer.encode(context, add_special_tokens=False),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)  # (1, L)

            generated = []
            for _ in range(max_new_tokens):
                attention_mask = (input_ids != self._tokenizer.pad_token_id).long()
                with torch.no_grad():
                    logits, _ = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                    )
                # Logits at the last position
                last_logits = logits[0, -1, :]   # (vocab_size,)
                if do_sample:
                    last_logits = last_logits / temperature
                    probs = torch.softmax(last_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_token = torch.argmax(last_logits, dim=-1, keepdim=True).squeeze(-1)

                # Stop if EOS
                if next_token.item() == self._tokenizer.eos_token_id:
                    break

                generated.append(next_token.item())
                input_ids = torch.cat(
                    [input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1
                )

                # Check stop strings
                generated_text = self._tokenizer.decode(
                    generated, skip_special_tokens=True
                )
                stop = False
                for seq in until:
                    if generated_text.endswith(seq):
                        stop = True
                        break
                if stop:
                    break

            completed = self._tokenizer.decode(generated, skip_special_tokens=True)
            res.append(completed)

        return res

    def loglikelihood_rolling(self, requests: List[Dict]) -> List[float]:
        """
        Loglikelihood of rolling windows (not used in our evaluation).
        """
        raise NotImplementedError(
            "Rolling loglikelihood not implemented for this model."
        )


# ------------------------------------------------------------------
# Evaluator class
# ------------------------------------------------------------------

class Evaluator:
    """
    Computes perplexity and downstream task metrics for a GPTModel.

    Args:
        model: The trained GPTModel instance.
        config: A dictionary representing the 'evaluation' section of the main config.
                Expected keys: tasks (list of str), fewshot_settings (dict mapping task->int),
                output_dir (str).
        tokenizer: HuggingFace tokenizer used for the model.
    """

    def __init__(
        self,
        model: GPTModel,
        config: Dict[str, Any],
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        self.output_dir = Path(config.get("output_dir", "./eval_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        self.device = next(self.model.parameters()).device

    def compute_perplexity(self, dataset: DataLoader) -> float:
        """
        Compute language modeling perplexity on a given dataloader.

        The dataloader is expected to yield batches with keys 'input_ids' and 'labels',
        where labels are the target sequence (shifted right). The loss is computed as
        mean cross‑entropy over all non‑padding tokens, and accumulated to compute
        overall perplexity over the entire dataset.

        Args:
            dataset: A PyTorch DataLoader yielding batches from the held‑out test set.

        Returns:
            The average perplexity (PPL) over all tokens.
        """
        total_nll = 0.0   # total negative log‑likelihood (sum)
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for batch in dataset:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Forward pass with labels
                _, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # Loss is the mean cross‑entropy over unmasked tokens (labels != -100)
                num_targets = (labels != -100).sum().item()
                if num_targets == 0:
                    continue

                total_nll += loss.item() * num_targets
                total_tokens += num_targets

        if total_tokens == 0:
            return float("inf")

        avg_nll = total_nll / total_tokens
        ppl = math.exp(avg_nll)
        return ppl

    def _run_lm_eval_tasks(self) -> Dict[str, float]:
        """
        Evaluate the model on a predefined set of few‑shot downstream tasks using
        the `lm_eval` library.

        The tasks and their few‑shot configurations are read from `self.config`.

        Returns:
            A dictionary mapping task name (as in the config) to its primary metric score.
            Example: {"hellaswag": 74.64, "mmlu": 60.82, ...}
        """
        tasks = self.config.get("tasks", [])
        fewshot_settings = self.config.get("fewshot_settings", {})

        if not tasks:
            logger.warning("No evaluation tasks specified in config.")
            return {}

        # Build the adapter that lm_eval will use
        lm = GatedAttentionLM(
            model=self.model,
            tokenizer=self.tokenizer,
            device=str(self.device),
            max_length=self.model.config.max_position_embeddings,
        )

        # Prepare per‑task few‑shot overrides
        num_fewshot = {}
        for task in tasks:
            if task in fewshot_settings:
                num_fewshot[task] = fewshot_settings[task]
            else:
                # default to 0 (some tasks require 0)
                num_fewshot[task] = 0

        # Run the evaluation via simple_evaluate
        results = simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=1,          # one request at a time (low memory)
            device=str(self.device),
            log_samples=False,     # do not store samples to save memory
        )

        # Extract primary metrics
        metric_names = {
            "hellaswag": "acc,none",
            "mmlu": "acc",
            "gsm8k": "exact_match,flexible-extract",
            "humaneval": "pass@1",
            "ceval": "acc",
            "cmmlu": "acc",
        }

        scores = {}
        for task in tasks:
            task_results = results.get("results", {}).get(task, {})
            metric_key = metric_names.get(task, None)
            if metric_key and metric_key in task_results:
                scores[task] = task_results[metric_key]
            else:
                # fallback: pick any metric that looks like an accuracy
                for key in task_results:
                    if "acc" in key and "stderr" not in key:
                        scores[task] = task_results[key]
                        break
                else:
                    logger.warning(
                        f"Could not find a suitable metric for task '{task}'. "
                        f"Available: {list(task_results.keys())}"
                    )

        # Log and save results
        with open(self.output_dir / "downstream_scores.json", "w") as f:
            json.dump(scores, f, indent=2)

        logger.info(f"Downstream evaluation scores: {scores}")
        return scores
