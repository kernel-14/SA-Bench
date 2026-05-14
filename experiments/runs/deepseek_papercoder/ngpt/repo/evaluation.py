"""
evaluation.py

Implements the ``Evaluator`` class for the NGPT reproduction project.  It computes
validation loss, runs downstream tasks (HellaSwag, PIQA, ARC‑Easy, ARC‑Challenge)
using the ``lm‑eval‑harness`` library, and performs a 5‑shot WMT14 French‑English
translation test with sacreBLEU.

All behaviour is driven by the global ``Config`` object.  The class works with
both the baseline GPT and the normalised Transformer (nGPT) without any
modification – normalisation is handled inside the model's forward pass.

References
----------
* lm-eval-harness (v0.4.1): https://github.com/EleutherAI/lm-evaluation-harness
* Radford et al. (2018): Language Models are Unsupervised Multitask Learners
  (GPT‑2) for the translation prompt format.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset  # Used for WMT14
import sacrebleu
from tqdm import tqdm

from config import Config
from model import GPTModel

# ---------------------------------------------------------------------------
# Optional import for lm‑eval‑harness (used for multiple‑choice tasks)
# ---------------------------------------------------------------------------
try:
    from lm_eval import simple_evaluate
    from lm_eval.api.model import LM
    _LM_AVAILABLE = True
except ImportError:
    simple_evaluate = None
    class LM:  # Dummy base class to avoid NameError
        pass
    _LM_AVAILABLE = False


# ===========================================================================
# Helper: create a causal attention mask of shape (1, 1, T, T)
# ===========================================================================

def _make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """
    Return a mask tensor where positions j > i are masked (with ``-inf``).

    Args:
        T: Sequence length.
        device: Target device.

    Returns:
        Tensor of shape ``(1, 1, T, T)``.
    """
    mask = torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


# ===========================================================================
# LM‑eval‑harness wrapper for our custom GPTModel
# ===========================================================================

class _ModelWrapper(LM):
    """
    Adapter that makes ``GPTModel`` compatible with the ``lm_eval`` library.

    It implements the ``loglikelihood`` method required by multiple‑choice
    tasks (HellaSwag, PIQA, ARC).  The model is expected to be in evaluation
    mode and living on the correct device.

    Parameters
    ----------
    model : GPTModel
        The language model.
    tokenizer : PreTrainedTokenizer
        The tokenizer (LLaMA‑2).
    device : torch.device
    max_length : int
        Maximum sequence length allowed by the model.
    batch_size : int
        Batch size used for evaluation (mostly informational).
    """

    def __init__(
        self,
        model: GPTModel,
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_length: int,
        batch_size: int,
    ):
        super().__init__()  # initialise LM base class
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.batch_size_per_gpu = batch_size

        # The lm_eval harness expects these attributes.
        self._tok_encode = self._tok_encode
        self._tok_decode = self._tok_decode
        # Disable caching (we will recompute logits each time)
        self.cache = None

    # ------------------------------------------------------------------
    # Tokenization helpers that the harness expects
    # ------------------------------------------------------------------
    def tok_encode(self, string: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    # ------------------------------------------------------------------
    # Required log‑likelihood method for multiple‑choice tasks
    # ------------------------------------------------------------------
    @torch.no_grad()
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Compute log‑probability of continuations given contexts.

        Each request is an object with ``.args = (context_str, continuation_str)``.

        Returns:
            List of ``(log_prob, is_greedy)`` tuples.
        """
        results = []
        for req in requests:
            context, continuation = req.args

            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)

            # Truncate context from the left if the total length exceeds max_length
            total_len = len(ctx_ids) + len(cont_ids)
            if total_len > self.max_length:
                excess = total_len - self.max_length
                ctx_ids = ctx_ids[excess:]

            full_ids = ctx_ids + cont_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)

            # Build causal mask (the model constructs its own mask inside forward,
            # but for safety we explicitly pass it here; the model will use it if
            # we adapt its interface – however, our GPTModel builds the mask from
            # its buffer, so we can simply call model(input_tensor).  The mask is
            # handled internally and does not need to be passed as argument.')
            # Actually, GPTModel.forward(idx, targets=None) does not accept mask;
            # it uses self.causal_mask[:T,:T].  So we can just call:
            logits, _ = self.model(input_tensor, targets=None)  # logits: (1, T, vocab)

            # The logits at position i predict token i+1.
            # For continuation tokens, use positions: ctx_len-1 .. total_len-2
            ctx_len = len(ctx_ids)
            cont_len = len(cont_ids)
            if cont_len == 0:
                results.append((0.0, True))
                continue

            # Gather the predictions for continuation tokens
            pred_logits = logits[0, ctx_len - 1 : total_len - 1, :]  # shape (cont_len, vocab)
            labels = torch.tensor(cont_ids, device=self.device)

            # Log‑softmax and extract the correct token probability
            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs[range(cont_len), labels]
            total_log_prob = token_log_probs.sum().item()

            # Check if the argmax decision at every step matches the true token
            greedy_tokens = pred_logits.argmax(dim=-1)
            is_greedy = bool((greedy_tokens == labels).all().item())

            results.append((total_log_prob, is_greedy))

        return results

    # We do not use generation for our selected tasks, so we can leave it unimplemented.
    def generate_until(self, requests):
        raise NotImplementedError("generate_until is not used in the evaluation pipeline.")


# ===========================================================================
# Evaluator class
# ===========================================================================

class Evaluator:
    """
    Evaluation suite for the language model.

    Parameters
    ----------
    model : GPTModel
        The trained model (should already be on the correct device).
    config : Config
        Global configuration containing evaluation settings.
    """

    def __init__(self, model: GPTModel, config: Config):
        self.model = model
        self.config = config

        # Tokenizer (LLaMA‑2, consistent with training)
        self.tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)

        # Determine device from model parameters
        self.device = next(model.parameters()).device

        # Put model into evaluation mode (dropout off, etc.)
        model.eval()

        # Useful token ids
        self.eos_token_id = self.tokenizer.eos_token_id
        self.newline_token_id = self.tokenizer.encode("\n")[0]

    # ------------------------------------------------------------------
    # Public API (matches the design specification)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_downstream(self) -> Dict[str, float]:
        """
        Run all downstream tasks and return a dictionary of metrics.

        The returned dictionary contains:
          - hellaswag_acc
          - piqa_acc
          - arc_easy_acc_norm
          - arc_challenge_acc_norm
          - wmt14_bleu
          - average (of the five values)

        Returns
        -------
        metrics : dict
            Metric name → value.
        """
        harness_scores = self._evaluate_harness_tasks()
        wmt_bleu = self.compute_wmt14_bleu()

        # Combine results
        results = {**harness_scores, "wmt14_bleu": wmt_bleu}

        # Compute average accuracy / BLEU (treat BLEU as percentage)
        avg = sum(results.values()) / len(results)
        results["average"] = avg
        return results

    def compute_wmt14_bleu(self) -> float:
        """
        Evaluate 5‑shot French‑English translation on WMT14.

        Returns
        -------
        bleu : float
            sacreBLEU score (percent).
        """
        return self._evaluate_wmt14()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_val_loss(self, val_loader: DataLoader) -> float:
        """
        Compute average cross‑entropy loss over a validation DataLoader.

        Args:
            val_loader: DataLoader yielding (x, y) batches.

        Returns:
            Average loss per token.
        """
        total_loss = 0.0
        total_tokens = 0
        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            # The model computes the loss inside forward().
            _, loss = self.model(x, targets=y)
            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
        return total_loss / total_tokens if total_tokens > 0 else float("inf")

    # ------------------------------------------------------------------
    # lm‑eval‑harness integration
    # ------------------------------------------------------------------
    def _evaluate_harness_tasks(self) -> Dict[str, float]:
        """
        Run HellaSwag, PIQA, ARC‑Easy, ARC‑Challenge via lm‑eval‑harness.

        Returns
        -------
        scores : dict
            ``{task_name: score}``.  The keys are chosen to match the paper's
            downstream tasks.
        """
        if not _LM_AVAILABLE:
            raise RuntimeError(
                "lm_eval is required for downstream evaluation. "
                "Install with `pip install lm_eval==0.4.1`"
            )

        # Create the wrapper
        wrapper = _ModelWrapper(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=self.config.model.max_seq_len,
            batch_size=self.config.training.batch_size,  # can be reduced if OOM
        )

        # Task list as used in the paper (zero‑shot)
        tasks = ["hellaswag", "piqa", "arc_easy", "arc_challenge"]

        # Run the evaluation
        results = simple_evaluate(
            model=wrapper,
            tasks=tasks,
            num_fewshot=0,
            batch_size=None,          # let the harness auto‑batch
            device=str(self.device),
        )

        # Extract the relevant metrics.
        # For HellaSwag and PIQA we use accuracy ("acc").
        # For ARC tasks we use normalised accuracy ("acc_norm").
        scores = {}
        metric_map = {
            "hellaswag": ("acc", "hellaswag_acc"),
            "piqa": ("acc", "piqa_acc"),
            "arc_easy": ("acc_norm", "arc_easy_acc_norm"),
            "arc_challenge": ("acc_norm", "arc_challenge_acc_norm"),
        }

        for task_name, (metric_key, output_key) in metric_map.items():
            task_results = results.get(task_name, {})
            # simple_evaluate may return a dict with filter key or direct metric
            if metric_key in task_results:
                score = task_results[metric_key]
            elif "acc" in task_results:
                score = task_results["acc"]  # fallback
            else:
                score = float("nan")
            scores[output_key] = score

        return scores

    # ------------------------------------------------------------------
    # WMT14 French‑English 5‑shot translation
    # ------------------------------------------------------------------

    def _evaluate_wmt14(self) -> float:
        """
        Core routine for the WMT14 5‑shot translation test.

        Returns
        -------
        bleu_score : float
            sacreBLEU score (percentage).
        """
        # Load WMT14 dataset (French—English)
        wmt14 = load_dataset("wmt14", "fr-en", trust_remote_code=True)
        train_data = wmt14["train"]
        test_data = wmt14["test"]

        # Randomly select 5 distinct few‑shot examples (fixed seed for reproducibility)
        rng = random.Random(42)
        few_shot_indices = rng.sample(range(len(train_data)), 5)
        few_shot_pairs = [
            (train_data[i]["translation"]["fr"], train_data[i]["translation"]["en"])
            for i in few_shot_indices
        ]

        # Maximum total tokens allowed for the prompt (including generation)
        max_seq_len = self.config.model.max_seq_len
        max_generation_len = 100  # as in GPT‑2 evaluation

        hypotheses = []
        references = []

        # Process each test sentence
        for example in tqdm(test_data, desc="WMT14 translation"):
            src_fr = example["translation"]["fr"]
            ref_en = example["translation"]["en"]

            # Build prompt: few‑shot pairs + test source
            prompt = ""
            for fr, en in few_shot_pairs:
                prompt += f"French: {fr}\nEnglish: {en}\n"
            prompt += f"French: {src_fr}\nEnglish:"

            # Tokenize the prompt (without adding special tokens)
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

            # If the prompt is too long, truncate the few‑shot part from the left
            # until it fits (reserving space for the generation).
            while len(prompt_ids) + max_generation_len > max_seq_len and len(few_shot_pairs) > 0:
                # Remove the oldest few‑shot pair and rebuild
                few_shot_pairs.pop(0)
                prompt = ""
                for fr, en in few_shot_pairs:
                    prompt += f"French: {fr}\nEnglish: {en}\n"
                prompt += f"French: {src_fr}\nEnglish:"
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

            # If still too long, truncate the prompt IDs from the left
            if len(prompt_ids) > max_seq_len:
                prompt_ids = prompt_ids[-max_seq_len:]

            # Convert to tensor
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

            # Autoregressive generation
            generated_ids = self._generate(
                prefix_ids=input_ids,
                max_new_tokens=max_generation_len,
                stop_token_ids=[self.eos_token_id, self.newline_token_id],
            )

            # Decode the full sequence
            full_text = self.tokenizer.decode(generated_ids[0].tolist())

            # Extract the translation: everything after the last "English:" until the
            # end of the string (or the next newline – the generation stops at newline).
            marker = "English:"
            idx = full_text.rfind(marker)
            if idx == -1:
                # Fallback: use the entire generated text (should not happen)
                translation = full_text
            else:
                translation = full_text[idx + len(marker):].strip()
                # Remove any trailing newline that might be the stop token
                if translation.endswith("\n"):
                    translation = translation[:-1]

            hypotheses.append(translation)
            references.append(ref_en)

        # Compute sacreBLEU
        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="13a", lowercase=True)
        return bleu.score

    @torch.no_grad()
    def _generate(
        self,
        prefix_ids: torch.Tensor,
        max_new_tokens: int,
        stop_token_ids: List[int],
    ) -> torch.Tensor:
        """
        Generate tokens greedily starting from a prefix.

        Args:
            prefix_ids: Tensor of shape ``(1, seq_len)``.
            max_new_tokens: Maximum number of tokens to append.
            stop_token_ids: Generation stops if any of these ids is predicted.

        Returns:
            Full sequence tensor (prefix + generated), shape ``(1, new_len)``.
        """
        generated = prefix_ids.clone()
        model = self.model
        model.eval()

        for _ in range(max_new_tokens):
            # Forward pass: the model internally builds the causal mask up to current length
            logits, _ = model(generated, targets=None)
            next_logits = logits[0, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            if next_token.item() in stop_token_ids:
                break
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # Safety: if the sequence exceeds the model's maximum config length, stop
            if generated.size(1) >= self.config.model.max_seq_len:
                break

        return generated

