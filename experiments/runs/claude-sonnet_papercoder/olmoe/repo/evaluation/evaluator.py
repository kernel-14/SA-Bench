## Code: evaluation/evaluator.py

```python
## evaluation/evaluator.py
"""In-loop pretraining evaluator for OLMoE-1B-7B.

Provides downstream task evaluation during pretraining, called every
eval_every_steps=1000 steps from training/trainer.py. Implements the
evaluation setup described in Appendix C (Table 11) of the paper.

Supported tasks and their settings (from config.yaml evaluation.pretraining_tasks):
  - ARC-Challenge: CF, 0-shot, char normalization, validation split
  - ARC-Easy: CF, 0-shot, no normalization, validation split
  - BoolQ: CF, 0-shot, no normalization, validation split
  - COPA: CF, 0-shot, no normalization, validation split
  - CommonsenseQA: CF, 0-shot, char normalization, validation split
  - HellaSwag: CF, 0-shot, char normalization, validation split
  - MMLU: MCF, 5-shot, no normalization, validation split
  - MMLU-Var: CF, 0-5 shot, char normalization, validation split
  - OpenBookQA: CF, 0-shot, char normalization, validation split
  - PIQA: CF, 0-shot, char normalization, validation split
  - SciQ: CF, 0-shot, no normalization, validation split
  - SocialIQA: CF, 0-shot, char normalization, validation split
  - Winogrande: CF, 0-shot, no normalization, validation split

Also supports perplexity evaluation on Paloma validation sets (Figure 24).

Configuration values used (from config.yaml):
  evaluation.pretraining_tasks: task configs with format/shots/norm/split
  model.max_seq_len: 4096  (truncation limit)
  model.vocab_size: 50304  (GPT-NeoX tokenizer)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.distributed import DistributedUtils
from utils.logging_utils import get_logger

logger: logging.Logger = get_logger("olmoe.evaluator")

# ---------------------------------------------------------------------------
# Optional HuggingFace datasets import.
# ---------------------------------------------------------------------------
try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
    DATASETS_AVAILABLE: bool = True
except ImportError:
    DATASETS_AVAILABLE = False
    HFDataset = None  # type: ignore[assignment,misc]
    load_dataset = None  # type: ignore[assignment]
    logger.warning(
        "HuggingFace 'datasets' library not available. "
        "Evaluation will be limited. Install with: pip install datasets"
    )

# ---------------------------------------------------------------------------
# Optional tqdm import for progress bars.
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
    TQDM_AVAILABLE: bool = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Task configuration map sourced directly from config.yaml
# (evaluation.pretraining_tasks section).
# ---------------------------------------------------------------------------
TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "arc_challenge": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "char",
        "split": "validation",
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
    },
    "arc_easy": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "none",
        "split": "validation",
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Easy",
    },
    "boolq": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "none",
        "split": "validation",
        "hf_path": "google/boolq",
        "hf_name": None,
    },
    "copa": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "none",
        "split": "validation",
        "hf_path": "super_glue",
        "hf_name": "copa",
    },
    "commonsenseqa": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "char",
        "split": "validation",
        "hf_path": "tau/commonsense_qa",
        "hf_name": None,
    },
    "hellaswag": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "char",
        "split": "validation",
        "hf_path": "Rowan/hellaswag",
        "hf_name": None,
    },
    "mmlu": {
        "format": "MCF",
        "num_shots": 5,
        "normalization": "none",
        "split": "validation",
        "hf_path": "cais/mmlu",
        "hf_name": "all",
    },
    "mmlu_var": {
        "format": "CF",
        "num_shots": "0-5",
        "normalization": "char",
        "split": "validation",
        "hf_path": "cais/mmlu",
        "hf_name": "all",
    },
    "openbookqa": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "char",
        "split": "validation",
        "hf_path": "allenai/openbookqa",
        "hf_name": "main",
    },
    "piqa": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "char",
        "split": "validation",
        "hf_path": "ybisk/piqa",
        "hf_name": None,
    },
    "sciq": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "none",
        "split": "validation",
        "hf_path": "allenai/sciq",
        "hf_name": None,
    },
    "socialiqa": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "char",
        "split": "validation",
        "hf_path": "allenai/social_i_qa",
        "hf_name": None,
    },
    "winogrande": {
        "format": "CF",
        "num_shots": 0,
        "normalization": "none",
        "split": "validation",
        "hf_path": "allenai/winogrande",
        "hf_name": "winogrande_xl",
    },
}

# ---------------------------------------------------------------------------
# MMLU subject list (57 subjects).
# Used for iterating over all MMLU subjects during evaluation.
# ---------------------------------------------------------------------------
MMLU_SUBJECTS: List[str] = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

# ---------------------------------------------------------------------------
# Maximum sequence length for evaluation (from config.yaml: model.max_seq_len).
# ---------------------------------------------------------------------------
MAX_SEQ_LEN: int = 4096

# ---------------------------------------------------------------------------
# Maximum number of examples to evaluate per task during in-loop evaluation.
# Using a subset keeps evaluation fast enough to not interrupt training.
# ---------------------------------------------------------------------------
MAX_EVAL_EXAMPLES: int = 500


class Evaluator:
    """In-loop pretraining evaluator for OLMoE-1B-7B.

    Evaluates the model on downstream tasks during pretraining using the
    Completion/Cloze Formulation (CF) and Multiple-Choice Formulation (MCF)
    as described in Appendix C (Table 11) of the paper.

    Datasets are loaded lazily on first use and cached to avoid repeated
    downloads during training. All evaluation runs with torch.no_grad() and
    model.eval() to prevent gradient accumulation and ensure correct behavior
    of any normalization layers.

    Attributes:
        model: The OLMoEModel (possibly FSDP-wrapped) to evaluate.
        tokenizer: GPT-NeoX tokenizer (vocab_size=50304).
        device: CUDA device for tensor operations.
        _dataset_cache: Lazy-loaded dataset cache keyed by task name.

    Example:
        >>> evaluator = Evaluator(model, tokenizer, device=torch.device("cuda:0"))
        >>> results = evaluator.evaluate_pretraining(["hellaswag", "mmlu"])
        >>> results["hellaswag"]
        0.523
        >>> results["mmlu"]
        0.287
    """

    def __init__(
        self,
        model: Any,  # OLMoEModel or FSDP-wrapped OLMoEModel
        tokenizer: Any,  # PreTrainedTokenizer or PreTrainedTokenizerFast
        device: str = "cuda",
    ) -> None:
        """Initialize Evaluator.

        Args:
            model: The OLMoEModel (or FSDP-wrapped OLMoEModel) to evaluate.
                   Must already be on the correct device. The evaluator calls
                   model.eval() before evaluation and model.train() after.
            tokenizer: GPT-NeoX tokenizer (EleutherAI/gpt-neox-20b).
                       Used for tokenizing prompts and completions.
                       vocab_size must be 50304 (config.yaml: model.vocab_size).
            device: Device string for tensor operations. Default: "cuda".
                    Should match the device the model is on.
                    Examples: "cuda", "cuda:0", "cpu".
        """
        self.model: Any = model
        self.tokenizer: Any = tokenizer
        self.device: str = device

        # Lazy-loaded dataset cache: task_name -> loaded dataset object.
        # Populated on first call to _load_task_dataset().
        self._dataset_cache: Dict[str, Any] = {}

        logger.info(
            f"Evaluator initialized: device='{device}', "
            f"max_eval_examples={MAX_EVAL_EXAMPLES}, "
            f"max_seq_len={MAX_SEQ_LEN}"
        )

    def evaluate_pretraining(
        self,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate the model on downstream tasks during pretraining.

        Runs evaluation on all specified tasks using the settings from
        config.yaml (evaluation.pretraining_tasks). Sets model to eval mode
        before evaluation and restores train mode after.

        Args:
            tasks: List of task names to evaluate. If None, evaluates all
                   tasks in TASK_CONFIG. Task names must be keys in TASK_CONFIG.
                   Example: ["hellaswag", "mmlu", "arc_challenge"]

        Returns:
            Dict mapping task name to accuracy (float in [0, 1]).
            Example: {"hellaswag": 0.523, "mmlu": 0.287, "arc_challenge": 0.412}
            Returns empty dict if datasets library is not available.

        Note:
            This method is safe to call during distributed training. All ranks
            run evaluation on the same data. Results are identical across ranks
            since the model weights are synchronized by FSDP.
        """
        if not DATASETS_AVAILABLE:
            logger.warning(
                "HuggingFace 'datasets' library not available. "
                "Skipping pretraining evaluation."
            )
            return {}

        if tasks is None:
            tasks = list(TASK_CONFIG.keys())

        # Validate task names.
        unknown_tasks: List[str] = [t for t in tasks if t not in TASK_CONFIG]
        if unknown_tasks:
            logger.warning(
                f"Unknown tasks will be skipped: {unknown_tasks}. "
                f"Valid tasks: {list(TASK_CONFIG.keys())}"
            )
            tasks = [t for t in tasks if t in TASK_CONFIG]

        if not tasks:
            return {}

        results: Dict[str, float] = {}

        # Set model to eval mode for evaluation.
        # With FSDP, this correctly propagates to all shards.
        self.model.eval()

        try:
            with torch.no_grad():
                for task_name in tasks:
                    try:
                        logger.info(
                            f"Evaluating task: '{task_name}' "
                            f"(rank={DistributedUtils.get_rank()})"
                        )
                        task_cfg: Dict[str, Any] = TASK_CONFIG[task_name]

                        if task_name == "mmlu":
                            accuracy: float = self._evaluate_mmlu_mcf(
                                num_shots=task_cfg["num_shots"]
                            )
                        elif task_name == "mmlu_var":
                            accuracy = self._evaluate_mmlu_var()
                        else:
                            accuracy = self._evaluate_cf_task(
                                task_name=task_name,
                                normalization=task_cfg["normalization"],
                                num_shots=task_cfg["num_shots"],
                            )

                        results[task_name] = accuracy
                        logger.info(
                            f"Task '{task_name}': accuracy={accuracy:.4f} "
                            f"({accuracy * 100:.1f}%)"
                        )

                    except Exception as exc:
                        logger.warning(
                            f"Evaluation of task '{task_name}' failed: "
                            f"{type(exc).__name__}: {exc}. "
                            f"Skipping this task."
                        )
                        results[task_name] = 0.0

        finally:
            # Always restore train mode, even if evaluation fails.
            self.model.train()

        logger.info(
            f"Pretraining evaluation complete: "
            f"{len(results)} tasks, "
            f"avg_accuracy={sum(results.values()) / max(len(results), 1):.4f}"
        )

        return results

    def evaluate_perplexity(self, dataset: Any) -> float:
        """Compute perplexity on a validation dataset.

        Used for Paloma validation sets (Books, Reddit, Stack, C4) as shown
        in Figure 24 of the paper. Tokenizes the dataset into seq_len=4096
        chunks and computes perplexity as exp(mean cross-entropy loss).

        Args:
            dataset: A HuggingFace Dataset or any iterable of dicts with a
                     "text" field. Each item should contain a text document.

        Returns:
            Perplexity as a float. Lower is better.
            Returns float("inf") if evaluation fails.

        Example:
            >>> c4_val = load_dataset("allenai/c4", "en", split="validation[:1%]")
            >>> ppl = evaluator.evaluate_perplexity(c4_val)
            >>> ppl
            12.4
        """
        self.model.eval()

        total_loss: float = 0.0
        total_tokens: int = 0

        try:
            with torch.no_grad():
                # Tokenize and pack all documents into seq_len chunks.
                all_token_ids: List[int] = []

                # Collect all tokens from the dataset.
                for example in dataset:
                    text: str = ""
                    if isinstance(example, dict):
                        text = example.get("text", example.get("content", ""))
                    elif isinstance(example, str):
                        text = example

                    if not text:
                        continue

                    token_ids: List[int] = self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )
                    all_token_ids.extend(token_ids)
                    # Add EOS between documents.
                    if self.tokenizer.eos_token_id is not None:
                        all_token_ids.append(self.tokenizer.eos_token_id)

                if not all_token_ids:
                    logger.warning("No tokens found in dataset for perplexity evaluation.")
                    return float("inf")

                # Split into non-overlapping chunks of MAX_SEQ_LEN.
                num_chunks: int = len(all_token_ids) // MAX_SEQ_LEN
                if num_chunks == 0:
                    logger.warning(
                        f"Dataset has fewer than {MAX_SEQ_LEN} tokens. "
                        f"Cannot compute perplexity."
                    )
                    return float("inf")

                for chunk_idx in range(num_chunks):
                    start: int = chunk_idx * MAX_SEQ_LEN
                    end: int = start + MAX_SEQ_LEN
                    chunk_ids: List[int] = all_token_ids[start:end]

                    input_ids: Tensor = torch.tensor(
                        [chunk_ids],
                        dtype=torch.long,
                        device=self.device,
                    )
                    labels: Tensor = input_ids.clone()

                    # Forward pass with labels to get CE loss.
                    output = self.model(
                        input_ids=input_ids,
                        labels=labels,
                    )

                    if output.ce_loss is not None:
                        # ce_loss is mean over tokens in this chunk.
                        # Weight by number of tokens for correct averaging.
                        chunk_tokens: int = MAX_SEQ_LEN - 1  # shifted labels
                        total_loss += output.ce_loss.item() * chunk_tokens
                        total_tokens += chunk_tokens

        finally:
            self.model.train()

        if total_tokens == 0:
            return float("inf")

        mean_loss: float = total_loss / total_tokens
        perplexity: float = math.exp(mean_loss)

        logger.info(f"Perplexity: {perplexity:.2f} (mean_loss={mean_loss:.4f})")
        return perplexity

    def evaluate_adaptation(self, tasks: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate the model after adaptation (SFT/DPO).

        Runs the adaptation evaluation suite from Appendix C of the paper.
        This is a simplified version — full adaptation evaluation requires
        external tools (AlpacaEval, IFEval, XSTest) not implemented here.

        Tasks supported here (subset of full adaptation eval):
            - mmlu: 0-shot exact match
            - hellaswag: 0-shot CF char normalization

        For full adaptation evaluation (GSM8k CoT, BBH, HumanEval, AlpacaEval,
        XSTest, IFEval), use evaluation/olmes_eval.py.

        Args:
            tasks: List of task names. If None, uses ["mmlu", "hellaswag"].

        Returns:
            Dict mapping task name to accuracy (float in [0, 1]).
        """
        if tasks is None:
            tasks = ["mmlu", "hellaswag"]

        # Reuse pretraining evaluation for supported tasks.
        supported: List[str] = [t for t in tasks if t in TASK_CONFIG]
        return self.evaluate_pretraining(tasks=supported)

    # =========================================================================
    # Core Scoring Methods
    # =========================================================================

    def _score_completion(
        self,
        prompt: str,
        completion: str,
    ) -> float:
        """Compute the log-probability of a completion given a prompt.

        Implements the Completion/Cloze Formulation (CF) scoring used for
        most in-loop evaluation tasks. The score is the sum of log-probabilities
        of each completion token given the preceding context.

        Causal LM shift: At position t, the model predicts token t+1.
        To score completion token at position p in full_ids, we use
        log_probs[0, p-1, full_ids[p]].

        Args:
            prompt: The prompt string (question/context). May be empty for
                    unconditional scoring (used in PMI normalization).
            completion: The completion string to score. Must be non-empty.

        Returns:
            Sum of log-probabilities over completion tokens (float, <= 0).
            Returns float("-inf") if completion tokenizes to empty or if
            the combined sequence exceeds MAX_SEQ_LEN.

        Example:
            >>> score = evaluator._score_completion(
            ...     "Question: What is the capital of France?\nAnswer:",
            ...     " Paris"
            ... )
            >>> score  # negative float, higher (less negative) = more likely
            -2.34
        """
        if not completion:
            return float("-inf")

        # Tokenize prompt and completion separately to find the boundary.
        # add_special_tokens=False: no BOS/EOS added (raw token IDs).
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        ) if prompt else []

        completion_ids: List[int] = self.tokenizer.encode(
            completion,
            add_special_tokens=False,
        )

        if not completion_ids:
            return float("-inf")

        # Concatenate prompt + completion token IDs.
        full_ids: List[int] = prompt_ids + completion_ids

        # Truncate to MAX_SEQ_LEN if needed.
        if len(full_ids) > MAX_SEQ_LEN:
            # Truncate from the left (keep the end, which contains the completion).
            # This preserves the completion tokens at the cost of truncating the prompt.
            full_ids = full_ids[-MAX_SEQ_LEN:]
            # Recompute where the completion starts after truncation.
            completion_start: int = max(0, len(full_ids) - len(completion_ids))
        else:
            completion_start = len(prompt_ids)

        if completion_start >= len(full_ids):
            # Entire completion was truncated away.
            return float("-inf")

        # Build input tensor: (1, seq_len).
        input_ids: Tensor = torch.tensor(
            [full_ids],
            dtype=torch.long,
            device=self.device,
        )

        # Forward pass: get logits of shape (1, seq_len, vocab_size).
        output = self.model(input_ids=input_ids)
        logits: Tensor = output.logits  # (1, seq_len, vocab_size)

        # Compute log-probabilities over vocabulary.
        # Shape: (1, seq_len, vocab_size)
        log_probs: Tensor = F.log_softmax(logits.float(), dim=-1)

        # Sum log-probabilities for completion tokens.
        # Causal LM: logits at position t predict token t+1.
        # Completion token at position p in full_ids is predicted by logits at p-1.
        # Completion spans positions [completion_start, len(full_ids)-1].
        # We use logits at positions [completion_start-1, len(full_ids)-2].
        total_log_prob: float = 0.0

        for pos in range(completion_start, len(full_ids)):
            # Logit position for predicting full_ids[pos] is pos-1.
            logit_pos: int = pos - 1
            if logit_pos < 0:
                # No context to predict the first token — skip.
                continue
            token_id: int = full_ids[pos]
            token_log_prob: float = log_probs[0, logit_pos, token_id].item()
            total_log_prob += token_log_prob

        return total_log_prob

    def _score_mcf(
        self,
        prompt: str,
        choices: List[str],
    ) -> int:
        """Score Multiple-Choice Formulation (MCF) by predicting the answer label.

        Scores each answer label (A, B, C, D, etc.) by looking at the logit
        for that label token at the last position of the prompt. Returns the
        index of the highest-scoring label.

        Used for MMLU 5-shot evaluation where the model predicts the answer
        label directly rather than scoring full answer text.

        Args:
            prompt: The formatted prompt ending with "Answer:" or similar.
                    The model predicts the next token (the answer label).
            choices: List of answer label strings. For MMLU: ["A", "B", "C", "D"].
                     For BoolQ: ["Yes", "No"]. The method handles any number of choices.

        Returns:
            Index of the predicted answer (0-indexed). For MMLU with choices
            ["A", "B", "C", "D"], returns 0 for A, 1 for B, etc.

        Example:
            >>> idx = evaluator._score_mcf(
            ...     "Question: What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:",
            ...     ["A", "B", "C", "D"]
            ... )
            >>> idx  # 1 = "B" = 4
            1
        """
        if not choices:
            return 0

        # Tokenize the prompt.
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        # Truncate prompt to MAX_SEQ_LEN if needed.
        if len(prompt_ids) > MAX_SEQ_LEN:
            prompt_ids = prompt_ids[-MAX_SEQ_LEN:]

        input_ids: Tensor = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.device,
        )

        # Forward pass: get logits at the last position.
        output = self.model(input_ids=input_ids)
        logits: Tensor = output.logits  # (1, seq_len, vocab_size)

        # Get logits at the last position (predicts the next token = answer label).
        last_logits: Tensor = logits[0, -1, :]  # (vocab_size,)

        # Score each choice label.
        # GPT-NeoX tokenizer: " A" (with space prefix) is the correct token
        # for answer labels that follow a space or newline in the prompt.
        scores: List[float] = []
        for choice in choices:
            # Try space-prefixed version first (standard for GPT-NeoX).
            space_prefixed: str = " " + choice.strip()
            token_ids_with_space: List[int] = self.tokenizer.encode(
                space_prefixed,
                add_special_tokens=False,
            )

            # Also try without space prefix.
            token_ids_no_space: List[int] = self.tokenizer.encode(
                choice.strip(),
                add_special_tokens=False,
            )

            # Use the first token of the space-prefixed version if it's a single token,
            # otherwise fall back to the no-space version.
            if token_ids_with_space:
                token_id: int = token_ids_with_space[0]
            elif token_ids_no_space:
                token_id = token_ids_no_space[0]
            else:
                # Fallback: use a dummy score.
                scores.append(float("-inf"))
                continue

            score: float = last_logits[token_id].item()
            scores.append(score)

        # Return the index of the highest-scoring choice.
        if not scores:
            return 0
        return int(max(range(len(scores)), key=lambda i: scores[i]))

    # =========================================================================
    # Task-Specific Evaluation Methods
    # =========================================================================

    def _evaluate_cf_task(
        self,
        task_name: str,
        normalization: str = "none",
        num_shots: int = 0,
    ) -> float:
        """Evaluate a task using Completion/Cloze Formulation (CF).

        Handles all CF tasks: ARC-C, ARC-E, BoolQ, COPA, CommonsenseQA,
        HellaSwag, OpenBookQA, PIQA, SciQ, SocialIQA, Winogrande.

        For each example:
          1. Format the prompt and candidate completions
          2. Score each completion with _score_completion()
          3. Apply normalization (none or char)
          4. Predict the highest-scoring completion
          5. Compare to ground truth

        Args:
            task_name: Task name key in TASK_CONFIG.
            normalization: Normalization method — "none" or "char".
                           "char": divide score by len(completion) in characters.
                           "none": use raw log-probability sum.
            num_shots: Number of few-shot examples to prepend. Default: 0.
                       Currently only 0-shot is implemented for CF tasks.

        Returns:
            Accuracy as float in [0, 1].
        """
        task_cfg: Dict[str, Any] = TASK_CONFIG[task_name]
        dataset: Any = self._load_task_dataset(task_name)

        if dataset is None:
            logger.warning(f"Could not load dataset for task '{task_name}'. Returning 0.0.")
            return 0.0

        # Limit number of examples for speed during training.
        examples: List[Any] = list(dataset)
        if len(examples) > MAX_EVAL_EXAMPLES:
            # Use a fixed subset for reproducibility.
            examples = examples[:MAX_EVAL_EXAMPLES]

        correct: int = 0
        total: int = 0

        for example in examples:
            try:
                prompt: str
                completions: List[str]
                correct_idx: int

                # Format prompt and completions based on task.
                prompt, completions, correct_idx = self._format_cf_example(
                    task_name=task_name,
                    example=example,
                )

                if not completions:
                    continue

                # Score each completion.
                scores: List[float] = []
                for completion in completions:
                    raw_score: float = self._score_completion(prompt, completion)

                    # Apply normalization.
                    if normalization == "char" and len(completion) > 0:
                        normalized_score: float = raw_score / len(completion)
                    else:
                        normalized_score = raw_score

                    scores.append(normalized_score)

                # Predict the highest-scoring completion.
                if not scores or all(s == float("-inf") for s in scores):
                    continue

                predicted_idx: int = int(
                    max(range(len(scores)), key=lambda i: scores[i])