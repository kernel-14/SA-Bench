```python
## evaluation/olmes_eval.py
"""OLMES-standard post-pretraining and post-adaptation evaluation for OLMoE-1B-7B.

Implements the evaluation protocol described in Appendix C (Table 11) of the paper:
  - After pretraining: OLMES standard with max(MCF, CF) formulation selection
  - After adaptation: MMLU 0-shot, GSM8k 8-shot CoT, BBH 3-shot, HumanEval Pass@10,
    AlpacaEval 1.0, XSTest F1, IFEval Loose Accuracy

OLMES evaluation settings (from config.yaml evaluation.olmes_tasks):
  - ARC-Challenge: max(MCF,CF), 5-shot, pmi normalization, test split
  - ARC-Easy: max(MCF,CF), 5-shot, char normalization, test split
  - BoolQ: max(MCF,CF), 5-shot, none normalization, validation split
  - CommonsenseQA: max(MCF,CF), 5-shot, pmi normalization, validation split
  - HellaSwag: max(MCF,CF), 5-shot, char normalization, validation split
  - MMLU: max(MCF,CF), 5-shot, char normalization, test split
  - OpenBookQA: max(MCF,CF), 5-shot, pmi normalization, test split
  - PIQA: max(MCF,CF), 5-shot, char normalization, validation split
  - SocialIQA: max(MCF,CF), 5-shot, char normalization, validation split
  - Winogrande: max(MCF,CF), 5-shot, none normalization, validation split

References:
  - Appendix C, Table 11: evaluation setup
  - Table 4: after-pretraining results
  - Table 5: after-adaptation results
  - Table 12: extended DCLM evaluation results
"""

import logging
import math
import os
import random
import re
import subprocess
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from utils.distributed import DistributedUtils
from utils.logging_utils import get_logger

logger: logging.Logger = get_logger("olmoe.olmes_eval")

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
        "OLMES evaluation will be disabled. Install with: pip install datasets"
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
# OLMES task configuration (from config.yaml evaluation.olmes_tasks and Table 11).
# Each entry encodes the exact evaluation settings from the paper.
# ---------------------------------------------------------------------------
OLMES_TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "arc_challenge": {
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Challenge",
        "train_split": "train",
        "eval_split": "test",
        "num_shots": 5,
        "normalization": "pmi",
        "question_key": "question",
        "choices_key": "choices",
        "answer_key": "answerKey",
        "task_type": "arc",
    },
    "arc_easy": {
        "hf_path": "allenai/ai2_arc",
        "hf_name": "ARC-Easy",
        "train_split": "train",
        "eval_split": "test",
        "num_shots": 5,
        "normalization": "char",
        "question_key": "question",
        "choices_key": "choices",
        "answer_key": "answerKey",
        "task_type": "arc",
    },
    "boolq": {
        "hf_path": "google/boolq",
        "hf_name": None,
        "train_split": "train",
        "eval_split": "validation",
        "num_shots": 5,
        "normalization": "none",
        "question_key": "question",
        "passage_key": "passage",
        "answer_key": "answer",
        "task_type": "boolq",
    },
    "commonsenseqa": {
        "hf_path": "tau/commonsense_qa",
        "hf_name": None,
        "train_split": "train",
        "eval_split": "validation",
        "num_shots": 5,
        "normalization": "pmi",
        "question_key": "question",
        "choices_key": "choices",
        "answer_key": "answerKey",
        "task_type": "commonsenseqa",
    },
    "hellaswag": {
        "hf_path": "Rowan/hellaswag",
        "hf_name": None,
        "train_split": "train",
        "eval_split": "validation",
        "num_shots": 5,
        "normalization": "char",
        "task_type": "hellaswag",
    },
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_name": "all",
        "train_split": "dev",
        "eval_split": "test",
        "num_shots": 5,
        "normalization": "char",
        "task_type": "mmlu",
    },
    "openbookqa": {
        "hf_path": "allenai/openbookqa",
        "hf_name": "main",
        "train_split": "train",
        "eval_split": "test",
        "num_shots": 5,
        "normalization": "pmi",
        "question_key": "question_stem",
        "choices_key": "choices",
        "answer_key": "answerKey",
        "task_type": "openbookqa",
    },
    "piqa": {
        "hf_path": "ybisk/piqa",
        "hf_name": None,
        "train_split": "train",
        "eval_split": "validation",
        "num_shots": 5,
        "normalization": "char",
        "task_type": "piqa",
    },
    "socialiqa": {
        "hf_path": "allenai/social_i_qa",
        "hf_name": None,
        "train_split": "train",
        "eval_split": "validation",
        "num_shots": 5,
        "normalization": "char",
        "task_type": "socialiqa",
    },
    "winogrande": {
        "hf_path": "allenai/winogrande",
        "hf_name": "winogrande_xl",
        "train_split": "train",
        "eval_split": "validation",
        "num_shots": 5,
        "normalization": "none",
        "task_type": "winogrande",
    },
}

# ---------------------------------------------------------------------------
# MMLU subject list (57 subjects) for per-subject evaluation.
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
# Maximum sequence length (from config.yaml: model.max_seq_len).
# ---------------------------------------------------------------------------
MAX_SEQ_LEN: int = 4096

# ---------------------------------------------------------------------------
# Maximum examples per task for in-loop evaluation (speed vs. accuracy tradeoff).
# Set to None to evaluate all examples.
# ---------------------------------------------------------------------------
MAX_EVAL_EXAMPLES: Optional[int] = None

# ---------------------------------------------------------------------------
# Fixed random seed for reproducible fewshot sampling.
# ---------------------------------------------------------------------------
FEWSHOT_SEED: int = 42

# ---------------------------------------------------------------------------
# GSM8k 8-shot CoT examples (fixed, from the original GSM8k paper).
# These are the standard 8-shot examples used in the paper's evaluation.
# ---------------------------------------------------------------------------
GSM8K_8SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. #### 29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. #### 8",
    },
]


class OLMESEvaluator:
    """OLMES-standard evaluator for OLMoE-1B-7B.

    Implements the evaluation protocol from Appendix C of the paper:
      - After pretraining: OLMES standard with max(MCF, CF) formulation selection,
        5-shot evaluation, and task-specific normalization (pmi, char, none)
      - After adaptation: MMLU 0-shot, GSM8k 8-shot CoT, BBH 3-shot,
        HumanEval Pass@10, AlpacaEval 1.0, XSTest F1, IFEval Loose Accuracy

    The evaluator runs with torch.no_grad() and model.eval() mode. It restores
    the model to train mode after evaluation.

    Attributes:
        model: The OLMoEModel (possibly FSDP-wrapped) to evaluate.
        tokenizer: GPT-NeoX tokenizer (vocab_size=50304).
        device: Device string for tensor operations.
        batch_size: Number of examples to process in parallel.
        _dataset_cache: Lazy-loaded dataset cache keyed by (path, name, split).
        _unconditional_cache: Cache for PMI unconditional log-probs.
        _fewshot_cache: Cache for sampled fewshot examples per task.

    Example:
        >>> evaluator = OLMESEvaluator(model, tokenizer, device="cuda:0")
        >>> results = evaluator.run_olmes(["arc_challenge", "hellaswag", "mmlu"])
        >>> results["arc_challenge"]
        0.621
        >>> results["average"]
        0.711
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        batch_size: int = 8,
    ) -> None:
        """Initialize OLMESEvaluator.

        Args:
            model: The OLMoEModel (or FSDP-wrapped OLMoEModel) to evaluate.
                   Must already be on the correct device. The evaluator calls
                   model.eval() before evaluation and model.train() after.
            tokenizer: GPT-NeoX tokenizer (EleutherAI/gpt-neox-20b).
                       vocab_size must be 50304 (config.yaml: model.vocab_size).
            device: Device string for tensor operations. Default: "cuda".
                    Should match the device the model is on.
            batch_size: Number of (prompt, completion) pairs to process in
                        parallel during scoring. Default: 8.
                        Reduce if OOM errors occur during evaluation.
        """
        self.model: Any = model
        self.tokenizer: Any = tokenizer
        self.device: str = device
        self.batch_size: int = batch_size

        # Lazy-loaded dataset cache: (hf_path, hf_name, split) -> dataset
        self._dataset_cache: Dict[Tuple[str, Optional[str], str], Any] = {}

        # PMI unconditional log-prob cache: completion_str -> float
        self._unconditional_cache: Dict[str, float] = {}

        # Fewshot examples cache: task_name -> List[Dict]
        self._fewshot_cache: Dict[str, List[Dict[str, Any]]] = {}

        # Ensure pad token is set for batched inference.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.debug(
                f"Set pad_token = eos_token = '{self.tokenizer.eos_token}' "
                f"(id={self.tokenizer.eos_token_id})"
            )

        logger.info(
            f"OLMESEvaluator initialized: device='{device}', "
            f"batch_size={batch_size}, "
            f"max_seq_len={MAX_SEQ_LEN}"
        )

    def run_olmes(
        self,
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Run OLMES-standard evaluation on specified tasks.

        Evaluates each task using both CF and MCF formulations and returns
        max(CF, MCF) as the final score per the OLMES protocol. Also computes
        the average across all evaluated tasks.

        Args:
            task_names: List of task names to evaluate. If None, evaluates all
                        10 OLMES tasks. Task names must be keys in OLMES_TASK_CONFIG.
                        Example: ["arc_challenge", "hellaswag", "mmlu"]

        Returns:
            Dict with the following structure:
                {
                    "arc_challenge": 0.621,       # max(MCF, CF)
                    "arc_challenge_cf": 0.615,    # CF score
                    "arc_challenge_mcf": 0.621,   # MCF score
                    ...
                    "average": 0.711,             # Average across all tasks
                }
            Returns empty dict if datasets library is not available.

        Note:
            All ranks run evaluation. Results are identical across ranks since
            model weights are synchronized by FSDP.
        """
        if not DATASETS_AVAILABLE:
            logger.warning(
                "HuggingFace 'datasets' library not available. "
                "Skipping OLMES evaluation."
            )
            return {}

        if task_names is None:
            task_names = list(OLMES_TASK_CONFIG.keys())

        # Validate task names.
        unknown: List[str] = [t for t in task_names if t not in OLMES_TASK_CONFIG]
        if unknown:
            logger.warning(
                f"Unknown OLMES tasks will be skipped: {unknown}. "
                f"Valid tasks: {list(OLMES_TASK_CONFIG.keys())}"
            )
            task_names = [t for t in task_names if t in OLMES_TASK_CONFIG]

        if not task_names:
            return {}

        results: Dict[str, float] = {}

        # Set model to eval mode.
        self.model.eval()

        try:
            with torch.no_grad():
                for task_name in task_names:
                    logger.info(
                        f"Running OLMES evaluation: task='{task_name}' "
                        f"(rank={DistributedUtils.get_rank()})"
                    )
                    try:
                        task_cfg: Dict[str, Any] = OLMES_TASK_CONFIG[task_name]

                        if task_name == "mmlu":
                            # MMLU requires per-subject evaluation.
                            cf_score, mcf_score = self._eval_mmlu_olmes(task_cfg)
                        else:
                            # Load datasets.
                            train_data: Any = self._load_dataset(
                                task_cfg["hf_path"],
                                task_cfg.get("hf_name"),
                                task_cfg["train_split"],
                            )
                            eval_data: Any = self._load_dataset(
                                task_cfg["hf_path"],
                                task_cfg.get("hf_name"),
                                task_cfg["eval_split"],
                            )

                            if train_data is None or eval_data is None:
                                logger.warning(
                                    f"Could not load dataset for task '{task_name}'. "
                                    f"Skipping."
                                )
                                continue

                            # Sample fewshot examples.
                            fewshot_examples: List[Dict[str, Any]] = (
                                self._sample_fewshots(
                                    train_data, task_name, n=task_cfg["num_shots"]
                                )
                            )

                            # Evaluate both formulations.
                            cf_score = self._evaluate_cf(
                                task_name, eval_data, fewshot_examples, task_cfg
                            )
                            mcf_score = self._evaluate_mcf(
                                task_name, eval_data, fewshot_examples, task_cfg
                            )

                        # Take max per OLMES protocol.
                        final_score: float = max(cf_score, mcf_score)
                        results[task_name] = final_score
                        results[f"{task_name}_cf"] = cf_score
                        results[f"{task_name}_mcf"] = mcf_score

                        logger.info(
                            f"OLMES '{task_name}': "
                            f"CF={cf_score:.4f} ({cf_score * 100:.1f}%), "
                            f"MCF={mcf_score:.4f} ({mcf_score * 100:.1f}%), "
                            f"max={final_score:.4f} ({final_score * 100:.1f}%)"
                        )

                    except Exception as exc:
                        logger.warning(
                            f"OLMES evaluation of task '{task_name}' failed: "
                            f"{type(exc).__name__}: {exc}. Skipping."
                        )
                        results[task_name] = 0.0
                        results[f"{task_name}_cf"] = 0.0
                        results[f"{task_name}_mcf"] = 0.0

        finally:
            # Always restore train mode.
            self.model.train()

        # Compute average over all successfully evaluated tasks.
        task_scores: List[float] = [
            results[t] for t in task_names if t in results
        ]
        if task_scores:
            results["average"] = sum(task_scores) / len(task_scores)
        else:
            results["average"] = 0.0

        logger.info(
            f"OLMES evaluation complete: "
            f"{len(task_scores)} tasks, "
            f"average={results.get('average', 0.0):.4f} "
            f"({results.get('average', 0.0) * 100:.1f}%)"
        )

        return results

    def run_adaptation_eval(
        self,
        task_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Run post-adaptation evaluation tasks.

        Evaluates the model on the adaptation benchmark suite from Table 5
        and Appendix C of the paper.

        Args:
            task_names: List of task names to evaluate. If None, evaluates
                        ["mmlu", "gsm8k", "bbh", "humaneval"].
                        Full suite: ["mmlu", "gsm8k", "bbh", "humaneval",
                                     "alpacaeval", "xstest", "ifeval"]
                        Note: alpacaeval requires OpenAI API key.

        Returns:
            Dict mapping task name to metric value:
                {
                    "mmlu": 0.519,          # 0-shot EM
                    "gsm8k": 0.405,         # 8-shot CoT EM
                    "bbh": 0.380,           # 3-shot EM
                    "humaneval": 0.516,     # 0-shot Pass@10
                    "alpacaeval": 0.840,    # % win rate (requires API)
                    "xstest": 0.826,        # F1
                    "ifeval": 0.481,        # Loose Accuracy
                }
        """
        if task_names is None:
            task_names = ["mmlu", "gsm8k", "bbh", "humaneval"]

        results: Dict[str, float] = {}

        self.model.eval()

        try:
            with torch.no_grad():
                for task in task_names:
                    logger.info(f"Running adaptation evaluation: task='{task}'")
                    try:
                        if task == "mmlu":
                            results["mmlu"] = self._eval_mmlu_0shot()
                        elif task == "gsm8k":
                            results["gsm8k"] = self._eval_gsm8k_8shot_cot()
                        elif task == "bbh":
                            results["bbh"] = self._eval_bbh_3shot()
                        elif task == "humaneval":
                            results["humaneval"] = self._eval_humaneval_pass_at_10()
                        elif task == "alpacaeval":
                            score: Optional[float] = self._eval_alpacaeval()
                            if score is not None:
                                results["alpacaeval"] = score
                        elif task == "xstest":
                            results["xstest"] = self._eval_xstest()
                        elif task == "ifeval":
                            results["ifeval"] = self._eval_ifeval()
                        else:
                            logger.warning(
                                f"Unknown adaptation task: '{task}'. Skipping."
                            )
                    except Exception as exc:
                        logger.warning(
                            f"Adaptation evaluation of task '{task}' failed: "
                            f"{type(exc).__name__}: {exc}. Skipping."
                        )

        finally:
            self.model.train()

        # Compute average over available tasks (excluding alpacaeval if missing).
        scored_tasks: List[str] = [
            t for t in ["mmlu", "gsm8k", "bbh", "humaneval", "xstest", "ifeval"]
            if t in results
        ]
        if scored_tasks:
            results["average"] = sum(results[t] for t in scored_tasks) / len(scored_tasks)

        logger.info(
            f"Adaptation evaluation complete: "
            f"tasks={list(results.keys())}, "
            f"average={results.get('average', 0.0):.4f}"
        )

        return results

    def run_dclm_eval(
        self,
        model_path: str,
        dclm_eval_script_path: str,
    ) -> Dict[str, float]:
        """Run DCLM evaluation using the official DCLM evaluation code.

        Per Appendix C: "we precisely follow their setup using the evaluation
        code released by the authors at https://github.com/mlfoundations/dclm."

        Args:
            model_path: Path to the model in HuggingFace format, or a
                        HuggingFace Hub model ID.
            dclm_eval_script_path: Path to the DCLM evaluation code directory
                                   (cloned from https://github.com/mlfoundations/dclm).

        Returns:
            Dict with DCLM Core and Extended task scores:
                {
                    "dclm_core": 0.472,
                    "dclm_extended": 0.325,
                }
            Returns empty dict if the DCLM evaluation script is not found or fails.
        """
        eval_script: str = os.path.join(dclm_eval_script_path, "eval.py")

        if not os.path.exists(eval_script):
            logger.warning(
                f"DCLM evaluation script not found at '{eval_script}'. "
                f"Clone the DCLM repo: git clone https://github.com/mlfoundations/dclm"
            )
            return {}

        logger.info(
            f"Running DCLM evaluation: model='{model_path}', "
            f"script='{eval_script}'"
        )

        try:
            result = subprocess.run(
                [
                    "python", eval_script,
                    "--model", model_path,
                    "--tasks", "core,extended",
                    "--output_format", "json",
                ],
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )

            if result.returncode != 0:
                logger.warning(
                    f"DCLM evaluation script failed with return code {result.returncode}. "
                    f"stderr: {result.stderr[:500]}"
                )
                return {}

            # Parse JSON output from the DCLM script.
            import json
            try:
                output_data: Dict[str, Any] = json.loads(result.stdout)
                dclm_results: Dict[str, float] = {
                    "dclm_core": float(output_data.get("core", 0.0)),
                    "dclm_extended": float(output_data.get("extended", 0.0)),
                }
                logger.info(
                    f"DCLM evaluation complete: "
                    f"core={dclm_results['dclm_core']:.4f}, "
                    f"extended={dclm_results['dclm_extended']:.4f}"
                )
                return dclm_results
            except (json.JSONDecodeError, KeyError) as parse_exc:
                logger.warning(
                    f"Failed to parse DCLM evaluation output: {parse_exc}. "
                    f"stdout: {result.stdout[:500]}"
                )
                return {}

        except subprocess.TimeoutExpired:
            logger.warning("DCLM evaluation timed out after 2 hours.")
            return {}
        except Exception as exc:
            logger.warning(
                f"DCLM evaluation failed: {type(exc).__name__}: {exc}"
            )
            return {}

    # =========================================================================
    # Core Scoring Methods
    # =========================================================================

    def _get_completion_logprob(
        self,
        prompt: str,
        completion: str,
    ) -> float:
        """Compute the sum of log-probabilities of completion tokens given prompt.

        Tokenizes prompt + completion jointly (more reliable than separate
        tokenization due to boundary effects). Uses the causal LM shift:
        logits at position t predict token t+1.

        Args:
            prompt: The prompt string (question/context). May be empty.
            completion: The completion string to score. Must be non-empty.

        Returns:
            Sum of log-probabilities over completion tokens (float, <= 0).
            Returns float("-inf") if completion is empty or sequence exceeds
            MAX_SEQ_LEN after truncation leaves no completion tokens.
        """
        if not completion:
            return float("-inf")

        # Tokenize jointly for reliable boundary detection.
        # add_special_tokens=False: no BOS/EOS added.
        prompt_ids: List[int] = (
            self.tokenizer.encode(prompt, add_special_tokens=False)
            if prompt
            else []
        )
        completion_ids: List[int] = self.tokenizer.encode(
            completion, add_special_tokens=False
        )

        if not completion_ids:
            return float("-inf")

        full_ids: List[int] = prompt_ids + completion_ids

        # Truncate from the left if exceeding MAX_SEQ_LEN.
        # Keep the completion tokens and as much prompt context as