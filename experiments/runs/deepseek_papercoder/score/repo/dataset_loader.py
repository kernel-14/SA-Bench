"""
dataset_loader.py

Provides all dataset loading, splitting, and prompt formatting required by SCoRe.
Implements the Lightman split for MATH, MBPP 3‑shot prompt construction,
HumanEval loading, and tokenization utilities. Also generates an offline pool
of base‑model first‑attempt responses for Stage II mixing.
"""

import json
import logging
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# Import configuration class – note that config.py must be present in the same package.
from config import Config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants for prompt templates
# --------------------------------------------------------------------------- #

# Full 3‑shot prompt for MBPP training, exactly as in the paper's Appendix C.
MBPP_3SHOT_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {task}\n"
    "Your code should pass these tests:\n"
    "{tests}\n"
    "[BEGIN]\n"
)

# The three canned examples (without their solutions) to be prepended.
MBPP_CANNED_EXAMPLES = [
    (
        "Write a function to find the similar elements from the given two tuple lists.",
        [
            "assert similar_elements((3, 4, 5, 6), (5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4), (5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13), (17, 15, 14, 13)) == (13, 14)",
        ],
        "def similar_elements(test_tup1, test_tup2):\n    res = tuple(set(test_tup1) & set(test_tup2))\n    return (res)\n[DONE]",
    ),
    (
        "Write a python function to identify non−prime numbers.",
        [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
        "import math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result\n[DONE]",
    ),
    (
        "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        [
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],3) == [85, 75, 65]",
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],2) == [85, 75]",
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],5) == [85, 75, 65, 58, 35]",
        ],
        "import heapq as hq\ndef heap_queue_largest(nums,n):\n    largest_nums = hq.nlargest(n, nums)\n    return largest_nums\n[DONE]",
    ),
]

# --------------------------------------------------------------------------- #
# DatasetLoader class
# --------------------------------------------------------------------------- #


class DatasetLoader:
    """Handles dataset ingestion, splitting, and prompt formatting for SCoRe."""

    def __init__(
        self,
        config: Config,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        """
        Args:
            config: Global configuration object.
            tokenizer: HuggingFace tokenizer matching the model.
        """
        self.config = config
        self.tokenizer = tokenizer

        # Set seed for reproducibility of randomized splits
        torch.manual_seed(config.training.seed)
        np.random.seed(config.training.seed)

    # ----------------------------------------------------------------------- #
    # Public dataset loading methods
    # ----------------------------------------------------------------------- #

    def load_math(self, split: str = "train") -> Dataset:
        """Load MATH dataset with the Lightman et al. augmentation split.

        Paper split: original training set (7500) + 4500 problems taken
        from the test set. The remaining 500 test problems form MATH500.

        Args:
            split: "train" or "eval" (returns MATH500 for eval).

        Returns:
            A HuggingFace Dataset with columns 'problem' and 'solution'.
        """
        raw_train = load_dataset("hendrycks/math", split="train", trust_remote_code=True)
        raw_test = load_dataset("hendrycks/math", split="test", trust_remote_code=True)

        # Fixed random selection of 4500 from test
        test_indices = list(range(len(raw_test)))
        seed = self.config.training.seed
        rng = np.random.default_rng(seed)
        rng.shuffle(test_indices)

        extra_indices = test_indices[: self.config.math.dataset.train.extra_from_test]
        eval_indices = test_indices[self.config.math.dataset.train.extra_from_test :]

        extra_train = raw_test.select(extra_indices)
        train = concatenate_datasets([raw_train, extra_train])

        if split == "train":
            return train
        elif split == "eval":
            return raw_test.select(eval_indices)
        else:
            raise ValueError(f"Unknown split '{split}', must be 'train' or 'eval'.")

    def load_mbpp(self, split: str = "train") -> Dataset:
        """Load MBPP dataset and construct the 3‑shot prompt for training.

        Args:
            split: e.g. "train". The MBPP dataset has splits "train", "validation", "test".

        Returns:
            Dataset with columns:
                - 'prompt'   : full 3‑shot prompt string (input for turn 1)
                - 'code'     : reference solution (may be used for offline eval)
                - 'test_list': list of assert strings for binary reward checking
        """
        raw_mbpp = load_dataset("mbpp", split=split, trust_remote_code=True)

        def build_full_prompt(example: Dict) -> Dict:
            # Canned examples with their own task descriptions and tests
            canned_blocks = []
            for task_desc, tests, code_solution in MBPP_CANNED_EXAMPLES:
                # Format the canned task as if it were the current task
                block = (
                    f"You are an expert Python programmer, and here is your task: {task_desc}\n"
                    f"Your code should pass these tests:\n"
                    + "\n".join(tests)
                    + "\n[BEGIN]\n"
                    + code_solution
                    + "\n"
                )
                canned_blocks.append(block)

            # Current task
            task = example["text"]
            tests = example["test_list"]
            tests_str = "\n".join(tests)
            current_block = MBPP_3SHOT_TEMPLATE.format(task=task, tests=tests_str)

            # Combine: canned examples (with their DONE) + current prompt
            full_prompt = "".join(canned_blocks) + current_block
            return {"prompt": full_prompt, "code": example["code"], "test_list": tests}

        dataset = raw_mbpp.map(build_full_prompt, remove_columns=raw_mbpp.column_names)
        return dataset

    def load_humaneval(self) -> Dataset:
        """Load HumanEval dataset for zero‑shot evaluation (no test cases given).

        Returns:
            Dataset with columns: 'prompt' (function signature), 'task_id', 'test'.
        """
        raw = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
        # The benchmark already contains 'prompt' column with the function signature.
        return raw

    # ----------------------------------------------------------------------- #
    # Prompt formatters
    # ----------------------------------------------------------------------- #

    def prepare_turn1_input(self, problem: str, task: str = "math") -> str:
        """Build the full first‑turn input string.

        For MATH, prepends the zero‑shot CoT instruction.
        For code, the problem string is already a fully formatted prompt.

        Args:
            problem: The raw problem statement (MATH) or already-formatted prompt (code).
            task: "math" or "code".

        Returns:
            A single string to be tokenized and fed to the model.
        """
        if task == "math":
            return f"{self.config.prompts.math_zero_shot}\nProblem: {problem}\n"
        else:  # code
            # The code prompt is already complete; return as-is.
            return problem

    def prepare_turn2_input(self, problem: str, y1: str, task: str = "math") -> str:
        """Build the second‑turn input that includes the previous attempt.

        The constructed string contains the original problem, the first‑turn response,
        and the self‑correction instruction (without revealing correctness).

        Args:
            problem: The raw problem (MATH) or fully formatted code prompt.
            y1: The model's first‑attempt output.
            task: "math" or "code".

        Returns:
            A single string.
        """
        turn1_input = self.prepare_turn1_input(problem, task)
        if task == "math":
            correction_instruction = self.config.prompts.math_self_correction
        else:
            correction_instruction = self.config.prompts.code_self_correction

        return f"{turn1_input}\n{y1}\n\n{correction_instruction}"

    # ----------------------------------------------------------------------- #
    # Tokenization helper
    # ----------------------------------------------------------------------- #

    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of prompt strings.

        Designed to be used with Dataset.map() for efficient preprocessing.
        Pads/truncates to config.model.max_seq_length.

        Args:
            examples: A batch dictionary containing a 'prompt' field (list of strings).

        Returns:
            A dictionary with 'input_ids' and 'attention_mask' tensors.
        """
        prompts = examples["prompt"]
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=self.config.model.max_seq_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    # ----------------------------------------------------------------------- #
    # Offline first‑attempt generation
    # ----------------------------------------------------------------------- #

    def generate_offline_y1(
        self,
        base_model: Any,  # expected to have a .generate() method (PolicyModel)
        reward_fn: Any,   # expected to have a .check_correct() or __call__ method
        train_dataset: Dataset,
        num_samples: int = 4,
        max_new_tokens: Optional[int] = None,
    ) -> Dataset:
        """Produce an offline dataset of base‑model first attempts with rewards.

        Used in Stage II to augment on‑policy data with fixed first‑turn responses.

        Args:
            base_model: The original (frozen) base model used for generation.
            reward_fn: A callable that computes the binary reward given (y, ground_truth).
            train_dataset: Dataset containing at least the raw problem/prompt and
                           ground truth information. Expected columns:
                           - 'problem' and 'solution' for MATH
                           - 'prompt', 'test_list' for code
            num_samples: Number of first‑attempt samples to generate per prompt.
            max_new_tokens: Maximum number of tokens for generation.
                            Defaults to based on config if None.

        Returns:
            A Dataset with entries: {'prompt': str, 'y1': str, 'r1': float,
                                     'ground_truth': Union[str, List[str]]}
        """
        if max_new_tokens is None:
            # Use a fraction of max sequence length (512 for MATH, 256 for code)
            max_new_tokens = self.config.model.max_seq_length // 4

        entries: List[Dict] = []

        # Determine task type from dataset presence of 'solution' (MATH) or 'test_list' (code)
        # We inspect the first example to decide.
        first_example = train_dataset[0]
        if "solution" in first_example:
            task = "math"
        elif "test_list" in first_example:
            task = "code"
        else:
            raise ValueError("Cannot infer task type from dataset columns.")

        # Iterate over all training prompts
        for example in train_dataset:
            if task == "math":
                raw_problem = example["problem"]
                ground_truth = example["solution"]
                turn1_input = self.prepare_turn1_input(raw_problem, task="math")
            else:
                # For code, the prompt is already the full turn‑1 string
                turn1_input = example["prompt"]
                raw_problem = turn1_input  # store as "problem" for reference
                ground_truth = example["test_list"]

            # Tokenize the turn‑1 input
            tokenized = self.tokenizer(
                turn1_input, return_tensors="pt", truncation=True,
                max_length=self.config.model.max_seq_length,
            )
            input_ids = tokenized["input_ids"][0]  # (1, L) -> (L,)

            # Generate multiple samples
            for _ in range(num_samples):
                # base_model.generate expects batch dimension
                generated_ids = base_model.generate(
                    input_ids=input_ids.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=True,  # explicitly sample
                )
                # Decode only the newly generated part
                new_tokens = generated_ids[0, len(input_ids):]
                y1 = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

                # Compute reward: for MATH it's exact answer match; for code, test execution.
                # reward_fn is expected to have a compute_reward(y, ground_truth) method,
                # but the design shows RewardFunction.__call__. We'll assume it's callable.
                try:
                    r1 = reward_fn(y1, ground_truth) if not isinstance(ground_truth, list) else reward_fn(y1, ground_truth)
                except Exception as e:
                    logger.warning(f"Reward computation failed: {e}. Setting r1=0.0")
                    r1 = 0.0

                entries.append({
                    "prompt": raw_problem,        # original problem string (used later to reconstruct turn2)
                    "y1": y1,
                    "r1": float(r1),
                    "ground_truth": ground_truth, # needed for r2_shaped when re‑using
                })

        return Dataset.from_list(entries)


# --------------------------------------------------------------------------- #
# Optional utility: load MBPP‑R (offline repair) – placeholder.
# This is not part of SCoRe training, only for evaluation if desired.
# --------------------------------------------------------------------------- #

def load_mbpp_r() -> Dataset:
    """Load MBPP‑R dataset if available (pre‑generated PaLM‑2 errors)."""
    try:
        # Assumes a file `mbpp_r.jsonl` in the data folder.
        data_path = "data/mbpp_r.jsonl"
        if not os.path.exists(data_path):
            logger.warning("MBPP‑R dataset not found. Skipping.")
            return Dataset.from_list([])
        with open(data_path, "r") as f:
            records = [json.loads(line) for line in f]
        return Dataset.from_list(records)
    except Exception as e:
        logger.error(f"Failed to load MBPP‑R: {e}")
        return Dataset.from_list([])
