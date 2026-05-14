"""
Dataset loading and preprocessing for MATH, MBPP, and HumanEval.

MATH split: following Lightman et al. (2023), augment training set with 4500
problems from the test set; evaluate on the remaining 500 (MATH500).

Code split: train on MBPP, evaluate on HumanEval (zero-shot transfer).
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

from prompts import (
    build_humaneval_first_turn_prompt,
    build_humaneval_second_turn_prompt,
    build_math_first_turn_prompt,
    build_math_second_turn_prompt,
    build_mbpp_first_turn_prompt,
    build_mbpp_second_turn_prompt,
)


@dataclass
class Problem:
    problem_id: str
    prompt_t1: str          # first-turn prompt fed to the model
    prompt_t2_template: str # second-turn prompt template (needs first_attempt filled in)
    answer: str             # ground-truth answer / solution
    task: str               # "math" | "mbpp" | "humaneval"
    metadata: Optional[Dict] = None

    def build_prompt_t2(self, first_attempt: str) -> str:
        """Fill in the first attempt to produce the second-turn prompt."""
        return self.prompt_t2_template.format(first_attempt=first_attempt)


# ---------------------------------------------------------------------------
# MATH
# ---------------------------------------------------------------------------

def _extract_math_answer(solution: str) -> str:
    """Extract the boxed answer from a MATH solution string."""
    import re
    match = re.search(r"\\boxed\{(.+?)\}", solution)
    if match:
        return match.group(1).strip()
    # Fallback: look for "The answer is X"
    match = re.search(r"[Tt]he (?:final )?answer is\s+(.+?)(?:\.|$)", solution)
    if match:
        return match.group(1).strip()
    return solution.strip()


def load_math_dataset(
    train_problems_from_test: int = 4500,
    eval_problems: int = 500,
    seed: int = 42,
) -> Tuple[List[Problem], List[Problem]]:
    """
    Load MATH dataset.

    Following Lightman et al. (2023): augment the MATH training set with
    `train_problems_from_test` problems from the test set; evaluate on the
    remaining `eval_problems` problems (MATH500).
    """
    rng = random.Random(seed)

    train_ds = load_dataset("hendrycks/competition_math", split="train", trust_remote_code=True)
    test_ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)

    test_indices = list(range(len(test_ds)))
    rng.shuffle(test_indices)
    train_from_test_indices = test_indices[:train_problems_from_test]
    eval_indices = test_indices[train_problems_from_test: train_problems_from_test + eval_problems]

    def _make_problem(example, idx: int) -> Problem:
        problem_text = example["problem"]
        answer = _extract_math_answer(example["solution"])
        prompt_t1 = build_math_first_turn_prompt(problem_text)
        # Template: {first_attempt} will be replaced at rollout time
        prompt_t2_template = (
            build_math_first_turn_prompt(problem_text)
            + "\n\n{first_attempt}\n\n"
            + "There might be an error in the solution above because of lack of "
            "understanding of the question. Please correct the error, if any, and "
            "rewrite the solution. Only output the final solution! At the end of the "
            "Solution, when you give your final answer, write it in the form "
            '"Final Answer: The final answer is $answer$. I hope it is correct."'
        )
        return Problem(
            problem_id=f"math_{idx}",
            prompt_t1=prompt_t1,
            prompt_t2_template=prompt_t2_template,
            answer=answer,
            task="math",
            metadata={"level": example.get("level"), "type": example.get("type")},
        )

    train_problems = [_make_problem(train_ds[i], i) for i in range(len(train_ds))]
    train_problems += [
        _make_problem(test_ds[i], len(train_ds) + i) for i in train_from_test_indices
    ]
    eval_problems_list = [
        _make_problem(test_ds[i], len(train_ds) + i) for i in eval_indices
    ]

    return train_problems, eval_problems_list


# ---------------------------------------------------------------------------
# MBPP
# ---------------------------------------------------------------------------

def _format_mbpp_tests(test_list: List[str]) -> str:
    return "\n".join(test_list)


def load_mbpp_dataset(seed: int = 42) -> Tuple[List[Problem], List[Problem]]:
    """Load MBPP dataset for training."""
    ds = load_dataset("mbpp", split="train", trust_remote_code=True)
    val_ds = load_dataset("mbpp", split="validation", trust_remote_code=True)

    def _make_problem(example, idx: int) -> Problem:
        problem_text = example["text"]
        test_cases = _format_mbpp_tests(example["test_list"])
        code_solution = example["code"]
        prompt_t1 = build_mbpp_first_turn_prompt(problem_text, test_cases)
        prompt_t2_template = (
            build_mbpp_first_turn_prompt(problem_text, test_cases)
            + "\n{first_attempt}\n[DONE]\n\n"
            + "There might be an error in the code above because of lack of "
            "understanding of the question. Please correct the error, if any, and "
            "rewrite the solution. Only output the final correct Python program!"
        )
        return Problem(
            problem_id=f"mbpp_{idx}",
            prompt_t1=prompt_t1,
            prompt_t2_template=prompt_t2_template,
            answer=code_solution,
            task="mbpp",
            metadata={"test_list": example["test_list"], "task_id": example.get("task_id")},
        )

    train_problems = [_make_problem(example, i) for i, example in enumerate(ds)]
    val_problems = [_make_problem(example, i) for i, example in enumerate(val_ds)]
    return train_problems, val_problems


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

def load_humaneval_dataset() -> List[Problem]:
    """Load HumanEval dataset for evaluation (zero-shot transfer from MBPP)."""
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)

    problems = []
    for example in ds:
        task_id = example["task_id"]
        prompt = example["prompt"]
        canonical_solution = example["canonical_solution"]
        test_code = example["test"]
        entry_point = example["entry_point"]

        prompt_t1 = build_humaneval_first_turn_prompt(prompt)
        prompt_t2_template = (
            build_humaneval_first_turn_prompt(prompt)
            + "\n\n{first_attempt}\n\n"
            + "There might be an error in the code above because of lack of "
            "understanding of the question. Please correct the error, if any, and "
            "rewrite the solution. Only output the final correct Python program!"
        )
        problems.append(
            Problem(
                problem_id=task_id,
                prompt_t1=prompt_t1,
                prompt_t2_template=prompt_t2_template,
                answer=canonical_solution,
                task="humaneval",
                metadata={
                    "test": test_code,
                    "entry_point": entry_point,
                    "prompt": prompt,
                },
            )
        )
    return problems


# ---------------------------------------------------------------------------
# MBPP-R: offline repair task (Ni et al., 2024)
# ---------------------------------------------------------------------------

def load_mbpp_repair_dataset(
    incorrect_solutions_path: str,
) -> List[Dict]:
    """
    Load MBPP-R offline repair dataset.

    The dataset consists of incorrect first-attempt programs generated from
    PaLM 2, which the model must repair. Following Ni et al. (2024).

    Args:
        incorrect_solutions_path: Path to JSONL file with fields:
            - problem_id, problem_text, test_cases, incorrect_solution
    """
    problems = []
    with open(incorrect_solutions_path) as f:
        for line in f:
            item = json.loads(line)
            test_cases = _format_mbpp_tests(item["test_cases"])
            prompt_t2 = build_mbpp_second_turn_prompt(
                item["problem_text"], test_cases, item["incorrect_solution"]
            )
            problems.append(
                {
                    "problem_id": item["problem_id"],
                    "prompt_t2": prompt_t2,
                    "test_list": item["test_cases"],
                    "incorrect_solution": item["incorrect_solution"],
                }
            )
    return problems


# ---------------------------------------------------------------------------
# PyTorch Dataset wrappers
# ---------------------------------------------------------------------------

class SCoReDataset(Dataset):
    """Wraps a list of Problem objects for use with DataLoader."""

    def __init__(self, problems: List[Problem]):
        self.problems = problems

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Problem:
        return self.problems[idx]


class OfflineTrajectoryDataset(Dataset):
    """
    Dataset of pre-collected two-turn trajectories for SFT-based baselines.

    Each item is a dict with keys:
        prompt_t1, response_t1, prompt_t2, response_t2,
        reward_t1, reward_t2, problem_id
    """

    def __init__(self, trajectories: List[Dict]):
        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict:
        return self.trajectories[idx]

    @classmethod
    def from_jsonl(cls, path: str) -> "OfflineTrajectoryDataset":
        trajectories = []
        with open(path) as f:
            for line in f:
                trajectories.append(json.loads(line))
        return cls(trajectories)

    def save_jsonl(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for item in self.trajectories:
                f.write(json.dumps(item) + "\n")


def collect_base_model_trajectories(
    model,
    problems: List[Problem],
    num_samples: int = 4,
    temperature: float = 1.0,
    reward_fn=None,
) -> List[Dict]:
    """
    Collect two-turn self-correction trajectories from the base model.

    Used to build D_STaR and D_SFT datasets for SFT baselines, and also
    to provide offline first-attempt prompts for Stage II (Section 5.3).

    Args:
        model: LLMPolicy instance
        problems: list of Problem objects
        num_samples: number of rollouts per problem
        temperature: sampling temperature
        reward_fn: callable(response, answer, task) -> float

    Returns:
        List of trajectory dicts
    """
    trajectories = []
    for problem in problems:
        for _ in range(num_samples):
            response_t1 = model.generate(problem.prompt_t1, temperature=temperature)
            prompt_t2 = problem.build_prompt_t2(response_t1)
            response_t2 = model.generate(prompt_t2, temperature=temperature)

            reward_t1 = reward_fn(response_t1, problem.answer, problem.task) if reward_fn else None
            reward_t2 = reward_fn(response_t2, problem.answer, problem.task) if reward_fn else None

            trajectories.append(
                {
                    "problem_id": problem.problem_id,
                    "prompt_t1": problem.prompt_t1,
                    "response_t1": response_t1,
                    "prompt_t2": prompt_t2,
                    "response_t2": response_t2,
                    "reward_t1": reward_t1,
                    "reward_t2": reward_t2,
                    "answer": problem.answer,
                    "task": problem.task,
                }
            )
    return trajectories
