"""Dataset loading and preprocessing for SCoRe experiments.

Supports:
- MATH dataset (Hendrycks et al. 2021) with MATH500 split per Lightman et al. 2023
- MBPP dataset (Austin et al. 2021)
- HumanEval dataset (Chen et al. 2021) for evaluation
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset


@dataclass
class MathProblem:
    """A single MATH problem."""
    problem_id: str
    problem_text: str
    answer: str
    level: str  # difficulty level (1-5)
    problem_type: str  # e.g., algebra, counting, etc.


@dataclass
class CodeProblem:
    """A single code generation problem (MBPP/HumanEval)."""
    problem_id: str
    task_description: str
    test_cases: str
    code_solution: str
    entry_point: str


class MATH500Dataset(Dataset):
    """MATH dataset with the MATH500 split.

    Training: 4500 problems from the test set (augmented to training set)
    Evaluation: remaining 500 problems (MATH500)
    Following Lightman et al. (2023).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split

        # Load all problems (training + test)
        problems = self._load_all_problems(data_dir)

        # Create MATH500 split
        random.seed(seed)
        all_test = [p for p in problems if p.get("source", "train") == "test"]
        if len(all_test) < 500:
            # If no source info, randomly split 500 from the full set
            all_problems = problems
            random.shuffle(all_problems)
            all_test = all_problems[:500]
            all_train = all_problems[500:]
        else:
            random.shuffle(all_test)
            test500 = all_test[:500]
            train_extra = all_test[500:]
            all_train = [p for p in problems if p.get("source", "train") == "train"]
            all_train.extend(train_extra)

        if split == "train":
            self.problems = all_train[:4500]  # 4500 training examples
        elif split == "val":
            # Small validation set from training
            random.shuffle(all_train)
            self.problems = all_train[4500:4580]  # 80 validation
        else:
            self.problems = test500

    def _load_all_problems(self, data_dir: str) -> List[Dict]:
        """Load all MATH problems from a directory or jsonl file."""
        problems = []
        math_path = os.path.join(data_dir, "MATH")

        if os.path.isdir(math_path):
            for problem_type in os.listdir(math_path):
                type_dir = os.path.join(math_path, problem_type)
                if os.path.isdir(type_dir):
                    for fname in os.listdir(type_dir):
                        if fname.endswith(".json"):
                            with open(os.path.join(type_dir, fname)) as f:
                                prob = json.load(f)
                                prob["problem_type"] = problem_type
                                prob["problem_id"] = fname.replace(".json", "")
                                problems.append(prob)
        else:
            # Load from jsonl file
            jsonl_path = os.path.join(data_dir, "math_train_test.jsonl")
            if os.path.exists(jsonl_path):
                with open(jsonl_path) as f:
                    for line in f:
                        problems.append(json.loads(line))
        return problems

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict:
        prob = self.problems[idx]
        return {
            "problem_id": prob.get("problem_id", str(idx)),
            "problem_text": prob["problem"],
            "answer": prob.get("answer", prob.get("solution", "")),
            "level": prob.get("level", ""),
            "problem_type": prob.get("problem_type", ""),
        }


class MBPPDataset(Dataset):
    """MBPP dataset for code generation.

    Training: MBPP train split
    Evaluation: HumanEval (separate dataset)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split

        if split == "train":
            self.problems = self._load_mbpp("train")
        elif split == "val":
            all_train = self._load_mbpp("train")
            random.shuffle(all_train)
            self.problems = all_train[:50]
        else:
            self.problems = self._load_mbpp("test")

    def _load_mbpp(self, split: str) -> List[Dict]:
        """Load MBPP problems from jsonl file."""
        mbpp_path = os.path.join(self.data_dir, "mbpp.jsonl")
        problems = []
        if os.path.exists(mbpp_path):
            with open(mbpp_path) as f:
                for line in f:
                    prob = json.loads(line)
                    if prob.get("split", "train") == split:
                        problems.append(prob)

        # If no split column, load all and manually split 80/20
        if not problems:
            with open(mbpp_path) as f:
                all_probs = [json.loads(line) for line in f]
            if split == "train":
                problems = all_probs[: int(0.8 * len(all_probs))]
            else:
                problems = all_probs[int(0.8 * len(all_probs)) :]

        return problems

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict:
        prob = self.problems[idx]
        # Build test cases string for prompt
        test_str = "\n".join(prob.get("test_list", prob.get("test", [])))
        return {
            "problem_id": str(prob.get("task_id", idx)),
            "task_description": prob.get("text", prob.get("prompt", "")),
            "test_cases": test_str,
            "code_solution": prob.get("code", ""),
            "entry_point": prob.get("entry_point", "solution"),
        }


class HumanEvalDataset(Dataset):
    """HumanEval dataset for evaluation."""

    def __init__(self, data_dir: str):
        super().__init__()
        human_eval_path = os.path.join(data_dir, "HumanEval.jsonl")
        self.problems = []
        with open(human_eval_path) as f:
            for line in f:
                self.problems.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict:
        prob = self.problems[idx]
        return {
            "problem_id": prob.get("task_id", str(idx)),
            "prompt": prob["prompt"],
            "entry_point": prob["entry_point"],
            "canonical_solution": prob.get("canonical_solution", ""),
            "test": prob["test"],
        }


class MBPPRepairDataset(Dataset):
    """MBPP-R: offline repair task (Ni et al. 2024).

    Requires correcting incorrect first-attempt programs from PaLM 2.
    We simulate this by taking incorrect base-model responses and pairing
    them with the correct solution as the target.
    """

    def __init__(
        self,
        incorrect_responses: List[Dict],
    ):
        self.incorrect_responses = incorrect_responses

    def __len__(self) -> int:
        return len(self.incorrect_responses)

    def __getitem__(self, idx: int) -> Dict:
        return self.incorrect_responses[idx]


def collate_math_batch(batch: List[Dict]) -> Dict:
    """Collate a batch of MATH problems."""
    return {
        "problem_ids": [item["problem_id"] for item in batch],
        "problem_texts": [item["problem_text"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "levels": [item["level"] for item in batch],
        "problem_types": [item["problem_type"] for item in batch],
    }


def collate_code_batch(batch: List[Dict]) -> Dict:
    """Collate a batch of code problems."""
    return {
        "problem_ids": [item["problem_id"] for item in batch],
        "task_descriptions": [item["task_description"] for item in batch],
        "test_cases": [item["test_cases"] for item in batch],
        "code_solutions": [item["code_solution"] for item in batch],
        "entry_points": [item["entry_point"] for item in batch],
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    is_code: bool = False,
) -> DataLoader:
    """Create a DataLoader with the appropriate collate function."""
    collate_fn = collate_code_batch if is_code else collate_math_batch
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
