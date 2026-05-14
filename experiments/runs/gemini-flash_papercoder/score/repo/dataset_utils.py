import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Problem:
    """
    Represents a single problem instance with its ID, text, ground truth, and optional metadata.
    """

    problem_id: str
    text: str
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Ensure problem_id is always a string.
        """
        self.problem_id = str(self.problem_id)


def load_math_dataset(path: str) -> List[Problem]:
    """
    Loads problems from a MATH dataset file in JSONL format.
    Assumes each JSON object has 'problem' for text and 'solution' for ground_truth.
    Problem IDs are generated sequentially if not explicitly present.

    Args:
        path: The file path to the MATH JSONL dataset.

    Returns:
        A list of Problem objects.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MATH dataset file not found at: {path}")

    problems: List[Problem] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                problem_id = data.get("id", f"math_problem_{idx}")
                problem_text = data.get("problem")
                ground_truth = data.get("solution")

                if problem_text is None or ground_truth is None:
                    raise ValueError(
                        f"Missing 'problem' or 'solution' key in line {idx+1} of {path}"
                    )

                # Store all other keys in metadata
                metadata = {k: v for k, v in data.items() if k not in ["id", "problem", "solution"]}

                problems.append(
                    Problem(
                        problem_id=problem_id,
                        text=problem_text,
                        ground_truth=ground_truth,
                        metadata=metadata,
                    )
                )
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Error decoding JSON in line {idx+1} of {path}: {e}",
                    e.doc,
                    e.pos,
                ) from e
            except ValueError as e:
                raise ValueError(
                    f"Data integrity error in line {idx+1} of {path}: {e}"
                ) from e
    return problems


def load_code_dataset(path: str) -> List[Problem]:
    """
    Loads problems from code generation dataset files (HumanEval, MBPP, MBPP-R)
    in JSONL format. Adapts parsing based on expected keys for each dataset type.

    HumanEval keys: 'task_id', 'prompt', 'canonical_solution', 'entry_point', 'test'
    MBPP keys: 'task_id', 'text', 'code' (for ground_truth), 'test_code'
    MBPP-R is expected to be similar to MBPP but might contain an 'incorrect_code' field.

    Args:
        path: The file path to the code JSONL dataset.

    Returns:
        A list of Problem objects.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON.
        ValueError: If a problem cannot be parsed due to missing critical keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Code dataset file not found at: {path}")

    problems: List[Problem] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())

                problem_id: Optional[str] = None
                problem_text: Optional[str] = None
                ground_truth: Optional[str] = None
                metadata: Dict[str, Any] = data.copy()

                # Try HumanEval schema first
                if "task_id" in data and "prompt" in data and "canonical_solution" in data:
                    problem_id = data["task_id"]
                    problem_text = data["prompt"]
                    ground_truth = data["canonical_solution"]
                    # Add other HumanEval specific keys to metadata
                    if "entry_point" in data:
                        metadata["entry_point"] = data["entry_point"]
                    if "test" in data:
                        metadata["test"] = data["test"]
                    del metadata["prompt"]
                    del metadata["canonical_solution"]
                # Then try MBPP schema
                elif "task_id" in data and "text" in data and "code" in data:
                    problem_id = data["task_id"]
                    problem_text = data["text"]
                    ground_truth = data["code"]
                    # Add other MBPP specific keys to metadata
                    if "test_code" in data:
                        metadata["test_code"] = data["test_code"]
                    del metadata["text"]
                    del metadata["code"]
                elif "problem_id" in data and "problem" in data and "ground_truth" in data:
                    # Generic code problem format, possibly for MBPP-R generated data
                    problem_id = data["problem_id"]
                    problem_text = data["problem"]
                    ground_truth = data["ground_truth"]
                    del metadata["problem"]
                    del metadata["ground_truth"]
                else:
                    raise ValueError(
                        "Problem does not match HumanEval, MBPP, or generic code problem schema."
                    )

                if problem_id is None or problem_text is None or ground_truth is None:
                    raise ValueError("Critical fields (id, text, ground_truth) missing after parsing.")

                problems.append(
                    Problem(
                        problem_id=problem_id,
                        text=problem_text,
                        ground_truth=ground_truth,
                        metadata=metadata,
                    )
                )
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Error decoding JSON in line {idx+1} of {path}: {e}",
                    e.doc,
                    e.pos,
                ) from e
            except ValueError as e:
                raise ValueError(
                    f"Data integrity error in line {idx+1} of {path}: {e}"
                ) from e
    return problems


class _ProblemDataset(Dataset):
    """
    A custom PyTorch Dataset to hold a list of Problem objects.
    """

    def __init__(self, problems: List[Problem]):
        """
        Initializes the dataset with a list of Problem objects.

        Args:
            problems: A list of Problem instances.
        """
        self.problems = problems

    def __len__(self) -> int:
        """
        Returns the total number of problems in the dataset.
        """
        return len(self.problems)

    def __getitem__(self, idx: int) -> Problem:
        """
        Retrieves a Problem object by its index.

        Args:
            idx: The index of the problem to retrieve.

        Returns:
            The Problem object at the specified index.
        """
        return self.problems[idx]


def build_dataloader(
    data: List[Problem], batch_size: int, shuffle: bool = True
) -> DataLoader:
    """
    Creates a PyTorch DataLoader from a list of Problem objects.

    Args:
        data: The list of Problem objects to be loaded.
        batch_size: The desired batch size for the DataLoader.
        shuffle: Whether to shuffle the data at the beginning of each epoch.

    Returns:
        A torch.utils.data.DataLoader instance.
    """
    dataset = _ProblemDataset(data)
    # No special collate_fn is needed as Problem objects are returned directly
    # and batching of these objects will be handled manually in trainers/evaluators.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # --- Example Usage for MATH dataset ---
    print("--- Testing MATH dataset loading ---")
    # Create a dummy MATH JSONL file for testing
    dummy_math_data = [
        {"id": "MATH_1", "problem": "What is 2+2?", "solution": "4", "level": "easy"},
        {"id": "MATH_2", "problem": "What is 5*3?", "solution": "15"},
        {"problem": "What is 10-7?", "solution": "3"}, # Missing ID
    ]
    math_test_file = "dummy_math_dataset.jsonl"
    with open(math_test_file, "w", encoding="utf-8") as f:
        for entry in dummy_math_data:
            f.write(json.dumps(entry) + "\n")

    try:
        math_problems = load_math_dataset(math_test_file)
        print(f"Loaded {len(math_problems)} MATH problems.")
        for p in math_problems:
            print(f"ID: {p.problem_id}, Text: {p.text[:20]}..., GT: {p.ground_truth}, Metadata: {p.metadata}")

        math_dataloader = build_dataloader(math_problems, batch_size=2, shuffle=False)
        print("\nTesting MATH DataLoader:")
        for i, batch in enumerate(math_dataloader):
            print(f"Batch {i+1}: Contains {len(batch)} problems.")
            for prob in batch:
                print(f"  - Problem ID: {prob.problem_id}")
    except Exception as e:
        print(f"Error loading MATH dataset: {e}")
    finally:
        if os.path.exists(math_test_file):
            os.remove(math_test_file)

    # --- Example Usage for Code dataset ---
    print("\n--- Testing Code dataset loading ---")
    # Create a dummy Code JSONL file for testing
    dummy_code_data = [
        {   # HumanEval-like
            "task_id": "HumanEval/0",
            "prompt": "def func(a, b):\n    \"\"\"Docstring\"\"\"\n",
            "canonical_solution": "    return a + b\n",
            "entry_point": "func",
            "test": "assert func(1, 2) == 3\n"
        },
        {   # MBPP-like
            "task_id": "MBPP/1",
            "text": "Write a function to add two numbers.",
            "code": "def add_numbers(x, y):\n    return x + y\n",
            "test_code": "assert add_numbers(1, 2) == 3\n",
            "difficulty": "easy"
        },
        {   # MBPP-R like generic
            "problem_id": "MBPP-R/100",
            "problem": "Given two lists, return their intersection.",
            "ground_truth": "def intersect(list1, list2):\n    return list(set(list1) & set(list2))\n",
            "incorrect_code": "def intersect(list1, list2):\n    return list1 + list2\n"
        },
        { # Invalid entry
            "invalid_key": "data"
        }
    ]
    code_test_file = "dummy_code_dataset.jsonl"
    with open(code_test_file, "w", encoding="utf-8") as f:
        for entry in dummy_code_data:
            f.write(json.dumps(entry) + "\n")

    try:
        code_problems = load_code_dataset(code_test_file)
        print(f"Loaded {len(code_problems)} Code problems.")
        for p in code_problems:
            print(f"ID: {p.problem_id}, Text: {p.text[:30]}..., GT: {p.ground_truth[:30]}..., Metadata keys: {list(p.metadata.keys())}")

        code_dataloader = build_dataloader(code_problems, batch_size=1, shuffle=True)
        print("\nTesting Code DataLoader:")
        for i, batch in enumerate(code_dataloader):
            print(f"Batch {i+1}: Contains {len(batch)} problems.")
            for prob in batch:
                print(f"  - Problem ID: {prob.problem_id}, First few chars of GT: {prob.ground_truth[:15]}")
    except Exception as e:
        print(f"Error loading Code dataset: {e}")
    finally:
        if os.path.exists(code_test_file):
            os.remove(code_test_file)

    # Test File Not Found
    print("\n--- Testing File Not Found ---")
    try:
        load_math_dataset("non_existent_file.jsonl")
    except FileNotFoundError as e:
        print(f"Caught expected error: {e}")

    # Test JSON Decode Error
    print("\n--- Testing JSON Decode Error ---")
    corrupt_jsonl_file = "corrupt.jsonl"
    with open(corrupt_jsonl_file, "w", encoding="utf-8") as f:
        f.write("{'id': 'p1', 'problem': 'test', 'solution': 'ans'}\n") # Invalid JSON
    try:
        load_math_dataset(corrupt_jsonl_file)
    except json.JSONDecodeError as e:
        print(f"Caught expected error: {e}")
    finally:
        if os.path.exists(corrupt_jsonl_file):
            os.remove(corrupt_jsonl_file)

