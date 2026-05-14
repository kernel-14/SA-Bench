"""
Data loading utilities for SCoRe experiments.

Handles:
- MATH dataset (Hendrycks et al., 2021)
- MBPP dataset (Austin et al., 2021)
- HumanEval dataset (Chen et al., 2021)

Train/test splits from Section 6 of the paper:
- MATH: augment training set with 4500 problems from test set,
  report results on remaining 500 problems (MATH500)
- Code: train on MBPP, report results on HumanEval
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MathProblem:
    """A MATH dataset problem."""
    problem: str
    solution: str
    answer: str
    level: str
    type: str
    unique_id: str


@dataclass
class CodeProblem:
    """A code generation problem."""
    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str
    test_cases: List[str]


def load_math_dataset(
    data_dir: str,
    split: str = "train",
) -> List[MathProblem]:
    """
    Load the MATH dataset.
    
    The MATH dataset is organized as:
    data_dir/
        train/
            algebra/
                problem1.json
                ...
            ...
        test/
            ...
    
    Each JSON file contains:
    {
        "problem": "...",
        "level": "Level X",
        "type": "Algebra",
        "solution": "...",
        "answer": "..."  (extracted from solution)
    }
    
    Args:
        data_dir: Path to MATH dataset directory
        split: "train" or "test"
        
    Returns:
        List of MathProblem objects
    """
    problems = []
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"MATH dataset not found at {split_dir}")
    
    for category in os.listdir(split_dir):
        category_dir = os.path.join(split_dir, category)
        if not os.path.isdir(category_dir):
            continue
        
        for filename in os.listdir(category_dir):
            if not filename.endswith(".json"):
                continue
            
            filepath = os.path.join(category_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            
            # Extract answer from solution
            answer = extract_answer_from_solution(data.get("solution", ""))
            
            problem = MathProblem(
                problem=data["problem"],
                solution=data["solution"],
                answer=answer,
                level=data.get("level", ""),
                type=data.get("type", category),
                unique_id=f"{category}/{filename}",
            )
            problems.append(problem)
    
    return problems


def extract_answer_from_solution(solution: str) -> str:
    """
    Extract the final answer from a MATH solution.
    Looks for \\boxed{...} pattern.
    
    Args:
        solution: Solution string
        
    Returns:
        Extracted answer string
    """
    # Look for \boxed{...}
    pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(pattern, solution)
    if match:
        return match.group(1).strip()
    
    return ""


def create_math500_split(
    all_test_problems: List[MathProblem],
    n_eval: int = 500,
    seed: int = 42,
) -> Tuple[List[MathProblem], List[MathProblem]]:
    """
    Create the MATH500 evaluation split.
    
    From Section 6: "following Lightman et al. (2023), we augment the MATH
    training set with 4500 problems from the test set, and report results
    on the remaining 500 problems (MATH500)"
    
    Args:
        all_test_problems: All test problems
        n_eval: Number of problems to keep for evaluation (500)
        seed: Random seed
        
    Returns:
        Tuple of (train_augmentation, eval_problems)
    """
    import random
    rng = random.Random(seed)
    
    indices = list(range(len(all_test_problems)))
    rng.shuffle(indices)
    
    eval_indices = set(indices[:n_eval])
    
    eval_problems = [all_test_problems[i] for i in range(len(all_test_problems)) if i in eval_indices]
    train_aug = [all_test_problems[i] for i in range(len(all_test_problems)) if i not in eval_indices]
    
    return train_aug, eval_problems


def load_mbpp_dataset(data_path: str) -> List[CodeProblem]:
    """
    Load the MBPP dataset.
    
    MBPP format (JSONL):
    {
        "task_id": 1,
        "text": "Write a function to find the similar elements...",
        "code": "def similar_elements(test_tup1, test_tup2): ...",
        "test_list": ["assert similar_elements(...) == ..."],
        "test_setup_code": "",
        "challenge_test_list": []
    }
    
    Args:
        data_path: Path to MBPP JSONL file
        
    Returns:
        List of CodeProblem objects
    """
    problems = []
    
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            
            problem = CodeProblem(
                task_id=str(data["task_id"]),
                prompt=data["text"],
                canonical_solution=data["code"],
                test="\n".join(data.get("test_list", [])),
                entry_point="",
                test_cases=data.get("test_list", []),
            )
            problems.append(problem)
    
    return problems


def load_humaneval_dataset(data_path: str) -> List[CodeProblem]:
    """
    Load the HumanEval dataset.
    
    HumanEval format (JSONL):
    {
        "task_id": "HumanEval/0",
        "prompt": "from typing import List\ndef has_close_elements...",
        "canonical_solution": "    for idx, elem in enumerate(numbers)...",
        "test": "def check(candidate):\n    assert...",
        "entry_point": "has_close_elements"
    }
    
    Args:
        data_path: Path to HumanEval JSONL file
        
    Returns:
        List of CodeProblem objects
    """
    problems = []
    
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract test cases from the test function
            test_cases = extract_humaneval_test_cases(data["test"], data["entry_point"])
            
            problem = CodeProblem(
                task_id=data["task_id"],
                prompt=data["prompt"],
                canonical_solution=data["canonical_solution"],
                test=data["test"],
                entry_point=data["entry_point"],
                test_cases=test_cases,
            )
            problems.append(problem)
    
    return problems


def extract_humaneval_test_cases(test_str: str, entry_point: str) -> List[str]:
    """
    Extract individual test cases from HumanEval test function.
    
    Args:
        test_str: Test function string
        entry_point: Function name being tested
        
    Returns:
        List of assert statements
    """
    # Find all assert statements
    assert_pattern = r"assert\s+.*"
    matches = re.findall(assert_pattern, test_str)
    
    # Replace 'candidate' with the actual function name
    test_cases = [m.replace("candidate", entry_point) for m in matches]
    
    return test_cases


def load_mbpp_repair_dataset(data_path: str) -> List[Dict]:
    """
    Load the MBPP-R (repair) dataset used for offline evaluation.
    
    From Section 6: "MBPP-R, an offline repair task that requires correcting
    incorrect first-attempt programs generated from PaLM 2"
    
    This dataset contains pre-generated incorrect programs that need to be repaired.
    
    Args:
        data_path: Path to MBPP-R dataset
        
    Returns:
        List of repair examples
    """
    examples = []
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    for item in data:
        examples.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "incorrect_solution": item["incorrect_solution"],
            "test_cases": item["test_cases"],
        })
    
    return examples


def prepare_math_training_data(
    train_problems: List[MathProblem],
    train_aug_problems: List[MathProblem],
    from_prompts,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Prepare MATH training data.
    
    Args:
        train_problems: Original training problems
        train_aug_problems: Augmentation problems from test set
        from_prompts: Function to build prompts
        
    Returns:
        Tuple of (problems, ground_truths, first_prompts)
    """
    all_problems = train_problems + train_aug_problems
    
    problems = [p.problem for p in all_problems]
    ground_truths = [p.answer for p in all_problems]
    first_prompts = [from_prompts(p.problem) for p in all_problems]
    
    return problems, ground_truths, first_prompts


def prepare_code_training_data(
    mbpp_problems: List[CodeProblem],
    from_prompts,
) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Prepare MBPP training data.
    
    Args:
        mbpp_problems: MBPP problems
        from_prompts: Function to build prompts
        
    Returns:
        Tuple of (problems, test_cases_list, first_prompts)
    """
    problems = [p.prompt for p in mbpp_problems]
    test_cases_list = [p.test_cases for p in mbpp_problems]
    first_prompts = [from_prompts(p.prompt, p.test_cases) for p in mbpp_problems]
    
    return problems, test_cases_list, first_prompts
