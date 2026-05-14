"""
Reward functions for MATH and code generation tasks.

The paper uses binary rewards during training: 1 if the model's answer matches
the ground truth (for MATH) or passes all test cases (for coding), 0 otherwise.
"""

import ast
import contextlib
import io
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
from typing import List, Optional


# ---------------------------------------------------------------------------
# MATH reward
# ---------------------------------------------------------------------------

def _normalize_math_answer(answer: str) -> str:
    """Normalize a math answer string for comparison."""
    answer = answer.strip()
    # Remove surrounding dollar signs
    answer = re.sub(r"^\$+|\$+$", "", answer).strip()
    # Remove \text{}, \mathrm{}, etc.
    answer = re.sub(r"\\(?:text|mathrm|mathbf|mathit|mathsf)\{([^}]*)\}", r"\1", answer)
    # Normalize fractions: \frac{a}{b} -> a/b
    answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", answer)
    # Remove spaces
    answer = answer.replace(" ", "")
    # Lowercase
    answer = answer.lower()
    return answer


def _extract_final_answer_math(response: str) -> Optional[str]:
    """
    Extract the final answer from a MATH response.

    The paper instructs the model to end with:
    "Final Answer: The final answer is $answer$. I hope it is correct."
    """
    # Try the paper's specific format first
    match = re.search(
        r"[Ff]inal [Aa]nswer[:\s]+[Tt]he final answer is\s*\$?([^$.\n]+)\$?",
        response,
    )
    if match:
        return match.group(1).strip()

    # Try boxed answer
    match = re.search(r"\\boxed\{(.+?)\}", response)
    if match:
        return match.group(1).strip()

    # Try "the answer is X"
    match = re.search(r"[Tt]he (?:final )?answer is[:\s]+(.+?)(?:\.|$)", response)
    if match:
        return match.group(1).strip()

    return None


def math_reward(response: str, ground_truth: str) -> float:
    """
    Binary reward for MATH: 1.0 if the extracted answer matches ground truth.

    Uses string normalization for comparison. For a more robust implementation,
    sympy-based symbolic comparison can be used.
    """
    predicted = _extract_final_answer_math(response)
    if predicted is None:
        return 0.0

    pred_norm = _normalize_math_answer(predicted)
    gt_norm = _normalize_math_answer(ground_truth)

    if pred_norm == gt_norm:
        return 1.0

    # Try numeric comparison
    try:
        pred_val = float(eval(pred_norm.replace("^", "**")))
        gt_val = float(eval(gt_norm.replace("^", "**")))
        if abs(pred_val - gt_val) < 1e-6:
            return 1.0
    except Exception:
        pass

    # Try sympy symbolic comparison
    try:
        from sympy import simplify, sympify
        pred_expr = sympify(pred_norm)
        gt_expr = sympify(gt_norm)
        if simplify(pred_expr - gt_expr) == 0:
            return 1.0
    except Exception:
        pass

    return 0.0


# ---------------------------------------------------------------------------
# Code reward
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def _extract_code_block(response: str) -> str:
    """Extract Python code from a model response."""
    # Try markdown code block
    match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try [BEGIN] ... [DONE] format (MBPP)
    match = re.search(r"\[BEGIN\](.*?)(?:\[DONE\]|$)", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return the whole response as code (HumanEval style)
    return response.strip()


def _run_code_with_tests(code: str, test_list: List[str], timeout: int = 10) -> bool:
    """
    Execute code with test assertions in a subprocess.

    Returns True if all tests pass, False otherwise.
    """
    test_code = "\n".join(test_list)
    full_code = f"{code}\n\n{test_code}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _run_humaneval_tests(code: str, test_code: str, entry_point: str, timeout: int = 10) -> bool:
    """
    Execute HumanEval solution with the provided test harness.

    HumanEval test code uses check(candidate) pattern.
    """
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def code_reward_mbpp(response: str, test_list: List[str]) -> float:
    """
    Binary reward for MBPP: 1.0 if the extracted code passes all test cases.
    """
    code = _extract_code_block(response)
    if not code:
        return 0.0
    passed = _run_code_with_tests(code, test_list)
    return 1.0 if passed else 0.0


def code_reward_humaneval(response: str, test_code: str, entry_point: str) -> float:
    """
    Binary reward for HumanEval: 1.0 if the extracted code passes all tests.
    """
    code = _extract_code_block(response)
    if not code:
        return 0.0
    passed = _run_humaneval_tests(code, test_code, entry_point)
    return 1.0 if passed else 0.0


# ---------------------------------------------------------------------------
# Unified reward interface
# ---------------------------------------------------------------------------

def compute_reward(response: str, problem_metadata: dict, task: str) -> float:
    """
    Compute binary reward for a given task.

    Args:
        response: model-generated response string
        problem_metadata: dict containing task-specific fields
            - math: {"answer": str}
            - mbpp: {"test_list": List[str]}
            - humaneval: {"test": str, "entry_point": str}
        task: "math" | "mbpp" | "humaneval"

    Returns:
        float in {0.0, 1.0}
    """
    if task == "math":
        return math_reward(response, problem_metadata["answer"])
    elif task == "mbpp":
        return code_reward_mbpp(response, problem_metadata["test_list"])
    elif task == "humaneval":
        return code_reward_humaneval(
            response,
            problem_metadata["test"],
            problem_metadata["entry_point"],
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# Shaped reward (Section 5.2)
# ---------------------------------------------------------------------------

def shaped_reward_t2(
    reward_t1: float,
    reward_t2: float,
    alpha: float = 10.0,
) -> float:
    """
    Compute the shaped reward bonus for the second attempt.

    From Section 5.2:
        b(y2 | y1, y*) = α * (r(y2, y*) - r(y1, y*))

    This bonus:
    - Rewards transitions that flip incorrect → correct (+α)
    - Penalizes transitions that flip correct → incorrect (-α)
    - Is zero when both attempts have the same correctness

    The total second-attempt reward used in Stage II is:
        r̂(y2, y*) + b(y2 | y1, y*)
    """
    return alpha * (reward_t2 - reward_t1)
