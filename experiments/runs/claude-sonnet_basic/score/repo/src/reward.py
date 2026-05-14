"""
Reward functions for SCoRe training.
Supports MATH (answer matching) and code (test case execution).
"""

import re
import subprocess
import tempfile
import os
from typing import Optional


def extract_math_answer(response: str) -> Optional[str]:
    """
    Extract the final answer from a MATH response.
    Looks for the pattern: "Final Answer: The final answer is $<answer>$. I hope it is correct."
    """
    # Try to find the standard format
    pattern = r"Final Answer: The final answer is \$?(.*?)\$?\. I hope it is correct"
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for boxed answer
    pattern2 = r"\\boxed\{([^}]+)\}"
    match2 = re.search(pattern2, response)
    if match2:
        return match2.group(1).strip()
    
    return None


def normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    if answer is None:
        return ""
    # Remove whitespace and common formatting
    answer = answer.strip()
    answer = re.sub(r'\s+', '', answer)
    # Remove dollar signs
    answer = answer.replace('$', '')
    # Normalize fractions
    answer = answer.replace('\\frac', 'frac')
    return answer.lower()


def math_reward(response: str, ground_truth: str) -> float:
    """
    Binary reward for MATH: 1.0 if answer matches ground truth, 0.0 otherwise.
    
    Args:
        response: Model's response string
        ground_truth: Ground truth answer string
        
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    predicted = extract_math_answer(response)
    if predicted is None:
        return 0.0
    
    pred_norm = normalize_math_answer(predicted)
    gt_norm = normalize_math_answer(ground_truth)
    
    return 1.0 if pred_norm == gt_norm else 0.0


def extract_code_from_response(response: str) -> str:
    """Extract Python code from a model response."""
    # Try to find code in markdown code blocks
    pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try [BEGIN] ... [DONE] format (MBPP style)
    pattern2 = r"\[BEGIN\](.*?)\[DONE\]"
    match2 = re.search(pattern2, response, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    
    # Return the whole response as code
    return response.strip()


def code_reward(response: str, test_cases: list, timeout: int = 10) -> float:
    """
    Binary reward for code generation: 1.0 if all test cases pass, 0.0 otherwise.
    
    Args:
        response: Model's code response
        test_cases: List of test case strings (assert statements)
        timeout: Execution timeout in seconds
        
    Returns:
        1.0 if all tests pass, 0.0 otherwise
    """
    code = extract_code_from_response(response)
    
    # Build test script
    test_script = code + "\n\n"
    for test in test_cases:
        test_script += test + "\n"
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            tmp_path = f.name
        
        result = subprocess.run(
            ['python', tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        os.unlink(tmp_path)
        return 1.0 if result.returncode == 0 else 0.0
        
    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return 0.0
    except Exception:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return 0.0


def compute_shaped_reward(
    r1: float,
    r2: float,
    alpha: float = 10.0
) -> float:
    """
    Compute the shaped reward bonus for the second attempt.
    
    From the paper (Section 5.2):
    b(y2 | y1, y*) = alpha * (r(y2, y*) - r(y1, y*))
    
    This bonus:
    - Rewards transitions that flip incorrect -> correct (positive bonus)
    - Penalizes transitions that flip correct -> incorrect (negative penalty)
    - Is zero when both attempts have the same correctness
    
    Args:
        r1: Reward for first attempt (0 or 1)
        r2: Reward for second attempt (0 or 1)
        alpha: Multiplier for the bonus (paper uses alpha=10)
        
    Returns:
        Shaped reward bonus
    """
    return alpha * (r2 - r1)


def compute_total_reward_stage2(
    r1: float,
    r2: float,
    alpha: float = 10.0
) -> tuple:
    """
    Compute total rewards for Stage II training.
    
    Returns (reward_t1, reward_t2_shaped) where:
    - reward_t1 = r1 (standard reward for first attempt)
    - reward_t2_shaped = r2 + alpha * (r2 - r1) (shaped reward for second attempt)
    
    Args:
        r1: Reward for first attempt
        r2: Reward for second attempt
        alpha: Shaping multiplier
        
    Returns:
        Tuple of (reward_t1, reward_t2_shaped)
    """
    bonus = compute_shaped_reward(r1, r2, alpha)
    reward_t2_shaped = r2 + bonus
    return r1, reward_t2_shaped
