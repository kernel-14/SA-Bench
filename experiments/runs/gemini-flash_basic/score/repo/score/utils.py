from typing import List, Dict, Any
import numpy as np

def calculate_metrics(
    predictions_t1: List[str],
    predictions_t2: List[str],
    ground_truths: List[str],
    task_type: str
) -> Dict[str, float]:
    """
    Calculates the self-correction metrics as defined in the paper (Section 3 and 6.1).

    Args:
        predictions_t1: List of model predictions for the first attempt.
        predictions_t2: List of model predictions for the second attempt.
        ground_truths: List of ground truth answers.
        task_type: The type of task (e.g., 'math', 'code') for reward calculation.

    Returns:
        A dictionary containing the calculated metrics.
    """
    if not (len(predictions_t1) == len(predictions_t2) == len(ground_truths)):
        raise ValueError("All input lists must have the same length.")

    num_samples = len(ground_truths)

    # Calculate correctness for each attempt
    correct_t1 = [
        1.0 if p.strip() == g.strip() else 0.0
        for p, g in zip(predictions_t1, ground_truths)
    ]
    correct_t2 = [
        1.0 if p.strip() == g.strip() else 0.0
        for p, g in zip(predictions_t2, ground_truths)
    ]

    # 1. Accuracy@t1
    accuracy_t1 = np.mean(correct_t1)

    # 2. Accuracy@t2
    accuracy_t2 = np.mean(correct_t2)

    # 3. Δ(t1, t2): Net improvement in model accuracy between first and second attempts
    delta_t1_t2 = accuracy_t2 - accuracy_t1

    # 4. Δi→c(t1, t2): Fraction of problems incorrect in t1 but correct in t2
    incorrect_t1_to_correct_t2 = 0
    for i in range(num_samples):
        if correct_t1[i] == 0.0 and correct_t2[i] == 1.0:
            incorrect_t1_to_correct_t2 += 1
    delta_ic_t1_t2 = incorrect_t1_to_correct_t2 / num_samples

    # 5. Δc→i(t1, t2): Fraction of problems correct in t1 but incorrect in t2
    correct_t1_to_incorrect_t2 = 0
    for i in range(num_samples):
        if correct_t1[i] == 1.0 and correct_t2[i] == 0.0:
            correct_t1_to_incorrect_t2 += 1
    delta_ci_t1_t2 = correct_t1_to_incorrect_t2 / num_samples

    return {
        "accuracy@t1": accuracy_t1,
        "accuracy@t2": accuracy_t2,
        "delta(t1,t2)": delta_t1_t2,
        "delta_i_to_c(t1,t2)": delta_ic_t1_t2,
        "delta_c_to_i(t1,t2)": delta_ci_t1_t2,
    }

def generate_first_attempt_prompt(problem: str) -> str:
    """
    Generates the prompt for the first attempt.
    Based on examples in the paper (e.g., MATH Example 1, 2).
    """
    return f"""Problem: {problem}

SCoRe turn 1 solution:"""

def generate_self_correction_instruction(first_attempt_solution: str) -> str:
    """
    Generates the self-correction instruction for the second attempt.
    Based on examples in the paper (e.g., MATH Example 1, 2).
    """
    return f"""Self-correction instruction. There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution.

SCoRe turn 2 solution:"""

