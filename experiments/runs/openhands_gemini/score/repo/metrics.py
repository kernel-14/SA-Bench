
from typing import List, Tuple, Callable

def calculate_metrics(
    eval_triplets: List[Tuple[str, str, str]], # List of (ground_truth, t1_response, t2_response)
    oracle_reward: Callable[[str, str], int] # Function that returns 1 if correct, 0 if incorrect
) -> dict:
    """
    Calculates the self-correction metrics described in the paper.

    Args:
        eval_triplets: A list of tuples, where each tuple contains:
                       (ground_truth_answer, turn1_response, turn2_response).
        oracle_reward: A callable function that takes (model_output, ground_truth)
                       and returns 1 for correct, 0 for incorrect.

    Returns:
        A dictionary containing the calculated metrics.
    """
    num_problems = len(eval_triplets)
    if num_problems == 0:
        return {
            "Accuracy@t1": 0.0,
            "Accuracy@t2": 0.0,
            "Delta(t1, t2)": 0.0,
            "i->c(t1, t2)": 0.0,
            "c->i(t1, t2)": 0.0,
        }

    t1_correct_count = 0
    t2_correct_count = 0
    incorrect_to_correct_count = 0 # i->c
    correct_to_incorrect_count = 0 # c->i

    for gt, t1_res, t2_res in eval_triplets:
        is_t1_correct = oracle_reward(t1_res, gt)
        is_t2_correct = oracle_reward(t2_res, gt)

        if is_t1_correct:
            t1_correct_count += 1
        if is_t2_correct:
            t2_correct_count += 1

        if not is_t1_correct and is_t2_correct:
            incorrect_to_correct_count += 1
        elif is_t1_correct and not is_t2_correct:
            correct_to_incorrect_count += 1

    acc_t1 = t1_correct_count / num_problems
    acc_t2 = t2_correct_count / num_problems
    delta_t1_t2 = acc_t2 - acc_t1
    ic_t1_t2 = incorrect_to_correct_count / num_problems
    ci_t1_t2 = correct_to_incorrect_count / num_problems

    return {
        "Accuracy@t1": acc_t1 * 100,
        "Accuracy@t2": acc_t2 * 100,
        "Delta(t1, t2)": delta_t1_t2 * 100,
        "i->c(t1, t2)": ic_t1_t2 * 100,
        "c->i(t1, t2)": ci_t1_t2 * 100,
    }

# --- Oracle Reward Functions (placeholders, actual implementation depends on dataset) ---
# These functions would need to be implemented based on the specific format
# of the dataset's ground truth and how model outputs need to be parsed and evaluated.

def math_oracle_reward(model_output: str, ground_truth: str) -> int:
    """
    Placeholder for MATH oracle reward function.
    This would typically involve parsing the model's 'Final Answer' and comparing it
    to the ground truth.
    """
    # Example: Simple string matching of the final answer part
    # In a real scenario, this would involve a robust math expression parser/evaluator.
    try:
        model_answer_tag = "Final Answer: The final answer is "
        if model_answer_tag in model_output:
            model_answer_start = model_output.rfind(model_answer_tag) + len(model_answer_tag)
            model_answer_end = model_output.rfind(". I hope it is correct.")
            if model_answer_end == -1: # Handle cases where the sentence might be cut short
                model_answer_end = len(model_output)
            parsed_model_answer = model_output[model_answer_start:model_answer_end].strip().replace('$', '')
            parsed_ground_truth = ground_truth.strip().replace('$', '')
            return 1 if parsed_model_answer == parsed_ground_truth else 0
        return 0
    except Exception:
        return 0


def code_oracle_reward(model_output: str, ground_truth: str) -> int:
    """
    Placeholder for Code oracle reward function.
    This would typically involve executing the generated code against test cases.
    For HumanEval/MBPP, the ground truth often includes test cases.
    """
    # This is a complex operation requiring a secure execution environment.
    # For now, it's a placeholder. In a real scenario, you'd use a sandboxed executor
    # to run `model_output` code and check if it passes `ground_truth` tests.
    # The `ground_truth` for coding problems is usually the canonical solution
    # and test cases, not just a single answer.
    # For this reproduction, we'll assume a simplified check or rely on an external
    # test harness. A basic placeholder for now might check for a specific output pattern.
    if "def " in model_output and "return" in model_output: # Very basic structural check
        # This needs a real code execution environment.
        # For simplicity in a static reproduction, we might simulate success.
        # E.g., if ground_truth is a set of tests, we'd run model_output against them.
        return 1 # Assume correct for now
    return 0

