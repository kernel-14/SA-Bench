import numpy as np

def calculate_base_reward(y: str, y_star: str, task_type: str) -> float:
    """
    Calculates the base reward for a given response.
    This simulates the oracle reward ^r(y, y*) from the paper.

    Args:
        y: The model's generated response.
        y_star: The ground truth response.
        task_type: The type of task (e.g., 'math', 'code').

    Returns:
        1.0 if the response is correct, 0.0 otherwise.
    """
    if task_type == "math":
        # For math, we assume a direct string comparison or a more sophisticated
        # math expression evaluator. For now, a simple equality check.
        # In a real implementation, this would involve parsing and evaluating.
        return 1.0 if y.strip() == y_star.strip() else 0.0
    elif task_type == "code":
        # For code, this would involve executing the code and running test cases.
        # For this static reproduction, we simulate it with an equality check.
        # In a real implementation, this would involve a robust code execution environment.
        return 1.0 if y.strip() == y_star.strip() else 0.0
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def calculate_shaped_reward(
    reward_y1: float,
    reward_y2: float,
    alpha: float = 2.0
) -> float:
    """
    Calculates the shaped reward for the second attempt, as described in Section 5.2.
    ^b(y2 | y1, y*) = alpha * (^r(y2, y*) - ^r(y1, y*))

    Args:
        reward_y1: The base reward for the first attempt (y1).
        reward_y2: The base reward for the second attempt (y2).
        alpha: A positive constant multiplier, ideally larger than 1.0.

    Returns:
        The shaped reward for the second attempt.
    """
    return alpha * (reward_y2 - reward_y1)

def get_total_stage_ii_reward(
    reward_y1: float,
    reward_y2: float,
    alpha: float = 2.0
) -> float:
    """
    Calculates the total reward for Stage II, incorporating the shaped reward.
    The objective is to maximize E[reward_y1 + (reward_y2 + shaped_reward_y2)]
    This is equivalent to: E[reward_y1 + reward_y2 + alpha * (reward_y2 - reward_y1)]

    Args:
        reward_y1: The base reward for the first attempt.
        reward_y2: The base reward for the second attempt.
        alpha: The alpha parameter for reward shaping.

    Returns:
        The total reward for the given two-turn rollout.
    """
    shaped_bonus = calculate_shaped_reward(reward_y1, reward_y2, alpha)
    return reward_y1 + reward_y2 + shaped_bonus

