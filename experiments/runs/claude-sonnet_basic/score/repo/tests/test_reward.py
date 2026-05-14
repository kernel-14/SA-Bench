"""
Unit tests for reward functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reward import (
    extract_math_answer,
    normalize_math_answer,
    math_reward,
    compute_shaped_reward,
    compute_total_reward_stage2,
)
from src.evaluation import compute_self_correction_metrics


def test_extract_math_answer():
    """Test answer extraction from MATH responses."""
    # Standard format
    response = "The answer is 3. Final Answer: The final answer is $3$. I hope it is correct."
    assert extract_math_answer(response) == "3"
    
    # Boxed format
    response2 = "Therefore, the answer is \\boxed{42}."
    assert extract_math_answer(response2) == "42"
    
    # No answer
    response3 = "I don't know."
    assert extract_math_answer(response3) is None


def test_math_reward():
    """Test binary math reward."""
    # Correct answer
    response = "Final Answer: The final answer is $3$. I hope it is correct."
    assert math_reward(response, "3") == 1.0
    
    # Wrong answer
    assert math_reward(response, "4") == 0.0
    
    # No answer
    assert math_reward("I don't know.", "3") == 0.0


def test_shaped_reward():
    """Test reward shaping bonus."""
    alpha = 10.0
    
    # i->c transition: r1=0, r2=1 -> bonus = alpha * (1 - 0) = alpha
    bonus = compute_shaped_reward(0.0, 1.0, alpha)
    assert bonus == alpha, f"Expected {alpha}, got {bonus}"
    
    # c->i transition: r1=1, r2=0 -> bonus = alpha * (0 - 1) = -alpha
    bonus = compute_shaped_reward(1.0, 0.0, alpha)
    assert bonus == -alpha, f"Expected {-alpha}, got {bonus}"
    
    # No change: r1=1, r2=1 -> bonus = 0
    bonus = compute_shaped_reward(1.0, 1.0, alpha)
    assert bonus == 0.0, f"Expected 0.0, got {bonus}"
    
    # No change: r1=0, r2=0 -> bonus = 0
    bonus = compute_shaped_reward(0.0, 0.0, alpha)
    assert bonus == 0.0, f"Expected 0.0, got {bonus}"


def test_total_reward_stage2():
    """Test total reward computation for Stage II."""
    alpha = 10.0
    
    # i->c: r1=0, r2=1
    r1, r2_shaped = compute_total_reward_stage2(0.0, 1.0, alpha)
    assert r1 == 0.0
    assert r2_shaped == 1.0 + alpha  # r2 + alpha * (r2 - r1) = 1 + 10 = 11
    
    # c->i: r1=1, r2=0
    r1, r2_shaped = compute_total_reward_stage2(1.0, 0.0, alpha)
    assert r1 == 1.0
    assert r2_shaped == 0.0 - alpha  # r2 + alpha * (r2 - r1) = 0 - 10 = -10


def test_evaluation_metrics():
    """Test self-correction evaluation metrics."""
    # Perfect self-correction: all i->c
    first_rewards = [0.0, 0.0, 0.0, 0.0]
    second_rewards = [1.0, 1.0, 1.0, 1.0]
    
    metrics = compute_self_correction_metrics(first_rewards, second_rewards)
    
    assert metrics["accuracy_t1"] == 0.0
    assert metrics["accuracy_t2"] == 1.0
    assert metrics["delta_t1_t2"] == 1.0
    assert metrics["delta_i_to_c"] == 1.0
    assert metrics["delta_c_to_i"] == 0.0
    
    # No self-correction: all c->i
    first_rewards = [1.0, 1.0, 1.0, 1.0]
    second_rewards = [0.0, 0.0, 0.0, 0.0]
    
    metrics = compute_self_correction_metrics(first_rewards, second_rewards)
    
    assert metrics["accuracy_t1"] == 1.0
    assert metrics["accuracy_t2"] == 0.0
    assert metrics["delta_t1_t2"] == -1.0
    assert metrics["delta_i_to_c"] == 0.0
    assert metrics["delta_c_to_i"] == 1.0
    
    # Mixed: some i->c, some c->i
    first_rewards = [0.0, 0.0, 1.0, 1.0]
    second_rewards = [1.0, 0.0, 1.0, 0.0]
    
    metrics = compute_self_correction_metrics(first_rewards, second_rewards)
    
    assert metrics["accuracy_t1"] == 0.5
    assert metrics["accuracy_t2"] == 0.5
    assert metrics["delta_t1_t2"] == 0.0
    assert metrics["delta_i_to_c"] == 0.5  # 1 out of 2 incorrect became correct
    assert metrics["delta_c_to_i"] == 0.5  # 1 out of 2 correct became incorrect


if __name__ == "__main__":
    test_extract_math_answer()
    print("test_extract_math_answer passed")
    
    test_math_reward()
    print("test_math_reward passed")
    
    test_shaped_reward()
    print("test_shaped_reward passed")
    
    test_total_reward_stage2()
    print("test_total_reward_stage2 passed")
    
    test_evaluation_metrics()
    print("test_evaluation_metrics passed")
    
    print("\nAll tests passed!")
