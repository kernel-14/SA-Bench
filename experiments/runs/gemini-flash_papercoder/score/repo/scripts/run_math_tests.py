import logging
from typing import List, Dict, Any, Tuple

from reward_utils import MathRewardFunction  # Import the class to be tested

# Configure logging for better visibility during testing
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_math_reward_tests() -> Tuple[int, int]:
    """
    Runs a series of tests for the MathRewardFunction to ensure it correctly
    calculates rewards based on model output and ground truth.

    Returns:
        A tuple (passed_tests, failed_tests) indicating the count of passed and failed tests.
    """
    logger.info("--- Starting MathRewardFunction Unit Tests ---")

    math_reward_calculator = MathRewardFunction()
    passed_tests = 0
    failed_tests = 0

    # Define test cases. Each dictionary contains:
    # 'model_output': A string simulating the LLM's response, adhering to the expected format.
    # 'ground_truth': The canonical correct answer string.
    # 'expected_reward': The binary reward (1.0 for correct, 0.0 for incorrect) that
    #                    MathRewardFunction should return.
    # 'description': A brief description of the test case.
    test_cases: List[Dict[str, Any]] = [
        # --- Exact Matches (Numerical) ---
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{42}. I hope it is correct.",
            "ground_truth": "42",
            "expected_reward": 1.0,
            "description": "Exact integer match",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{3.14}. I hope it is correct.",
            "ground_truth": "3.14",
            "expected_reward": 1.0,
            "description": "Exact float match",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{-10}. I hope it is correct.",
            "ground_truth": "-10",
            "expected_reward": 1.0,
            "description": "Exact negative integer match",
        },
        # --- Numerical Equivalences ---
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{0.5}. I hope it is correct.",
            "ground_truth": "1/2",
            "expected_reward": 1.0,
            "description": "Float vs fraction equivalence",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{1/2}. I hope it is correct.",
            "ground_truth": "0.5",
            "expected_reward": 1.0,
            "description": "Fraction vs float equivalence",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{2/4}. I hope it is correct.",
            "ground_truth": "1/2",
            "expected_reward": 1.0,
            "description": "Simplified fraction equivalence",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{3.0}. I hope it is correct.",
            "ground_truth": "3",
            "expected_reward": 1.0,
            "description": "Float with trailing zero vs integer equivalence",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{sqrt(9)}. I hope it is correct.",
            "ground_truth": "3",
            "expected_reward": 1.0,
            "description": "Symbolic expression (sqrt) vs integer equivalence",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{2*pi}. I hope it is correct.",
            "ground_truth": "6.283185307179586",  # Approx 2*pi
            "expected_reward": 1.0,
            "description": "Symbolic constant (pi) numerical equivalence",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{1.234567891234}. I hope it is correct.",
            "ground_truth": "1.234567891235",
            "expected_reward": 1.0,
            "description": "Close float values (within tolerance)",
        },
        # --- Symbolic Equivalences ---
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{x+y}. I hope it is correct.",
            "ground_truth": "y+x",
            "expected_reward": 1.0,
            "description": "Algebraic commutative equivalence",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{2*x + x}. I hope it is correct.",
            "ground_truth": "3*x",
            "expected_reward": 1.0,
            "description": "Algebraic simplification equivalence",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{(x+1)*(x-1)}. I hope it is correct.",
            "ground_truth": "x**2 - 1",
            "expected_reward": 1.0,
            "description": "Algebraic expansion equivalence",
        },
        # --- Incorrect Answers ---
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{5}. I hope it is correct.",
            "ground_truth": "4",
            "expected_reward": 0.0,
            "description": "Incorrect integer value",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{1/3}. I hope it is correct.",
            "ground_truth": "0.33",
            "expected_reward": 0.0,  # 1/3 is not exactly 0.33
            "description": "Incorrect float approximation",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{x+z}. I hope it is correct.",
            "ground_truth": "y+x",
            "expected_reward": 0.0,
            "description": "Incorrect algebraic expression (different variable)",
        },
        # --- Malformed Model Outputs (should result in 0.0 reward) ---
        {
            "model_output": "Solution: The final answer is 42. I hope it is correct.",
            "ground_truth": "42",
            "expected_reward": 0.0,
            "description": "Missing \\boxed{} format in model output",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{}. I hope it is correct.",
            "ground_truth": "42",
            "expected_reward": 0.0,
            "description": "Empty \\boxed{} in model output",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{invalid_math_expression}. I hope it is correct.",
            "ground_truth": "10",
            "expected_reward": 0.0,
            "description": "Unparsable math expression in \\boxed{}",
        },
        {
            "model_output": "Solution... No final answer was given.",
            "ground_truth": "10",
            "expected_reward": 0.0,
            "description": "No final answer pattern found at all",
        },
        # --- Malformed Ground Truth (should log error and result in 0.0 reward) ---
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{10}. I hope it is correct.",
            "ground_truth": "unparsable_gt",
            "expected_reward": 0.0,
            "description": "Unparsable ground truth",
        },
        # --- Complex Cases ---
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{sin(pi/2)}. I hope it is correct.",
            "ground_truth": "1",
            "expected_reward": 1.0,
            "description": "Trigonometric function equivalence",
        },
        {
            "model_output": "Solution... Final Answer: The final answer is \\boxed{2*x**2 + 4*x - 6}. I hope it is correct.",
            "ground_truth": "2*(x+3)*(x-1)",
            "expected_reward": 1.0,
            "description": "Polynomial factorization equivalence",
        },
    ]

    for i, test_case in enumerate(test_cases):
        model_output = test_case["model_output"]
        ground_truth = test_case["ground_truth"]
        expected_reward = test_case["expected_reward"]
        description = test_case["description"]

        actual_reward = math_reward_calculator.calculate_reward(model_output, ground_truth)

        if actual_reward == expected_reward:
            logger.info(f"PASS Test {i+1} ({description}): Model output '{model_output.split('Final Answer: ')[-1].strip()}' (GT: '{ground_truth}') -> Reward: {actual_reward} (Expected: {expected_reward})")
            passed_tests += 1
        else:
            logger.error(f"FAIL Test {i+1} ({description}): Model output '{model_output.split('Final Answer: ')[-1].strip()}' (GT: '{ground_truth}') -> Reward: {actual_reward} (Expected: {expected_reward})")
            failed_tests += 1

    logger.info(f"--- MathRewardFunction Tests Complete ---")
    logger.info(f"Total Tests: {len(test_cases)}, Passed: {passed_tests}, Failed: {failed_tests}")

    return passed_tests, failed_tests


if __name__ == "__main__":
    passed, failed = run_math_reward_tests()

    if failed > 0:
        logger.error(f"{failed} MathRewardFunction tests failed. Please review the logs.")
        import sys
        sys.exit(1)
    else:
        logger.info("All MathRewardFunction tests passed successfully!")
