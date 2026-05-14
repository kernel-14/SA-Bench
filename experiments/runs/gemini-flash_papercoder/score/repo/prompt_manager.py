"""
This module defines the PromptManager class, which is responsible for constructing
the specific prompts used for generating first and second attempts by the LLM.
It loads prompt templates from the configuration and formats them with problem
details and previous responses.
"""

from typing import Dict, Any, Literal

from config import Config


class PromptManager:
    """
    Manages the creation of prompts for different tasks and turns in the
    self-correction process. It uses templates defined in the Config.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the PromptManager with prompt templates from the configuration.

        Args:
            config: An instance of the Config class containing all experiment settings.
        """
        self.config: Config = config

        # Extract task-specific prompt instructions
        if self.config.prompts is None:
            raise ValueError("Prompt configurations are missing in config.yaml.")

        # MATH prompts
        self.math_first_turn_instruction: str = self.config.prompts.get(
            "math_first_turn", ""
        )
        self.math_second_turn_instruction: str = self.config.prompts.get(
            "math_second_turn_instruction", ""
        )

        # Code prompts (MBPP and HumanEval)
        self.mbpp_first_turn_prefix: str = self.config.prompts.get(
            "mbpp_first_turn_prefix", ""
        )
        # The paper uses the same instruction for HumanEval self-correction as MBPP
        self.code_second_turn_instruction: str = self.config.prompts.get(
            "mbpp_second_turn_instruction", ""
        )
        # HumanEval first turn prefix, expected to be empty for zero-shot
        self.human_eval_first_turn_prefix: str = self.config.prompts.get(
            "human_eval_first_turn_prefix", ""
        )

        # Hardcoded MBPP 3-shot examples from Appendix C
        self.mbpp_3_shot_examples: str = """
Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:
assert similar_elements((3, 4, 5, 6), (5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4), (5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13), (17, 15, 14, 13)) == (13, 14)
[BEGIN]
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return (res)
[DONE]

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
# [BEGIN]
import math
def is_not_prime(n):
    result = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
[DONE]

You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]
[BEGIN]
import heapq as hq
def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
[DONE]
        """.strip() + "\n" # Ensure consistent newline at end

    def get_first_turn_prompt(
        self,
        problem_text: str,
        task_context: Literal["math", "mbpp_train", "human_eval_eval", "mbpp_r_eval"],
    ) -> str:
        """
        Constructs the prompt for the first turn generation by the LLM.

        Args:
            problem_text: The raw problem description from the dataset.
            task_context: A string indicating the specific context (e.g., "math", "mbpp_train").

        Returns:
            The formatted string prompt for the first turn.

        Raises:
            ValueError: If an unsupported task_context is provided.
        """
        if task_context == "math":
            return f"{self.math_first_turn_instruction}\nProblem. {problem_text}"
        elif task_context in ["mbpp_train", "mbpp_r_eval"]:
            # For MBPP training and MBPP-R evaluation, use the 3-shot prompt.
            # The problem_text will be appended after the 3-shot examples.
            return f"{self.mbpp_first_turn_prefix}\n{self.mbpp_3_shot_examples}\n{problem_text}"
        elif task_context == "human_eval_eval":
            # HumanEval uses zero-shot prompting, so the prefix is empty.
            return f"{self.human_eval_first_turn_prefix}{problem_text}"
        else:
            raise ValueError(f"Unsupported task_context for first turn: {task_context}")

    def get_second_turn_prompt(
        self,
        problem_text: str,
        first_response: str,
        task_context: Literal["math", "mbpp_train", "human_eval_eval", "mbpp_r_eval"],
    ) -> str:
        """
        Constructs the prompt for the second turn (self-correction) generation by the LLM.

        Args:
            problem_text: The original problem description.
            first_response: The model's generated response from the first turn.
            task_context: A string indicating the specific context (e.g., "math", "mbpp_train").

        Returns:
            The formatted string prompt for the second turn.

        Raises:
            ValueError: If an unsupported task_context is provided.
        """
        if task_context == "math":
            # Example format from Appendix E for MATH
            return (
                f"Problem. {problem_text}\n"
                f"SCoRe turn 1 solution (incorrect).\n"
                f"{first_response}\n"
                f"Self-correction instruction. {self.math_second_turn_instruction}"
            )
        elif task_context in ["mbpp_train", "human_eval_eval", "mbpp_r_eval"]:
            # Example format from Appendix E for HumanEval/Code
            # Uses the general code self-correction instruction for all code tasks.
            return (
                f"# Problem: {problem_text}\n"
                f"# Turn 1 solution (incorrect):\n"
                f"{first_response}\n"
                f"# Self-correction instruction. {self.code_second_turn_instruction}"
            )
        else:
            raise ValueError(f"Unsupported task_context for second turn: {task_context}")


if __name__ == "__main__":
    # Example usage and testing
    # Create a dummy config for testing
    class MockConfig(Config):
        def __init__(self):
            super().__init__()
            self.prompts = {
                "math_first_turn": "You are a math expert.",
                "math_second_turn_instruction": "Correct the math error.",
                "mbpp_first_turn_prefix": "You are an expert Python programmer.",
                "mbpp_second_turn_instruction": "Correct the code error.",
                "human_eval_first_turn_prefix": "",
                "human_eval_second_turn_instruction": "Correct the code error.", # Redundant, but for completeness
            }

    print("--- Initializing PromptManager ---")
    mock_config = MockConfig()
    prompt_manager = PromptManager(mock_config)
    print("PromptManager initialized successfully.")

    # --- Test MATH prompts ---
    print("\n--- Testing MATH Prompts ---")
    math_problem_text = "What is the square root of 9?"
    math_first_response = "Solution: sqrt(9) = 2. Final Answer: \\boxed{2}."

    # First turn MATH
    math_first_turn_prompt = prompt_manager.get_first_turn_prompt(
        math_problem_text, "math"
    )
    print("MATH First Turn Prompt:")
    print(math_first_turn_prompt)
    assert (
        "You are a math expert." in math_first_turn_prompt
        and "Problem. What is the square root of 9?" in math_first_turn_prompt
    )

    # Second turn MATH
    math_second_turn_prompt = prompt_manager.get_second_turn_prompt(
        math_problem_text, math_first_response, "math"
    )
    print("\nMATH Second Turn Prompt:")
    print(math_second_turn_prompt)
    assert (
        "Problem. What is the square root of 9?" in math_second_turn_prompt
        and "SCoRe turn 1 solution (incorrect).\nSolution: sqrt(9) = 2. Final Answer: \\boxed{2}."
        in math_second_turn_prompt
        and "Self-correction instruction. Correct the math error."
        in math_second_turn_prompt
    )

    # --- Test Code (MBPP) prompts ---
    print("\n--- Testing MBPP (Train/R-Eval) Prompts ---")
    mbpp_problem_text = "Write a function to sum two numbers.\n[BEGIN]"
    mbpp_first_response = "def sum_two(a, b):\n    return a * b\n[DONE]"

    # First turn MBPP (training context)
    mbpp_first_turn_prompt = prompt_manager.get_first_turn_prompt(
        mbpp_problem_text, "mbpp_train"
    )
    print("MBPP First Turn Prompt (train context):")
    print(mbpp_first_turn_prompt)
    assert (
        "You are an expert Python programmer." in mbpp_first_turn_prompt
        and "[DONE]\n\nWrite a function to sum two numbers.\n[BEGIN]" in mbpp_first_turn_prompt
    )
    assert len(prompt_manager.mbpp_3_shot_examples) > 100 # Check if examples are present

    # Second turn MBPP (training context)
    mbpp_second_turn_prompt = prompt_manager.get_second_turn_prompt(
        mbpp_problem_text, mbpp_first_response, "mbpp_train"
    )
    print("\nMBPP Second Turn Prompt (train context):")
    print(mbpp_second_turn_prompt)
    assert (
        "# Problem: Write a function to sum two numbers.\n[BEGIN]" in mbpp_second_turn_prompt
        and "# Turn 1 solution (incorrect):\ndef sum_two(a, b):\n    return a * b\n[DONE]"
        in mbpp_second_turn_prompt
        and "# Self-correction instruction. Correct the code error."
        in mbpp_second_turn_prompt
    )

    # --- Test HumanEval prompts ---
    print("\n--- Testing HumanEval (Eval) Prompts ---")
    human_eval_problem_text = "def factorial(n):\n    \"\"\"Docstring\"\"\"\n"
    human_eval_first_response = "def factorial(n):\n    return n\n"

    # First turn HumanEval (evaluation context)
    human_eval_first_turn_prompt = prompt_manager.get_first_turn_prompt(
        human_eval_problem_text, "human_eval_eval"
    )
    print("HumanEval First Turn Prompt (eval context):")
    print(human_eval_first_turn_prompt)
    assert (
        human_eval_first_turn_prompt == human_eval_problem_text
    ) # Empty prefix means just the problem text

    # Second turn HumanEval (evaluation context)
    human_eval_second_turn_prompt = prompt_manager.get_second_turn_prompt(
        human_eval_problem_text, human_eval_first_response, "human_eval_eval"
    )
    print("\nHumanEval Second Turn Prompt (eval context):")
    print(human_eval_second_turn_prompt)
    assert (
        "# Problem: def factorial(n):\n    \"\"\"Docstring\"\"\"\n" in human_eval_second_turn_prompt
        and "# Turn 1 solution (incorrect):\ndef factorial(n):\n    return n\n"
        in human_eval_second_turn_prompt
        and "# Self-correction instruction. Correct the code error."
        in human_eval_second_turn_prompt
    )

    # --- Test unsupported task_context ---
    print("\n--- Testing unsupported task_context ---")
    try:
        prompt_manager.get_first_turn_prompt("dummy", "unsupported_task")
    except ValueError as e:
        print(f"Caught expected error for first turn: {e}")

    try:
        prompt_manager.get_second_turn_prompt("dummy", "dummy", "unsupported_task")
    except ValueError as e:
        print(f"Caught expected error for second turn: {e}")

    print("\nAll tests passed!")

