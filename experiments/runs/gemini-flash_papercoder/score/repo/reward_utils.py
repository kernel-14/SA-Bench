import abc
import logging
import re
import shutil
import subprocess
import tempfile
from typing import Any, Optional

import sympy  # For MathRewardFunction
from config import Config
from dataset_utils import Problem  # Imported for type hinting/conceptual understanding


logger = logging.getLogger(__name__)


class BaseRewardFunction(abc.ABC):
    """
    Abstract base class for calculating rewards and reward shaping bonuses.
    """

    def __init__(self) -> None:
        """
        Initializes the BaseRewardFunction.
        """
        pass

    @abc.abstractmethod
    def calculate_reward(self, response: str, ground_truth: str) -> float:
        """
        Calculates a binary reward (1.0 for correct, 0.0 for incorrect) for a model's response.

        Args:
            response: The model's generated response string.
            ground_truth: The ground truth information required for evaluation.
                          For MATH, this is the canonical answer string.
                          For CODE, this is expected to be the string containing all test cases.

        Returns:
            1.0 if the response is correct, 0.0 otherwise.
        """
        raise NotImplementedError

    def calculate_bonus(self, reward_t1: float, reward_t2: float, alpha: float) -> float:
        """
        Calculates the reward shaping bonus as defined in Equation 5 of the paper.
        bonus = alpha * (reward_t2 - reward_t1)

        Args:
            reward_t1: Binary reward for the first turn's response.
            reward_t2: Binary reward for the second turn's response.
            alpha: The reward shaping multiplier.

        Returns:
            The calculated reward bonus.
        """
        reward_difference = reward_t2 - reward_t1
        return alpha * reward_difference


class MathRewardFunction(BaseRewardFunction):
    """
    Concrete implementation of BaseRewardFunction for mathematical reasoning problems.
    Evaluates correctness based on extracting and comparing final numerical/symbolic answers.
    """

    # Regex to extract the final answer from the model's response, based on the prompt format.
    # Expects "Final Answer: The final answer is \boxed{{answer}}\ ."
    # The \boxed{{answer}} part captures the content.
    MATH_ANSWER_PATTERN = re.compile(r"Final Answer: The final answer is \\boxed{(.*?)}\\.")

    def __init__(self) -> None:
        """
        Initializes the MathRewardFunction.
        """
        super().__init__()

    def _extract_and_canonicalize_math_answer(self, text: str) -> Optional[Any]:
        """
        Extracts the final answer from a given text using regex and attempts to canonicalize
        it using sympy.

        Args:
            text: The text containing the potential final answer.

        Returns:
            A canonicalized sympy expression or Python number (int/float) if extraction
            and parsing are successful, otherwise None.
        """
        match = self.MATH_ANSWER_PATTERN.search(text)
        if not match:
            return None

        extracted_answer_str = match.group(1).strip()
        if not extracted_answer_str:
            return None

        try:
            # Attempt to parse as a mathematical expression
            canonical_answer = sympy.sympify(extracted_answer_str, evaluate=True)
            return canonical_answer
        except (sympy.SympifyError, TypeError, ValueError):
            # Fallback for simple numbers that sympify might struggle with or
            # non-mathematical strings that might be numerical
            try:
                # Try integer first, then float to preserve precision if it's an int
                if extracted_answer_str.isdigit() or (extracted_answer_str.startswith('-') and extracted_answer_str[1:].isdigit()):
                    return int(extracted_answer_str)
                return float(extracted_answer_str)
            except ValueError:
                logger.debug(
                    f"Could not sympify or convert to int/float: '{extracted_answer_str}'"
                )
                return None
        except Exception as e:
            logger.error(
                f"Unexpected error during sympy parsing for '{extracted_answer_str}': {e}"
            )
            return None

    def calculate_reward(self, response: str, ground_truth: str) -> float:
        """
        Calculates a binary reward for a math problem.
        1. Extracts the final answer from the response.
        2. Canonicalizes both the extracted answer and the ground truth using sympy.
        3. Compares the canonicalized forms for numerical or symbolic equivalence.

        Args:
            response: The model's generated response string.
            ground_truth: The canonical ground truth answer string (e.g., "4", "1/2", "x+y").

        Returns:
            1.0 if the answers match, 0.0 otherwise.
        """
        model_answer = self._extract_and_canonicalize_math_answer(response)
        if model_answer is None:
            logger.debug(
                f"Math reward: Could not extract/parse model answer from response:\n{response}"
            )
            return 0.0

        try:
            canonical_ground_truth = sympy.sympify(ground_truth, evaluate=True)
        except (sympy.SympifyError, TypeError, ValueError):
            logger.error(f"Math reward: Could not parse ground truth '{ground_truth}' with sympy.")
            return 0.0
        except Exception as e:
            logger.error(
                f"Unexpected error during sympy parsing for ground truth '{ground_truth}': {e}"
            )
            return 0.0

        # Attempt direct comparison first for simple cases (e.g., int/float literals)
        if model_answer == canonical_ground_truth:
            return 1.0

        # Numerical comparison for numbers (int, float, sympy.Number)
        # Convert to float for comparison if possible and both are numbers
        if isinstance(model_answer, (int, float, sympy.Number)) and isinstance(
            canonical_ground_truth, (int, float, sympy.Number)
        ):
            try:
                model_val = float(model_answer)
                gt_val = float(canonical_ground_truth)
                if abs(model_val - gt_val) < 1e-6:  # Small tolerance for float comparison
                    return 1.0
            except (ValueError, TypeError):
                # Fallback if direct float conversion fails, e.g., complex numbers
                pass

        # Symbolic comparison for algebraic expressions
        try:
            # Simplify both expressions and compare their difference to zero.
            # This handles algebraic equivalence (e.g., 'x+y' == 'y+x').
            if sympy.simplify(model_answer - canonical_ground_truth) == 0:
                return 1.0
        except Exception as e:
            logger.debug(
                f"Symbolic comparison failed for model: '{model_answer}', gt: '{canonical_ground_truth}'. Error: {e}"
            )

        return 0.0


def _text_code_indent(code_string: str, indent: str = "    ") -> str:
    """Helper to indent code strings for embedding in other code."""
    return "\n".join([indent + line for line in code_string.splitlines()])


class CodeRewardFunction(BaseRewardFunction):
    """
    Concrete implementation of BaseRewardFunction for code generation problems.
    Evaluates correctness by executing the generated code against provided test cases
    in a temporary, sandboxed Python environment.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the CodeRewardFunction.

        Args:
            config: The Config object, primarily for evaluation settings like timeout
                    and maximum code/output lengths.
        """
        super().__init__()
        self.config = config
        self.timeout_seconds: int = self.config.evaluation.get("code_execution_timeout", 10)
        self.max_code_length: int = self.config.evaluation.get("max_code_length", 2048)
        self.max_output_length: int = self.config.evaluation.get("max_output_length", 1024)

    def calculate_reward(self, response: str, test_code: str) -> float:
        """
        Calculates a binary reward for a code generation problem.
        Executes the model's generated code and provided test cases in a temporary,
        sandboxed Python environment.

        Args:
            response: The model's generated Python code string. This is expected to be
                      the complete function/program.
            test_code: The Python code containing the test cases (e.g., assert statements,
                       or test functions). This is assumed to be extracted from
                       Problem.metadata['test'] or Problem.metadata['test_code'] by the caller.

        Returns:
            1.0 if the code executes successfully and passes all tests, 0.0 otherwise.
        """
        reward = 0.0
        temp_dir = None
        original_cwd = None
        try:
            # 1. Create a temporary directory for sandboxing
            temp_dir = tempfile.mkdtemp()
            # Change to the temporary directory to isolate execution
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            # 2. Prepare the code files for execution
            # Write the model's generated code to 'solution.py'
            solution_file_path = "solution.py"
            with open(solution_file_path, "w", encoding="utf-8") as f:
                f.write(response[:self.max_code_length])  # Truncate if too long

            # Prepare a test execution wrapper. This imports the solution and runs tests.
            # This is a common and safer pattern for evaluating generated code.
            test_executor_code = f"""
import sys
import os
import io
import contextlib

# Add current directory to path to import solution.py
sys.path.insert(0, os.path.dirname(__file__))

# Redirect stdout/stderr to prevent printing to console during tests
# and capture it for debugging if needed, without it affecting our pass/fail logic
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

try:
    with suppress_stdout_stderr():
        # Dynamically import solution from solution.py
        # This makes sure functions from solution.py are available for `test_code`
        # Using a direct import for simplicity for now, more robust might be importlib
        from solution import * 

    # Execute the test cases. Assume test_code contains assert statements.
    # The `test_code` itself might contain import statements or definitions,
    # so we execute it as a script, but after `solution.py` is available.
    exec(compile(_text_code_indent(test_code, indent=''), '<string>', 'exec'))

    print("ALL_TESTS_PASSED") # Signal for successful completion
except AssertionError:
    sys.exit(1) # Indicate test failure due to failed assertion
except Exception as e:
    # Catch any other runtime errors (e.g., NameError if function not defined)
    print(f"ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(2) # Indicate general runtime error
            """

            test_executor_file_path = "test_executor.py"
            with open(test_executor_file_path, "w", encoding="utf-8") as f:
                f.write(test_executor_code)

            # 3. Execute the test executor in a subprocess
            process = subprocess.run(
                ["python", test_executor_file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,  # Do not raise CalledProcessError for non-zero exit codes
            )

            # 4. Evaluate results based on return code and stdout signal
            if process.returncode == 0 and "ALL_TESTS_PASSED" in process.stdout:
                reward = 1.0
            else:
                logger.debug(
                    f"Code execution failed or tests did not pass. Return code: {process.returncode}"
                )
                logger.debug(f"Stdout:\n{process.stdout[:self.max_output_length]}")
                logger.debug(f"Stderr:\n{process.stderr[:self.max_output_length]}")

        except subprocess.TimeoutExpired:
            logger.debug(f"Code execution timed out after {self.timeout_seconds} seconds.")
            # Kill the process if it timed out to ensure it doesn't linger
            if 'process' in locals() and process.poll() is None: # Check if process is still running
                process.kill()
                process.wait()
        except FileNotFoundError:
            logger.error("Python interpreter not found. Ensure Python is in PATH.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during code evaluation: {e}")
        finally:
            # 5. Clean up temporary files and restore original working directory
            if original_cwd:
                os.chdir(original_cwd)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        return reward


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # --- Test BaseRewardFunction ---
    print("\n--- Testing BaseRewardFunction ---")
    brf = BaseRewardFunction()
    # Test calculate_bonus
    assert brf.calculate_bonus(0.0, 1.0, 10.0) == 10.0  # Incorrect -> Correct
    assert brf.calculate_bonus(1.0, 0.0, 10.0) == -10.0  # Correct -> Incorrect
    assert brf.calculate_bonus(0.0, 0.0, 10.0) == 0.0  # Both Incorrect
    assert brf.calculate_bonus(1.0, 1.0, 10.0) == 0.0  # Both Correct
    print("BaseRewardFunction.calculate_bonus tests passed.")

    # --- Test MathRewardFunction ---
    print("\n--- Testing MathRewardFunction ---")
    mrf = MathRewardFunction()

    # Test cases for MathRewardFunction
    math_test_cases = [
        # Correct numerical answers
        ("Solution: ... Final Answer: The final answer is \\boxed{4.0}. I hope it is correct.", "4", 1.0),
        ("Solution: ... Final Answer: The final answer is \\boxed{1/2}. I hope it is correct.", "0.5", 1.0),
        ("Solution: ... Final Answer: The final answer is \\boxed{2/4}. I hope it is correct.", "1/2", 1.0),
        ("Solution: ... Final Answer: The final answer is \\boxed{sqrt(9)}. I hope it is correct.", "3", 1.0),
        ("Solution: ... Final Answer: The final answer is \\boxed{3.00}. I hope it is correct.", "3", 1.0),
        ("Solution: ... Final Answer: The final answer is \\boxed{1.23456789}. I hope it is correct.", "1.23456789", 1.0),
        ("Solution: ... Final Answer: The final answer is \\boxed{-5}. I hope it is correct.", "-5", 1.0), # Negative int
        # Correct symbolic answers
        ("Solution: ... Final Answer: The final answer is \\boxed{x+y}. I hope it is correct.", "y+x", 1.0),
        ("Solution: ... Final Answer: The final answer is \\boxed{2*x+x}. I hope it is correct.", "3*x", 1.0),
        # Incorrect answers
        ("Solution: ... Final Answer: The final answer is \\boxed{5}. I hope it is correct.", "4", 0.0),
        ("Solution: ... Final Answer: The final answer is \\boxed{x+z}. I hope it is correct.", "y+x", 0.0),
        # No final answer pattern
        ("Solution: This is just a solution without a final answer.", "4", 0.0),
        ("Solution: ... Final Answer: The final answer is \\boxed{}. I hope it is correct.", "4", 0.0), # Empty box
        # Invalid math expression in model output
        ("Solution: ... Final Answer: The final answer is \\boxed{invalid_math}. I hope it is correct.", "4", 0.0),
        # Invalid math expression in ground truth (should log error and return 0.0)
        ("Solution: ... Final Answer: The final answer is \\boxed{4}. I hope it is correct.", "invalid_gt", 0.0),
        # Problem where both are numbers but one is sympy.Number and other python float
        ("Solution: ... Final Answer: The final answer is \\boxed{1.0}. I hope it is correct.", "1", 1.0),
    ]

    for i, (response, gt, expected_reward) in enumerate(math_test_cases):
        reward = mrf.calculate_reward(response, gt)
        print(f"Math Test {i+1}: Response='{response[:50]}...', GT='{gt}', Reward={reward}, Expected={expected_reward}")
        assert (
            reward == expected_reward
        ), f"Math Test {i+1} Failed: {response}, GT={gt}, Got {reward}, Expected {expected_reward}"
    print("MathRewardFunction tests passed.")

    # --- Test CodeRewardFunction ---
    print("\n--- Testing CodeRewardFunction ---")

    class MockConfigForCode(Config):
        def __init__(self):
            super().__init__()
            self.evaluation = {
                "code_execution_timeout": 5,
                "max_code_length": 2048,
                "max_output_length": 1024,
            }

    code_config = MockConfigForCode()
    crf = CodeRewardFunction(code_config)

    # Test case 1: Correct code, passing tests
    correct_code = """
def add_two_numbers(a, b):
    return a + b
"""
    passing_test_code = """
assert add_two_numbers(1, 2) == 3
assert add_two_numbers(-1, 1) == 0
"""
    print("\nCode Test 1: Correct code")
    reward = crf.calculate_reward(correct_code, passing_test_code)
    print(f"  Reward: {reward}, Expected: 1.0")
    assert reward == 1.0

    # Test case 2: Incorrect code, failing tests
    incorrect_code = """
def multiply_two_numbers(a, b):
    return a * b
"""
    failing_test_code = """
assert multiply_two_numbers(2, 3) == 5 # Should be 6
"""
    print("\nCode Test 2: Incorrect code (failing tests)")
    reward = crf.calculate_reward(incorrect_code, failing_test_code)
    print(f"  Reward: {reward}, Expected: 0.0")
    assert reward == 0.0

    # Test case 3: Syntax error in code
    syntax_error_code = """
def syntax_error_func(a, b)
    return a + b
"""
    simple_test_code = """
# No specific test logic, just to trigger syntax check on import
"""
    print("\nCode Test 3: Syntax error in code")
    reward = crf.calculate_reward(syntax_error_code, simple_test_code)
    print(f"  Reward: {reward}, Expected: 0.0")
    assert reward == 0.0

    # Test case 4: Runtime error in code (handled by test)
    runtime_error_code = """
def divide_by_zero(a, b):
    return a / 0
"""
    runtime_test_code = """
try:
    divide_by_zero(1, 0)
    assert False, "Should have raised ZeroDivisionError"
except ZeroDivisionError:
    pass
"""
    print("\nCode Test 4: Runtime error in code (handled by test)")
    reward = crf.calculate_reward(runtime_error_code, runtime_test_code)
    print(f"  Reward: {reward}, Expected: 1.0")
    assert reward == 1.0  # Test code correctly catches the error

    runtime_error_code_uncaught = """
def divide_by_zero_uncaught(a, b):
    return a / 0
"""
    runtime_test_code_uncaught = """
# This test doesn't handle the ZeroDivisionError
divide_by_zero_uncaught(1, 0)
"""
    print("\nCode Test 5: Runtime error in code (uncaught by test)")
    reward = crf.calculate_reward(runtime_error_code_uncaught, runtime_test_code_uncaught)
    print(f"  Reward: {reward}, Expected: 0.0")
    assert reward == 0.0

    # Test case 6: Timeout
    infinite_loop_code = """
def infinite_loop():
    while True:
        pass
"""
    infinite_loop_test_code = """
infinite_loop()
"""
    print("\nCode Test 6: Timeout")
    reward = crf.calculate_reward(infinite_loop_code, infinite_loop_test_code)
    print(f"  Reward: {reward}, Expected: 0.0")
    assert reward == 0.0

    # Test case 7: Empty code
    empty_code = ""
    print("\nCode Test 7: Empty code")
    reward = crf.calculate_reward(empty_code, passing_test_code)
    print(f"  Reward: {reward}, Expected: 0.0")
    assert reward == 0.0

    print("\nCodeRewardFunction tests passed.")

