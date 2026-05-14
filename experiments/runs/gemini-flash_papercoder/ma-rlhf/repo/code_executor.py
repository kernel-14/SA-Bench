"""
This module provides the CodeExecutor class responsible for executing generated
code snippets against test cases and calculating rewards based on compilation
and execution results, primarily for the APPS dataset.
"""

import os
import subprocess
import tempfile
import uuid
import re
import sys
from typing import List, Tuple, Optional
from loguru import logger
from omegaconf import DictConfig # Use DictConfig to avoid circular import with config.py

# Define markers for communication between the test runner script and CodeExecutor
_TEST_PASS_MARKER = "_TEST_PASS_"
_TEST_FAIL_MARKER = "_TEST_FAIL_"
_RUNTIME_ERROR_MARKER_INTERNAL = "_RUNTIME_ERROR_" # Internal marker from runner for a specific test failure
_COMPILE_ERROR_MARKER_INTERNAL = "_COMPILE_ERROR_" # Internal marker from runner for import/syntax errors

# Final summary markers for easier parsing of stderr from the subprocess
_FINAL_RESULTS_PASS_MARKER = "FINAL_RESULTS_PASS:"
_FINAL_RESULTS_FAIL_MARKER = "FINAL_RESULTS_FAIL:"
_FINAL_RESULTS_RUNTIME_ERROR_FLAG_MARKER = "FINAL_RESULTS_RUNTIME_ERROR:"

# Template for the Python test runner script
# This script will be dynamically generated and executed by the CodeExecutor.
# It handles importing the user's solution, executing provided test cases,
# and reporting results and errors in a structured way.
_TEST_RUNNER_TEMPLATE = """
import sys
import io
import traceback
import importlib.util
import os
import json # For safer passing of list of strings

# Markers for parsing output
TEST_PASS_MARKER = "{_TEST_PASS_MARKER_}"
TEST_FAIL_MARKER = "{_TEST_FAIL_MARKER_}"
RUNTIME_ERROR_MARKER_INTERNAL = "{_RUNTIME_ERROR_MARKER_INTERNAL_}"
COMPILE_ERROR_MARKER_INTERNAL = "{_COMPILE_ERROR_MARKER_INTERNAL_}"

SOLUTION_FILE = os.path.abspath("{solution_file_path}")
TEST_CASES_JSON = '{test_cases_json}' # JSON string of list of test code strings

def run_tests():
    global_n_pass = 0
    global_n_fail = 0
    has_runtime_error_flag = False

    # Dynamically import the solution
    try:
        spec = importlib.util.spec_from_file_location("solution_module", SOLUTION_FILE)
        solution_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(solution_module)
        
        # Assume the main function for the problem is named 'solve'
        solve_func = getattr(solution_module, 'solve', None)
        if solve_func is None:
            raise AttributeError("Function 'solve' not found in solution file. "
                                 "Make sure your solution defines a 'solve' function.")
    except Exception as e:
        sys.stderr.write(f"{COMPILE_ERROR_MARKER_INTERNAL}: {{e}}\\n")
        sys.stderr.write(traceback.format_exc())
        sys.exit(1) # Indicate compilation/import error for the solution file

    try:
        test_cases = json.loads(TEST_CASES_JSON)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Error decoding test cases JSON: {{e}}\\n")
        sys.exit(1)

    # Prepare globals for exec to make solve_func available to test cases
    exec_globals = solution_module.__dict__.copy()
    exec_globals['solve'] = solve_func # Ensure 'solve' is directly accessible by test code
    exec_globals['TEST_PASS_MARKER'] = TEST_PASS_MARKER
    exec_globals['TEST_FAIL_MARKER'] = TEST_FAIL_MARKER

    for i, test_case_code in enumerate(test_cases):
        # Redirect stdout to capture test case specific output
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            # Execute the test case code. Test cases are expected to call
            # 'solve' and print TEST_PASS_MARKER or TEST_FAIL_MARKER.
            exec(test_case_code, exec_globals)
            output = redirected_output.getvalue()

            if TEST_PASS_MARKER in output:
                global_n_pass += 1
            elif TEST_FAIL_MARKER in output:
                global_n_fail += 1
            else:
                # If no clear pass/fail marker, assume failure for this test
                global_n_fail += 1
                sys.stderr.write(f"Test {{i}} did not emit pass/fail marker. Output: {{output.strip()}}\\n")

        except Exception as e:
            has_runtime_error_flag = True
            global_n_fail += 1 # Any unhandled exception during a test means test failure
            sys.stderr.write(f"{RUNTIME_ERROR_MARKER_INTERNAL}: Runtime error in test {{i}}: {{e}}\\n")
            sys.stderr.write(traceback.format_exc()) # Print full traceback to stderr
        finally:
            sys.stdout = old_stdout # Restore stdout

    sys.stderr.write(f"{_FINAL_RESULTS_PASS_MARKER}{{global_n_pass}}\\n")
    sys.stderr.write(f"{_FINAL_RESULTS_FAIL_MARKER}{{global_n_fail}}\\n")
    sys.stderr.write(f"{_FINAL_RESULTS_RUNTIME_ERROR_FLAG_MARKER}{'True' if has_runtime_error_flag else 'False'}\\n")

if __name__ == "__main__":
    run_tests()
"""


class CodeExecutor:
    """
    Executes generated Python code against test cases and computes rewards.
    It manages temporary files and subprocess execution for sandboxed evaluation.
    """

    def __init__(self, config: DictConfig):
        """
        Initializes the CodeExecutor instance.

        Args:
            config: A DictConfig object containing global configuration parameters,
                    including 'code_execution_timeout'.
        """
        self.config = config
        self.code_execution_timeout: int = config.global.get("code_execution_timeout", 10)
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None
        logger.info(f"CodeExecutor initialized with timeout: {self.code_execution_timeout} seconds.")

    def _get_temp_dir(self) -> str:
        """
        Ensures a temporary directory exists and returns its path.
        """
        if self.temp_dir is None:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="ma_rlhf_code_exec_")
            logger.debug(f"Created temporary directory: {self.temp_dir.name}")
        return self.temp_dir.name

    def _cleanup_temp_dir(self):
        """
        Cleans up the temporary directory.
        """
        if self.temp_dir:
            self.temp_dir.cleanup()
            logger.debug(f"Cleaned up temporary directory: {self.temp_dir.name}")
            self.temp_dir = None

    def execute_code(self, code_string: str, test_cases: List[str]) -> Tuple[int, int, str]:
        """
        Executes the provided Python code string against a list of test cases.

        Args:
            code_string: The Python code generated by the model (e.g., a function definition).
            test_cases: A list of Python code snippets, each representing a single test case.
                        Each test case is expected to call the `solve` function from `code_string`
                        and print `_TEST_PASS_` or `_TEST_FAIL_` based on assertions.

        Returns:
            A tuple (N_pass, N_fail, error_type):
                - N_pass (int): Number of test cases passed.
                - N_fail (int): Number of test cases failed.
                - error_type (str): 'success', 'compile_error', 'runtime_error', or 'timeout'.
        """
        n_pass = 0
        n_fail = 0
        error_type = 'success'
        temp_dir_path = self._get_temp_dir()

        solution_filename = os.path.join(temp_dir_path, f"solution_{uuid.uuid4().hex}.py")
        runner_filename = os.path.join(temp_dir_path, f"runner_{uuid.uuid4().hex}.py")

        try:
            # 1. Write the generated code to a temporary solution file
            with open(solution_filename, 'w') as f:
                f.write(code_string)

            # 2. Prepare the test cases JSON for embedding in the runner script
            # Escape single quotes within the JSON string for safe embedding in Python string
            test_cases_json = json.dumps(test_cases).replace("'", "\\'")

            # 3. Format and write the test runner script
            runner_script_content = _TEST_RUNNER_TEMPLATE.format(
                _TEST_PASS_MARKER_=_TEST_PASS_MARKER,
                _TEST_FAIL_MARKER_=_TEST_FAIL_MARKER,
                _RUNTIME_ERROR_MARKER_INTERNAL_=_RUNTIME_ERROR_MARKER_INTERNAL,
                _COMPILE_ERROR_MARKER_INTERNAL_=_COMPILE_ERROR_MARKER_INTERNAL,
                solution_file_path=solution_filename,
                test_cases_json=test_cases_json
            )
            with open(runner_filename, 'w') as f:
                f.write(runner_script_content)

            # 4. Execute the test runner script in a subprocess
            logger.debug(f"Executing solution '{os.path.basename(solution_filename)}' "
                         f"with runner '{os.path.basename(runner_filename)}'")
            result = subprocess.run(
                [sys.executable, runner_filename],
                capture_output=True,
                text=True,
                timeout=self.code_execution_timeout,
                check=False  # Do not raise CalledProcessError for non-zero exit codes
            )
            logger.debug(f"Subprocess finished with return code: {result.returncode}")

            # 5. Parse results
            stderr_output = result.stderr
            stdout_output = result.stdout # Not used for final metrics, but might contain prints
            
            # Check for compile error first (e.g., SyntaxError, or failed import in runner)
            if result.returncode != 0:
                if _COMPILE_ERROR_MARKER_INTERNAL in stderr_output or "SyntaxError" in stderr_output or "IndentationError" in stderr_output or "AttributeError: function 'solve' not found" in stderr_output:
                    error_type = 'compile_error'
                    logger.warning(f"Compile error detected for solution {os.path.basename(solution_filename)}")
                else:
                    # Generic error, possibly an uncaught runtime error in the runner itself or solution
                    error_type = 'runtime_error'
                    logger.warning(f"Subprocess exited with non-zero code and no clear compile error for {os.path.basename(solution_filename)}. Assuming runtime error.")
            
            if error_type == 'success': # If not already marked as compile error
                # Check for runtime errors reported by the runner script
                if _RUNTIME_ERROR_MARKER_INTERNAL in stderr_output or \
                   f"{_FINAL_RESULTS_RUNTIME_ERROR_FLAG_MARKER}True" in stderr_output:
                    error_type = 'runtime_error'
                    logger.debug(f"Runtime error(s) reported by runner for {os.path.basename(solution_filename)}")

                # Extract pass/fail counts
                pass_match = re.search(rf"{_FINAL_RESULTS_PASS_MARKER}(\d+)", stderr_output)
                fail_match = re.search(rf"{_FINAL_RESULTS_FAIL_MARKER}(\d+)", stderr_output)

                if pass_match:
                    n_pass = int(pass_match.group(1))
                if fail_match:
                    n_fail = int(fail_match.group(1))
                
                # If error_type is still 'success' but n_fail > 0, it means logical errors, not compile/runtime.
                if n_fail > 0 and error_type == 'success':
                    error_type = 'fail' # Indicate logical failure, not system error

        except subprocess.TimeoutExpired:
            error_type = 'timeout'
            logger.warning(f"Code execution timed out after {self.code_execution_timeout}s for solution {os.path.basename(solution_filename)}")
            # If timed out, all tests implicitly fail
            n_pass = 0
            n_fail = len(test_cases)
        except Exception as e:
            error_type = 'runtime_error' # Catch any other unexpected errors during CodeExecutor's management
            logger.error(f"Unexpected error during code execution setup or parsing: {e}")
            logger.error(traceback.format_exc())
            n_pass = 0
            n_fail = len(test_cases)
        finally:
            # 6. Cleanup temporary files
            if os.path.exists(solution_filename):
                os.remove(solution_filename)
            if os.path.exists(runner_filename):
                os.remove(runner_filename)
            logger.debug(f"Cleaned up temporary files for {os.path.basename(solution_filename)}")

        return n_pass, n_fail, error_type

    def get_compiler_reward(self, n_pass: int, n_fail: int, error_type: str) -> float:
        """
        Calculates the reward score for a generated code snippet based on the
        compiler signal formula provided in the paper (Appendix B.5).

        Args:
            n_pass (int): Number of test cases passed.
            n_fail (int): Number of test cases failed.
            error_type (str): 'success', 'compile_error', 'runtime_error', 'timeout', or 'fail'.

        Returns:
            float: The calculated reward value.
        """
        if error_type == 'compile_error':
            return -1.0
        elif error_type == 'runtime_error' or error_type == 'timeout':
            return -0.6
        else:  # 'success' or 'fail' (meaning logical failures)
            total_tests = n_pass + n_fail
            if total_tests == 0:
                logger.warning("No tests were executed/reported. Returning default low reward.")
                return -1.0 # Fallback for edge case where no tests run
            
            pass_ratio = n_pass / total_tests
            reward = -0.3 + 1.3 * pass_ratio
            return reward

    def __del__(self):
        """
        Ensures the temporary directory is cleaned up when the object is destroyed.
        """
        self._cleanup_temp_dir()


if __name__ == "__main__":
    # Example usage and testing
    logger.remove() # Remove default logger
    logger.add(sys.stderr, level="INFO") # Use custom logger for demonstration

    # Mock a minimal config
    mock_config = DictConfig({
        "global": {
            "code_execution_timeout": 5  # 5 seconds timeout for tests
        }
    })
    executor = CodeExecutor(mock_config)

    # --- Test Case 1: All tests pass ---
    logger.info("\n--- Test Case 1: All tests pass ---")
    good_code = """
def solve(a, b):
    return a + b
"""
    good_test_cases = [
        "assert solve(1, 2) == 3; print(TEST_PASS_MARKER)",
        "assert solve(-1, 1) == 0; print(TEST_PASS_MARKER)",
        "assert solve(0, 0) == 0; print(TEST_PASS_MARKER)"
    ]
    n_pass, n_fail, error_type = executor.execute_code(good_code, good_test_cases)
    reward = executor.get_compiler_reward(n_pass, n_fail, error_type)
    logger.info(f"Result: Pass={n_pass}, Fail={n_fail}, ErrorType='{error_type}', Reward={reward:.2f}")
    assert n_pass == 3 and n_fail == 0 and error_type == 'success' and abs(reward - 1.0) < 1e-6

    # --- Test Case 2: Some tests fail logically ---
    logger.info("\n--- Test Case 2: Some tests fail logically ---")
    buggy_code = """
def solve(a, b):
    return a * b # Should be a + b
"""
    buggy_test_cases = [
        "assert solve(1, 2) == 3; print(TEST_PASS_MARKER)", # This will fail
        "assert solve(-1, 1) == -1; print(TEST_PASS_MARKER)", # This will pass
        "assert solve(0, 0) == 0; print(TEST_PASS_MARKER)"  # This will pass
    ]
    n_pass, n_fail, error_type = executor.execute_code(buggy_code, buggy_test_cases)
    reward = executor.get_compiler_reward(n_pass, n_fail, error_type)
    logger.info(f"Result: Pass={n_pass}, Fail={n_fail}, ErrorType='{error_type}', Reward={reward:.2f}")
    assert n_pass == 2 and n_fail == 1 and error_type == 'fail' and abs(reward - (-0.3 + 1.3 * (2/3))) < 1e-6

    # --- Test Case 3: Compile Error (Syntax Error) ---
    logger.info("\n--- Test Case 3: Compile Error (Syntax Error) ---")
    syntax_error_code = """
def solve(a, b):
    return a + b
    if True:
        pass # missing indentation here
"""
    n_pass, n_fail, error_type = executor.execute_code(syntax_error_code, good_test_cases)
    reward = executor.get_compiler_reward(n_pass, n_fail, error_type)
    logger.info(f"Result: Pass={n_pass}, Fail={n_fail}, ErrorType='{error_type}', Reward={reward:.2f}")
    assert error_type == 'compile_error' and abs(reward - -1.0) < 1e-6

    # --- Test Case 4: Runtime Error (e.g., ZeroDivisionError) ---
    logger.info("\n--- Test Case 4: Runtime Error (ZeroDivisionError) ---")
    runtime_error_code = """
def solve(a, b):
    return a / b
"""
    runtime_test_cases = [
        "assert solve(4, 2) == 2; print(TEST_PASS_MARKER)",
        "assert solve(1, 0) == 0; print(TEST_FAIL_MARKER)", # This test case is poorly written to catch ZeroDivisionError, it would crash
    ]
    n_pass, n_fail, error_type = executor.execute_code(runtime_error_code, runtime_test_cases)
    reward = executor.get_compiler_reward(n_pass, n_fail, error_type)
    logger.info(f"Result: Pass={n_pass}, Fail={n_fail}, ErrorType='{error_type}', Reward={reward:.2f}")
    assert error_type == 'runtime_error' and abs(reward - -0.6) < 1e-6

    # --- Test Case 5: Timeout ---
    logger.info("\n--- Test Case 5: Timeout ---")
    timeout_code = """
def solve(a, b):
    import time
    time.sleep(100) # Sleep for a long time
    return a + b
"""
    n_pass, n_fail, error_type = executor.execute_code(timeout_code, good_test_cases)
    reward = executor.get_compiler_reward(n_pass, n_fail, error_type)
    logger.info(f"Result: Pass={n_pass}, Fail={n_fail}, ErrorType='{error_type}', Reward={reward:.2f}")
    assert error_type == 'timeout' and abs(reward - -0.6) < 1e-6
    
    # --- Test Case 6: Test case that assumes `solve` is in global scope ---
    logger.info("\n--- Test Case 6: Test case with direct `solve` call ---")
    direct_call_code = """
def solve(x):
    return x * 2
"""
    direct_call_tests = [
        "result = solve(5); assert result == 10; print(TEST_PASS_MARKER)",
        "result = solve(3); assert result == 7; print(TEST_FAIL_MARKER)"
    ]
    n_pass, n_fail, error_type = executor.execute_code(direct_call_code, direct_call_tests)
    reward = executor.get_compiler_reward(n_pass, n_fail, error_type)
    logger.info(f"Result: Pass={n_pass}, Fail={n_fail}, ErrorType='{error_type}', Reward={reward:.2f}")
    assert n_pass == 1 and n_fail == 1 and error_type == 'fail' and abs(reward - (-0.3 + 1.3 * (1/2))) < 1e-6


    logger.info("\nAll CodeExecutor tests completed.")
