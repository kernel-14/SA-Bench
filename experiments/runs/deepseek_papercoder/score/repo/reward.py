"""
reward.py

Implements the binary reward function used in SCoRe.  For MATH problems the
reward is 1.0 if the final boxed answer extracted from the model response
exactly matches the ground‑truth answer (after normalisation); otherwise 0.0.
For code tasks the reward is 1.0 if executing the generated code together with
the provided test cases completes without any `AssertionError` or timeout;
otherwise 0.0.

The interface follows the design specification:

    class RewardFunction {
        +__init__(tokenizer: PreTrainedTokenizer, is_code: bool)
        +extract_answer(text: str) -> str
        +check_correct(y: str, y_star: str) -> float
        +check_code(test_cases: dict, code: str) -> float
        +__call__(y: str, y_star: Union[str, dict], is_code: bool) -> float
    }

The module is intended to be used for both training (on‑policy RL) and
evaluation.  Timeout and other constant values are hard‑coded; no external
configuration is required beyond what is passed to the constructor.
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Dict, Optional, Union

# We do not import Config here to avoid circular dependency; the design
# specifies the constructor receives only tokenizer and is_code.
# The tokenizer is accepted for API consistency but not used.

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Module‑level constants
# --------------------------------------------------------------------------- #

# Maximum time (seconds) allowed for one code execution; if exceeded, the
# reward is 0.  The paper does not state a timeout; 5 seconds is a conservative
# choice that should be sufficient for typical MBPP / HumanEval problems.
CODE_EXECUTION_TIMEOUT = 5

# Regular expression for extracting the content of the *first* \boxed{} after
# the "Final Answer:" marker (case‑insensitive).
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

# --------------------------------------------------------------------------- #
# RewardFunction
# --------------------------------------------------------------------------- #


class RewardFunction:
    """
    Binary reward function for math‑answer matching and code‑execution.

    When `is_code` is True, the reward is determined by running the supplied
    code against test assertions.  Otherwise the reward compares the final
    boxed answer extracted from the model output with the reference answer.
    """

    def __init__(self, tokenizer, is_code: bool):
        """
        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
            The tokenizer associated with the model.  Not used internally;
            accepted solely for compatibility with the design specification.
        is_code : bool
            If True, the reward operates on code tasks (MBPP/HumanEval);
            otherwise it treats the task as a MATH problem.
        """
        self.tokenizer = tokenizer
        self.is_code = is_code

        # The timeout for code execution can be changed by the caller if
        # required (e.g. for very slow problems).
        self.code_timeout = CODE_EXECUTION_TIMEOUT

    # ----------------------------------------------------------------------- #
    # Answer extraction (MATH)
    # ----------------------------------------------------------------------- #

    def extract_answer(self, text: str) -> str:
        """
        Extract the final boxed answer from a MATH response.

        Searches for the *last* occurrence of the phrase "Final Answer:"
        (case‑insensitive) and then retrieves the content of the first
        ``\\boxed{}`` that appears after that point.

        Parameters
        ----------
        text : str
            The (possibly multi‑line) model output.

        Returns
        -------
        str
            The extracted answer string (stripped of surrounding whitespace)
            if found, otherwise an empty string.
        """
        # Locate the final "Final Answer:" marker.
        lower_text = text.lower()
        marker = "final answer:"
        idx = lower_text.rfind(marker)
        if idx == -1:
            return ""

        # Search for \boxed{...} after the marker.
        tail = text[idx:]
        match = _BOXED_RE.search(tail)
        if match is None:
            return ""

        return match.group(1).strip()

    def _normalize_answer(self, ans: str) -> str:
        """
        Normalise an extracted answer string for comparison.

        The normalisation includes:
          - stripping leading/trailing whitespace,
          - collapsing multiple spaces,
          - removing a single trailing period (if present).

        Parameters
        ----------
        ans : str
            The extracted answer.

        Returns
        -------
        str
            The normalised answer.
        """
        ans = ans.strip()
        # Collapse runs of whitespace.
        ans = " ".join(ans.split())
        # Remove trailing period if it is the last character.
        if ans.endswith("."):
            ans = ans[:-1].strip()
        return ans

    # ----------------------------------------------------------------------- #
    # Correctness checks
    # ----------------------------------------------------------------------- #

    def check_correct(self, y: str, y_star: str) -> float:
        """
        Compare the model output ``y`` with the ground‑truth answer ``y_star``.

        Both are expected to contain a ``\\boxed{}`` string (the reference may
        already be pre‑formatted that way).  The answers are extracted and
        compared using exact string matching after normalisation.

        Parameters
        ----------
        y : str
            The model‑generated response.
        y_star : str
            The ground‑truth answer (e.g. from the MATH dataset).

        Returns
        -------
        float
            1.0 if the answers match, else 0.0.
        """
        answer = self.extract_answer(y)
        ref_answer = self.extract_answer(y_star)
        if not answer or not ref_answer:
            return 0.0
        if self._normalize_answer(answer) == self._normalize_answer(ref_answer):
            return 1.0
        return 0.0

    def check_code(self, test_cases: dict, code: str) -> float:
        """
        Execute the supplied ``code`` together with the test assertions and
        return 1.0 if all tests pass without error.

        The ``test_cases`` dictionary must contain one of the following keys
        (checked in order): ``"test_code"`` (a full test script as a string),
        ``"test_list"`` (a list of assert strings), or ``"test"`` (a single
        string containing one or more assert statements).  The execution runs
        in a subprocess with a timeout; any exception or timeout yields 0.0.

        Parameters
        ----------
        test_cases : dict
            Dictionary providing the test assertions.
        code : str
            The raw Python code generated by the model.

        Returns
        -------
        float
            1.0 if all tests pass, else 0.0.
        """
        code = code.strip()
        if not code:
            return 0.0

        # Derive the test string from the dictionary.
        test_str = ""
        if "test_code" in test_cases:
            test_str = str(test_cases["test_code"])
        elif "test_list" in test_cases:
            test_str = "\n".join(str(t) for t in test_cases["test_list"])
        elif "test" in test_cases:
            test_str = str(test_cases["test"])
        else:
            logger.warning("check_code: no recognised test keys in test_cases dict")
            return 0.0

        # Write the code and tests to a temporary Python file.
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                f.write("\n\n")
                f.write(test_str)
                temp_path = f.name

            # Use the same Python interpreter that is running this script.
            python_exe = sys.executable
            proc = subprocess.run(
                [python_exe, temp_path],
                capture_output=True,
                text=True,
                timeout=self.code_timeout,
            )

            # Clean up the temporary file.
            os.unlink(temp_path)

            # If the subprocess returned non‑zero exit code, a test failed.
            if proc.returncode != 0:
                return 0.0

            # In some environments an AssertionError may be raised but not
            # reflected in the return code.  Check stderr as a safety net.
            if "AssertionError" in proc.stderr:
                return 0.0

            return 1.0

        except subprocess.TimeoutExpired:
            logger.warning("Code execution timed out.")
            return 0.0
        except Exception as e:
            logger.warning(f"Code execution error: {e}")
            return 0.0

    # ----------------------------------------------------------------------- #
    # Unified call interface
    # ----------------------------------------------------------------------- #

    def __call__(
        self,
        y: str,
        y_star: Union[str, Dict],
        is_code: Optional[bool] = None,
    ) -> float:
        """
        Compute the binary reward for a model response.

        When `is_code` is True (or `self.is_code` is True and no override is
        given), `y_star` must be a dictionary containing test assertions (see
        `check_code`).  Otherwise `y_star` must be a string containing the
        ground‑truth answer in boxed format.

        Parameters
        ----------
        y : str
            Model‑generated output.
        y_star : str or dict
            Ground‑truth reference (answer string for MATH, test‑case dict
            for code).
        is_code : bool, optional
            Override the task type.  If None, `self.is_code` is used.

        Returns
        -------
        float
            1.0 if the response is correct, 0.0 otherwise.
        """
        if is_code is None:
            is_code = self.is_code

        if is_code:
            if not isinstance(y_star, dict):
                logger.error("For code tasks, y_star must be a dict.")
                return 0.0
            return self.check_code(test_cases=y_star, code=y)
        else:
            if not isinstance(y_star, str):
                logger.error("For MATH tasks, y_star must be a string.")
                return 0.0
            return self.check_correct(y=y, y_star=y_star)


# --------------------------------------------------------------------------- #
# Simple self‑contained tests (run with `python reward.py`)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Dummy tokenizer to satisfy the constructor.
    class DummyTokenizer:
        pass

    tokenizer = DummyTokenizer()

    # ---------------------- MATH tests ---------------------- #
    rf_math = RewardFunction(tokenizer, is_code=False)

    # Example: correct answer extraction and comparison.
    y = (
        "Final Answer: The final answer is \\boxed{42}. I hope it is correct."
    )
    y_star = "\\boxed{42}"
    assert rf_math(y, y_star) == 1.0, "MATH exact match failed."

    # Incorrect.
    assert rf_math(y, "\\boxed{41}") == 0.0

    # No boxed answer.
    assert rf_math("Some text without a box", "\\boxed{42}") == 0.0

    # Normalisation: extra spaces, trailing period.
    y2 = "Final Answer: The final answer is \\boxed{  42 . }."
    y_star2 = "\\boxed{42}"
    assert rf_math(y2, y_star2) == 1.0, "Normalisation failed."

    print("MATH reward tests passed.")

    # ---------------------- CODE tests ---------------------- #
    rf_code = RewardFunction(tokenizer, is_code=True)

    # A simple correct function.
    correct_code = textwrap.dedent("""\
    def add(a, b):
        return a + b
    """)

    test_dict = {
        "test_list": [
            "assert add(2, 3) == 5",
            "assert add(-1, 1) == 0",
        ]
    }
    assert rf_code(correct_code, test_dict) == 1.0, "Correct code should pass."

    # A function that fails an assertion.
    incorrect_code = textwrap.dedent("""\
    def add(a, b):
        return a - b
    """)
    assert rf_code(incorrect_code, test_dict) == 0.0, "Incorrect code should fail."

    # Syntax error.
    broken_code = "def add(a, b): retrn a + b"
    assert rf_code(broken_code, test_dict) == 0.0, "Syntax error should fail."

    # Timeout – infinite loop.
    infinite_code = textwrap.dedent("""\
    def add(a, b):
        while True:
            pass
    """)
    # Use a very short timeout to ensure we catch it.
    saved_timeout = rf_code.code_timeout
    rf_code.code_timeout = 1  # seconds
    assert rf_code(infinite_code, test_dict) == 0.0, "Timeout should fail."
    rf_code.code_timeout = saved_timeout

    print("Code reward tests passed.")
