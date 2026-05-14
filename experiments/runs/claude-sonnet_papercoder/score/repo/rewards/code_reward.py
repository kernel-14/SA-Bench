## rewards/code_reward.py
"""Binary reward function for code generation tasks in SCoRe.

This module implements CodeReward, which computes a binary correctness
reward (1.0 or 0.0) for model-generated code on MBPP and HumanEval
problems. The reward is 1.0 if and only if the generated code passes
all provided test cases when executed in an isolated subprocess.

The paper states (Section 6): "we use binary rewards during training,
indicating whether the model's answer matches the ground truth one ...
or passes all test cases (for coding)."

Config values used (from config.yaml):
    reward.code.reward_type: "binary_test_pass"
    reward.code.execution_timeout: 5  (seconds per execution)
    reward.code.max_memory_mb: 512    (memory limit per subprocess)

Design invariants:
    - Every failure mode maps to False (→ 0.0 reward). The reward function
      must never raise an exception during training.
    - Each compute() call spawns a fresh subprocess for full isolation.
    - The last code block in a response is extracted (handles self-correction
      within a turn, where the model produces corrected code at the end).
    - Memory limiting is applied on Linux via resource.setrlimit; silently
      skipped on other platforms.

Typical usage:
    from rewards.code_reward import CodeReward

    reward_fn = CodeReward(timeout=5)
    code = reward_fn.extract_code(model_response)
    score = reward_fn.compute(code, test_cases)
    # score is 1.0 or 0.0
"""

import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — guarded so the module can be imported even if these
# packages are not installed (tests, linting, CI without full deps).
# ---------------------------------------------------------------------------
try:
    from func_timeout import func_timeout, FunctionTimedOut

    _FUNC_TIMEOUT_AVAILABLE: bool = True
except ImportError:
    _FUNC_TIMEOUT_AVAILABLE = False
    logger.warning(
        "func_timeout is not installed. Timeout wrapping for _run_with_timeout "
        "will fall back to a basic threading approach. Install func-timeout==4.3.5 "
        "for reliable timeout enforcement."
    )

# resource module is Linux/macOS only — not available on Windows
try:
    import resource as _resource_module

    _RESOURCE_AVAILABLE: bool = True
except ImportError:
    _RESOURCE_AVAILABLE = False
    logger.debug(
        "resource module not available (likely Windows). "
        "Memory limiting for subprocess execution will be disabled."
    )

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default execution timeout in seconds (matches config.yaml default)
_DEFAULT_TIMEOUT_SECONDS: int = 5

# Default memory limit in MB (matches config.yaml default)
_DEFAULT_MAX_MEMORY_MB: int = 512

# Sentinel comment appended to the test script to verify test execution
# completed (guards against sys.exit(0) false positives)
_SENTINEL_PRINT: str = "\nprint('__SCORE_TESTS_PASSED__')\n"

# Expected sentinel string in subprocess stdout
_SENTINEL_STRING: str = "__SCORE_TESTS_PASSED__"

# Python executable path — use the same interpreter running this process
_PYTHON_EXECUTABLE: str = sys.executable


class CodeReward:
    """Binary reward function for code generation tasks.

    Computes a binary correctness reward (1.0 or 0.0) by executing
    model-generated code against test case assertions in an isolated
    subprocess. Each call to compute() spawns a fresh subprocess to
    prevent state pollution between rollouts.

    Attributes:
        timeout: Maximum execution time in seconds per subprocess call.
            Sourced from config.yaml: reward.code.execution_timeout.
        max_memory_mb: Maximum memory in MB for each subprocess.
            Sourced from config.yaml: reward.code.max_memory_mb.
            Applied only on Linux via resource.setrlimit; silently
            skipped on other platforms.
    """

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
        max_memory_mb: int = _DEFAULT_MAX_MEMORY_MB,
    ) -> None:
        """Initialize CodeReward.

        Args:
            timeout: Maximum execution time in seconds for each subprocess
                invocation. Default 5 matches config.yaml
                reward.code.execution_timeout. Must be a positive integer.
            max_memory_mb: Maximum virtual memory in MB for each subprocess.
                Default 512 matches config.yaml reward.code.max_memory_mb.
                Applied only on Linux; silently ignored on other platforms.

        Raises:
            ValueError: If timeout is not a positive integer.
        """
        if timeout <= 0:
            raise ValueError(
                f"timeout must be a positive integer (got {timeout}). "
                "This value comes from config.yaml: "
                "reward.code.execution_timeout."
            )
        self.timeout: int = timeout
        self.max_memory_mb: int = max_memory_mb

        logger.debug(
            "CodeReward initialized: timeout=%ds, max_memory_mb=%dMB, "
            "func_timeout_available=%s, resource_available=%s.",
            self.timeout,
            self.max_memory_mb,
            _FUNC_TIMEOUT_AVAILABLE,
            _RESOURCE_AVAILABLE,
        )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def compute(self, code: str, test_cases: List[str]) -> float:
        """Compute the binary correctness reward for generated code.

        This is the main entry point called by RewardFunction.compute_reward().
        Returns 1.0 if the code passes all test cases, 0.0 otherwise.

        All failure modes (syntax errors, runtime errors, assertion failures,
        timeouts, memory errors) are caught and mapped to 0.0. This method
        never raises an exception.

        Args:
            code: The extracted Python code string to evaluate. Should be
                the output of extract_code() applied to the raw model
                response.
            test_cases: List of test assertion strings or a list containing
                a single HumanEval test function string. For MBPP, each
                element is an assertion like
                "assert func(args) == expected". For HumanEval, the list
                typically contains one element: the full test function
                string including the check() call.

        Returns:
            1.0 if all test cases pass, 0.0 otherwise.
        """
        # Guard: empty code or no test cases → immediate failure
        if not code or not code.strip():
            logger.debug(
                "compute() called with empty code string. Returning 0.0."
            )
            return 0.0

        if not test_cases:
            logger.debug(
                "compute() called with empty test_cases list. Returning 0.0."
            )
            return 0.0

        try:
            result: bool = self._execute_in_sandbox(code, test_cases)
            reward: float = 1.0 if result else 0.0
            logger.debug(
                "compute() result: %s (reward=%.1f).", result, reward
            )
            return reward
        except Exception as exc:
            # Belt-and-suspenders: _execute_in_sandbox should never raise,
            # but catch anything that slips through to protect the training loop.
            logger.debug(
                "compute() caught unexpected exception: %s. Returning 0.0.",
                exc,
            )
            return 0.0

    def extract_code(self, response: str) -> str:
        """Extract Python code from a model response string.

        The model response may contain:
            1. A markdown Python code block: ```python\\n...\\n```
            2. A generic markdown code block: ```\\n...\\n```
            3. Raw Python code with no delimiters
            4. Mixed prose + code (explanation followed by code block)

        Takes the LAST code block found in the response. This handles
        self-correction within a turn (Appendix D of the paper): the model
        often produces an explanation followed by corrected code, and the
        last block is the final corrected version.

        Args:
            response: The full model response string, potentially containing
                markdown code fences, prose, and Python code.

        Returns:
            The extracted Python code string with normalized indentation
            via textwrap.dedent(). Returns an empty string if the response
            is empty or whitespace-only.
        """
        if not response or not response.strip():
            return ""

        # ------------------------------------------------------------------
        # Strategy 1: Find all ```python ... ``` blocks (case-insensitive
        # language tag). Take the LAST one.
        # ------------------------------------------------------------------
        python_fence_pattern: re.Pattern = re.compile(
            r"```python\s*\n(.*?)\n?\s*```",
            re.DOTALL | re.IGNORECASE,
        )
        python_matches: List[re.Match] = list(
            python_fence_pattern.finditer(response)
        )
        if python_matches:
            last_match: re.Match = python_matches[-1]
            extracted: str = last_match.group(1)
            return textwrap.dedent(extracted).strip()

        # ------------------------------------------------------------------
        # Strategy 2: Find all generic ``` ... ``` blocks. Take the LAST one.
        # This handles models that omit the "python" language tag.
        # ------------------------------------------------------------------
        generic_fence_pattern: re.Pattern = re.compile(
            r"```\s*\n(.*?)\n?\s*```",
            re.DOTALL,
        )
        generic_matches: List[re.Match] = list(
            generic_fence_pattern.finditer(response)
        )
        if generic_matches:
            last_generic: re.Match = generic_matches[-1]
            extracted = last_generic.group(1)
            return textwrap.dedent(extracted).strip()

        # ------------------------------------------------------------------
        # Strategy 3: No fences found — treat the entire response as raw
        # Python code. This handles models that output code directly without
        # markdown formatting (common for MBPP 3-shot format with [BEGIN]/[DONE]).
        # ------------------------------------------------------------------
        # Check for [BEGIN] / [DONE] delimiters from the MBPP few-shot format
        begin_done_pattern: re.Pattern = re.compile(
            r"\[BEGIN\]\s*\n(.*?)\n?\s*\[DONE\]",
            re.DOTALL,
        )
        begin_done_matches: List[re.Match] = list(
            begin_done_pattern.finditer(response)
        )
        if begin_done_matches:
            last_bd: re.Match = begin_done_matches[-1]
            extracted = last_bd.group(1)
            return textwrap.dedent(extracted).strip()

        # ------------------------------------------------------------------
        # Strategy 4: Return the full response as-is (last resort).
        # The code may be raw Python without any delimiters.
        # ------------------------------------------------------------------
        logger.debug(
            "extract_code: No code fences or [BEGIN]/[DONE] delimiters found. "
            "Returning full response as raw code (length=%d).",
            len(response),
        )
        return textwrap.dedent(response).strip()

    # -------------------------------------------------------------------------
    # Private execution helpers
    # -------------------------------------------------------------------------

    def _run_with_timeout(
        self, func: Callable[[], Any], timeout: int
    ) -> Any:
        """Execute a callable with a hard timeout.

        Uses func_timeout if available (preferred — uses threading-based
        interruption). Falls back to a basic threading approach if
        func_timeout is not installed.

        This method is used to wrap the subprocess.run() call with an
        additional outer timeout layer. The subprocess itself also has
        a timeout via subprocess.run(timeout=...), providing two layers
        of protection.

        Args:
            func: A zero-argument callable to execute with timeout.
            timeout: Maximum execution time in seconds.

        Returns:
            The return value of func(), or None if the timeout fires or
            any exception occurs.
        """
        if _FUNC_TIMEOUT_AVAILABLE:
            try:
                return func_timeout(timeout, func)
            except FunctionTimedOut:
                logger.debug(
                    "_run_with_timeout: func_timeout fired after %ds.",
                    timeout,
                )
                return None
            except Exception as exc:
                logger.debug(
                    "_run_with_timeout: func raised exception: %s.", exc
                )
                return None
        else:
            # Fallback: use threading with a join timeout
            import threading

            result_container: List[Any] = [None]
            exception_container: List[Optional[Exception]] = [None]

            def _target() -> None:
                try:
                    result_container[0] = func()
                except Exception as exc:
                    exception_container[0] = exc

            thread: threading.Thread = threading.Thread(
                target=_target, daemon=True
            )
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                logger.debug(
                    "_run_with_timeout (threading fallback): timed out "
                    "after %ds.",
                    timeout,
                )
                # Cannot forcibly kill a Python thread; the daemon flag
                # ensures it won't block process exit.
                return None

            if exception_container[0] is not None:
                logger.debug(
                    "_run_with_timeout (threading fallback): func raised "
                    "exception: %s.",
                    exception_container[0],
                )
                return None

            return result_container[0]

    def _build_memory_limit_preexec_fn(self) -> Optional[Callable[[], None]]:
        """Build a preexec_fn that sets memory limits for a subprocess.

        Uses the resource module (Linux/macOS only) to set RLIMIT_AS
        (virtual address space limit) to max_memory_mb megabytes.

        Returns:
            A zero-argument callable suitable for subprocess.Popen's
            preexec_fn argument, or None if the resource module is not
            available (Windows) or if setting limits is not supported.
        """
        if not _RESOURCE_AVAILABLE:
            return None

        # Convert MB to bytes
        limit_bytes: int = self.max_memory_mb * 1024 * 1024

        def _set_limits() -> None:
            """Set virtual memory limit in the child process."""
            try:
                _resource_module.setrlimit(
                    _resource_module.RLIMIT_AS,
                    (limit_bytes, limit_bytes),
                )
            except (ValueError, _resource_module.error) as exc:
                # Non-fatal: if we can't set the limit, proceed without it.
                # This can happen if the limit exceeds the system maximum.
                logger.debug(
                    "_set_limits: Could not set RLIMIT_AS to %d bytes: %s. "
                    "Proceeding without memory limit.",
                    limit_bytes,
                    exc,
                )

        return _set_limits

    def _build_test_script(
        self, code: str, test_cases: List[str]
    ) -> str:
        """Construct the complete Python script to execute in the sandbox.

        Combines the generated code with test assertions and a sentinel
        print statement. The sentinel guards against false positives from
        code that calls sys.exit(0) before the tests run.

        For MBPP: test_cases is a list of assertion strings.
            Combined script:
                {generated_code}

                {assertion_1}
                {assertion_2}
                ...
                print('__SCORE_TESTS_PASSED__')

        For HumanEval: test_cases typically contains one element — the
            full test function string (including the check() call).
            Combined script:
                {generated_code}

                {test_function_and_check_call}
                print('__SCORE_TESTS_PASSED__')

        Args:
            code: The extracted Python code string.
            test_cases: List of test assertion strings or HumanEval test
                function strings.

        Returns:
            The complete Python script string ready for execution.
        """
        # Join test cases with newlines
        tests_block: str = "\n".join(test_cases)

        # Combine: code + blank line + tests + sentinel
        script: str = (
            code
            + "\n\n"
            + tests_block
            + _SENTINEL_PRINT
        )
        return script

    def _execute_in_sandbox(
        self, code: str, test_cases: List[str]
    ) -> bool:
        """Execute code + test cases in an isolated subprocess.

        This is the core safety-critical method. It:
            1. Builds the combined test script.
            2. Writes it to a temporary file (avoids shell argument length
               limits for long code strings).
            3. Launches a subprocess using the same Python interpreter.
            4. Enforces a hard timeout via subprocess.run(timeout=...).
            5. Checks both the return code AND the sentinel string in stdout
               to confirm all tests ran to completion.
            6. Cleans up the temp file in a finally block.

        All exceptions are caught and mapped to False. This method must
        never raise.

        Args:
            code: The extracted Python code string (non-empty, pre-validated
                by compute()).
            test_cases: List of test assertion strings (non-empty,
                pre-validated by compute()).

        Returns:
            True if the code passes all test cases (subprocess exits 0 AND
            sentinel string appears in stdout), False otherwise.
        """
        # Build the complete test script
        script: str = self._build_test_script(code, test_cases)

        # Write to a temporary file to avoid shell argument length limits
        tmp_file_path: Optional[str] = None
        try:
            # Use delete=False so we control cleanup in the finally block
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="score_code_reward_",
                delete=False,
                encoding="utf-8",
            ) as tmp_file:
                tmp_file.write(script)
                tmp_file_path = tmp_file.name

            logger.debug(
                "_execute_in_sandbox: Wrote %d chars to temp file '%s'.",
                len(script),
                tmp_file_path,
            )

            # Build the preexec_fn for memory limiting (Linux only)
            preexec_fn: Optional[Callable[[], None]] = (
                self._build_memory_limit_preexec_fn()
            )

            # Define the subprocess call as a lambda for _run_with_timeout
            def _run_subprocess() -> Optional[subprocess.CompletedProcess]:
                """Execute the temp file in a subprocess."""
                try:
                    # Use subprocess.run with a hard timeout.
                    # stdout=PIPE to capture sentinel output.
                    # stderr=PIPE to suppress error output from training logs.
                    completed: subprocess.CompletedProcess = subprocess.run(
                        [_PYTHON_EXECUTABLE, tmp_file_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=self.timeout,
                        # preexec_fn is only valid on Unix
                        preexec_fn=preexec_fn if os.name != "nt" else None,
                    )
                    return completed
                except subprocess.TimeoutExpired:
                    logger.debug(
                        "_run_subprocess: subprocess.TimeoutExpired after %ds.",
                        self.timeout,
                    )
                    return None
                except Exception as exc:
                    logger.debug(
                        "_run_subprocess: subprocess.run raised exception: %s.",
                        exc,
                    )
                    return None

            # Execute with outer timeout via _run_with_timeout
            # The outer timeout is slightly longer than the subprocess timeout
            # to allow subprocess.run's own timeout to fire first.
            outer_timeout: int = self.timeout + 2
            completed_process: Optional[subprocess.CompletedProcess] = (
                self._run_with_timeout(_run_subprocess, outer_timeout)
            )

            # Evaluate the result
            if completed_process is None:
                # Timeout or exception in subprocess launch
                logger.debug(
                    "_execute_in_sandbox: subprocess returned None "
                    "(timeout or launch failure). Returning False."
                )
                return False

            # Check return code: 0 = no unhandled exceptions
            if completed_process.returncode != 0:
                logger.debug(
                    "_execute_in_sandbox: subprocess exited with code %d. "
                    "stderr: %s",
                    completed_process.returncode,
                    completed_process.stderr.decode("utf-8", errors="replace")[
                        :200
                    ],
                )
                return False

            # Check sentinel string in stdout to confirm tests ran to
            # completion (guards against sys.exit(0) before tests execute)
            stdout_text: str = completed_process.stdout.decode(
                "utf-8", errors="replace"
            )
            if _SENTINEL_STRING not in stdout_text:
                logger.debug(
                    "_execute_in_sandbox: sentinel '%s' not found in stdout. "
                    "Tests may not have completed. stdout: '%s'",
                    _SENTINEL_STRING,
                    stdout_text[:200],
                )
                return False

            logger.debug(
                "_execute_in_sandbox: All tests passed (returncode=0, "
                "sentinel found)."
            )
            return True

        except Exception as exc:
            # Catch-all: protect the training loop from any unexpected error
            logger.debug(
                "_execute_in_sandbox: Unexpected exception: %s. "
                "Returning False.",
                exc,
            )
            return False

        finally:
            # Always clean up the temp file
            if tmp_file_path is not None:
                try:
                    os.unlink(tmp_file_path)
                    logger.debug(
                        "_execute_in_sandbox: Deleted temp file '%s'.",
                        tmp_file_path,
                    )
                except OSError as exc:
                    logger.debug(
                        "_execute_in_sandbox: Could not delete temp file "
                        "'%s': %s.",
                        tmp_file_path,
                        exc,
                    )
