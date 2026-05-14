```python
## rewards/math_reward.py
"""Binary reward function for MATH problems in SCoRe.

This module implements MathReward, which computes a binary correctness
reward (1.0 or 0.0) for model responses on MATH dataset problems. The
reward is 1.0 if and only if the model's extracted answer is mathematically
equivalent to the ground truth answer.

The paper states (Section 6): "we use binary rewards during training,
indicating whether the model's answer matches the ground truth one (for
MATH)." The config confirms: reward.math.reward_type: "binary_answer_match"
and reward.math.use_sympy: true.

Two-stage comparison strategy:
    1. Symbolic equivalence via sympy (primary) — handles fractions,
       square roots, equivalent algebraic forms.
    2. Normalized string matching (fallback) — handles cases where sympy
       parsing fails.

Typical usage:
    from rewards.math_reward import MathReward

    reward_fn = MathReward()
    score = reward_fn.compute(model_response, ground_truth_answer)
    # score is 1.0 or 0.0
"""

import logging
import re
import unicodedata
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the module can be imported even if
# these packages are not installed (tests, linting, etc.).
# ---------------------------------------------------------------------------
try:
    import sympy
    from sympy import simplify, sympify
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )

    _SYMPY_AVAILABLE: bool = True
except ImportError:
    _SYMPY_AVAILABLE = False
    logger.warning(
        "sympy is not installed. Symbolic equivalence checking will be "
        "disabled; only string matching will be used."
    )

try:
    from latex2sympy2 import latex2sympy

    _LATEX2SYMPY_AVAILABLE: bool = True
except ImportError:
    _LATEX2SYMPY_AVAILABLE = False
    logger.warning(
        "latex2sympy2 is not installed. LaTeX-to-sympy conversion will "
        "fall back to sympy.sympify() directly."
    )

try:
    from func_timeout import func_timeout, FunctionTimedOut

    _FUNC_TIMEOUT_AVAILABLE: bool = True
except ImportError:
    _FUNC_TIMEOUT_AVAILABLE = False
    logger.warning(
        "func_timeout is not installed. Sympy calls will not be "
        "time-bounded — pathological inputs may cause hangs."
    )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Timeout in seconds for a single sympy equivalence check.
# Prevents pathological sympy inputs from stalling the training loop.
_SYMPY_TIMEOUT_SECONDS: int = 5

# Tolerance for floating-point numeric comparison fallback.
_FLOAT_TOLERANCE: float = 1e-6


class MathReward:
    """Binary reward function for MATH dataset problems.

    Computes a binary correctness reward (1.0 or 0.0) by:
        1. Extracting the answer from the model's response using the
           canonical "Final Answer: The final answer is $answer$. I hope
           it is correct." format (Appendix C).
        2. Normalizing both the extracted answer and the ground truth.
        3. Checking equivalence via sympy (primary) then string match
           (fallback).

    All regex patterns are pre-compiled in __init__ for efficiency at
    training scale (thousands of reward evaluations per step).

    Attributes:
        _final_answer_pattern: Compiled regex for extracting the answer
            from the canonical response format.
        _boxed_pattern: Compiled regex for extracting content from
            \\boxed{...} expressions.
        _frac_pattern: Compiled regex for converting \\frac{a}{b} to (a)/(b).
        _sqrt_pattern: Compiled regex for converting \\sqrt{x} to sqrt(x).
        _text_pattern: Compiled regex for extracting content from \\text{...}.
        _whitespace_pattern: Compiled regex for collapsing whitespace.
        _percent_pattern: Compiled regex for percentage expressions.
        _unicode_minus_pattern: Compiled regex for Unicode minus signs.
    """

    def __init__(self) -> None:
        """Initialize MathReward with pre-compiled regex patterns.

        Pre-compiling all patterns here avoids repeated compilation overhead
        during training, where compute() is called thousands of times per
        training step.
        """
        # ------------------------------------------------------------------
        # Pattern 1: Extract answer from canonical response format.
        # Appendix C format: "Final Answer: The final answer is $answer$.
        # I hope it is correct."
        # - Case-insensitive for robustness.
        # - re.DOTALL so the answer can span multiple lines.
        # - Captures everything between "is" and ". I hope" (or end of
        #   string if the model truncates the sentence).
        # - Takes the LAST match (model may self-correct within a turn,
        #   Appendix D: "the model learns to occasionally self-correct
        #   within a turn").
        # ------------------------------------------------------------------
        self._final_answer_pattern: re.Pattern = re.compile(
            r"[Ff]inal\s+[Aa]nswer\s*:\s*[Tt]he\s+final\s+answer\s+is"
            r"\s+(.*?)\s*\.?\s*[Ii]\s+hope",
            re.DOTALL | re.IGNORECASE,
        )

        # Fallback pattern: "Final Answer: The final answer is X" at end of
        # string (model truncated "I hope it is correct.")
        self._final_answer_fallback_pattern: re.Pattern = re.compile(
            r"[Ff]inal\s+[Aa]nswer\s*:\s*[Tt]he\s+final\s+answer\s+is"
            r"\s+(.*?)\s*$",
            re.DOTALL | re.IGNORECASE,
        )

        # ------------------------------------------------------------------
        # Pattern 2: Extract content from \boxed{...}.
        # Used both in extract_answer (fallback) and _normalize_answer.
        # Simple pattern for non-nested content; nested braces handled
        # by _extract_boxed_content() which uses brace counting.
        # ------------------------------------------------------------------
        self._boxed_simple_pattern: re.Pattern = re.compile(
            r"\\boxed\s*\{", re.DOTALL
        )

        # ------------------------------------------------------------------
        # Pattern 3: Convert \frac{a}{b} → (a)/(b).
        # Handles single-level fractions; nested fractions require
        # iterative application.
        # ------------------------------------------------------------------
        self._frac_pattern: re.Pattern = re.compile(
            r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}"
        )

        # ------------------------------------------------------------------
        # Pattern 4: Convert \sqrt{x} → sqrt(x) and \sqrt[n]{x} → x**(1/n).
        # ------------------------------------------------------------------
        self._sqrt_pattern: re.Pattern = re.compile(
            r"\\sqrt\s*(?:\[([^\]]*)\])?\s*\{([^{}]*)\}"
        )

        # ------------------------------------------------------------------
        # Pattern 5: Extract content from \text{...}.
        # ------------------------------------------------------------------
        self._text_pattern: re.Pattern = re.compile(
            r"\\text\s*\{([^{}]*)\}"
        )

        # ------------------------------------------------------------------
        # Pattern 6: Collapse multiple whitespace characters to a single
        # space.
        # ------------------------------------------------------------------
        self._whitespace_pattern: re.Pattern = re.compile(r"\s+")

        # ------------------------------------------------------------------
        # Pattern 7: Percentage expressions.
        # Matches "42%", "3.14%", "42 \%", "3.14 \\%".
        # ------------------------------------------------------------------
        self._percent_pattern: re.Pattern = re.compile(
            r"([\d]+(?:\.[\d]+)?)\s*(?:\\%|%)"
        )

        # ------------------------------------------------------------------
        # Pattern 8: Unicode minus sign (U+2212) → ASCII hyphen-minus.
        # ------------------------------------------------------------------
        self._unicode_minus_pattern: re.Pattern = re.compile(r"\u2212")

        # ------------------------------------------------------------------
        # Pattern 9: Dollar sign delimiters.
        # ------------------------------------------------------------------
        self._dollar_pattern: re.Pattern = re.compile(r"\$")

        # ------------------------------------------------------------------
        # Pattern 10: LaTeX display/inline math delimiters.
        # ------------------------------------------------------------------
        self._latex_delimiters_pattern: re.Pattern = re.compile(
            r"\\[\(\)\[\]]"
        )

        # ------------------------------------------------------------------
        # Pattern 11: Trailing/leading punctuation that is not part of a
        # number (commas, periods not preceded by a digit).
        # ------------------------------------------------------------------
        self._trailing_punct_pattern: re.Pattern = re.compile(
            r"[,;]\s*$|^\s*[,;]"
        )

        # ------------------------------------------------------------------
        # Pattern 12: LaTeX spacing commands that carry no mathematical
        # meaning.
        # ------------------------------------------------------------------
        self._latex_spacing_pattern: re.Pattern = re.compile(
            r"\\(?:,|!|;|quad|qquad|hspace\{[^}]*\}|vspace\{[^}]*\})"
        )

        # ------------------------------------------------------------------
        # Pattern 13: \left and \right delimiters (keep the bracket itself).
        # ------------------------------------------------------------------
        self._left_right_pattern: re.Pattern = re.compile(
            r"\\(?:left|right)\s*"
        )

        # ------------------------------------------------------------------
        # Pattern 14: \cdot (multiplication dot) → *.
        # ------------------------------------------------------------------
        self._cdot_pattern: re.Pattern = re.compile(r"\\cdot")

        # ------------------------------------------------------------------
        # Pattern 15: \times → *.
        # ------------------------------------------------------------------
        self._times_pattern: re.Pattern = re.compile(r"\\times")

        # ------------------------------------------------------------------
        # Pattern 16: \pm → ± (keep as-is for string comparison; sympy
        # handles it).
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Pattern 17: Detect if a string looks like a pure integer or float.
        # ------------------------------------------------------------------
        self._numeric_pattern: re.Pattern = re.compile(
            r"^-?\s*[\d]+(?:\.[\d]+)?$"
        )

        logger.debug("MathReward initialized with pre-compiled regex patterns.")

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def compute(self, prediction: str, ground_truth: str) -> float:
        """Compute the binary correctness reward for a model response.

        This is the main entry point called by RewardFunction.compute_reward().
        Returns 1.0 if the model's answer is mathematically equivalent to
        the ground truth, 0.0 otherwise.

        The two-stage comparison strategy:
            1. Symbolic equivalence via sympy (primary) — handles fractions,
               square roots, equivalent algebraic forms.
            2. Normalized string matching (fallback) — handles cases where
               sympy parsing fails.

        Args:
            prediction: The full model response string, expected to contain
                the canonical "Final Answer: The final answer is $answer$.
                I hope it is correct." format from Appendix C.
            ground_truth: The ground truth answer string. May be in raw
                LaTeX form (e.g., "\\frac{3}{7}") or plain text (e.g., "3/7").
                Some MATH dataset answers are stored with \\boxed{} wrapping.

        Returns:
            1.0 if the prediction is correct, 0.0 otherwise.
        """
        # Step 1: Extract the answer from the model's response
        extracted_pred: str = self.extract_answer(prediction)

        # Step 2: Normalize both the extracted prediction and ground truth
        normalized_pred: str = self._normalize_answer(extracted_pred)
        normalized_gt: str = self._normalize_answer(ground_truth)

        # Step 3: Early exit if either normalized string is empty
        if not normalized_pred or not normalized_gt:
            logger.debug(
                "Empty answer after normalization. pred='%s', gt='%s'. "
                "Returning 0.0.",
                normalized_pred,
                normalized_gt,
            )
            return 0.0

        # Step 4: Primary — symbolic equivalence via sympy
        if _SYMPY_AVAILABLE and _LATEX2SYMPY_AVAILABLE:
            try:
                if self._sympy_equivalence(normalized_pred, normalized_gt):
                    logger.debug(
                        "Sympy equivalence: CORRECT. pred='%s', gt='%s'.",
                        normalized_pred,
                        normalized_gt,
                    )
                    return 1.0
            except Exception as exc:
                logger.debug(
                    "Sympy equivalence raised unexpected exception: %s. "
                    "Falling back to string match.",
                    exc,
                )

        # Step 5: Fallback — normalized string matching
        if self._string_match(normalized_pred, normalized_gt):
            logger.debug(
                "String match: CORRECT. pred='%s', gt='%s'.",
                normalized_pred,
                normalized_gt,
            )
            return 1.0

        logger.debug(
            "INCORRECT. pred='%s', gt='%s'.",
            normalized_pred,
            normalized_gt,
        )
        return 0.0

    def extract_answer(self, response: str) -> str:
        """Extract the answer string from a model response.

        Parses the model's full response text and extracts just the answer
        portion. The model is instructed (Appendix C) to always end with:
            "Final Answer: The final answer is $answer$. I hope it is correct."

        Takes the LAST match when multiple "Final Answer" patterns appear
        (handles within-turn self-correction, Appendix D).

        Fallback chain:
            1. Primary regex: "Final Answer: The final answer is X. I hope"
            2. Fallback regex: "Final Answer: The final answer is X" (at end)
            3. \\boxed{...} extraction anywhere in the response
            4. Last non-empty line of the response

        Args:
            response: The full model response string.

        Returns:
            The extracted answer string (not yet normalized). Returns an
            empty string if no answer can be extracted.
        """
        if not response or not response.strip():
            return ""

        # ------------------------------------------------------------------
        # Attempt 1: Primary pattern — "Final Answer: The final answer is
        # $answer$. I hope it is correct."
        # Take the LAST match (within-turn self-correction, Appendix D).
        # ------------------------------------------------------------------
        primary_matches: List[re.Match] = list(
            self._final_answer_pattern.finditer(response)
        )
        if primary_matches:
            last_match: re.Match = primary_matches[-1]
            raw_answer: str = last_match.group(1).strip()
            # Strip surrounding $ delimiters
            raw_answer = self._dollar_pattern.sub("", raw_answer).strip()
            if raw_answer:
                return raw_answer

        # ------------------------------------------------------------------
        # Attempt 2: Fallback pattern — "Final Answer: The final answer is X"
        # at end of string (model truncated "I hope it is correct.").
        # ------------------------------------------------------------------
        fallback_matches: List[re.Match] = list(
            self._final_answer_fallback_pattern.finditer(response)
        )
        if fallback_matches:
            last_fallback: re.Match = fallback_matches[-1]
            raw_answer = last_fallback.group(1).strip()
            raw_answer = self._dollar_pattern.sub("", raw_answer).strip()
            if raw_answer:
                return raw_answer

        # ------------------------------------------------------------------
        # Attempt 3: Extract from \boxed{...} anywhere in the response.
        # Some models produce boxed answers without the full sentence.
        # ------------------------------------------------------------------
        boxed_answer: str = self._extract_last_boxed(response)
        if boxed_answer:
            return boxed_answer

        # ------------------------------------------------------------------
        # Attempt 4: Last resort — return the last non-empty line.
        # This will likely fail the reward check but avoids crashing.
        # ------------------------------------------------------------------
        lines: List[str] = [
            line.strip() for line in response.split("\n") if line.strip()
        ]
        if lines:
            logger.debug(
                "Could not extract answer via primary/fallback patterns or "
                "\\boxed{}. Using last non-empty line as fallback: '%s'.",
                lines[-1][:80],
            )
            return lines[-1]

        return ""

    # -------------------------------------------------------------------------
    # Private normalization helpers
    # -------------------------------------------------------------------------

    def _normalize_answer(self, answer: str) -> str:
        """Transform an answer string into a canonical form.

        Applied to both predictions and ground truth answers before
        comparison. This method is idempotent — calling it twice produces
        the same result as calling it once.

        Normalization steps (applied in order):
            1. Strip surrounding whitespace.
            2. Remove $ delimiters.
            3. Remove LaTeX display/inline math delimiters (\\(, \\), \\[, \\]).
            4. Extract from \\boxed{...} if present.
            5. Replace Unicode minus with ASCII hyphen-minus.
            6. Convert \\frac{a}{b} → (a)/(b).
            7. Convert \\sqrt{x} → sqrt(x), \\sqrt[n]{x} → x**(1/n).
            8. Extract content from \\text{...}.
            9. Remove LaTeX spacing commands.
            10. Remove \\left and \\right (keep the bracket).
            11. Convert \\cdot and \\times → *.
            12. Convert \\pi → pi, \\infty → oo, \\e → E.
            13. Collapse whitespace.
            14. Strip trailing/leading punctuation (commas, semicolons).
            15. Convert to lowercase.

        Args:
            answer: The raw answer string to normalize.

        Returns:
            The normalized answer string. Returns empty string if input
            is empty or whitespace-only.
        """
        if not answer or not answer.strip():
            return ""

        s: str = answer.strip()

        # Step 1: Remove $ delimiters
        s = self._dollar_pattern.sub("", s)

        # Step 2: Remove LaTeX display/inline math delimiters
        s = self._latex_delimiters_pattern.sub("", s)

        # Step 3: Extract from \boxed{...} if present
        boxed_content: str = self._extract_last_boxed(s)
        if boxed_content:
            s = boxed_content

        # Step 4: Replace Unicode minus (U+2212) with ASCII hyphen-minus
        s = self._unicode_minus_pattern.sub("-", s)

        # Step 5: Normalize other Unicode characters
        # Normalize to NFC form to handle composed vs decomposed characters
        s = unicodedata.normalize("NFC", s)

        # Step 6: Convert \frac{a}{b} → (a)/(b)
        # Apply iteratively to handle nested fractions
        prev: str = ""
        max_iterations: int = 10
        iteration: int = 0
        while prev != s and iteration < max_iterations:
            prev = s
            s = self._frac_pattern.sub(r"(\1)/(\2)", s)
            iteration += 1

        # Step 7: Convert \sqrt[n]{x} → x**(1/n) and \sqrt{x} → sqrt(x)
        def _replace_sqrt(m: re.Match) -> str:
            """Replace \\sqrt[n]{x} or \\sqrt{x} with sympy-compatible form."""
            n_part: Optional[str] = m.group(1)  # Optional index n
            x_part: str = m.group(2)  # Content under the radical
            if n_part and n_part.strip():
                return f"({x_part})**(1/({n_part}))"
            return f"sqrt({x_part})"

        s = self._sqrt_pattern.sub(_replace_sqrt, s)

        # Step 8: Extract content from \text{...}
        s = self._text_pattern.sub(r"\1", s)

        # Step 9: Remove LaTeX spacing commands
        s = self._latex_spacing_pattern.sub("", s)

        # Step 10: Remove \left and \right (keep the bracket character)
        s = self._left_right_pattern.sub("", s)

        # Step 11: Convert \cdot → * and \times → *
        s = self._cdot_pattern.sub("*", s)
        s = self._times_pattern.sub("*", s)

        # Step 12: Convert named constants
        # Order matters: \pi before \prime, \infty before \in
        s = re.sub(r"\\pi\b", "pi", s)
        s = re.sub(r"\\infty\b", "oo", s)
        s = re.sub(r"\\e\b", "E", s)
        s = re.sub(r"\\log\b", "log", s)
        s = re.sub(r"\\ln\b", "ln", s)
        s = re.sub(r"\\sin\b", "sin", s)
        s = re.sub(r"\\cos\b", "cos", s)
        s = re.sub(r"\\tan\b", "tan", s)

        # Step 13: Remove remaining backslash commands that have no value
        # (e.g., \!, \,, \;, \:) — already handled by spacing pattern,
        # but catch any remaining single-letter commands that are purely
        # formatting.
        s = re.sub(r"\\[a-zA-Z]+\b", "", s)

        # Step 14: Collapse whitespace
        s = self._whitespace_pattern.sub(" ", s).strip()

        # Step 15: Strip trailing/leading punctuation
        s = self._trailing_punct_pattern.sub("", s).strip()

        # Step 16: Convert to lowercase for string comparison
        s = s.lower()

        return s.strip()

    def _extract_last_boxed(self, text: str) -> str:
        """Extract the content of the last \\boxed{...} in a string.

        Uses brace counting to handle nested braces correctly
        (e.g., \\boxed{\\frac{1}{2}} → "\\frac{1}{2}").

        Args:
            text: The string to search for \\boxed{...} expressions.

        Returns:
            The content of the last \\boxed{...}, or empty string if none
            found.
        """
        # Find all starting positions of \boxed{
        matches: List[re.Match] = list(
            self._boxed_simple_pattern.finditer(text)
        )
        if not matches:
            return ""

        # Take the last \boxed{ occurrence
        last_match: re.Match = matches[-1]
        start_pos: int = last_match.end()  # Position after the opening {

        # Walk forward counting brace depth to find the matching closing }
        depth: int = 1
        pos: int = start_pos
        content_chars: List[str] = []

        while pos < len(text) and depth > 0:
            char: str = text[pos]
            if char == "{":
                depth += 1
                content_chars.append(char)
            elif char == "}":
                depth -= 1
                if depth > 0:
                    content_chars.append(char)
            else:
                content_chars.append(char)
            pos += 1

        if depth != 0:
            logger.debug(
                "Unbalanced braces in \\boxed{} extraction. "
                "Returning partial content."
            )

        return "".join(content_chars).strip()

    # -------------------------------------------------------------------------
    # Private comparison methods
    # -------------------------------------------------------------------------

    def _sympy_equivalence(self, pred: str, gt: str) -> bool:
        """Check mathematical equivalence using sympy.

        This is the primary comparison method. Uses latex2sympy2 for LaTeX
        parsing (better support than raw sympy.sympify for MATH dataset
        answers), with a hard timeout via func_timeout to prevent hangs.

        Args:
            pred: Normalized predicted answer string.
            gt: Normalized ground truth answer string.

        Returns:
            True if the expressions are mathematically equivalent,
            False if not equivalent or if parsing/evaluation fails.
        """
        if not _SYMPY_AVAILABLE:
            return False

        def _check_equivalence() -> bool:
            """Inner function wrapped by timeout."""
            pred_expr = self._parse_to_sympy(pred)
            gt_expr = self._parse_to_sympy(gt)

            if pred_expr is None or gt_expr is None:
                return False

            # Method 1: simplify(pred - gt) == 0
            try:
                diff = simplify(pred_expr - gt_expr)
                if diff == 0:
                    return True
            except Exception:
                pass

            # Method 2: .equals() — uses a different internal algorithm
            try:
                if pred_expr.equals(gt_expr):
                    return True
            except Exception:
                pass

            # Method 3: Direct equality after simplification
            try:
                pred_simplified = simplify(pred_expr)
                gt_simplified = simplify(gt_expr)
                if pred_simplified == gt_simplified:
                    return True
            except Exception:
                pass

            # Method 4: Numeric evaluation at a test point
            # Useful for expressions that are symbolically complex but
            # numerically equal (e.g., different forms of the same constant)
            try:
                pred_val = complex(pred_expr.evalf())
                gt_val = complex(gt_expr.evalf())
                if abs(pred_val - gt_val) < _FLOAT_TOLERANCE:
                    return True
            except Exception:
                pass

            return False

        # Apply timeout protection
        if _FUNC_TIMEOUT_AVAILABLE:
            try:
                result: bool = func_timeout(
                    _SYMPY_TIMEOUT_SECONDS, _check_equivalence
                )
                return result
            except FunctionTimedOut:
                logger.debug(
                    "Sympy equivalence check timed out after %ds for "
                    "pred='%s', gt='%s'. Returning False.",
                    _SYMPY_TIMEOUT_SECONDS,
                    pred[:50],
                    gt[:50],
                )
                return False
            except Exception as exc:
                logger.debug(
                    "Sympy equivalence check failed with exception: %s. "
                    "pred='%s', gt='%s'.",
                    exc,
                    pred[:50],
                    gt[:50],
                )
                return False
        else:
            # No timeout protection — run directly
            try:
                return _check_equivalence()
            except Exception as exc:
                logger.debug(
                    "Sympy equivalence check failed: %s. "
                    "pred='%s', gt='%s'.",
                    exc,
                    pred[:50],
                    gt[:50],
                )
                return False

    def _parse_to_sympy(self, expr_str: str) -> Optional[object]:
        """Parse a normalized answer string into a sympy expression.

        Tries multiple parsing strategies in order:
            1. latex2sympy2.latex2sympy() — best for LaTeX expressions.
            2. sympy.sympify() with standard transformations.
            3. sympy.sympify() with implicit multiplication.
            4. Direct float conversion for pure numeric strings.

        Args:
            expr_str: A normalized answer string.

        Returns:
            A sympy expression object, or None if all parsing attempts fail.
        """
        if not expr_str:
            return None

        # Strategy 1: latex2sympy2 (best for LaTeX)
        if _LATEX2SYMPY_AVAILABLE:
            try:
                result = latex2sympy(expr_str)
                if result is not None:
                    return result
            except Exception:
                pass

        if not _SYMPY_AVAILABLE:
            return None

        # Strategy 2: sympy.sympify with standard transformations
        try:
            transformations = standard_transformations + (
                implicit_multiplication_application,
            )
            result = parse_expr(
                expr_str,
                transformations=transformations,
                evaluate=True,
            )
            return result
        except Exception:
            pass

        # Strategy 3: Basic sympy.sympify
        try:
            result = sympify(expr_str, evaluate=True)
            return result
        except Exception:
            pass

        # Strategy 4: Direct float conversion for pure numeric strings
        try:
            float_val: float = float(expr_str)
            return sympify(float_val)
        except (ValueError, TypeError):
            pass

        return None

    def _string_match(self, pred: str, gt: str) -> bool:
        """Fallback string-based comparison when sympy parsing fails.

        Applies multiple comparison strategies in order of strictness.
        Conservative by design — prefers false negatives over false
        positives to avoid corrupting the RL training signal.

        Args:
            pred: Normalized predicted answer string.
            gt: Normalized ground truth answer string.

        Returns:
            True if the strings are considered equivalent by any strategy,
            False otherwise.
        """
        # Strategy 1: Direct normalized string equality
        if pred == gt:
            return True

        # Strategy 2: Remove all whitespace and compare
        pred_no_space: str = re.sub(r"\s+", "", pred)
        gt_no_space: str = re.sub(r"\s+", "", gt)
        if pred_no_space == gt_no_space:
            return True

        # Strategy 3: Numeric float comparison with tolerance
        # Handles cases like "0.333..." vs "1/3" that sympy might miss
        pred_float: Optional[float] = self._try_parse_float(pred)
        gt_float: Optional[float] = self._try_parse_float(gt)
        if pred_float is not None and gt_float is not None:
            if abs(pred_float - gt_float) < _FLOAT_TOLERANCE:
                return True

        # Strategy 4: Integer comparison
        pred_int: Optional[int] = self._try_parse_int(pred)
        gt_int: Optional[int] = self._try_parse_int(gt)
        if pred_int is not None and gt_int is not None:
            if pred_int == gt_int:
                return True

        # Strategy 5: Compare after stripping common LaTeX wrappers
        # that normalization might have missed
        pred_stripped: str = self._aggressive_strip(pred)
        gt_stripped: str = self._aggressive_strip(gt)
        if pred_stripped and gt_stripped and pred_stripped == gt_stripped:
            return True

        return False

    def _try_parse_float(self, s: str) -> Optional[float]:
        """Attempt to parse a string as a float.

        Handles simple fractions (e.g., "1/2" → 0.5) in addition to
        standard float strings.

        Args:
            s: The string to parse.

        Returns:
            The float value, or None if parsing fails.
        """
        # Direct float conversion
        try:
            return float(s)
        except (ValueError, TypeError):
            pass

        # Simple fraction: "a/b" or "(a)/(b)"
        # Remove parentheses first
        s_clean: str = re.sub(r"[()]", "", s).strip()
        fraction_match: Optional[re.Match] = re.match(
            r"^(-?\s*[\d]+(?:\.[\d]+)?)\s*/\s*(-?\s*[\d]+(?:\.[\d]+)?)$",
            s_clean,
        )
        if fraction_match:
            try