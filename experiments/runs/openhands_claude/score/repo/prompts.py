"""
All prompts and self-correction instructions from Appendix C of the paper.
"""

# ---------------------------------------------------------------------------
# MATH prompts (Appendix C)
# ---------------------------------------------------------------------------

MATH_SYSTEM_PROMPT = (
    "You are a math expert. When you respond, respond only with the Solution of the "
    "final Problem, thinking step by step. At the end of the Solution, when you give "
    "your final answer, write it in the form "
    '"Final Answer: The final answer is $answer$. I hope it is correct."'
)

MATH_SELF_CORRECTION_INSTRUCTION = (
    "There might be an error in the solution above because of lack of understanding "
    "of the question. Please correct the error, if any, and rewrite the solution. "
    "Only output the final solution! At the end of the Solution, when you give your "
    "final answer, write it in the form "
    '"Final Answer: The final answer is $answer$. I hope it is correct."'
)

# ---------------------------------------------------------------------------
# MBPP prompts (Appendix C) — 3-shot canonical prompt
# ---------------------------------------------------------------------------

MBPP_SHOT_1 = """\
You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:

assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)

[BEGIN]
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return (res)
[DONE]"""

MBPP_SHOT_2 = """\
You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True

[BEGIN]
import math
def is_not_prime(n):
    result = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
[DONE]"""

MBPP_SHOT_3 = """\
You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:

assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]

[BEGIN]
import heapq as hq
def heap_queue_largest(nums, n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
[DONE]"""

MBPP_FEW_SHOT_PREFIX = "\n\n".join([MBPP_SHOT_1, MBPP_SHOT_2, MBPP_SHOT_3])

MBPP_SELF_CORRECTION_INSTRUCTION = (
    "There might be an error in the code above because of lack of understanding of "
    "the question. Please correct the error, if any, and rewrite the solution. "
    "Only output the final correct Python program!"
)

# HumanEval uses zero-shot prompting (Section 6, Evaluation prompts)
HUMANEVAL_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the following Python function. "
    "Only output the complete function implementation."
)

HUMANEVAL_SELF_CORRECTION_INSTRUCTION = MBPP_SELF_CORRECTION_INSTRUCTION


def build_math_first_turn_prompt(problem: str) -> str:
    """Build the zero-shot CoT prompt for the first MATH attempt."""
    return f"{MATH_SYSTEM_PROMPT}\n\nProblem: {problem}"


def build_math_second_turn_prompt(problem: str, first_attempt: str) -> str:
    """Build the self-correction prompt for the second MATH attempt."""
    first_turn = build_math_first_turn_prompt(problem)
    return (
        f"{first_turn}\n\n{first_attempt}\n\n"
        f"{MATH_SELF_CORRECTION_INSTRUCTION}"
    )


def build_mbpp_first_turn_prompt(problem: str, test_cases: str) -> str:
    """Build the 3-shot prompt for the first MBPP attempt."""
    task_description = (
        f"You are an expert Python programmer, and here is your task: {problem} "
        f"Your code should pass these tests:\n\n{test_cases}\n\n[BEGIN]"
    )
    return f"{MBPP_FEW_SHOT_PREFIX}\n\n{task_description}"


def build_mbpp_second_turn_prompt(
    problem: str, test_cases: str, first_attempt: str
) -> str:
    """Build the self-correction prompt for the second MBPP attempt."""
    first_turn = build_mbpp_first_turn_prompt(problem, test_cases)
    return (
        f"{first_turn}\n{first_attempt}\n[DONE]\n\n"
        f"{MBPP_SELF_CORRECTION_INSTRUCTION}"
    )


def build_humaneval_first_turn_prompt(prompt: str) -> str:
    """Build the zero-shot prompt for the first HumanEval attempt."""
    return f"{HUMANEVAL_SYSTEM_PROMPT}\n\n{prompt}"


def build_humaneval_second_turn_prompt(prompt: str, first_attempt: str) -> str:
    """Build the self-correction prompt for the second HumanEval attempt."""
    first_turn = build_humaneval_first_turn_prompt(prompt)
    return (
        f"{first_turn}\n\n{first_attempt}\n\n"
        f"{HUMANEVAL_SELF_CORRECTION_INSTRUCTION}"
    )
