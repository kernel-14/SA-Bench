"""
Prompt templates from Appendix C of the paper.

Contains the exact prompts used for:
- MATH zero-shot evaluation
- MATH self-correction instruction
- MBPP 3-shot training prompt
- MBPP/HumanEval self-correction instruction

These prompts are critical for reproducing the paper's results since
self-correction behavior is sensitive to prompt design.
"""

# =============================================================================
# MATH Prompts (Appendix C)
# =============================================================================

MATH_ZERO_SHOT_PROMPT = """You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct.\""""

MATH_SELF_CORRECTION_INSTRUCTION = """There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct.\""""


def build_math_prompt(problem: str) -> str:
    """
    Build a zero-shot MATH prompt for first-attempt solving.
    
    Follows the format from Appendix C.
    """
    return f"""You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."

Problem: {problem}"""


def build_math_correction_prompt(problem: str, previous_solution: str) -> str:
    """
    Build a MATH self-correction prompt for second-attempt correction.
    
    Includes the problem, previous solution, and correction instruction.
    The correction instruction does NOT reveal answer correctness.
    """
    return f"""You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct."

Problem: {problem}

Your previous solution:
{previous_solution}

There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct.\""""


# =============================================================================
# MBPP Prompts (Appendix C)
# =============================================================================

MBPP_THREE_SHOT_PROMPT = """You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:

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

[BEGIN]
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
[DONE]"""

MBPP_SELF_CORRECTION_INSTRUCTION = """There might be an error in the code above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program!"""

HUMANEVAL_ZERO_SHOT_PROMPT = """You are an expert Python programmer. Complete the following function. Only output the completed function.

"""

HUMANEVAL_SELF_CORRECTION_INSTRUCTION = """There might be an error in the code above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program!"""


def build_code_prompt(
    problem: str,
    use_3shot: bool = False,
) -> str:
    """
    Build a code generation prompt.
    
    For MBPP training: uses 3-shot prompt from Appendix C.
    For HumanEval evaluation: uses zero-shot prompt.
    """
    if use_3shot:
        return f"""{MBPP_THREE_SHOT_PROMPT}

You are an expert Python programmer, and here is your task: {problem}

[BEGIN]
"""
    else:
        return f"""{HUMANEVAL_ZERO_SHOT_PROMPT}{problem}"""


def build_code_correction_prompt(
    problem: str,
    previous_solution: str,
    use_3shot: bool = False,
) -> str:
    """
    Build a code self-correction prompt.
    
    Includes the problem, previous solution, and correction instruction.
    """
    base = build_code_prompt(problem, use_3shot=use_3shot)
    return f"""{base}

Your previous solution:
{previous_solution}

{MBPP_SELF_CORRECTION_INSTRUCTION}"""


__all__ = [
    "MATH_ZERO_SHOT_PROMPT",
    "MATH_SELF_CORRECTION_INSTRUCTION",
    "MBPP_THREE_SHOT_PROMPT",
    "MBPP_SELF_CORRECTION_INSTRUCTION",
    "HUMANEVAL_ZERO_SHOT_PROMPT",
    "HUMANEVAL_SELF_CORRECTION_INSTRUCTION",
    "build_math_prompt",
    "build_math_correction_prompt",
    "build_code_prompt",
    "build_code_correction_prompt",
]
