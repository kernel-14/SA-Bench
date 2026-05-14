"""
Prompts used in SCoRe training and evaluation.
From Appendix C of the paper.
"""

# ============================================================
# MATH Prompts
# ============================================================

MATH_SYSTEM_PROMPT = (
    "You are a math expert. When you respond, respond only with the Solution of the "
    "final Problem, thinking step by step. At the end of the Solution, when you give "
    "your final answer, write it in the form "
    '"Final Answer: The final answer is $<answer>$. I hope it is correct."'
)

MATH_SELF_CORRECTION_INSTRUCTION = (
    "There might be an error in the solution above because of lack of understanding "
    "of the question. Please correct the error, if any, and rewrite the solution. "
    "Only output the final solution! At the end of the Solution, when you give your "
    "final answer, write it in the form "
    '"Final Answer: The final answer is $<answer>$. I hope it is correct."'
)

# ============================================================
# MBPP/HumanEval Prompts
# ============================================================

MBPP_3SHOT_EXAMPLES = [
    {
        "task": "Write a function to find the similar elements from the given two tuple lists.",
        "tests": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
        "solution": (
            "def similar_elements(test_tup1, test_tup2):\n"
            "    res = tuple(set(test_tup1) & set(test_tup2))\n"
            "    return (res)"
        ),
    },
    {
        "task": "Write a python function to identify non-prime numbers.",
        "tests": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
        "solution": (
            "import math\n"
            "def is_not_prime(n):\n"
            "    result = False\n"
            "    for i in range(2, int(math.sqrt(n)) + 1):\n"
            "        if n % i == 0:\n"
            "            result = True\n"
            "    return result"
        ),
    },
    {
        "task": (
            "Write a function to find the largest integers from a given list of numbers "
            "using heap queue algorithm."
        ),
        "tests": [
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]",
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]",
            "assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]",
        ],
        "solution": (
            "import heapq as hq\n"
            "def heap_queue_largest(nums, n):\n"
            "    largest_nums = hq.nlargest(n, nums)\n"
            "    return largest_nums"
        ),
    },
]

CODE_SELF_CORRECTION_INSTRUCTION = (
    "There might be an error in the code above because of lack of understanding of "
    "the question. Please correct the error, if any, and rewrite the solution. "
    "Only output the final correct Python program!"
)


def build_mbpp_3shot_prompt(task: str, tests: list) -> str:
    """
    Build the 3-shot MBPP prompt for a given task.
    
    Args:
        task: Task description
        tests: List of test case strings
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    
    # Add 3-shot examples
    for example in MBPP_3SHOT_EXAMPLES:
        prompt += f"You are an expert Python programmer, and here is your task: {example['task']}"
        prompt += " Your code should pass these tests:\n\n"
        for test in example["tests"]:
            prompt += f"{test}\n"
        prompt += "\n[BEGIN]\n\n"
        prompt += example["solution"] + "\n\n"
        prompt += "[DONE]\n\n"
    
    # Add the actual task
    prompt += f"You are an expert Python programmer, and here is your task: {task}"
    prompt += " Your code should pass these tests:\n\n"
    for test in tests:
        prompt += f"{test}\n"
    prompt += "\n[BEGIN]\n\n"
    
    return prompt


def build_math_prompt(problem: str) -> str:
    """
    Build the zero-shot MATH prompt.
    
    Args:
        problem: Math problem text
        
    Returns:
        Formatted prompt string
    """
    return f"{MATH_SYSTEM_PROMPT}\n\nProblem: {problem}"


def build_self_correction_prompt(
    original_prompt: str,
    first_attempt: str,
    task_type: str = "math"
) -> str:
    """
    Build the self-correction prompt for the second attempt.
    
    Args:
        original_prompt: The original problem prompt
        first_attempt: The model's first attempt response
        task_type: Either "math" or "code"
        
    Returns:
        Formatted self-correction prompt
    """
    if task_type == "math":
        instruction = MATH_SELF_CORRECTION_INSTRUCTION
    else:
        instruction = CODE_SELF_CORRECTION_INSTRUCTION
    
    return f"{original_prompt}\n\n{first_attempt}\n\n{instruction}"
