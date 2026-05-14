"""Prompts and self-correction instructions from Appendix C of the paper."""

MATH_ZERO_SHOT_PROMPT = (
    "You are a math expert. When you respond, respond only with the Solution of the "
    "final Problem, thinking step by step. At the end of the Solution, when you give "
    "your final answer, write it in the form "
    '"Final Answer: The final answer is $\\boxed{answer}$. I hope it is correct."'
)

MATH_SELF_CORRECTION_INSTRUCTION = (
    "There might be an error in the solution above because of lack of understanding "
    "of the question. Please correct the error, if any, and rewrite the solution. "
    "Only output the final solution! At the end of the Solution, when you give your "
    "final answer, write it in the form "
    '"Final Answer: The final answer is $\\boxed{answer}$. I hope it is correct."'
)

# MBPP 3-shot prompt from Appendix C
MBPP_THREE_SHOT_PROMPT_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {task_description}\n\n"
    "[BEGIN]\n"
    "{code_solution}\n"
    "[DONE]\n\n"
)

MBPP_SELF_CORRECTION_INSTRUCTION = (
    "There might be an error in the code above because of lack of understanding of "
    "the question. Please correct the error, if any, and rewrite the solution. Only "
    "output the final correct Python program!"
)

# HumanEval zero-shot prompt (used at evaluation)
HUMANEVAL_ZERO_SHOT_PROMPT = (
    "You are an expert Python programmer. Complete the following function.\n\n"
)

HUMANEVAL_SELF_CORRECTION_INSTRUCTION = (
    "There might be an error in the code above because of lack of understanding of "
    "the question. Please correct the error, if any, and rewrite the solution. Only "
    "output the final correct Python program!"
)

# Self-Refine prompt (Madaan et al. 2023)
SELF_REFINE_FEEDBACK_PROMPT = (
    "Review your previous answer and identify any mistakes. "
    "Then provide a corrected solution."
)

SELF_REFINE_REFINEMENT_PROMPT = (
    "Based on your review, provide the corrected solution. "
    "Only output the final solution!"
)


def build_math_first_turn_prompt(problem: str) -> str:
    """Build the first-turn prompt for MATH problems."""
    return f"{MATH_ZERO_SHOT_PROMPT}\n\nProblem: {problem}\n\nSolution:"


def build_math_second_turn_prompt(
    problem: str, first_response: str
) -> str:
    """Build the second-turn prompt for MATH (self-correction)."""
    return (
        f"{MATH_ZERO_SHOT_PROMPT}\n\n"
        f"Problem: {problem}\n\n"
        f"Solution:\n{first_response}\n\n"
        f"{MATH_SELF_CORRECTION_INSTRUCTION}\n\n"
        f"Solution:"
    )


def build_mbpp_first_turn_prompt(problem_description: str, test_cases: str) -> str:
    """Build the first-turn prompt for MBPP (3-shot format)."""
    # The full 3-shot prompt is composed by the data module
    return (
        f"You are an expert Python programmer, and here is your task: "
        f"{problem_description} Your code should pass these tests:\n\n"
        f"{test_cases}\n\n"
        f"[BEGIN]\n"
    )


def build_mbpp_second_turn_prompt(
    problem_description: str,
    test_cases: str,
    first_response: str,
) -> str:
    """Build the second-turn prompt for MBPP (self-correction)."""
    return (
        f"You are an expert Python programmer, and here is your task: "
        f"{problem_description} Your code should pass these tests:\n\n"
        f"{test_cases}\n\n"
        f"Your previous solution:\n```python\n{first_response}\n```\n\n"
        f"{MBPP_SELF_CORRECTION_INSTRUCTION}\n\n"
        f"[BEGIN]\n"
    )


def build_humaneval_first_turn_prompt(problem_prompt: str) -> str:
    """Build the first-turn prompt for HumanEval (zero-shot)."""
    return f"{HUMANEVAL_ZERO_SHOT_PROMPT}{problem_prompt}"


def build_humaneval_second_turn_prompt(
    problem_prompt: str, first_response: str
) -> str:
    """Build the second-turn prompt for HumanEval (self-correction)."""
    return (
        f"{HUMANEVAL_ZERO_SHOT_PROMPT}{problem_prompt}\n\n"
        f"Your previous solution:\n```python\n{first_response}\n```\n\n"
        f"{HUMANEVAL_SELF_CORRECTION_INSTRUCTION}\n\n"
    )


def build_self_refine_first_prompt(problem: str) -> str:
    """First turn for Self-Refine baseline."""
    return f"{MATH_ZERO_SHOT_PROMPT}\n\nProblem: {problem}\n\nSolution:"


def build_self_refine_feedback_prompt(
    problem: str, first_response: str
) -> str:
    """Feedback turn for Self-Refine baseline."""
    return (
        f"MATH_ZERO_SHOT_PROMPT}\n\n"
        f"Problem: {problem}\n\n"
        f"Your previous solution:\n{first_response}\n\n"
        f"{SELF_REFINE_FEEDBACK_PROMPT}"
    )


def build_self_refine_refinement_prompt(
    problem: str, first_response: str, feedback: str
) -> str:
    """Refinement turn for Self-Refine baseline."""
    return (
        f"{MATH_ZERO_SHOT_PROMPT}\n\n"
        f"Problem: {problem}\n\n"
        f"Your previous solution:\n{first_response}\n\n"
        f"Your review:\n{feedback}\n\n"
        f"{SELF_REFINE_REFINEMENT_PROMPT}\n\n"
        f"Solution:"
    )
