
from config import config

MATH_ZERO_SHOT_PROMPT = config.MATH_ZERO_SHOT_PROMPT
MATH_SELF_CORRECTION_INSTRUCTION = config.MATH_SELF_CORRECTION_INSTRUCTION

MBPP_HUMANEVAL_ZERO_SHOT_PROMPT_TEMPLATE = config.MBPP_HUMANEVAL_ZERO_SHOT_PROMPT_TEMPLATE
MBPP_HUMANEVAL_SELF_CORRECTION_INSTRUCTION = config.MBPP_HUMANEVAL_SELF_CORRECTION_INSTRUCTION

# Helper functions to format prompts
def format_math_problem(problem_text: str) -> str:
    return MATH_ZERO_SHOT_PROMPT.format(problem=problem_text)

def format_math_self_correction_prompt(problem_text: str, previous_solution: str) -> str:
    initial_prompt = format_math_problem(problem_text)
    return f"{initial_prompt}\n{MATH_SELF_CORRECTION_INSTRUCTION.format(previous_solution=previous_solution)}"

def format_code_problem(problem_description: str, test_cases: str, seed_code: str = "") -> str:
    return MBPP_HUMANEVAL_ZERO_SHOT_PROMPT_TEMPLATE.format(
        problem_description=problem_description,
        test_cases=test_cases,
        seed_code=seed_code
    )

def format_code_self_correction_prompt(problem_description: str, test_cases: str, previous_code: str, seed_code: str = "") -> str:
    initial_prompt = format_code_problem(problem_description, test_cases, seed_code)
    return f"{initial_prompt}\n{MBPP_HUMANEVAL_SELF_CORRECTION_INSTRUCTION.format(previous_code=previous_code)}"

