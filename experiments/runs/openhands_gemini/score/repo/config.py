
import os
import torch

class Config:
    # General settings
    PROJECT_NAME = "SCoRe_Reproduce"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # Model settings
    # The paper mentions Gemini 1.5 Flash and Gemini 1.0 Pro,
    # but as they are proprietary, we will use an open-source alternative for reproduction.
    # For now, let's use a placeholder. A suitable open-source model like Llama-2 or CodeLlama
    # would be used in a real implementation.
    BASE_MODEL_MATH = "google/gemma-2b" # Placeholder for Gemini 1.5 Flash
    BASE_MODEL_CODE = "google/gemma-2b" # Placeholder for Gemini 1.0 Pro

    # Dataset settings
    # For MATH, they augment training with 4500 problems from test and report on 500 (MATH500)
    # For Code, they train on MBPP and report on HumanEval.
    MATH_DATASET_PATH = "competition_math" # "hendrycks/math" is not directly available, competition_math is a common alternative.
    MBPP_DATASET_PATH = "mbpp" # "openai_mbpp" is not directly available, mbpp is a common alternative.
    HUMANEVAL_DATASET_PATH = "openai_humaneval" # "openai_humaneval" is commonly used.

    # Training hyperparameters (Table 5) - MATH
    MATH_HYPERPARAMS = {
        "optimizer": "Adam",
        "learning_rate": 5e-6,
        "training_steps": 3000, # Per stage
        "batch_size": 512,
        "sampling_temperature": 1.0, # For generating responses
        "alpha": 10.0, # Reward shaping multiplier
        "beta1": 0.01, # KL divergence penalty for main RL objective
        "beta2": 0.1, # KL divergence penalty for Stage I (first attempt)
    }

    # Training hyperparameters (Table 5) - MBPP (Code)
    CODE_HYPERPARAMS = {
        "optimizer": "Adam",
        "learning_rate": 1e-5,
        "training_steps": 1500, # Per stage
        "batch_size": 128,
        "sampling_temperature": 1.0,
        "alpha": 10.0,
        "beta1": 0.01,
        "beta2": 0.25,
    }

    # RL specific settings
    # The paper mentions REINFORCE with KL divergence penalty (Ahmadian et al., 2024).
    # For implementation, we will likely use a PPO-like framework from TRL with a custom reward.
    # The `beta1` and `beta2` directly correspond to the KL penalty coefficients.
    RL_METHOD = "REINFORCE_KL"

    # Prompt settings (Appendix C)
    MATH_ZERO_SHOT_PROMPT = """You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is \\$answer\\$. I hope it is correct."\n\nProblem. {problem}"""
    MATH_SELF_CORRECTION_INSTRUCTION = """There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final solution! At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is \\$answer\\$. I hope it is correct."\n\nPrevious solution:\n{previous_solution}"""

    # MBPP uses 3-shot prompting for first attempt, HumanEval uses zero-shot.
    # We will need to construct these dynamically. The template is for a single shot.
    MBPP_HUMANEVAL_ZERO_SHOT_PROMPT_TEMPLATE = """You are an expert Python programmer, and here is your task: {problem_description}\nYour code should pass these tests:\n{test_cases}\n\n[BEGIN]\n{seed_code}\n[DONE]"""
    MBPP_HUMANEVAL_SELF_CORRECTION_INSTRUCTION = """There might be an error in the code above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program!\n\nPrevious code:\n{previous_code}"""

    # Output directories
    OUTPUT_DIR = "output"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Instantiate the config
config = Config()
