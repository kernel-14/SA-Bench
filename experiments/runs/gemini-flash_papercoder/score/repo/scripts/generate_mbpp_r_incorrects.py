import argparse
import copy
import logging
import os
import random
from typing import Any, Dict, List, Optional

# Local imports
from config import Config
from dataset_utils import Problem, load_code_dataset, save_problems_to_jsonl
from model_utils import LLMForSelfCorrection
from prompt_manager import PromptManager
from reward_utils import CodeRewardFunction

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_mbpp_r_incorrects(config_path: str) -> None:
    """
    Generates a dataset for the MBPP-R task by using the base LLM to produce
    first-attempt solutions and filtering for those that are incorrect.
    These incorrect solutions are then stored along with the original problem
    in a new JSONL file.

    Args:
        config_path: Path to the main configuration YAML file.
    """
    logger.info(f"Loading configuration from {config_path}...")
    config = Config()
    config.load_from_yaml(config_path)

    # Validate and retrieve required config parameters for this script
    raw_mbpp_problems_path: Optional[str] = getattr(
        config, "mbpp_raw_problems_for_incorrect_gen_path", None
    )
    output_mbpp_r_path: Optional[str] = getattr(config, "mbpp_r_path", None)
    evaluation_settings: Optional[Dict[str, Any]] = getattr(config, "evaluation", None)
    seed: Optional[int] = getattr(config, "seed", None)

    if raw_mbpp_problems_path is None:
        raise ValueError(
            "config.mbpp_raw_problems_for_incorrect_gen_path must be specified in config.yaml"
        )
    if output_mbpp_r_path is None:
        raise ValueError("config.mbpp_r_path must be specified in config.yaml")
    if evaluation_settings is None:
        raise ValueError("Evaluation settings must be specified in config.yaml")

    sampling_temperature: float = evaluation_settings.get("sampling_temperature", 0.0)
    max_new_tokens: int = evaluation_settings.get("max_new_tokens", 1024)

    logger.info(f"Setting random seed to {seed} for reproducibility.")
    if seed is not None:
        random.seed(seed)

    logger.info(f"Loading raw MBPP problems from: {raw_mbpp_problems_path}")
    raw_mbpp_problems: List[Problem] = load_code_dataset(raw_mbpp_problems_path)
    logger.info(f"Loaded {len(raw_mbpp_problems)} raw MBPP problems.")

    logger.info("Initializing LLM for generation...")
    # This LLMForSelfCorrection instance represents the base model for generating responses.
    # We set `is_ref_model=False` because it's generating responses, not just providing log-probs.
    llm_wrapper = LLMForSelfCorrection(config, is_ref_model=False)

    prompt_manager = PromptManager(config)
    code_reward_function = CodeRewardFunction(config)

    generated_incorrect_dataset: List[Problem] = []

    # Ensure parent directories for the output file exist
    os.makedirs(os.path.dirname(output_mbpp_r_path), exist_ok=True)

    for i, problem in enumerate(raw_mbpp_problems):
        logger.info(f"Processing problem {i+1}/{len(raw_mbpp_problems)}: {problem.problem_id}")

        # Construct the first-turn prompt for code generation.
        # For MBPP-R, we use the 'mbpp_train' context to ensure 3-shot examples are included,
        # mimicking the setup described in the paper's Appendix C for MBPP training samples.
        full_prompt = prompt_manager.get_first_turn_prompt(
            problem_text=problem.text, task_context="mbpp_train"
        )

        # Generate the first attempt solution using the base LLM.
        # The sampling_temperature is taken from the evaluation settings (typically 0.0 for greedy).
        generated_code_response: str
        _log_probs: List[float]  # Not used for this script, but returned by generate
        generated_code_response, _log_probs = llm_wrapper.generate(
            prompt=full_prompt,
            temperature=sampling_temperature,
            max_new_tokens=max_new_tokens,
        )

        # Extract test cases from problem metadata.
        # MBPP problems typically store test cases in 'test_code', HumanEval in 'test'.
        test_code_for_problem: str = problem.metadata.get(
            "test_code", problem.metadata.get("test", "")
        )
        if not test_code_for_problem:
            logger.warning(
                f"Problem {problem.problem_id} has no 'test_code' or 'test' in metadata. Skipping evaluation."
            )
            continue

        # Evaluate the correctness of the generated code.
        # The CodeRewardFunction handles execution in a sandboxed environment.
        is_correct: float = code_reward_function.calculate_reward(
            response=generated_code_response,
            test_code=test_code_for_problem,
        )

        if is_correct == 0.0:
            logger.info(f"  Generated solution for {problem.problem_id} is INCORRECT. Storing.")
            # Create a new Problem object (deep copy to avoid modifying the original)
            # and store the generated incorrect attempt in its metadata.
            incorrect_problem_entry: Problem = copy.deepcopy(problem)
            incorrect_problem_entry.metadata["incorrect_first_attempt"] = generated_code_response
            generated_incorrect_dataset.append(incorrect_problem_entry)
        else:
            logger.info(
                f"  Generated solution for {problem.problem_id} is CORRECT. Not storing for MBPP-R."
            )

    logger.info(
        f"Total {len(generated_incorrect_dataset)} incorrect problems generated and collected for MBPP-R."
    )

    if generated_incorrect_dataset:
        logger.info(f"Saving generated MBPP-R dataset to: {output_mbpp_r_path}")
        save_problems_to_jsonl(generated_incorrect_dataset, output_mbpp_r_path)
        logger.info("MBPP-R dataset generation complete.")
    else:
        logger.warning("No incorrect problems were generated. MBPP-R dataset will be empty.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an MBPP-R dataset by having a base LLM generate "
        "first-attempt solutions and filtering for incorrect ones."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the main configuration YAML file.",
    )
    args = parser.parse_args()

    try:
        generate_mbpp_r_incorrects(args.config)
    except Exception as e:
        logger.error(
            f"An error occurred during MBPP-R dataset generation: {e}", exc_info=True
        )
        import sys

        sys.exit(1)

