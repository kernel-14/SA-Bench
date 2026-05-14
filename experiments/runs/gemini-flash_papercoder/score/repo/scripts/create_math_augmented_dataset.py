import argparse
import logging
import os
import random
from typing import Optional

# Assume these are available in the project structure
from config import Config
from dataset_utils import Problem, load_math_dataset, save_problems_to_jsonl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_math_augmented_dataset(config_path: str) -> None:
    """
    Prepares the MATH dataset by augmenting the training set with a specified
    number of problems from the test set, as described in the paper.
    The remaining test problems form the new evaluation set.

    Args:
        config_path: Path to the YAML configuration file.
    """
    logging.info(f"Loading configuration from {config_path}...")
    config = Config()
    config.load_from_yaml(config_path)

    # --- Start: Assuming Config class is extended for these fields based on plan ---
    # These fields are crucial for this script but might not be in the initial config.yaml template.
    # We retrieve them using getattr with None defaults and then validate their presence.
    original_math_train_path_glob: Optional[str] = getattr(
        config, "original_math_train_path_glob", None
    )
    original_math_test_path_glob: Optional[str] = getattr(
        config, "original_math_test_path_glob", None
    )
    math_aug_test_samples: Optional[int] = getattr(config, "math_aug_test_samples", None)
    # --- End: Assuming Config class is extended ---

    math_train_output_path: Optional[str] = config.math_train_path
    math_eval_output_path: Optional[str] = config.math_eval_path
    seed: Optional[int] = config.seed

    # Validate all required configuration values
    if not all(
        [
            original_math_train_path_glob,
            original_math_test_path_glob,
            math_aug_test_samples is not None,
            math_train_output_path,
            math_eval_output_path,
            seed is not None,
        ]
    ):
        logging.error(
            "Missing required configuration for MATH dataset creation. "
            "Ensure 'original_math_train_path_glob', 'original_math_test_path_glob', "
            "'math_aug_test_samples', 'math_train_path', 'math_eval_path', and 'seed' "
            "are defined in your config.yaml and Config class."
        )
        raise ValueError("Incomplete configuration for MATH dataset preparation.")

    logging.info(f"Setting random seed to {seed} for reproducibility.")
    random.seed(seed)

    logging.info(f"Loading original MATH training problems from: {original_math_train_path_glob}")
    original_math_train_problems: list[Problem] = load_math_dataset(
        original_math_train_path_glob
    )
    logging.info(
        f"Loaded {len(original_math_train_problems)} original MATH training problems."
    )

    logging.info(f"Loading original MATH test problems from: {original_math_test_path_glob}")
    original_math_test_problems: list[Problem] = load_math_dataset(
        original_math_test_path_glob
    )
    logging.info(f"Loaded {len(original_math_test_problems)} original MATH test problems.")

    # Shuffle the original test set for unbiased selection
    logging.info("Shuffling original MATH test problems.")
    random.shuffle(original_math_test_problems)

    # Select problems for augmentation and for the new evaluation set (MATH500)
    if math_aug_test_samples > len(original_math_test_problems):
        logging.warning(
            f"Requested {math_aug_test_samples} test samples for augmentation, "
            f"but only {len(original_math_test_problems)} are available. "
            "Using all available test samples for augmentation."
        )
        selected_test_problems_for_augmentation: list[Problem] = original_math_test_problems
        new_math_eval_problems: list[Problem] = []  # No remaining problems
    else:
        selected_test_problems_for_augmentation = original_math_test_problems[
            :math_aug_test_samples
        ]
        new_math_eval_problems = original_math_test_problems[math_aug_test_samples:]

    logging.info(
        f"Selected {len(selected_test_problems_for_augmentation)} problems from test set for augmentation."
    )
    logging.info(
        f"Remaining {len(new_math_eval_problems)} problems form the new evaluation set (MATH500)."
    )

    # Create the augmented training set
    augmented_math_train_problems: list[Problem] = (
        original_math_train_problems + selected_test_problems_for_augmentation
    )
    logging.info(f"Created augmented training set with {len(augmented_math_train_problems)} problems.")

    # Save the new datasets
    logging.info(f"Saving augmented training data to: {math_train_output_path}")
    save_problems_to_jsonl(augmented_math_train_problems, math_train_output_path)

    logging.info(f"Saving new evaluation data (MATH500) to: {math_eval_output_path}")
    save_problems_to_jsonl(new_math_eval_problems, math_eval_output_path)

    logging.info("MATH dataset augmentation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare MATH dataset by augmenting training set "
        "with test samples and creating MATH500 evaluation set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the main configuration YAML file.",
    )
    args = parser.parse_args()

    try:
        create_math_augmented_dataset(args.config)
    except Exception as e:
        logging.error(
            f"An error occurred during MATH dataset preparation: {e}", exc_info=True
        )
        import sys

        sys.exit(1)

