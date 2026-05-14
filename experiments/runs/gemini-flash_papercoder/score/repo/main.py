import argparse
import logging
import os
import random
import json # Required for json.JSONDecodeError in dataset loading error handling
import numpy as np
import torch
from typing import List, Union

# Local imports
from config import Config
from dataset_utils import Problem, build_dataloader, load_code_dataset, load_math_dataset
from model_utils import LLMForSelfCorrection
from prompt_manager import PromptManager
from reward_utils import CodeRewardFunction, MathRewardFunction, BaseRewardFunction
from rl_trainer import Stage1RLTrainer, Stage2RLTrainer
from evaluation import Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Set seeds for reproducibility across different libraries."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Global seed set to {seed} for reproducibility.")
    else:
        logger.warning("No seed provided, reproducibility not guaranteed.")


def main() -> None:
    """
    Main function to orchestrate the SCoRe (Self-Correction via Reinforcement Learning)
    pipeline. This includes loading configuration, initializing components, running
    Stage I and Stage II RL training, and finally evaluating the trained model.
    """
    parser = argparse.ArgumentParser(
        description="Run SCoRe (Self-Correction via Reinforcement Learning) pipeline."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["math", "code"],
        help="Override task type specified in config.yaml (e.g., 'math' or 'code').",
    )
    args = parser.parse_args()

    # --- 1. Configuration Loading ---
    logger.info(f"Loading configuration from {args.config_path}...")
    config = Config()
    try:
        config.load_from_yaml(args.config_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Override task_type if provided via command line
    if args.task_type:
        config.task_type = args.task_type
        logger.info(f"Task type overridden by CLI to: {config.task_type}")

    if config.task_type is None:
        logger.error("Task type is not specified in config.yaml or via command line. Exiting.")
        return

    # Set seeds for reproducibility
    seed_everything(config.seed)

    # Create checkpoint and log directories if they don't exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {config.checkpoint_dir}")
    logger.info(f"Logs will be stored in: {config.log_dir}")


    # --- 2. Initialize Components ---

    # Prompt Manager
    prompt_manager = PromptManager(config)
    logger.info("PromptManager initialized.")

    # Reward Function
    reward_function: BaseRewardFunction
    if config.task_type == "math":
        reward_function = MathRewardFunction()
        logger.info("MathRewardFunction initialized.")
    elif config.task_type == "code":
        # CodeRewardFunction requires the config object for execution settings (timeout, etc.)
        reward_function = CodeRewardFunction(config)
        logger.info("CodeRewardFunction initialized.")
    else:
        logger.error(f"Unsupported task_type: {config.task_type}. Must be 'math' or 'code'. Exiting.")
        return

    # Dataset and DataLoaders
    train_problems: List[Problem] = []
    eval_problems: List[Problem] = []
    try:
        if config.task_type == "math":
            train_problems = load_math_dataset(config.math_train_path)
            eval_problems = load_math_dataset(config.math_eval_path)
        elif config.task_type == "code":
            train_problems = load_code_dataset(config.mbpp_train_path)
            # For evaluation, the paper uses HumanEval for general eval.
            # MBPP-R is handled by the Evaluator class's specific method.
            eval_problems = load_code_dataset(config.human_eval_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load datasets for task type '{config.task_type}': {e}. Exiting.")
        return

    # Get batch sizes from stage-specific configs, as they might differ
    train_batch_size_stage1 = config.get_stage_hyperparameters(1).get("batch_size", 16)
    eval_batch_size = config.evaluation.get("batch_size", 32)

    train_dataloader = build_dataloader(
        train_problems, train_batch_size_stage1, shuffle=True
    )
    eval_dataloader = build_dataloader(
        eval_problems, eval_batch_size, shuffle=False
    )
    logger.info(f"Loaded {len(train_problems)} training problems and {len(eval_problems)} evaluation problems.")
    logger.info("DataLoaders initialized.")

    # LLM Model Wrappers
    logger.info(f"Initializing policy model from {config.base_model_name}...")
    # This is the model that will be fine-tuned. `is_ref_model=False` means it can be trained
    # and it will internally load its own fixed reference model for KL computations.
    model_wrapper_policy = LLMForSelfCorrection(config, is_ref_model=False)
    initial_base_model_path = os.path.join(config.checkpoint_dir, "base_model")
    model_wrapper_policy.save_pretrained(initial_base_model_path)
    logger.info(f"Initial base model saved to {initial_base_model_path}.")

    logger.info(f"Initializing external reference model from {config.base_model_name}...")
    # This is a separate, dedicated reference model wrapper.
    # It's fixed and used by the RL Trainers for KL divergence calculations.
    ref_model_wrapper = LLMForSelfCorrection(config, is_ref_model=True)
    logger.info("External reference model initialized.")


    # --- 3. Stage I Training ---
    logger.info("\n--- Starting Stage I Training ---")
    stage1_trainer = Stage1RLTrainer(
        model_wrapper=model_wrapper_policy,
        ref_model_wrapper=ref_model_wrapper, # External ref model for KL
        dataloader=train_dataloader,
        prompt_manager=prompt_manager,
        reward_function=reward_function,
        config=config,
    )
    stage1_trainer.train()

    stage1_checkpoint_path = os.path.join(config.checkpoint_dir, "stage1_model")
    # Save the current state of model_wrapper_policy, which now contains Stage I training results
    model_wrapper_policy.save_pretrained(stage1_checkpoint_path)
    logger.info(f"Stage I trained model saved to {stage1_checkpoint_path}.")


    # --- 4. Stage II Training ---
    logger.info("\n--- Starting Stage II Training ---")
    # The `model_wrapper_policy` object has been updated in-place by Stage I training.
    # We directly use this updated model as the starting point for Stage II.
    # No need to create a new LLMForSelfCorrection instance and load the checkpoint explicitly.
    # However, we must update the dataloader's batch_size if Stage 2 has a different batch size.
    train_batch_size_stage2 = config.get_stage_hyperparameters(2).get("batch_size", 16)
    if train_batch_size_stage2 != train_batch_size_stage1:
        logger.info(f"Updating train dataloader batch size from {train_batch_size_stage1} to {train_batch_size_stage2} for Stage II.")
        train_dataloader = build_dataloader(train_problems, train_batch_size_stage2, shuffle=True)

    stage2_trainer = Stage2RLTrainer(
        model_wrapper=model_wrapper_policy, # Continues training the same policy model
        ref_model_wrapper=ref_model_wrapper, # External ref model for KL
        dataloader=train_dataloader,
        prompt_manager=prompt_manager,
        reward_function=reward_function,
        config=config,
    )
    stage2_trainer.train()

    final_score_model_path = os.path.join(config.checkpoint_dir, "score_final_model")
    # Save the final state of model_wrapper_policy, which now contains Stage II training results
    model_wrapper_policy.save_pretrained(final_score_model_path)
    logger.info(f"Final SCoRe model saved to {final_score_model_path}.")


    # --- 5. Evaluation ---
    logger.info("\n--- Starting Final Evaluation ---")
    evaluator = Evaluator(
        model_wrapper=model_wrapper_policy, # Evaluate the final SCoRe model
        dataloader=eval_dataloader,
        prompt_manager=prompt_manager,
        reward_function=reward_function,
        config=config,
    )
    metrics = evaluator.evaluate()

    logger.info("\n--- Evaluation Results ---")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("--- SCoRe Pipeline Completed Successfully ---")


if __name__ == "__main__":
    main()

