"""
main.py: Entry point for the MA-RLHF reproduction pipeline.

This script orchestrates the entire MA-RLHF training and evaluation process,
including Supervised Fine-Tuning (SFT), Reward Model (RM) training,
Macro-Action Proximal Policy Optimization (MA-PPO), and various evaluation stages.
It parses command-line arguments, loads configurations, initializes all necessary
components, and executes the specified stages.
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# Import custom modules
from config import load_config, Config
from utils import TokenizerWrapper # log_metrics is used internally by trainers/eval managers
from data_loader import DataLoader
from models import SFTModel, RewardModel, PolicyModel, ValueModel # Imported explicitly for clarity, but RLHFTrainer manages them
from macro_action_handler import MacroActionHandler # Imported explicitly, but RLHFTrainer manages it
from ppo_algorithm import PPOAlgorithm # Imported explicitly, but RLHFTrainer manages it
from rlhf_trainer import RLHFTrainer
from evaluation_manager import EvaluationManager
from code_executor import CodeExecutor


def main():
    """
    Main function to parse arguments, set up the environment, and run
    the MA-RLHF training and evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run MA-RLHF training and evaluation pipeline."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Which stages to run (sft, rm, ppo, eval, or all, comma-separated).",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "tldr_summarization",
            "hh_rlhf",
            "webgpt_comparison",
            "apps_code_gen",
        ],
        help="The research task to perform.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=[
            "gemma_2b",
            "gemma_7b",
            "gemma_27b",
            "codegemma_2b",
            "codegemma_7b",
            "llama_3_2_3b",
        ],
        help="Identifier for the base model size (e.g., gemma_2b).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="A unique name for the current experiment run. Defaults to task_model_size_timestamp.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint from which to resume training (for SFT, RM, or PPO).",
    )

    args = parser.parse_args()

    # --- 1. Load Configuration ---
    try:
        # Use a temporary OmegaConf to get the model name for tokenizer before full config load
        # This is a workaround as load_config needs model_id to apply overrides.
        temp_base_cfg = OmegaConf.load(args.config_path)
        if args.model_size not in temp_base_cfg.model_configs:
            raise ValueError(f"Model ID '{args.model_size}' not found in config.yaml model_configs.")
        
        # Pass `args.model_size` and `args.task` to `load_config` for dynamic resolution.
        cfg: Config = load_config(args.config_path, args.model_size, args.task)
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}. Please ensure --config_path is correct.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading or resolving configuration: {e}")
        sys.exit(1)

    # --- 2. Global Setup ---
    # Set run_name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{args.task}_{args.model_size}_{timestamp}"

    # Update global output/logging directories with run_name
    cfg.global.output_dir = os.path.join(cfg.global.output_dir, args.run_name)
    cfg.global.logging_dir = os.path.join(cfg.global.logging_dir, args.run_name)
    os.makedirs(cfg.global.output_dir, exist_ok=True)
    os.makedirs(cfg.global.logging_dir, exist_ok=True)

    # Configure Loguru logger
    logger.remove()  # Remove default logger
    logger.add(sys.stderr, level="INFO")  # Add console output
    logger.add(
        os.path.join(cfg.global.logging_dir, "experiment.log"),
        level="INFO",
        rotation="10 MB",
        compression="zip",
    )
    logger.info(f"Starting experiment: {args.run_name}")
    logger.info(f"Config loaded:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seeds for reproducibility
    if cfg.global.seed is not None:
        random.seed(cfg.global.seed)
        np.random.seed(cfg.global.seed)
        torch.manual_seed(cfg.global.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.global.seed)
            torch.backends.cudnn.deterministic = True # For CUDA operations
            torch.backends.cudnn.benchmark = False
        logger.info(f"Global seed set to {cfg.global.seed}")

    # Parse stages to run
    stages_to_run = (
        ["sft", "rm", "ppo", "eval"] if args.stage == "all" else args.stage.split(",")
    )
    logger.info(f"Stages to run: {stages_to_run}")

    # --- 3. Initialize Core Components ---
    tokenizer_model_name = cfg.model_configs[args.model_size].name
    tokenizer_wrapper = TokenizerWrapper(tokenizer_model_name)
    data_loader = DataLoader(cfg, tokenizer_wrapper)

    # RLHFTrainer manages the SFT, RM, Policy, Value Models and PPOAlgorithm internally
    trainer = RLHFTrainer(config=cfg)

    code_executor: Optional[CodeExecutor] = None
    if args.task == "apps_code_gen":
        code_executor = CodeExecutor(config=cfg)
        logger.info("CodeExecutor initialized for APPS task.")

    # EvaluationManager needs references to the models managed by the trainer
    eval_manager = EvaluationManager(
        config=cfg,
        data_loader=data_loader,
        policy_model=trainer.ppo_algorithm.policy_model, # The policy model being optimized
        reward_model=trainer.ppo_algorithm.reward_model, # The reward model providing scores
        sft_model=trainer.sft_model, # The SFT model, potentially for baselines
        tokenizer_wrapper=tokenizer_wrapper,
        code_executor=code_executor,
    )
    logger.info("EvaluationManager initialized.")

    # --- 4. Execute Stages ---
    try:
        if "sft" in stages_to_run:
            trainer.train_sft()

        if "rm" in stages_to_run:
            trainer.train_rm()

        if "ppo" in stages_to_run:
            trainer.train_ppo()

        if "eval" in stages_to_run:
            logger.info("Starting evaluation stage...")
            results = {}

            # Evaluate RM scores
            rm_scores_results = eval_manager.evaluate_rm_scores(task=args.task)
            results["rm_scores"] = rm_scores_results

            # Evaluate GPT-4
            gpt4_results = eval_manager.evaluate_gpt4(task=args.task)
            results["gpt4_evaluation"] = gpt4_results

            # Prepare for Human Evaluation (Note: This saves data for manual annotation)
            human_eval_prep_results = eval_manager.evaluate_human(task=args.task)
            results["human_evaluation_prep"] = human_eval_prep_results

            # Evaluate pass@k for Code Generation
            if args.task == "apps_code_gen":
                pass_at_k_results = eval_manager.evaluate_pass_at_k(task=args.task)
                results["pass_at_k"] = pass_at_k_results
            
            logger.info(f"Evaluation Results for {args.run_name}: {results}")
            
            # Save evaluation results to a JSON file
            eval_output_path = os.path.join(cfg.global.output_dir, "eval_results", "final_evaluation.json")
            with open(eval_output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Final evaluation results saved to {eval_output_path}")

    except Exception as e:
        logger.exception(f"An error occurred during experiment execution: {e}")
        sys.exit(1)

    logger.info(f"Experiment '{args.run_name}' completed successfully.")


if __name__ == "__main__":
    main()

