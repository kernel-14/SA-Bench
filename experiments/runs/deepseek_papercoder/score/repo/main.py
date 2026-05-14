"""
main.py – Entry point for SCoRe reproduction.

Orchestrates: configuration loading → data preparation → two‑stage RL training
→ evaluation on held‑out benchmarks.

Usage:
    python main.py --task math [--config config.yaml] [--seed 42] [--no_offline]
"""

import argparse
import logging
import os
import random
import sys
import time
from typing import Optional, Union

import numpy as np
import torch

# Project‑internal modules (assumed to be in the same package / working directory)
from config import Config, load_config_from_yaml
from dataset_loader import DatasetLoader
from reward import RewardFunction
from policy_model import PolicyModel
from rl_trainer import RLTrainer
from evaluation import Evaluator

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helper: seed everything for reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


# --------------------------------------------------------------------------- #
# Helper: deep‑clone and freeze model for reference
# --------------------------------------------------------------------------- #
def create_reference_model(policy: PolicyModel) -> PolicyModel:
    """
    Clones the policy model and freezes all parameters to serve as π_ref.
    The cloned model shares the same tokenizer.
    """
    import copy
    ref_model = copy.deepcopy(policy.model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Build a new PolicyModel wrapper that holds the frozen model.
    ref_policy = PolicyModel.__new__(PolicyModel)
    ref_policy.model = ref_model
    ref_policy.tokenizer = policy.tokenizer  # share tokenizer
    ref_policy.model_name = policy.model_name
    ref_policy.device_map = policy.device_map
    ref_policy.max_seq_length = policy.max_seq_length
    # Minimal additional attributes needed by the trainer.
    ref_policy.device = next(ref_model.parameters()).device
    # The following are not used for a frozen reference but are required for the API.
    ref_policy.generate = policy.generate  # not used in training (logprobs only)
    ref_policy.compute_logprobs = lambda *args, **kwargs: ref_policy._logprobs(*args, **kwargs)
    ref_policy._logprobs = policy.compute_logprobs  # use the same method but with no grad
    ref_policy.compute_kl_penalty = PolicyModel.compute_kl_penalty  # static method
    ref_policy.decode_tokens = policy.decode_tokens
    ref_policy.tokenize_prompts = policy.tokenize_prompts

    return ref_policy


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCoRe: Self‑Correction via Reinforcement Learning"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="math",
        choices=["math", "code", "all"],
        help="Which task to train/evaluate. 'all' runs both sequentially.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override random seed."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory for saving model checkpoints (overrides config).",
    )
    parser.add_argument(
        "--no_offline",
        action="store_true",
        help="Disable offline first‑attempt augmentation in Stage II.",
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------------- #
    # Load and configure
    # ----------------------------------------------------------------------- #
    overrides = {}
    if args.checkpoint_dir:
        overrides["training.checkpoint_dir"] = args.checkpoint_dir
    config = load_config_from_yaml(args.config, overrides=overrides)
    if args.seed is not None:
        config.training.seed = args.seed
    set_seed(config.training.seed)

    # Determine device map – simple single‑GPU or CPU for simplicity.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_map = {"": "cuda:0"}
    else:
        device = torch.device("cpu")
        device_map = {"": "cpu"}
    logger.info("Using device: %s", device)

    # Tasks to run
    tasks = ["math", "code"] if args.task == "all" else [args.task]

    # ----------------------------------------------------------------------- #
    # Loop over tasks (usually one)
    # ----------------------------------------------------------------------- #
    for task in tasks:
        logger.info("========== Starting SCoRe for task: %s ==========", task)
        t_start = time.time()

        task_config = config.math if task == "math" else config.code
        is_code = task == "code"

        # Safety: ensure dataset paths exist (for HuggingFace downloads, it’s automatic)
        # ------------------------------------------------------------------ #
        # Dataset loading
        # ------------------------------------------------------------------ #
        logger.info("Loading datasets...")
        tokenizer = None  # will be set after model load
        # We need a temporary tokenizer to init DatasetLoader; we'll pass it later.
        # But DatasetLoader requires a tokenizer at __init__. We can load a minimal tokenizer now.
        # Better: initialize DatasetLoader after base model is loaded to get the actual tokenizer.
        # However, DatasetLoader methods only need the tokenizer for tokenization callbacks;
        # we can create the loader with a placeholder and later set tokenizer.
        # For simplicity, we'll load the model first, then create DatasetLoader with its tokenizer.
        logger.info("Loading base model...")
        policy = PolicyModel(
            model_name=config.model.name,
            device_map=device_map,
            max_seq_length=config.model.max_seq_length,
        )
        tokenizer = policy.tokenizer

        # DatasetLoader
        ds_loader = DatasetLoader(config, tokenizer)

        # Load train/eval datasets according to task
        if task == "math":
            train_dataset = ds_loader.load_math(split="train")
            eval_dataset = ds_loader.load_math(split="eval")
            # MATH dataset has fields: 'problem', 'solution', 'level', 'type'
            # We need to keep problem and solution (ground truth)
            # Keep original columns for reward computation.
            train_prompts = [{"problem": ex["problem"], "solution": ex["solution"]}
                             for ex in train_dataset]
            eval_prompts = [{"problem": ex["problem"], "solution": ex["solution"]}
                            for ex in eval_dataset]
        else:  # code
            train_dataset = ds_loader.load_mbpp(split="train")
            # MBPP dataset has 'prompt', 'code', 'test_list'
            # For training we need the formatted prompt and test cases.
            train_prompts = []
            for ex in train_dataset:
                prompt = ex["prompt"]  # already 3‑shot format
                test_cases = ex["test_list"]
                # wrap tests in a dict for RewardFunction compatibility
                test_dict = {"test_list": test_cases}
                train_prompts.append({"prompt": prompt, "test": test_dict, "code": ex.get("code", "")})

            eval_dataset = ds_loader.load_humaneval()
            # HumanEval has 'prompt' (function signature) and 'test' (string)
            eval_prompts = []
            for ex in eval_dataset:
                eval_prompts.append({
                    "prompt": ex["prompt"],
                    "test": {"test": ex["test"]}  # store test string
                })

        logger.info("Training examples: %d, Eval examples: %d",
                     len(train_prompts), len(eval_prompts))

        # ------------------------------------------------------------------ #
        # Reward function
        # ------------------------------------------------------------------ #
        reward_fn = RewardFunction(tokenizer, is_code=is_code)

        # ------------------------------------------------------------------ #
        # Reference model (frozen base)
        # ------------------------------------------------------------------ #
        logger.info("Creating frozen reference model...")
        ref_model = create_reference_model(policy)

        # ------------------------------------------------------------------ #
        # Offline first‑attempt data for Stage II (optional)
        # ------------------------------------------------------------------ #
        offline_data = None
        if not args.no_offline:
            logger.info("Generating offline first‑attempt dataset...")
            # Use the frozen base model to generate y1
            # We need to provide ground truths for reward computation.
            # For code tasks, we need test cases per problem; we can extract from train_prompts.
            # generate_offline_y1 expects a Dataset; we'll convert train_prompts to a Dataset.
            from datasets import Dataset as HfDataset
            if task == "math":
                raw_dataset = HfDataset.from_list([{"problem": p["problem"],
                                                    "solution": p["solution"]}
                                                   for p in train_prompts])
            else:
                raw_dataset = HfDataset.from_list([{"prompt": p["prompt"],
                                                    "test_list": p["test"]["test_list"]}
                                                   for p in train_prompts])
            offline_dataset_obj = ds_loader.generate_offline_y1(
                base_model=ref_model,  # use frozen base
                reward_fn=reward_fn,
                train_dataset=raw_dataset,
                num_samples=task_config.offline.num_samples_per_prompt,
            )
            # Convert to list of dicts for RLTrainer
            offline_data = []
            for entry in offline_dataset_obj:
                offline_data.append({
                    "problem": entry["prompt"],  # raw problem string
                    "y1": entry["y1"],
                    "r1": entry["r1"],
                    "ground_truth": entry["ground_truth"],
                })
            logger.info("Offline dataset contains %d entries.", len(offline_data))
        else:
            logger.info("Offline augmentation disabled.")

        # ------------------------------------------------------------------ #
        # Stage I training
        # ------------------------------------------------------------------ #
        logger.info("=== Stage I training ===")
        rl_trainer_stage1 = RLTrainer(
            policy=policy,
            ref=ref_model,
            reward_fn=reward_fn,
            config=task_config,
            offline_dataset=None,  # no offline in Stage I
            task=task,
        )
        best_stage1_path = rl_trainer_stage1.run_stage1(train_prompts)
        logger.info("Stage I best checkpoint: %s", best_stage1_path)
        # Load best Stage I checkpoint into policy
        rl_trainer_stage1.load_checkpoint(best_stage1_path)

        # ------------------------------------------------------------------ #
        # Stage II training
        # ------------------------------------------------------------------ #
        logger.info("=== Stage II training ===")
        rl_trainer_stage2 = RLTrainer(
            policy=policy,
            ref=ref_model,
            reward_fn=reward_fn,
            config=task_config,
            offline_dataset=offline_data,   # may be None
            task=task,
        )
        best_stage2_path = rl_trainer_stage2.run_stage2(train_prompts)
        logger.info("Stage II best checkpoint: %s", best_stage2_path)
        # Load best Stage II checkpoint
        rl_trainer_stage2.load_checkpoint(best_stage2_path)

        # ------------------------------------------------------------------ #
        # Evaluation
        # ------------------------------------------------------------------ #
        logger.info("=== Evaluation on %s test set ===", task)
        evaluator = Evaluator(
            policy=policy,
            reward_fn=reward_fn,
            config=config,
        )
        metrics = evaluator.evaluate_self_correction(
            dataset=eval_prompts,
            temperature=config.evaluation.temperature,
        )
        logger.info("Self‑correction metrics:")
        for k, v in metrics.items():
            logger.info("  %s: %.4f", k, v)

        # Optional: inference‑compute scaling experiment (only if explicitly wanted)
        if config.evaluation.inference_scaling is not None:
            K = config.evaluation.inference_scaling.num_parallel_samples_K
            temp = config.evaluation.inference_scaling.temperature
            logger.info("Running inference‑compute scaling with K=%d, temp=%.2f", K, temp)
            scaling_metrics = evaluator.run_parallel_and_sequential(
                dataset=eval_prompts,
                K=K,
                temperature=temp,
            )
            logger.info("Inference scaling metrics:")
            for k, v in scaling_metrics.items():
                logger.info("  %s: %.4f", k, v)

        t_end = time.time()
        logger.info("Task '%s' completed in %.2f minutes.", task, (t_end - t_start) / 60)


if __name__ == "__main__":
    main()
