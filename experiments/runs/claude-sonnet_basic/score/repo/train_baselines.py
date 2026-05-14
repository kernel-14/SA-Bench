"""
Training script for baseline methods (STaR, Pair-SFT, Self-Refine).

Usage:
    # Train STaR baseline
    python train_baselines.py --method star --task math \
        --model_name google/gemma-2b --math_data_dir /path/to/MATH

    # Train Pair-SFT baseline
    python train_baselines.py --method pair_sft --task math \
        --model_name google/gemma-2b --math_data_dir /path/to/MATH

    # Evaluate Self-Refine (no training needed)
    python train_baselines.py --method self_refine --task math \
        --model_name google/gemma-2b --math_data_dir /path/to/MATH
"""

import argparse
import json
import logging
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.baselines import STaRTrainer, PairSFTTrainer, SelfRefine, BaselineConfig
from src.reward import math_reward, code_reward
from src.prompts import (
    build_math_prompt,
    build_mbpp_3shot_prompt,
    MATH_SELF_CORRECTION_INSTRUCTION,
    CODE_SELF_CORRECTION_INSTRUCTION,
)
from src.data_utils import (
    load_math_dataset,
    load_mbpp_dataset,
    load_humaneval_dataset,
    create_math500_split,
)
from src.evaluation import (
    compute_self_correction_metrics,
    evaluate_model_on_dataset,
    format_results_table,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline methods")
    
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["star", "pair_sft", "self_refine"],
        help="Baseline method to train"
    )
    parser.add_argument(
        "--task", type=str, default="math",
        choices=["math", "code"],
    )
    parser.add_argument(
        "--model_name", type=str, default="google/gemma-2b",
    )
    parser.add_argument(
        "--math_data_dir", type=str, default="./data/MATH",
    )
    parser.add_argument(
        "--mbpp_data_path", type=str, default="./data/mbpp.jsonl",
    )
    parser.add_argument(
        "--humaneval_data_path", type=str, default="./data/HumanEval.jsonl",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/baselines",
    )
    parser.add_argument(
        "--include_correct_to_correct", action="store_true",
        help="Include correct->correct pairs (D+ variants)"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--star_iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = os.path.join(args.output_dir, f"{args.method}_{args.task}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    
    # Setup data and reward
    if args.task == "math":
        reward_fn = math_reward
        correction_instruction = MATH_SELF_CORRECTION_INSTRUCTION
        
        train_problems = load_math_dataset(args.math_data_dir, split="train")
        test_problems = load_math_dataset(args.math_data_dir, split="test")
        train_aug, eval_problems = create_math500_split(test_problems, n_eval=500)
        
        all_train = train_problems + train_aug
        problems = [p.problem for p in all_train]
        ground_truths = [p.answer for p in all_train]
        first_prompts = [build_math_prompt(p.problem) for p in all_train]
        
        eval_probs = [p.problem for p in eval_problems]
        eval_gts = [p.answer for p in eval_problems]
        eval_prompts = [build_math_prompt(p.problem) for p in eval_problems]
        
    else:
        correction_instruction = CODE_SELF_CORRECTION_INSTRUCTION
        mbpp_problems = load_mbpp_dataset(args.mbpp_data_path)
        
        problems = [p.prompt for p in mbpp_problems]
        test_cases_list = [p.test_cases for p in mbpp_problems]
        first_prompts = [
            build_mbpp_3shot_prompt(p.prompt, p.test_cases)
            for p in mbpp_problems
        ]
        
        def reward_fn(response, gt):
            return code_reward(response, gt)
        
        ground_truths = test_cases_list
        
        humaneval_problems = load_humaneval_dataset(args.humaneval_data_path)
        eval_probs = [p.prompt for p in humaneval_problems]
        eval_gts = [p.test_cases for p in humaneval_problems]
        eval_prompts = [p.prompt for p in humaneval_problems]
    
    config = BaselineConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        task_type=args.task,
        star_iterations=args.star_iterations,
        include_correct_to_correct=args.include_correct_to_correct,
    )
    
    if args.method == "self_refine":
        # No training needed - just evaluate
        logger.info("Self-Refine: evaluating without training...")
        
        self_refine = SelfRefine(model, tokenizer)
        
        first_rewards = []
        second_rewards = []
        
        for prob, gt, prompt in zip(eval_probs, eval_gts, eval_prompts):
            y1, y2 = self_refine.self_correct(prompt, correction_instruction)
            r1 = reward_fn(y1, gt)
            r2 = reward_fn(y2, gt)
            first_rewards.append(r1)
            second_rewards.append(r2)
        
        metrics = compute_self_correction_metrics(first_rewards, second_rewards)
        
    elif args.method == "star":
        logger.info(f"STaR: running {args.star_iterations} iterations...")
        
        trainer = STaRTrainer(model, tokenizer, config, reward_fn)
        
        for iteration in range(args.star_iterations):
            logger.info(f"STaR iteration {iteration + 1}/{args.star_iterations}")
            
            # Collect data
            training_data = trainer.collect_star_data(
                problems=problems,
                ground_truths=ground_truths,
                first_prompts=first_prompts,
                correction_instruction=correction_instruction,
            )
            
            logger.info(f"Collected {len(training_data)} training examples")
            
            # Train
            train_metrics = trainer.train_iteration(training_data)
            logger.info(f"Training loss: {train_metrics['loss']:.4f}")
        
        # Evaluate
        metrics = evaluate_model_on_dataset(
            model, tokenizer,
            eval_probs, eval_gts, eval_prompts,
            correction_instruction, reward_fn,
        )
        
    elif args.method == "pair_sft":
        logger.info("Pair-SFT: building dataset and training...")
        
        trainer = PairSFTTrainer(model, tokenizer, config, reward_fn)
        
        # Build dataset
        training_data = trainer.build_pair_sft_dataset(
            problems=problems,
            ground_truths=ground_truths,
            first_prompts=first_prompts,
            correction_instruction=correction_instruction,
        )
        
        logger.info(f"Built {len(training_data)} training examples")
        
        # Train
        train_metrics = trainer.train(training_data)
        logger.info(f"Training loss: {train_metrics['loss']:.4f}")
        
        # Evaluate
        metrics = evaluate_model_on_dataset(
            model, tokenizer,
            eval_probs, eval_gts, eval_prompts,
            correction_instruction, reward_fn,
        )
    
    # Save results
    logger.info("Results:")
    logger.info(json.dumps(metrics, indent=2))
    
    results = {args.method: metrics}
    save_results(
        results,
        os.path.join(output_dir, "results.json"),
        task=args.task.upper(),
    )
    
    # Save model
    model.save_pretrained(os.path.join(output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "model"))


if __name__ == "__main__":
    main()
