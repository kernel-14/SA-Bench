"""
Main training script for SCoRe.

Usage:
    # Train on MATH
    python train_score.py --task math --model_name google/gemma-2b \
        --math_data_dir /path/to/MATH --output_dir ./outputs/score_math

    # Train on MBPP/HumanEval
    python train_score.py --task code --model_name google/gemma-2b \
        --mbpp_data_path /path/to/mbpp.jsonl --output_dir ./outputs/score_code
"""

import argparse
import json
import logging
import os
import random
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.score_trainer import SCoReConfig, SCoReTrainer
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
from src.evaluation import compute_self_correction_metrics, format_results_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SCoRe model")
    
    # Task
    parser.add_argument(
        "--task", type=str, default="math",
        choices=["math", "code"],
        help="Task type: math or code"
    )
    
    # Model
    parser.add_argument(
        "--model_name", type=str, default="google/gemma-2b",
        help="HuggingFace model name or path"
    )
    
    # Data paths
    parser.add_argument(
        "--math_data_dir", type=str, default="./data/MATH",
        help="Path to MATH dataset directory"
    )
    parser.add_argument(
        "--mbpp_data_path", type=str, default="./data/mbpp.jsonl",
        help="Path to MBPP dataset JSONL file"
    )
    parser.add_argument(
        "--humaneval_data_path", type=str, default="./data/HumanEval.jsonl",
        help="Path to HumanEval dataset JSONL file"
    )
    
    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/score",
        help="Output directory for checkpoints and results"
    )
    
    # Training hyperparameters (from Appendix B)
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate (default: 5e-6 for MATH, 1e-5 for MBPP)")
    parser.add_argument("--training_steps", type=int, default=None,
                        help="Total training steps (default: 3000 for MATH, 1500 for MBPP)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: 512 for MATH, 128 for MBPP)")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Reward shaping multiplier (default: 10)")
    parser.add_argument("--beta1", type=float, default=0.01,
                        help="KL penalty for Stage II (default: 0.01)")
    parser.add_argument("--beta2", type=float, default=None,
                        help="KL penalty for first-turn in Stage I (default: 0.1 for MATH, 0.25 for MBPP)")
    parser.add_argument("--stage1_fraction", type=float, default=0.33,
                        help="Fraction of total steps for Stage I (default: 0.33)")
    
    # Evaluation
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--n_eval_problems", type=int, default=100,
                        help="Number of problems for evaluation during training")
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    
    return parser.parse_args()


def setup_config(args) -> SCoReConfig:
    """Set up SCoRe config with task-specific defaults from Appendix B."""
    config = SCoReConfig()
    config.task_type = args.task
    config.model_name = args.model_name
    
    # Task-specific defaults from Table 5 in Appendix B
    if args.task == "math":
        config.learning_rate = args.learning_rate or 5e-6
        config.training_steps = args.training_steps or 3000
        config.batch_size = args.batch_size or 512
        config.beta2 = args.beta2 or 0.1
    else:  # code
        config.learning_rate = args.learning_rate or 1e-5
        config.training_steps = args.training_steps or 1500
        config.batch_size = args.batch_size or 128
        config.beta2 = args.beta2 or 0.25
    
    config.alpha = args.alpha
    config.beta1 = args.beta1
    config.stage1_steps = int(config.training_steps * args.stage1_fraction)
    config.eval_every = args.eval_every
    
    return config


def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    # Load reference model (frozen copy of base model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    return model, ref_model, tokenizer


def main():
    args = parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup config
    config = setup_config(args)
    
    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)
    
    logger.info(f"Config: {config}")
    
    # Load model
    model, ref_model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.device
    )
    
    # Setup reward function and data
    if args.task == "math":
        reward_fn = math_reward
        correction_instruction = MATH_SELF_CORRECTION_INSTRUCTION
        
        # Load MATH data
        logger.info("Loading MATH dataset...")
        train_problems = load_math_dataset(args.math_data_dir, split="train")
        test_problems = load_math_dataset(args.math_data_dir, split="test")
        
        # Create MATH500 split
        train_aug, eval_problems = create_math500_split(test_problems, n_eval=500)
        
        # Combine training data
        all_train = train_problems + train_aug
        
        problems = [p.problem for p in all_train]
        ground_truths = [p.answer for p in all_train]
        first_prompts = [build_math_prompt(p.problem) for p in all_train]
        
        eval_probs = [p.problem for p in eval_problems]
        eval_gts = [p.answer for p in eval_problems]
        eval_prompts = [build_math_prompt(p.problem) for p in eval_problems]
        
    else:  # code
        correction_instruction = CODE_SELF_CORRECTION_INSTRUCTION
        
        # Load MBPP data
        logger.info("Loading MBPP dataset...")
        mbpp_problems = load_mbpp_dataset(args.mbpp_data_path)
        
        problems = [p.prompt for p in mbpp_problems]
        test_cases_list = [p.test_cases for p in mbpp_problems]
        first_prompts = [
            build_mbpp_3shot_prompt(p.prompt, p.test_cases)
            for p in mbpp_problems
        ]
        
        # For code, reward_fn needs test cases
        def reward_fn(response, gt):
            # gt is a list of test cases for code
            return code_reward(response, gt)
        
        ground_truths = test_cases_list
        
        # Load HumanEval for evaluation
        logger.info("Loading HumanEval dataset...")
        humaneval_problems = load_humaneval_dataset(args.humaneval_data_path)
        
        eval_probs = [p.prompt for p in humaneval_problems]
        eval_gts = [p.test_cases for p in humaneval_problems]
        eval_prompts = [p.prompt for p in humaneval_problems]
    
    # Create trainer
    trainer = SCoReTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config,
        reward_fn=reward_fn,
    )
    
    logger.info(f"Starting SCoRe training for {config.training_steps} steps")
    logger.info(f"Stage I: {config.stage1_steps} steps")
    logger.info(f"Stage II: {config.training_steps - config.stage1_steps} steps")
    
    # Training loop
    best_eval_delta = float("-inf")
    
    for step in range(config.training_steps):
        # Determine current stage
        stage = 1 if step < config.stage1_steps else 2
        
        # Sample a batch of problems
        batch_indices = random.sample(
            range(len(problems)),
            min(config.batch_size, len(problems))
        )
        
        batch_problems = [problems[i] for i in batch_indices]
        batch_gts = [ground_truths[i] for i in batch_indices]
        batch_prompts = [first_prompts[i] for i in batch_indices]
        
        # Collect rollouts
        rollout_batch = trainer.collect_rollouts(
            problems=batch_problems,
            ground_truths=batch_gts,
            first_prompts=batch_prompts,
            correction_instruction=correction_instruction,
            temperature=config.sampling_temperature,
        )
        
        # Training step
        metrics = trainer.train_step(rollout_batch, stage=stage)
        trainer.global_step = step + 1
        
        # Log metrics
        if step % 10 == 0:
            logger.info(
                f"Step {step}/{config.training_steps} | Stage {stage} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Acc@t1: {metrics['accuracy_t1']:.3f} | "
                f"Acc@t2: {metrics['accuracy_t2']:.3f} | "
                f"Delta: {metrics['delta_t1_t2']:+.3f}"
            )
        
        # Evaluate
        if step % config.eval_every == 0 or step == config.training_steps - 1:
            logger.info(f"Evaluating at step {step}...")
            
            # Sample eval problems
            eval_indices = random.sample(
                range(len(eval_probs)),
                min(args.n_eval_problems, len(eval_probs))
            )
            
            eval_metrics = trainer.evaluate(
                eval_problems=[eval_probs[i] for i in eval_indices],
                eval_ground_truths=[eval_gts[i] for i in eval_indices],
                eval_first_prompts=[eval_prompts[i] for i in eval_indices],
                correction_instruction=correction_instruction,
            )
            
            logger.info(
                f"Eval | Acc@t1: {eval_metrics['eval/accuracy_t1']:.3f} | "
                f"Acc@t2: {eval_metrics['eval/accuracy_t2']:.3f} | "
                f"Delta: {eval_metrics['eval/delta_t1_t2']:+.3f}"
            )
            
            # Save best checkpoint (by training reward, as per paper)
            if eval_metrics["eval/delta_t1_t2"] > best_eval_delta:
                best_eval_delta = eval_metrics["eval/delta_t1_t2"]
                checkpoint_path = os.path.join(args.output_dir, "best_checkpoint")
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                logger.info(f"Saved best checkpoint to {checkpoint_path}")
    
    # Final evaluation on full eval set
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate(
        eval_problems=eval_probs,
        eval_ground_truths=eval_gts,
        eval_first_prompts=eval_prompts,
        correction_instruction=correction_instruction,
    )
    
    logger.info("Final evaluation results:")
    logger.info(json.dumps(final_metrics, indent=2))
    
    # Save final results
    with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    logger.info(f"Training complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
