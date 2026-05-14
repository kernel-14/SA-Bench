"""
Main training entry point for SCoRe and baseline methods.

Usage:
    python -m score.train \
        --task math \
        --model_name_or_path <path> \
        --output_dir ./outputs

This script orchestrates:
1. Data loading (MATH or MBPP/HumanEval)
2. Model initialization with reference (frozen) model
3. SCoRe two-stage training (Stage I + Stage II)
4. Evaluation with self-correction metrics
5. Baseline comparisons (STaR, Pair-SFT)

Hyperparameters are configured per Table 5 (Appendix B).
"""

import argparse
import logging
import os
import sys
from typing import Dict, Optional

import torch
import numpy as np

from score.training.score_trainer import SCoReTrainer, SCoReConfig
from score.training.sft_trainer import (
    STaRTrainer, PairSFTTrainer, SFTConfig
)
from score.data.dataset import (
    load_math_dataset, load_mbpp_dataset, load_humaneval_dataset,
    prepare_training_data, TRAIN_TEST_SPLITS,
)
from score.prompts.templates import (
    build_math_prompt, build_math_correction_prompt,
    build_code_prompt, build_code_correction_prompt,
)
from score.evaluation.metrics import (
    SelfCorrectionMetrics, compute_edit_distance_ratio,
    analyze_behavior_collapse,
)
from score.utils.logging import setup_logging, TrainingLogger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SCoRe: Training Language Models to Self-Correct via RL"
    )
    
    # Task
    parser.add_argument("--task", type=str, default="math",
                        choices=["math", "code"],
                        help="Task: math (MATH dataset) or code (MBPP/HumanEval)")
    
    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or HuggingFace model name")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to base model for reference (defaults to model_name_or_path)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for checkpoints and logs")
    
    # Stage I
    parser.add_argument("--stage1_steps", type=int, default=1500,
                        help="Number of Stage I training steps")
    parser.add_argument("--stage1_beta2", type=float, default=0.1,
                        help="KL penalty for first turn in Stage I")
    parser.add_argument("--stage1_beta1", type=float, default=0.01,
                        help="KL penalty for second turn in Stage I")
    
    # Stage II
    parser.add_argument("--stage2_steps", type=int, default=1500,
                        help="Number of Stage II training steps")
    parser.add_argument("--stage2_beta1", type=float, default=0.01,
                        help="KL penalty for both turns in Stage II")
    parser.add_argument("--stage2_alpha", type=float, default=10.0,
                        help="Reward shaping progress bonus multiplier")
    
    # Training
    parser.add_argument("--total_steps", type=int, default=3000,
                        help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size per update")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--sampling_temperature", type=float, default=1.0,
                        help="Temperature for training sampling")
    
    # Evaluation
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_temperature", type=float, default=0.0,
                        help="Temperature for evaluation (0=greedy)")
    
    # Data
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset (overrides HuggingFace download)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training samples (for debugging)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Maximum number of evaluation samples")
    
    # Ablation options
    parser.add_argument("--ablation", type=str, default=None,
                        choices=[None, "no_stage1", "no_reward_shaping", "single_turn", "star_stage2"],
                        help="Run an ablation study")
    parser.add_argument("--run_sft_baselines", action="store_true",
                        help="Also run STaR and Pair-SFT baselines")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Load a HuggingFace model and tokenizer."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    
    logger.info(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    
    return model, tokenizer


def load_reference_model(model_path: str, device: str = "auto"):
    """Load a frozen reference model."""
    # Use the same path but a separate instance
    return load_model_and_tokenizer(model_path, device)


def main():
    args = parse_args()
    
    setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        log_file=os.path.join(args.output_dir, "train.log"),
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model and reference
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, device)
    ref_model, _ = load_reference_model(
        args.base_model_path or args.model_name_or_path, device
    )
    
    # Load datasets
    logger.info(f"Loading datasets for task: {args.task}")
    
    if args.task == "math":
        train_problems = load_math_dataset(
            data_path=args.data_path,
            split="train",
            num_samples=args.max_train_samples,
        )
        test_problems = load_math_dataset(
            data_path=args.data_path,
            split="test",
            num_samples=args.max_eval_samples or 500,
        )
        
        # MATH500: use last 500 of test set
        if len(test_problems) > 500:
            test_problems = test_problems[-500:]
        
        train_problems = prepare_training_data(train_problems, task="math")
        test_problems = prepare_training_data(test_problems, task="math")
        
    else:  # code
        train_problems = load_mbpp_dataset(
            data_path=args.data_path,
            split="train",
        )
        test_problems = load_humaneval_dataset(
            data_path=args.data_path,
        )
        
        if args.max_train_samples:
            train_problems = train_problems[:args.max_train_samples]
        if args.max_eval_samples:
            test_problems = test_problems[:args.max_eval_samples]
        
        train_problems = prepare_training_data(train_problems, task="code")
        test_problems = prepare_training_data(test_problems, task="code")
    
    logger.info(f"Train problems: {len(train_problems)}")
    logger.info(f"Test problems: {len(test_problems)}")
    
    # Build prompts for training
    if args.task == "math":
        train_prompts_t1 = [build_math_prompt(p["problem"]) for p in train_problems]
    else:
        train_prompts_t1 = [
            build_code_prompt(p["problem"], use_3shot=True) 
            for p in train_problems
        ]
    
    # SCoRe Configuration
    score_config = SCoReConfig(
        model_name=args.model_name_or_path,
        base_model_path=args.base_model_path or args.model_name_or_path,
        stage1_steps=args.stage1_steps,
        stage1_beta2=args.stage1_beta2,
        stage1_beta1=args.stage1_beta1,
        stage2_steps=args.stage2_steps,
        stage2_beta1=args.stage2_beta1,
        stage2_alpha=args.stage2_alpha,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        sampling_temperature=args.sampling_temperature,
        eval_temperature=args.eval_temperature,
        save_dir=os.path.join(args.output_dir, "checkpoints"),
        save_every=args.save_every,
        eval_every=args.eval_every,
        task=args.task,
    )
    
    # Initialize SCoRe trainer
    trainer = SCoReTrainer(
        model=model,
        reference_model=ref_model,
        tokenizer=tokenizer,
        config=score_config,
    )
    
    # Evaluate base model
    logger.info("Evaluating base model...")
    base_metrics = trainer.evaluate(test_problems)
    logger.info(f"Base model metrics: {base_metrics}")
    
    # Run ablation if specified
    if args.ablation == "no_stage1":
        logger.info("Running ablation: w/o Stage I (Stage II directly from base model)")
        stage2_results = trainer.run_stage2(train_problems, test_problems)
        ablation_metrics = trainer.evaluate(test_problems)
        logger.info(f"Ablation (no Stage I) metrics: {ablation_metrics}")
        
    elif args.ablation == "no_reward_shaping":
        logger.info("Running ablation: w/o reward shaping")
        stage1_results = trainer.run_stage1(train_problems, test_problems)
        # Temporarily set alpha to 0
        original_alpha = trainer.rl_trainer.config.alpha
        trainer.rl_trainer.config.alpha = 0.0
        stage2_results = trainer.run_stage2(train_problems, test_problems)
        trainer.rl_trainer.config.alpha = original_alpha
        ablation_metrics = trainer.evaluate(test_problems)
        logger.info(f"Ablation (no reward shaping) metrics: {ablation_metrics}")
        
    elif args.ablation == "single_turn":
        logger.info("Running ablation: single-turn RL only")
        # Single turn: just optimize first attempt
        from score.training.reinforce import REINFORCEPolicyGradient
        # ... would implement single-turn loop here
        logger.info("Single-turn ablation - would optimize Acc@t1 only")
        
    elif args.ablation == "star_stage2":
        logger.info("Running ablation: STaR instead of REINFORCE in Stage II")
        stage1_results = trainer.run_stage1(train_problems, test_problems)
        
        # Run STaR from Stage I checkpoint
        sft_config = SFTConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            task=args.task,
        )
        star_trainer = STaRTrainer(model, tokenizer, sft_config)
        
        # Build correction prompts
        if args.task == "math":
            prompts_t2 = [
                build_math_correction_prompt(p["problem"], "") 
                for p in train_problems
            ]
            prompts_t1_correct = [
                build_math_prompt(p["problem"]) for p in train_problems
            ]
        else:
            prompts_t2 = [
                build_code_correction_prompt(p["problem"], "", use_3shot=True)
                for p in train_problems
            ]
            prompts_t1_correct = [
                build_code_prompt(p["problem"], use_3shot=True)
                for p in train_problems
            ]
        
        star_result = star_trainer.train(
            train_problems, prompts_t1_correct, prompts_t2,
            num_iterations=3,
        )
        ablation_metrics = trainer.evaluate(test_problems)
        logger.info(f"Ablation (STaR Stage II) metrics: {ablation_metrics}")
        
    else:
        # Full SCoRe training
        logger.info("Starting SCoRe training...")
        
        # Stage I
        logger.info("=" * 60)
        logger.info("Stage I: Training initialization to decouple attempts")
        logger.info("=" * 60)
        stage1_results = trainer.run_stage1(train_problems, test_problems)
        logger.info(f"Stage I results: {stage1_results}")
        
        # Stage II
        logger.info("=" * 60)
        logger.info("Stage II: Multi-turn RL with reward shaping")
        logger.info("=" * 60)
        stage2_results = trainer.run_stage2(train_problems, test_problems)
        logger.info(f"Stage II results: {stage2_results}")
        
        # Final evaluation
        logger.info("=" * 60)
        logger.info("Final Evaluation")
        logger.info("=" * 60)
        
        final_metrics = trainer.evaluate(test_problems)
        logger.info(f"Final SCoRe metrics: {final_metrics}")
        
        # Format as table row
        logger.info(
            f"SCoRe (Ours) & {final_metrics['accuracy_t1']:.1%} & "
            f"{final_metrics['accuracy_t2']:.1%} & "
            f"{final_metrics['delta_t1_t2']:.1%} & "
            f"{final_metrics['i_to_c']:.1%} & "
            f"{final_metrics['c_to_i']:.1%}"
        )
    
    # Run SFT baselines if requested
    if args.run_sft_baselines:
        logger.info("=" * 60)
        logger.info("Running SFT Baselines (Section 4)")
        logger.info("=" * 60)
        
        # Reload model for baselines
        base_model, _ = load_model_and_tokenizer(
            args.model_name_or_path, device
        )
        
        # STaR
        sft_config = SFTConfig(
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            task=args.task,
        )
        star_trainer = STaRTrainer(base_model, tokenizer, sft_config)
        
        star_result = star_trainer.train(
            train_problems, train_prompts_t1, prompts_t2,
            num_iterations=3,
            include_correct_to_correct=True,  # D_STaR^+
        )
        # ... evaluate, etc.
    
    # Save final model
    trainer.save_checkpoint("final")
    logger.info(f"Training complete. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
