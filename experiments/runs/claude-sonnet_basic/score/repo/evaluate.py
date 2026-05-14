"""
Evaluation script for comparing SCoRe against baselines.

Reproduces Tables 2 and 3 from the paper.

Usage:
    python evaluate.py --task math \
        --score_model_path ./outputs/score_math/best_checkpoint \
        --base_model_name google/gemma-2b \
        --math_data_dir /path/to/MATH \
        --output_dir ./outputs/evaluation
"""

import argparse
import json
import logging
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    compute_edit_distance_ratios,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SCoRe and baselines")
    
    parser.add_argument("--task", type=str, default="math", choices=["math", "code"])
    parser.add_argument("--base_model_name", type=str, default="google/gemma-2b")
    parser.add_argument("--score_model_path", type=str, default=None,
                        help="Path to trained SCoRe model")
    parser.add_argument("--star_model_path", type=str, default=None,
                        help="Path to trained STaR model")
    parser.add_argument("--pair_sft_model_path", type=str, default=None,
                        help="Path to trained Pair-SFT model")
    
    parser.add_argument("--math_data_dir", type=str, default="./data/MATH")
    parser.add_argument("--mbpp_data_path", type=str, default="./data/mbpp.jsonl")
    parser.add_argument("--humaneval_data_path", type=str, default="./data/HumanEval.jsonl")
    
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation")
    parser.add_argument("--n_eval", type=int, default=500,
                        help="Number of evaluation problems")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    
    return parser.parse_args()


def load_model(model_path: str, device: str = "auto"):
    """Load a model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    
    return model, tokenizer


def main():
    args = parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load evaluation data
    if args.task == "math":
        reward_fn = math_reward
        correction_instruction = MATH_SELF_CORRECTION_INSTRUCTION
        
        test_problems = load_math_dataset(args.math_data_dir, split="test")
        _, eval_problems = create_math500_split(test_problems, n_eval=500)
        
        # Subsample if needed
        if args.n_eval < len(eval_problems):
            eval_problems = random.sample(eval_problems, args.n_eval)
        
        eval_probs = [p.problem for p in eval_problems]
        eval_gts = [p.answer for p in eval_problems]
        eval_prompts = [build_math_prompt(p.problem) for p in eval_problems]
        
    else:
        correction_instruction = CODE_SELF_CORRECTION_INSTRUCTION
        
        humaneval_problems = load_humaneval_dataset(args.humaneval_data_path)
        
        if args.n_eval < len(humaneval_problems):
            humaneval_problems = random.sample(humaneval_problems, args.n_eval)
        
        eval_probs = [p.prompt for p in humaneval_problems]
        eval_gts = [p.test_cases for p in humaneval_problems]
        eval_prompts = [p.prompt for p in humaneval_problems]
        
        def reward_fn(response, gt):
            return code_reward(response, gt)
    
    all_results = {}
    
    # Evaluate each model
    models_to_eval = {
        "Base model": args.base_model_name,
    }
    
    if args.score_model_path:
        models_to_eval["SCoRe (Ours)"] = args.score_model_path
    if args.star_model_path:
        models_to_eval["STaR"] = args.star_model_path
    if args.pair_sft_model_path:
        models_to_eval["Pair-SFT"] = args.pair_sft_model_path
    
    for method_name, model_path in models_to_eval.items():
        logger.info(f"Evaluating {method_name} from {model_path}...")
        
        model, tokenizer = load_model(model_path, args.device)
        
        metrics = evaluate_model_on_dataset(
            model=model,
            tokenizer=tokenizer,
            problems=eval_probs,
            ground_truths=eval_gts,
            first_prompts=eval_prompts,
            correction_instruction=correction_instruction,
            reward_fn=reward_fn,
            temperature=0.0,  # Greedy decoding as per paper
        )
        
        all_results[method_name] = metrics
        
        logger.info(f"{method_name}: {metrics}")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Print and save results table
    task_name = "MATH" if args.task == "math" else "HumanEval"
    print(format_results_table(all_results, task=task_name))
    
    save_results(
        all_results,
        os.path.join(args.output_dir, f"results_{args.task}.json"),
        task=task_name,
    )
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
