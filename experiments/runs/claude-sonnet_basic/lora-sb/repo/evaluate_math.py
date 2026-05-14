"""
Evaluation script for GSM8K and MATH benchmarks.

Evaluates a fine-tuned model on mathematical reasoning tasks.
Extracts the final numeric answer and computes accuracy.

Usage:
    python evaluate_math.py --model_path ./outputs/math/mistral-7b_lora_sb_r96 \
                            --base_model mistralai/Mistral-7B-v0.1 \
                            --benchmark gsm8k
"""

import argparse
import os
import re
import json
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on math benchmarks")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model or adapter weights")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name (if loading adapter separately)")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        choices=["gsm8k", "math"])
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples to evaluate (None = all)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--output_file", type=str, default=None)
    
    return parser.parse_args()


# Prompt templates
GSM8K_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{question}\n\n### Response:"
)

MATH_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{question}\n\n### Response:"
)


def extract_answer_gsm8k(text: str) -> Optional[str]:
    """Extract the final numeric answer from GSM8K response."""
    # Look for #### pattern (GSM8K format)
    match = re.search(r'####\s*([\d,.-]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    
    # Look for "The answer is X" pattern
    match = re.search(r'[Tt]he answer is\s*([\d,.-]+)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    
    # Extract last number
    numbers = re.findall(r'[-+]?\d*\.?\d+', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    
    return None


def extract_answer_math(text: str) -> Optional[str]:
    """Extract the final answer from MATH response."""
    # Look for \boxed{} pattern
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    
    # Look for "The answer is" pattern
    match = re.search(r'[Tt]he answer is\s*(.+?)(?:\.|$)', text)
    if match:
        return match.group(1).strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    
    # Remove whitespace
    answer = answer.strip()
    
    # Remove trailing zeros after decimal
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        pass
    
    return answer.lower()


def evaluate_gsm8k(model, tokenizer, n_samples=None, device="cuda", max_new_tokens=512):
    """Evaluate on GSM8K benchmark."""
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))
    
    correct = 0
    total = 0
    results = []
    
    model.eval()
    
    for item in tqdm(dataset, desc="Evaluating GSM8K"):
        question = item["question"]
        gold_answer = item["answer"]
        
        # Extract gold answer (after ####)
        gold_match = re.search(r'####\s*([\d,.-]+)', gold_answer)
        if gold_match:
            gold_num = gold_match.group(1).replace(',', '').strip()
        else:
            gold_num = gold_answer.strip()
        
        prompt = GSM8K_PROMPT.format(question=question)
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        pred_answer = extract_answer_gsm8k(generated)
        
        is_correct = (normalize_answer(pred_answer) == normalize_answer(gold_num))
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "gold": gold_num,
            "predicted": pred_answer,
            "generated": generated,
            "correct": is_correct,
        })
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"GSM8K Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy, results


def evaluate_math(model, tokenizer, n_samples=None, device="cuda", max_new_tokens=512):
    """Evaluate on MATH benchmark."""
    dataset = load_dataset("hendrycks/competition_math", split="test")
    
    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))
    
    correct = 0
    total = 0
    results = []
    
    model.eval()
    
    for item in tqdm(dataset, desc="Evaluating MATH"):
        question = item["problem"]
        gold_answer = item["solution"]
        
        # Extract gold answer from \boxed{}
        gold_match = re.search(r'\\boxed\{([^}]+)\}', gold_answer)
        gold_num = gold_match.group(1).strip() if gold_match else gold_answer.strip()
        
        prompt = MATH_PROMPT.format(question=question)
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        pred_answer = extract_answer_math(generated)
        
        is_correct = (normalize_answer(pred_answer) == normalize_answer(gold_num))
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "gold": gold_num,
            "predicted": pred_answer,
            "generated": generated,
            "correct": is_correct,
        })
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"MATH Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy, results


def main():
    args = parse_args()
    
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    # Load model
    if args.base_model:
        # Load base model and apply adapter
        from lora_sb import apply_lora_sb, LoRASBLinear
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        
        # Load adapter weights
        adapter_path = os.path.join(args.model_path, "adapter_weights.pt")
        if os.path.exists(adapter_path):
            state_dict = torch.load(adapter_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
    else:
        # Load full model from path
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = args.device
    
    # Evaluate
    if args.benchmark == "gsm8k":
        accuracy, results = evaluate_gsm8k(
            model, tokenizer, args.n_samples, device, args.max_new_tokens
        )
    elif args.benchmark == "math":
        accuracy, results = evaluate_math(
            model, tokenizer, args.n_samples, device, args.max_new_tokens
        )
    
    # Save results
    if args.output_file:
        output = {
            "benchmark": args.benchmark,
            "accuracy": accuracy,
            "results": results,
        }
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
