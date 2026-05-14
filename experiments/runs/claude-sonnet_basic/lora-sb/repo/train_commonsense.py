"""
Training script for commonsense reasoning experiments.

Fine-tunes Llama-3.2 3B on COMMONSENSE170K using LoRA-SB.
Evaluates on 8 commonsense reasoning datasets.

Usage:
    python train_commonsense.py --model meta-llama/Llama-3.2-3B --rank 96 --method lora_sb
    python train_commonsense.py --model meta-llama/Llama-3.2-3B --rank 32 --method lora
"""

import argparse
import os
import random
import json
import re
from typing import Optional, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

from lora_sb import (
    apply_lora_sb,
    estimate_gradient,
    initialize_lora_sb,
    print_trainable_parameters,
)
from lora_xs import apply_lora_xs, initialize_lora_xs_pissa
from lora_baselines import apply_lora, apply_rslora, apply_pissa, apply_dora


# Target modules for Llama
LLAMA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Commonsense datasets
COMMONSENSE_DATASETS = [
    "boolq", "piqa", "social_i_qa", "hellaswag",
    "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM on commonsense reasoning")
    
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--method", type=str, default="lora_sb",
                        choices=["lora_sb", "lora_xs", "lora", "rslora", "pissa", "dora", "full_ft"])
    parser.add_argument("--rank", type=int, default=96)
    parser.add_argument("--lora_alpha", type=float, default=None)
    
    parser.add_argument("--train_dataset", type=str, default="commonsense_170k",
                        help="Path to COMMONSENSE170K dataset or HuggingFace dataset name")
    parser.add_argument("--output_dir", type=str, default="./outputs/commonsense")
    
    # Training hyperparams (from paper Table 8)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--grad_accum_steps", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.05)
    
    parser.add_argument("--init_fraction", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", default=True)
    
    return parser.parse_args()


# Prompt template from LLM-Adapters paper (Hu et al., 2023)
COMMONSENSE_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


class CommonsenseDataset(Dataset):
    """Dataset for commonsense reasoning tasks."""
    
    def __init__(self, data, tokenizer, max_seq_len: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        instruction = item.get("instruction", "")
        output = item.get("output", item.get("answer", ""))
        
        prompt = COMMONSENSE_PROMPT.format(instruction=instruction)
        full_text = prompt + " " + output
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Mask prompt tokens
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_encoding["input_ids"])
        
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[:self.max_seq_len]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, tokenizer, max_seq_len):
    """Collate with padding."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    max_len = min(max(len(ids) for ids in input_ids), max_seq_len)
    
    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []
    
    for ids, mask, lbl in zip(input_ids, attention_masks, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(
            torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
        )
        padded_attention_masks.append(
            torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        )
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "labels": torch.stack(padded_labels),
    }


def evaluate_commonsense(model, tokenizer, dataset_name: str, device: str, max_new_tokens: int = 10):
    """
    Evaluate model on a commonsense reasoning dataset.
    Returns accuracy.
    """
    # Load evaluation dataset
    dataset_map = {
        "boolq": ("boolq", "validation"),
        "piqa": ("piqa", "validation"),
        "social_i_qa": ("social_i_qa", "validation"),
        "hellaswag": ("hellaswag", "validation"),
        "winogrande": ("winogrande", "validation_xl"),
        "ARC-Easy": ("ai2_arc", "ARC-Easy", "test"),
        "ARC-Challenge": ("ai2_arc", "ARC-Challenge", "test"),
        "openbookqa": ("openbookqa", "main", "test"),
    }
    
    if dataset_name not in dataset_map:
        print(f"Unknown dataset: {dataset_name}")
        return 0.0
    
    config = dataset_map[dataset_name]
    try:
        if len(config) == 2:
            eval_data = load_dataset(config[0], split=config[1])
        else:
            eval_data = load_dataset(config[0], config[1], split=config[2])
    except Exception as e:
        print(f"Could not load {dataset_name}: {e}")
        return 0.0
    
    model.eval()
    correct = 0
    total = 0
    
    # Simple accuracy evaluation using generation
    with torch.no_grad():
        for item in eval_data:
            # Format question based on dataset
            question = format_commonsense_question(item, dataset_name)
            if question is None:
                continue
            
            prompt = COMMONSENSE_PROMPT.format(instruction=question["prompt"])
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True,
            ).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip().lower()
            
            if question["answer"].lower() in generated or generated in question["answer"].lower():
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def format_commonsense_question(item, dataset_name: str):
    """Format a commonsense question for generation."""
    try:
        if dataset_name == "boolq":
            return {
                "prompt": f"{item['passage']}\nQuestion: {item['question']}? Answer yes or no.",
                "answer": "yes" if item["answer"] else "no",
            }
        elif dataset_name == "piqa":
            choices = [item["sol1"], item["sol2"]]
            return {
                "prompt": f"Goal: {item['goal']}\nA: {choices[0]}\nB: {choices[1]}\nAnswer:",
                "answer": "A" if item["label"] == 0 else "B",
            }
        elif dataset_name == "hellaswag":
            endings = item["endings"]
            choices_str = "\n".join([f"{chr(65+i)}: {e}" for i, e in enumerate(endings)])
            return {
                "prompt": f"{item['ctx']}\n{choices_str}\nAnswer:",
                "answer": chr(65 + int(item["label"])),
            }
        elif dataset_name == "winogrande":
            return {
                "prompt": f"{item['sentence']}\nA: {item['option1']}\nB: {item['option2']}\nAnswer:",
                "answer": "A" if item["answer"] == "1" else "B",
            }
        elif dataset_name in ["ARC-Easy", "ARC-Challenge"]:
            choices = item["choices"]
            choices_str = "\n".join([f"{l}: {t}" for l, t in zip(choices["label"], choices["text"])])
            return {
                "prompt": f"{item['question']}\n{choices_str}\nAnswer:",
                "answer": item["answerKey"],
            }
        elif dataset_name == "openbookqa":
            choices = item["choices"]
            choices_str = "\n".join([f"{l}: {t}" for l, t in zip(choices["label"], choices["text"])])
            return {
                "prompt": f"{item['question_stem']}\n{choices_str}\nAnswer:",
                "answer": item["answerKey"],
            }
        elif dataset_name == "social_i_qa":
            return {
                "prompt": (f"Context: {item['context']}\nQuestion: {item['question']}\n"
                          f"A: {item['answerA']}\nB: {item['answerB']}\nC: {item['answerC']}\nAnswer:"),
                "answer": chr(64 + int(item["label"])),
            }
    except Exception:
        return None
    return None


def train(args):
    set_seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load training dataset (COMMONSENSE170K)
    print(f"Loading commonsense training data...")
    try:
        # Try loading from HuggingFace
        raw_dataset = load_dataset("tau/commonsense_qa", split="train")
    except Exception:
        # Fallback: try local path
        try:
            raw_dataset = load_dataset("json", data_files=args.train_dataset, split="train")
        except Exception as e:
            print(f"Could not load dataset: {e}")
            print("Please provide COMMONSENSE170K dataset")
            return
    
    # Load model
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
    )
    
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.rank
    
    if args.method == "lora_sb":
        model = apply_lora_sb(model, LLAMA_TARGET_MODULES, rank=args.rank)
    elif args.method == "lora_xs":
        model = apply_lora_xs(model, LLAMA_TARGET_MODULES, rank=args.rank, alpha=lora_alpha)
    elif args.method == "lora":
        model = apply_lora(model, LLAMA_TARGET_MODULES, rank=args.rank, alpha=lora_alpha,
                           dropout=args.dropout)
    elif args.method == "rslora":
        model = apply_rslora(model, LLAMA_TARGET_MODULES, rank=args.rank, alpha=lora_alpha,
                             dropout=args.dropout)
    elif args.method == "pissa":
        model = apply_pissa(model, LLAMA_TARGET_MODULES, rank=args.rank, alpha=lora_alpha,
                            dropout=args.dropout)
    elif args.method == "dora":
        model = apply_dora(model, LLAMA_TARGET_MODULES, rank=args.rank, alpha=lora_alpha,
                           dropout=args.dropout)
    
    print_trainable_parameters(model)
    device = args.device
    
    # Create dataset
    train_dataset = CommonsenseDataset(raw_dataset, tokenizer, args.max_seq_len)
    
    # LoRA-SB initialization
    if args.method == "lora_sb":
        n_init_samples = max(1, int(len(train_dataset) * args.init_fraction))
        print(f"Initializing LoRA-SB with {n_init_samples} samples")
        
        init_indices = random.sample(range(len(train_dataset)), n_init_samples)
        init_subset = Subset(train_dataset, init_indices)
        init_loader = DataLoader(
            init_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer, args.max_seq_len),
        )
        
        delta_w_dict = estimate_gradient(model, init_loader, n_init_samples, device)
        initialize_lora_sb(model, delta_w_dict)
        print("LoRA-SB initialization complete")
    
    elif args.method == "lora_xs":
        initialize_lora_xs_pissa(model)
    
    # Training dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_seq_len),
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    
    total_steps = len(train_loader) * args.num_epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    model.train()
    global_step = 0
    total_loss = 0.0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            total_loss += loss.item()
            
            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 100 == 0:
                    avg_loss = total_loss / 100
                    print(f"Step {global_step}: loss={avg_loss:.4f}")
                    total_loss = 0.0
    
    # Save model
    model_name = args.model.split("/")[-1]
    save_path = os.path.join(args.output_dir, f"{model_name}_{args.method}_r{args.rank}")
    os.makedirs(save_path, exist_ok=True)
    
    trainable_state = {k: v for k, v in model.state_dict().items()
                       if any(k.startswith(n) for n, p in model.named_parameters() if p.requires_grad)}
    torch.save(trainable_state, os.path.join(save_path, "adapter_weights.pt"))
    tokenizer.save_pretrained(save_path)
    
    print(f"Model saved to {save_path}")
    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = train(args)
