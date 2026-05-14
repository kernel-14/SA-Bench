"""
Training script for arithmetic reasoning experiments.

Fine-tunes Mistral-7B or Gemma-2 9B on MetaMathQA using LoRA-SB.
Evaluates on GSM8K and MATH benchmarks.

Usage:
    python train_math.py --model mistral-7b --rank 96 --method lora_sb
    python train_math.py --model gemma-2-9b --rank 64 --method lora_sb
    python train_math.py --model mistral-7b --rank 32 --method lora_xs
    python train_math.py --model mistral-7b --rank 32 --method lora
"""

import argparse
import os
import math
import random
import json
from typing import Optional, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

from lora_sb import (
    apply_lora_sb,
    estimate_gradient,
    initialize_lora_sb,
    print_trainable_parameters,
    LoRASBLinear,
)
from lora_xs import apply_lora_xs
from lora_baselines import apply_lora, apply_rslora, apply_pissa


# Target modules for causal LMs (key, value, query, output, and FC layers)
CAUSAL_LM_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM on MetaMathQA with LoRA-SB")
    
    # Model
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="Model name or path")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Short model name for logging (e.g., mistral-7b)")
    
    # Method
    parser.add_argument("--method", type=str, default="lora_sb",
                        choices=["lora_sb", "lora_xs", "lora", "rslora", "pissa", "full_ft"],
                        help="Fine-tuning method")
    parser.add_argument("--rank", type=int, default=96,
                        help="Rank for low-rank methods")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha (default: same as rank)")
    
    # Data
    parser.add_argument("--dataset", type=str, default="meta-math/MetaMathQA",
                        help="Training dataset")
    parser.add_argument("--n_train_samples", type=int, default=50000,
                        help="Number of training samples")
    parser.add_argument("--init_fraction", type=float, default=0.001,
                        help="Fraction of data for initialization (0.1%)")
    
    # Training
    parser.add_argument("--output_dir", type=str, default="./outputs/math",
                        help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Evaluation
    parser.add_argument("--eval_gsm8k", action="store_true", default=True)
    parser.add_argument("--eval_math", action="store_true", default=True)
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", default=True)
    
    return parser.parse_args()


class MetaMathDataset(Dataset):
    """Dataset for MetaMathQA training."""
    
    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    )
    
    def __init__(self, data, tokenizer, max_seq_len: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format prompt
        instruction = item.get("query", item.get("instruction", ""))
        response = item.get("response", item.get("output", ""))
        
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)
        full_text = prompt + response
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Create labels: mask prompt tokens with -100
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
    """Collate function with padding."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad to max length in batch
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


def setup_model(args, tokenizer):
    """Load and configure model with specified fine-tuning method."""
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" else None,
    )
    
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.rank
    
    if args.method == "lora_sb":
        print(f"Applying LoRA-SB with rank={args.rank}")
        model = apply_lora_sb(model, CAUSAL_LM_TARGET_MODULES, rank=args.rank)
    elif args.method == "lora_xs":
        print(f"Applying LoRA-XS with rank={args.rank}")
        model = apply_lora_xs(model, CAUSAL_LM_TARGET_MODULES, rank=args.rank,
                              alpha=lora_alpha)
    elif args.method == "lora":
        print(f"Applying LoRA with rank={args.rank}")
        model = apply_lora(model, CAUSAL_LM_TARGET_MODULES, rank=args.rank,
                           alpha=lora_alpha)
    elif args.method == "rslora":
        print(f"Applying rsLoRA with rank={args.rank}")
        model = apply_rslora(model, CAUSAL_LM_TARGET_MODULES, rank=args.rank,
                             alpha=lora_alpha)
    elif args.method == "pissa":
        print(f"Applying PiSSA with rank={args.rank}")
        model = apply_pissa(model, CAUSAL_LM_TARGET_MODULES, rank=args.rank,
                            alpha=lora_alpha)
    elif args.method == "full_ft":
        print("Full fine-tuning")
        # All parameters trainable
    
    print_trainable_parameters(model)
    return model


def train(args):
    set_seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    raw_dataset = load_dataset(args.dataset, split="train")
    
    # Subsample training data
    if args.n_train_samples < len(raw_dataset):
        indices = random.sample(range(len(raw_dataset)), args.n_train_samples)
        raw_dataset = raw_dataset.select(indices)
    
    # Setup model
    model = setup_model(args, tokenizer)
    device = args.device
    
    # Create dataset
    train_dataset = MetaMathDataset(raw_dataset, tokenizer, args.max_seq_len)
    
    # LoRA-SB initialization
    if args.method == "lora_sb":
        n_init_samples = max(1, int(args.n_train_samples * args.init_fraction))
        print(f"Initializing LoRA-SB with {n_init_samples} samples ({args.init_fraction*100:.1f}%)")
        
        # Create init dataloader
        init_indices = random.sample(range(len(train_dataset)), n_init_samples)
        init_subset = Subset(train_dataset, init_indices)
        init_loader = DataLoader(
            init_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer, args.max_seq_len),
        )
        
        # Estimate gradient
        delta_w_dict = estimate_gradient(model, init_loader, n_init_samples, device)
        
        # Initialize LoRA-SB
        initialize_lora_sb(model, delta_w_dict)
        print("LoRA-SB initialization complete")
    
    elif args.method == "lora_xs":
        # LoRA-XS uses SVD of pre-trained weights (PiSSA-style)
        from lora_xs import initialize_lora_xs_pissa
        initialize_lora_xs_pissa(model)
    
    # Create training dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_seq_len),
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * args.num_epochs // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
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
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            
            # Gradient accumulation
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
                    print(f"Step {global_step}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
                    total_loss = 0.0
    
    # Save model
    model_name = args.model_name or args.model.split("/")[-1]
    save_path = os.path.join(args.output_dir, f"{model_name}_{args.method}_r{args.rank}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save only trainable parameters
    trainable_state = {k: v for k, v in model.state_dict().items() 
                       if any(k.startswith(n) for n, p in model.named_parameters() if p.requires_grad)}
    torch.save(trainable_state, os.path.join(save_path, "adapter_weights.pt"))
    tokenizer.save_pretrained(save_path)
    
    print(f"Model saved to {save_path}")
    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = train(args)
