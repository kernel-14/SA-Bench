#!/usr/bin/env python3
"""
Example script for fine-tuning LLMs on math/commonsense reasoning using LoRA-SB.

Follows the experimental setup from the paper:
- Section 3.1: Mistral-7B and Gemma-2 9B on MetaMathQA (GSM8K/MATH eval)
- Section 3.2: Llama-3.2 3B on COMMONSENSE170K

Key hyperparameters (from Table 8):
- Mistral-7B / Gemma-2 9B: lr=1e-4, batch_size=1, grad_acc=32, epochs=1, cosine schedule
- Llama-3.2 3B: lr=2e-3, batch_size=6, grad_acc=24, epochs=2, linear schedule
"""

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lora_sb import init_lora_sb
from lora_sb.train import (
    train_epoch,
    evaluate,
    count_lora_sb_parameters,
    get_lora_sb_optimizer,
    merge_and_save,
)
from lora_sb.gradient_opt import LoRASBOptimizerWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def run_experiment(args):
    """Run a math/commonsense fine-tuning experiment with LoRA-SB."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )

    # Target all attention + FC layers as per paper
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                     'gate_proj', 'up_proj', 'down_proj']

    # Prepare a dataloader for initialization (using a small subset)
    from datasets import load_dataset

    if args.dataset == 'metamath':
        dataset = load_dataset('meta-math/MetaMathQA', split='train')
    elif args.dataset == 'commonsense':
        dataset = load_dataset('commonsense_qa', split='train')
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Take a subset if needed
    if args.max_train_samples and args.max_train_samples < len(dataset):
        dataset = dataset.select(range(args.max_train_samples))

    # Tokenize
    def tokenize_fn(examples):
        if args.dataset == 'metamath':
            texts = [f"Question: {q}\nAnswer: {a}" for q, a in
                    zip(examples['query'], examples['response'])]
        else:
            texts = examples.get('question', examples.get('text', ''))

        result = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=args.max_seq_length,
            return_tensors='pt',
        )
        result['labels'] = result['input_ids'].clone()
        return result

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type='torch')

    # Determine initialization samples (0.1%)
    num_init_samples = max(1, len(dataset) // 1000)
    logger.info(f"Using {num_init_samples} samples for initialization")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Initialize LoRA-SB
    logger.info(f"Initializing LoRA-SB with rank={args.rank}")
    model = init_lora_sb(
        model,
        dataloader=train_loader,
        rank=args.rank,
        num_init_samples=num_init_samples,
        scaling=1.0,
        learning_rate=args.lr,
        use_layerwise=True,
        target_modules=target_modules,
    )

    # Log params
    stats = count_lora_sb_parameters(model)
    logger.info(f"Parameter stats: {stats}")

    # Optimizer
    optimizer = get_lora_sb_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    num_training_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)

    if args.scheduler == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    gradient_opt_wrapper = LoRASBOptimizerWrapper(model, use_shortcut=True)

    # Train
    logger.info(f"Training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        metrics = train_epoch(
            model, train_loader, optimizer,
            lr_scheduler=lr_scheduler,
            gradient_opt_wrapper=gradient_opt_wrapper,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_interval=args.log_interval,
        )
        logger.info(f"Loss: {metrics['loss']:.4f}")

    # Save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"lora_sb_r{args.rank}.pt")
        merge_and_save(model, out_path)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LoRA-SB Math/Commonsense Fine-tuning')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1',
                       help='Model name')
    parser.add_argument('--dataset', type=str, default='metamath',
                       choices=['metamath', 'commonsense'])
    parser.add_argument('--rank', type=int, default=32,
                       help='Low-rank decomposition rank')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Micro batch size')
    parser.add_argument('--max_seq_length', type=int, default=512,
                       help='Max sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32,
                       help='Gradient accumulation steps')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.02)
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'linear'])
    parser.add_argument('--use_bf16', action='store_true')
    parser.add_argument('--max_train_samples', type=int, default=50000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    run_experiment(args)
