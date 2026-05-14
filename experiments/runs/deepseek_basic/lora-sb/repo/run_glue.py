#!/usr/bin/env python3
"""
Example script for fine-tuning RoBERTa-large on GLUE using LoRA-SB.

This follows the experimental setup from the paper (Section 3.3, Table 3).
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lora_sb import LoRA_SB_Layer, init_lora_sb
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


# GLUE task configurations (from Table 9 in paper)
GLUE_CONFIGS = {
    'cola': {'num_labels': 2, 'metric': 'matthews_correlation'},
    'sst2': {'num_labels': 2, 'metric': 'accuracy'},
    'mrpc': {'num_labels': 2, 'metric': 'accuracy'},
    'rte': {'num_labels': 2, 'metric': 'accuracy'},
    'qnli': {'num_labels': 2, 'metric': 'accuracy'},
    'stsb': {'num_labels': 1, 'metric': 'pearson'},
}


def load_glue_dataset(task_name, tokenizer, max_seq_length=512, split='train'):
    """Load a GLUE dataset."""
    from datasets import load_dataset

    dataset = load_dataset('glue', task_name, split=split)

    def tokenize_fn(examples):
        if task_name == 'stsb':
            return tokenizer(
                examples['sentence1'],
                examples['sentence2'],
                truncation=True,
                padding='max_length',
                max_length=max_seq_length,
            )
        elif task_name in ['cola', 'sst2']:
            return tokenizer(
                examples['sentence'],
                truncation=True,
                padding='max_length',
                max_length=max_seq_length,
            )
        else:
            return tokenizer(
                examples['question'] if task_name == 'qnli' else examples['sentence1'],
                examples['sentence'] if task_name == 'qnli' else examples['sentence2'],
                truncation=True,
                padding='max_length',
                max_length=max_seq_length,
            )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset


def run_glue_experiment(args):
    """Run a GLUE experiment with LoRA-SB."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    model_name = args.model_name or 'roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = GLUE_CONFIGS[args.task]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config['num_labels'],
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
    )
    model = model.to(device)

    # Load datasets
    train_dataset = load_glue_dataset(args.task, tokenizer, args.max_seq_length, 'train')
    eval_dataset = load_glue_dataset(args.task, tokenizer, args.max_seq_length, 'validation')

    # Determine number of initialization samples (0.1% of training data)
    num_init_samples = max(1, len(train_dataset) // 1000)
    logger.info(f"Using {num_init_samples} samples for initialization "
                f"({num_init_samples / len(train_dataset) * 100:.2f}% of training data)")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Target modules for RoBERTa: self-attention layers only (as per paper Section 3.3)
    target_modules = ['query', 'key', 'value', 'output.dense']

    # Initialize LoRA-SB
    logger.info(f"Initializing LoRA-SB with rank={args.rank}, scaling={args.scaling}")
    model = init_lora_sb(
        model,
        dataloader=train_loader,
        rank=args.rank,
        num_init_samples=num_init_samples,
        scaling=args.scaling,
        learning_rate=args.lr,
        use_layerwise=True,
        target_modules=target_modules,
    )

    # Log parameter counts
    param_stats = count_lora_sb_parameters(model)
    logger.info(f"Parameter stats: {param_stats}")

    # Create optimizer (only R is trainable)
    optimizer = get_lora_sb_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Gradient optimization wrapper (with orthonormal B, A, this is a no-op)
    gradient_opt_wrapper = LoRASBOptimizerWrapper(model, use_shortcut=True)

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler=lr_scheduler,
            gradient_opt_wrapper=gradient_opt_wrapper,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad_norm=args.clip_grad_norm,
            log_interval=args.log_interval,
        )
        logger.info(f"Train loss: {train_metrics['loss']:.4f}")

        eval_metrics = evaluate(model, eval_loader)
        logger.info(f"Eval loss: {eval_metrics['loss']:.4f}")

    # Save the model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"lora_sb_{args.task}_r{args.rank}.pt")
        merge_and_save(model, output_path)

    logger.info("Training complete!")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LoRA-SB GLUE fine-tuning')
    parser.add_argument('--task', type=str, default='sst2',
                       choices=list(GLUE_CONFIGS.keys()),
                       help='GLUE task name')
    parser.add_argument('--model_name', type=str, default='roberta-large',
                       help='Model name or path')
    parser.add_argument('--rank', type=int, default=8,
                       help='Rank of low-rank decomposition')
    parser.add_argument('--scaling', type=float, default=1.0,
                       help='Scaling factor s (default 1.0 for orthonormal B, A)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--max_seq_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.06,
                       help='Warmup ratio')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--clip_grad_norm', type=float, default=None,
                       help='Gradient clipping norm')
    parser.add_argument('--use_bf16', action='store_true',
                       help='Use bfloat16 precision')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Log every N steps')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_glue_experiment(args)
