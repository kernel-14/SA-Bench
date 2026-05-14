"""
Training script for GLUE benchmark experiments.

Fine-tunes RoBERTa-large on GLUE tasks using LoRA-SB.
Evaluates on CoLA, RTE, MRPC, STS-B, QNLI, SST-2.

Usage:
    python train_glue.py --task cola --method lora_sb --rank 24
    python train_glue.py --task rte --method lora --rank 8
    python train_glue.py --task mrpc --method lora_xs --rank 16
"""

import argparse
import os
import random
import json
import math
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, accuracy_score

from lora_sb import (
    apply_lora_sb,
    estimate_gradient,
    initialize_lora_sb,
    print_trainable_parameters,
    LoRASBLinear,
)
from lora_xs import apply_lora_xs, initialize_lora_xs_pissa
from lora_baselines import apply_lora, apply_rslora, apply_pissa, apply_dora


# For RoBERTa, only apply to self-attention layers
ROBERTA_TARGET_MODULES = ["query", "key", "value", "dense"]


# GLUE task configurations
GLUE_CONFIGS = {
    "cola": {
        "num_labels": 2,
        "metric": "matthews_corrcoef",
        "text_fields": ["sentence"],
        "max_seq_len": 128,
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 1e-3,
        "warmup_ratio": 0.06,
    },
    "rte": {
        "num_labels": 2,
        "metric": "accuracy",
        "text_fields": ["sentence1", "sentence2"],
        "max_seq_len": 128,
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 1e-3,
        "warmup_ratio": 0.06,
    },
    "mrpc": {
        "num_labels": 2,
        "metric": "accuracy",
        "text_fields": ["sentence1", "sentence2"],
        "max_seq_len": 128,
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 1e-3,
        "warmup_ratio": 0.06,
    },
    "stsb": {
        "num_labels": 1,
        "metric": "pearson",
        "text_fields": ["sentence1", "sentence2"],
        "max_seq_len": 128,
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 1e-3,
        "warmup_ratio": 0.06,
    },
    "qnli": {
        "num_labels": 2,
        "metric": "accuracy",
        "text_fields": ["question", "sentence"],
        "max_seq_len": 256,
        "batch_size": 32,
        "epochs": 15,
        "learning_rate": 1e-3,
        "warmup_ratio": 0.06,
    },
    "sst2": {
        "num_labels": 2,
        "metric": "accuracy",
        "text_fields": ["sentence"],
        "max_seq_len": 128,
        "batch_size": 32,
        "epochs": 15,
        "learning_rate": 1e-3,
        "warmup_ratio": 0.06,
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RoBERTa-large on GLUE with LoRA-SB")
    
    parser.add_argument("--model", type=str, default="roberta-large")
    parser.add_argument("--task", type=str, default="cola",
                        choices=list(GLUE_CONFIGS.keys()))
    parser.add_argument("--method", type=str, default="lora_sb",
                        choices=["lora_sb", "lora_xs", "lora", "rslora", "pissa", "dora", "full_ft"])
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha (paper uses 16 for LoRA-XS on GLUE)")
    
    parser.add_argument("--output_dir", type=str, default="./outputs/glue")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--init_fraction", type=float, default=0.001)
    
    # Override hyperparams
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    
    return parser.parse_args()


class GLUEDataset(Dataset):
    """Dataset for GLUE tasks."""
    
    def __init__(self, data, tokenizer, task: str, max_seq_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.task = task
        self.max_seq_len = max_seq_len
        self.config = GLUE_CONFIGS[task]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text_fields = self.config["text_fields"]
        
        if len(text_fields) == 1:
            encoding = self.tokenizer(
                item[text_fields[0]],
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                item[text_fields[0]],
                item[text_fields[1]],
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        
        if "token_type_ids" in encoding:
            result["token_type_ids"] = encoding["token_type_ids"].squeeze()
        
        label = item["label"]
        if self.config["metric"] == "pearson":
            result["labels"] = torch.tensor(label, dtype=torch.float)
        else:
            result["labels"] = torch.tensor(label, dtype=torch.long)
        
        return result


def compute_metrics(task: str, predictions, labels):
    """Compute task-specific metrics."""
    config = GLUE_CONFIGS[task]
    metric = config["metric"]
    
    if metric == "matthews_corrcoef":
        preds = np.argmax(predictions, axis=1)
        return {"matthews_corrcoef": matthews_corrcoef(labels, preds)}
    elif metric == "accuracy":
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": accuracy_score(labels, preds)}
    elif metric == "pearson":
        corr, _ = pearsonr(predictions.squeeze(), labels)
        return {"pearson": corr}
    else:
        raise ValueError(f"Unknown metric: {metric}")


def setup_model(args, task_config):
    """Load and configure model."""
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=task_config["num_labels"],
        torch_dtype=dtype,
    )
    
    lora_alpha = args.lora_alpha
    
    if args.method == "lora_sb":
        model = apply_lora_sb(model, ROBERTA_TARGET_MODULES, rank=args.rank)
    elif args.method == "lora_xs":
        model = apply_lora_xs(model, ROBERTA_TARGET_MODULES, rank=args.rank, alpha=lora_alpha)
    elif args.method == "lora":
        model = apply_lora(model, ROBERTA_TARGET_MODULES, rank=args.rank, alpha=lora_alpha)
    elif args.method == "rslora":
        model = apply_rslora(model, ROBERTA_TARGET_MODULES, rank=args.rank, alpha=lora_alpha)
    elif args.method == "pissa":
        model = apply_pissa(model, ROBERTA_TARGET_MODULES, rank=args.rank, alpha=lora_alpha)
    elif args.method == "dora":
        model = apply_dora(model, ROBERTA_TARGET_MODULES, rank=args.rank, alpha=lora_alpha)
    
    print_trainable_parameters(model)
    return model


def train(args):
    set_seed(args.seed)
    
    task_config = GLUE_CONFIGS[args.task]
    
    # Override hyperparams if specified
    batch_size = args.batch_size or task_config["batch_size"]
    num_epochs = args.epochs or task_config["epochs"]
    learning_rate = args.learning_rate or task_config["learning_rate"]
    max_seq_len = task_config["max_seq_len"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load dataset
    glue_task = args.task if args.task != "stsb" else "stsb"
    raw_dataset = load_dataset("glue", glue_task)
    
    train_data = raw_dataset["train"]
    val_data = raw_dataset["validation"]
    
    # Setup model
    model = setup_model(args, task_config)
    device = args.device
    model = model.to(device)
    
    # Create datasets
    train_dataset = GLUEDataset(train_data, tokenizer, args.task, max_seq_len)
    val_dataset = GLUEDataset(val_data, tokenizer, args.task, max_seq_len)
    
    # LoRA-SB initialization
    if args.method == "lora_sb":
        n_init_samples = max(1, int(len(train_dataset) * args.init_fraction))
        print(f"Initializing LoRA-SB with {n_init_samples} samples")
        
        init_indices = random.sample(range(len(train_dataset)), n_init_samples)
        init_subset = Subset(train_dataset, init_indices)
        init_loader = DataLoader(init_subset, batch_size=batch_size, shuffle=False)
        
        delta_w_dict = estimate_gradient(model, init_loader, n_init_samples, device)
        initialize_lora_sb(model, delta_w_dict)
        print("LoRA-SB initialization complete")
    
    elif args.method == "lora_xs":
        initialize_lora_xs_pissa(model)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * task_config["warmup_ratio"])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    best_metric = -float('inf')
    best_results = {}
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                
                outputs = model(**batch)
                logits = outputs.logits
                
                if task_config["metric"] == "pearson":
                    all_preds.extend(logits.cpu().numpy().tolist())
                else:
                    all_preds.extend(logits.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = compute_metrics(args.task, all_preds, all_labels)
        metric_value = list(metrics.values())[0]
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, {metrics}")
        
        if metric_value > best_metric:
            best_metric = metric_value
            best_results = metrics
    
    print(f"\nBest results: {best_results}")
    
    # Save results
    results = {
        "task": args.task,
        "method": args.method,
        "rank": args.rank,
        "seed": args.seed,
        "best_metric": best_metric,
        "metrics": best_results,
    }
    
    save_path = os.path.join(
        args.output_dir,
        f"{args.task}_{args.method}_r{args.rank}_seed{args.seed}.json"
    )
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return best_results


if __name__ == "__main__":
    args = parse_args()
    results = train(args)
    print(f"Final results: {results}")
