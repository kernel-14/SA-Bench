"""Train RoBERTa-large on GLUE tasks and evaluate.

Supports: LoRA-SB, LoRA-XS, LoRA, rsLoRA, PiSSA, DoRA, LoRA-Pro, Full FT
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
import yaml

from src.data import (
    GLUE_NUM_LABELS,
    GLUE_TASKS,
    get_glue_dataloader,
    get_init_dataloader,
    load_glue_task,
)
from src.evaluate import make_glue_metrics_fn
from src.initialization import initialize_lora_sb
from src.model import (
    SEQ_CLS_TARGET_MODULES,
    apply_lora,
    apply_lora_sb,
    apply_lora_xs,
    count_trainable_parameters,
    load_seq_cls,
)
from src.train import TrainingConfig, evaluate_glue, train_glue


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on GLUE")
    parser.add_argument("--config", type=str, default="config/glue.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--method", type=str, default=None,
                        choices=["lora_sb", "lora_xs", "lora", "rslora", "pissa", "dora", "lora_pro", "full_ft"])
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="GLUE tasks to run (default: all)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_glue_task(
    task_name: str,
    cfg: Dict,
    method: str,
    rank: int,
    alpha: float,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Dict:
    """Run a single GLUE task."""
    task_cfg = cfg["task_configs"][task_name]
    num_labels = task_cfg["num_labels"]
    batch_size = task_cfg["batch_size"]
    max_seq_len = task_cfg["max_seq_len"]
    metric_name = task_cfg["metric"]

    print(f"\n{'='*60}")
    print(f"Task: {task_name.upper()}, Method: {method}, Rank: {rank}")
    print(f"{'='*60}")

    # Load model fresh for each task
    model, tokenizer = load_seq_cls(cfg["model_name"], num_labels=num_labels, torch_dtype=dtype)

    # Load data
    dataset = load_glue_task(task_name, tokenizer, max_seq_len=max_seq_len)
    train_data = dataset["train"]
    eval_data = dataset["validation"] if "validation" in dataset else dataset["validation_matched"]

    # Compute init samples (0.1% of training set)
    n_init = max(1, int(len(train_data) * cfg.get("init_n_samples_ratio", 0.001)))
    dropout = cfg.get("dropout", 0.0)
    target_modules = cfg.get("target_modules", SEQ_CLS_TARGET_MODULES)

    if method == "lora_sb":
        # For GLUE, we need a DataLoader that yields batches with labels
        # Use a simple DataLoader from the tokenized dataset
        from torch.utils.data import DataLoader, Subset
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        init_subset = Subset(train_data, indices[:n_init])
        init_loader = DataLoader(init_subset, batch_size=1, shuffle=False)

        print(f"Computing LoRA-SB initialization with {n_init} samples...")
        init_dict = initialize_lora_sb(
            model=model,
            dataloader=init_loader,
            target_module_names=target_modules,
            rank=rank,
            n_samples=n_init,
            device=device,
            use_lowrank_svd=False,  # Use full SVD for smaller RoBERTa matrices
        )
        model = apply_lora_sb(model, target_modules, init_dict, dropout=dropout)

    elif method == "lora_xs":
        model = apply_lora_xs(model, target_modules, rank=rank, alpha=alpha, dropout=dropout)

    elif method in ("lora", "rslora", "pissa", "dora", "lora_pro"):
        model = apply_lora(model, target_modules, rank=rank, alpha=alpha, dropout=dropout, method=method)

    elif method == "full_ft":
        for param in model.parameters():
            param.requires_grad_(True)

    n_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {n_params:,} ({n_params / 1000:.2f}K)")

    train_loader = get_glue_dataloader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = get_glue_dataloader(eval_data, batch_size=batch_size, shuffle=False)

    train_cfg = TrainingConfig(
        optimizer=cfg.get("optimizer", "adamw"),
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
        lr_scheduler=cfg["lr_scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        num_epochs=cfg["num_epochs"],
        batch_size=batch_size,
        gradient_accumulation_steps=1,  # GLUE uses no gradient accumulation
        dropout=dropout,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        logging_steps=cfg.get("logging_steps", 10),
        output_dir=os.path.join(cfg.get("output_dir", "outputs/glue"), task_name),
        seed=seed,
        dtype=cfg.get("dtype", "bfloat16"),
    )

    compute_metrics = make_glue_metrics_fn(task_name)

    start_time = time.time()
    result = train_glue(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=train_cfg,
        device=device,
        task_name=task_name,
        compute_metrics_fn=compute_metrics,
    )
    elapsed = time.time() - start_time

    best_metric = result["best_metric"]
    print(f"Best {metric_name}: {best_metric:.4f} (trained in {elapsed/60:.1f}min)")

    return {
        "task": task_name,
        "method": method,
        "rank": rank,
        "n_params": n_params,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "training_time_min": elapsed / 60,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.model_name:
        cfg["model_name"] = args.model_name
    if args.method:
        cfg["method"] = args.method
    if args.rank:
        cfg["rank"] = args.rank
    if args.alpha:
        cfg["alpha"] = args.alpha
    if args.seed:
        cfg["seed"] = args.seed
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float32

    method = cfg["method"]
    rank = cfg["rank"]
    alpha = cfg.get("alpha", 16)  # LoRA-XS uses alpha=16 for GLUE

    tasks = args.tasks or cfg.get("glue_tasks", GLUE_TASKS)

    all_results = []
    for task_name in tasks:
        result = run_glue_task(
            task_name=task_name,
            cfg=cfg,
            method=method,
            rank=rank,
            alpha=alpha,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        all_results.append(result)

    # Print summary table
    print("\n" + "="*80)
    print(f"GLUE Results — Method: {method}, Rank: {rank}")
    print("="*80)
    print(f"{'Task':<12} {'Metric':<25} {'Value':>10} {'#Params':>12}")
    print("-"*60)
    for r in all_results:
        print(f"{r['task']:<12} {r['metric_name']:<25} {r['best_metric']:>10.4f} {r['n_params']:>12,}")

    metrics = [r["best_metric"] for r in all_results]
    print(f"\nAverage: {sum(metrics)/len(metrics):.4f}")

    # Save results
    output_dir = cfg.get("output_dir", "outputs/glue")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
