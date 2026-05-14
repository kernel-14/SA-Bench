"""Train on COMMONSENSE170K and evaluate on 8 commonsense reasoning datasets.

Supports: LoRA-SB, LoRA-XS, LoRA, rsLoRA, PiSSA, DoRA, LoRA-Pro, Full FT
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict

import numpy as np
import torch
import yaml

from src.data import (
    COMMONSENSE_DATASETS,
    get_causal_lm_dataloader,
    get_init_dataloader,
    load_commonsense170k,
    load_commonsense_eval,
)
from src.evaluate import evaluate_commonsense
from src.initialization import initialize_lora_sb
from src.model import (
    CAUSAL_LM_TARGET_MODULES,
    apply_lora,
    apply_lora_sb,
    apply_lora_xs,
    count_trainable_parameters,
    load_causal_lm,
)
from src.train import TrainingConfig, train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on COMMONSENSE170K")
    parser.add_argument("--config", type=str, default="config/commonsense_reasoning.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--method", type=str, default=None,
                        choices=["lora_sb", "lora_xs", "lora", "rslora", "pissa", "dora", "lora_pro", "full_ft"])
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


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

    set_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float32

    print(f"Method: {cfg['method']}, Model: {cfg['model_name']}, Rank: {cfg['rank']}")

    model, tokenizer = load_causal_lm(cfg["model_name"], torch_dtype=dtype)

    print("Loading COMMONSENSE170K...")
    train_dataset = load_commonsense170k(
        tokenizer=tokenizer,
        max_seq_len=cfg["max_seq_len"],
        seed=cfg["seed"],
    )

    method = cfg["method"]
    rank = cfg["rank"]
    alpha = cfg.get("alpha", rank)
    dropout = cfg.get("dropout", 0.0)
    target_modules = cfg.get("target_modules", CAUSAL_LM_TARGET_MODULES)

    if method == "lora_sb":
        print(f"Computing LoRA-SB initialization with {cfg['init_n_samples']} samples...")
        init_dataloader = get_init_dataloader(
            dataset=train_dataset,
            tokenizer=tokenizer,
            n_samples=cfg["init_n_samples"],
            batch_size=cfg["init_batch_size"],
            seed=cfg["seed"],
        )
        init_dict = initialize_lora_sb(
            model=model,
            dataloader=init_dataloader,
            target_module_names=target_modules,
            rank=rank,
            n_samples=cfg["init_n_samples"],
            device=device,
            use_lowrank_svd=True,
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
    print(f"Trainable parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    train_cfg = TrainingConfig(
        optimizer=cfg.get("optimizer", "adamw"),
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
        lr_scheduler=cfg["lr_scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        num_epochs=cfg["num_epochs"],
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        dropout=dropout,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        logging_steps=cfg.get("logging_steps", 10),
        output_dir=cfg.get("output_dir", "outputs/commonsense"),
        seed=cfg["seed"],
        dtype=cfg.get("dtype", "bfloat16"),
    )

    train_dataloader = get_causal_lm_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        seed=cfg["seed"],
    )

    print("Starting training...")
    start_time = time.time()
    history = train(model, train_dataloader, train_cfg, device)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed / 3600:.2f}h")

    # Evaluate on all 8 commonsense datasets
    eval_results = {}
    eval_datasets = cfg.get("eval_datasets", COMMONSENSE_DATASETS)

    for dataset_name in eval_datasets:
        print(f"Evaluating on {dataset_name}...")
        try:
            eval_data = load_commonsense_eval(dataset_name)
            metrics = evaluate_commonsense(
                model=model,
                tokenizer=tokenizer,
                dataset=eval_data,
                device=device,
                dataset_name=dataset_name,
                max_new_tokens=32,
                batch_size=8,
            )
            eval_results[dataset_name] = metrics["accuracy"]
            print(f"  {dataset_name}: {metrics['accuracy']:.2f}%")
        except Exception as e:
            print(f"  {dataset_name} evaluation failed: {e}")
            eval_results[dataset_name] = 0.0

    avg_accuracy = sum(eval_results.values()) / len(eval_results) if eval_results else 0.0
    print(f"\nAverage accuracy: {avg_accuracy:.2f}%")

    results = {
        "method": method,
        "model": cfg["model_name"],
        "rank": rank,
        "n_params": n_params,
        "eval_results": eval_results,
        "avg_accuracy": avg_accuracy,
        "training_time_hours": elapsed / 3600,
    }
    print("\nResults:", results)

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    with open(os.path.join(train_cfg.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
