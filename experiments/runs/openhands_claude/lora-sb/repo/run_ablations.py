"""Ablation experiments from Section 4 of the paper.

Ablation 1 (Table 4): Initialization strategies
  - trunc_SVD(Kaiming)
  - trunc_SVD(ΔW_avg + N_{μ=1e-2})
  - trunc_SVD(ΔW_avg + N_{μ=1e-3})
  - trunc_SVD(ΔW_avg + N_{μ=1e-4})
  - trunc_SVD(ΔW_avg + N_{μ=1e-5})
  - LoRA-SB: trunc_SVD(ΔW_avg)

Ablation 2 (Table 5): Number of initialization samples
  - 1, 5, 25, 50, 100, 200, 500 samples

Ablation 3 (Figure 3): Optimal gradient approximation
  - LoRA-SB (orthonormal B and A)
  - Non-orthogonal variant (B = U*S, A = Vh, R = I)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from src.data import (
    get_causal_lm_dataloader,
    get_init_dataloader,
    load_gsm8k,
    load_math_dataset,
    load_metamath,
)
from src.evaluate import evaluate_math_generation
from src.initialization import compute_gradient_approximation, truncated_svd_init
from src.lora_layers import LoRASBLayer, LinearWithLoRASB
from src.model import (
    CAUSAL_LM_TARGET_MODULES,
    apply_lora_sb,
    count_trainable_parameters,
    freeze_base_model,
    load_causal_lm,
)
from src.train import TrainingConfig, train
from src.utils import (
    init_kaiming_svd,
    init_non_orthogonal_svd,
    init_noisy_delta_w_svd,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA-SB Ablation Experiments")
    parser.add_argument("--config", type=str, default="config/math_reasoning.yaml")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["init_strategy", "n_samples", "grad_approx"])
    parser.add_argument("--output_dir", type=str, default="outputs/ablations")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_lora_sb_from_init(
    model: torch.nn.Module,
    init_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    target_modules: List[str],
    dropout: float = 0.0,
) -> torch.nn.Module:
    """Apply LoRA-SB with a custom init_dict (for ablations)."""
    return apply_lora_sb(model, target_modules, init_dict, dropout=dropout)


def run_training_and_eval(
    model: torch.nn.Module,
    train_dataset,
    tokenizer,
    cfg: Dict,
    device: torch.device,
    output_dir: str,
) -> Dict:
    """Run training and evaluation, return metrics."""
    train_cfg = TrainingConfig(
        learning_rate=cfg["learning_rate"],
        lr_scheduler=cfg["lr_scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        num_epochs=cfg["num_epochs"],
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        dropout=cfg.get("dropout", 0.0),
        output_dir=output_dir,
        seed=cfg["seed"],
    )

    train_loader = get_causal_lm_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        seed=cfg["seed"],
    )

    train(model, train_loader, train_cfg, device)

    gsm8k_data = load_gsm8k(split="test")
    gsm8k_metrics = evaluate_math_generation(model, tokenizer, gsm8k_data, device, task="gsm8k")

    try:
        math_data = load_math_dataset(split="test")
        math_metrics = evaluate_math_generation(model, tokenizer, math_data, device, task="math")
    except Exception:
        math_metrics = {"accuracy": 0.0}

    return {
        "gsm8k": gsm8k_metrics["accuracy"],
        "math": math_metrics["accuracy"],
    }


def ablation_init_strategy(cfg: Dict, device: torch.device, output_dir: str) -> None:
    """Table 4: Compare initialization strategies."""
    set_seed(cfg["seed"])
    dtype = torch.bfloat16

    model_name = cfg["model_name"]
    rank = cfg["rank"]
    target_modules = cfg.get("target_modules", CAUSAL_LM_TARGET_MODULES)

    # Load training data once
    model_base, tokenizer = load_causal_lm(model_name, torch_dtype=dtype)
    train_dataset = load_metamath(tokenizer, max_seq_len=cfg["max_seq_len"],
                                  n_train=cfg["n_train_samples"], seed=cfg["seed"])

    # Compute ΔW_avg once
    init_loader = get_init_dataloader(train_dataset, tokenizer,
                                      n_samples=cfg["init_n_samples"],
                                      batch_size=1, seed=cfg["seed"])
    delta_w_dict = compute_gradient_approximation(
        model=model_base,
        dataloader=init_loader,
        target_module_names=target_modules,
        n_samples=cfg["init_n_samples"],
        device=device,
    )

    strategies = [
        ("lora_sb", None),           # LoRA-SB: trunc_SVD(ΔW_avg)
        ("kaiming", None),            # trunc_SVD(Kaiming)
        ("noisy_1e-3", 1e-3),        # trunc_SVD(ΔW_avg + N_{1e-3})
        ("noisy_1e-4", 1e-4),        # trunc_SVD(ΔW_avg + N_{1e-4})
        ("noisy_1e-5", 1e-5),        # trunc_SVD(ΔW_avg + N_{1e-5})
    ]

    all_results = {}

    for strategy_name, noise_std in strategies:
        print(f"\nRunning ablation: {strategy_name}")
        model, tokenizer = load_causal_lm(model_name, torch_dtype=dtype)

        init_dict = {}
        for name, delta_w in delta_w_dict.items():
            module = dict(model.named_modules())[name]
            w_dtype = module.weight.dtype

            if strategy_name == "lora_sb":
                B, R, A = truncated_svd_init(delta_w, rank=rank, dtype=w_dtype)
            elif strategy_name == "kaiming":
                B, R, A = init_kaiming_svd(module.weight.data, rank=rank)
            else:
                B, R, A = init_noisy_delta_w_svd(delta_w, rank=rank, noise_std=noise_std)

            init_dict[name] = (B, R, A)

        model = build_lora_sb_from_init(model, init_dict, target_modules)
        metrics = run_training_and_eval(
            model, train_dataset, tokenizer, cfg, device,
            os.path.join(output_dir, f"init_{strategy_name}"),
        )
        all_results[strategy_name] = metrics
        print(f"  GSM8K: {metrics['gsm8k']:.2f}%, MATH: {metrics['math']:.2f}%")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "init_strategy_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nInit strategy ablation results:", all_results)


def ablation_n_samples(cfg: Dict, device: torch.device, output_dir: str) -> None:
    """Table 5: Effect of number of initialization samples."""
    set_seed(cfg["seed"])
    dtype = torch.bfloat16

    model_name = cfg["model_name"]
    rank = cfg["rank"]
    target_modules = cfg.get("target_modules", CAUSAL_LM_TARGET_MODULES)

    model_base, tokenizer = load_causal_lm(model_name, torch_dtype=dtype)
    train_dataset = load_metamath(tokenizer, max_seq_len=cfg["max_seq_len"],
                                  n_train=cfg["n_train_samples"], seed=cfg["seed"])

    sample_counts = [1, 5, 25, 50, 100, 200, 500]
    all_results = {}

    for n_samples in sample_counts:
        print(f"\nRunning ablation: n_samples={n_samples}")
        model, tokenizer = load_causal_lm(model_name, torch_dtype=dtype)

        init_loader = get_init_dataloader(train_dataset, tokenizer,
                                          n_samples=n_samples, batch_size=1, seed=cfg["seed"])
        delta_w_dict = compute_gradient_approximation(
            model=model,
            dataloader=init_loader,
            target_module_names=target_modules,
            n_samples=n_samples,
            device=device,
        )

        init_dict = {}
        for name, delta_w in delta_w_dict.items():
            module = dict(model.named_modules())[name]
            B, R, A = truncated_svd_init(delta_w, rank=rank, dtype=module.weight.dtype)
            init_dict[name] = (B, R, A)

        model = build_lora_sb_from_init(model, init_dict, target_modules)
        metrics = run_training_and_eval(
            model, train_dataset, tokenizer, cfg, device,
            os.path.join(output_dir, f"n_samples_{n_samples}"),
        )
        all_results[n_samples] = metrics
        print(f"  GSM8K: {metrics['gsm8k']:.2f}%, MATH: {metrics['math']:.2f}%")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "n_samples_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nN-samples ablation results:", all_results)


def ablation_grad_approx(cfg: Dict, device: torch.device, output_dir: str) -> None:
    """Figure 3: Optimal gradient approximation vs non-orthogonal variant."""
    set_seed(cfg["seed"])
    dtype = torch.bfloat16

    model_name = cfg["model_name"]
    rank = cfg["rank"]
    target_modules = cfg.get("target_modules", CAUSAL_LM_TARGET_MODULES)

    model_base, tokenizer = load_causal_lm(model_name, torch_dtype=dtype)
    train_dataset = load_metamath(tokenizer, max_seq_len=cfg["max_seq_len"],
                                  n_train=cfg["n_train_samples"], seed=cfg["seed"])

    init_loader = get_init_dataloader(train_dataset, tokenizer,
                                      n_samples=cfg["init_n_samples"],
                                      batch_size=1, seed=cfg["seed"])
    delta_w_dict = compute_gradient_approximation(
        model=model_base,
        dataloader=init_loader,
        target_module_names=target_modules,
        n_samples=cfg["init_n_samples"],
        device=device,
    )

    variants = [
        ("lora_sb_orthonormal", False),   # LoRA-SB: B^T B = I, A A^T = I
        ("non_orthogonal", True),          # B = U*S, A = Vh, R = I
    ]

    all_results = {}

    for variant_name, use_non_ortho in variants:
        print(f"\nRunning ablation: {variant_name}")
        model, tokenizer = load_causal_lm(model_name, torch_dtype=dtype)

        init_dict = {}
        for name, delta_w in delta_w_dict.items():
            module = dict(model.named_modules())[name]
            w_dtype = module.weight.dtype

            if use_non_ortho:
                B, R, A = init_non_orthogonal_svd(delta_w, rank=rank)
                B, R, A = B.to(w_dtype), R.to(w_dtype), A.to(w_dtype)
            else:
                B, R, A = truncated_svd_init(delta_w, rank=rank, dtype=w_dtype)

            init_dict[name] = (B, R, A)

        model = build_lora_sb_from_init(model, init_dict, target_modules)
        metrics = run_training_and_eval(
            model, train_dataset, tokenizer, cfg, device,
            os.path.join(output_dir, variant_name),
        )
        all_results[variant_name] = metrics
        print(f"  GSM8K: {metrics['gsm8k']:.2f}%, MATH: {metrics['math']:.2f}%")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "grad_approx_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nGrad approx ablation results:", all_results)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ablation == "init_strategy":
        ablation_init_strategy(cfg, device, args.output_dir)
    elif args.ablation == "n_samples":
        ablation_n_samples(cfg, device, args.output_dir)
    elif args.ablation == "grad_approx":
        ablation_grad_approx(cfg, device, args.output_dir)


if __name__ == "__main__":
    main()
