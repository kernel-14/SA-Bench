#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py – LoRA‑SB reproduction entry point.

This script orchestrates the full experimental pipeline:
  1. Parse command‑line arguments and configuration.
  2. Load and preprocess datasets.
  3. Load the pre‑trained model and (if not a baseline) initialise LoRA‑SB matrices.
  4. Train the model (only the R matrices are updated for LoRA‑SB).
  5. Evaluate on the appropriate benchmarks (arithmetic, commonsense, GLUE).

Baselines (LoRA, LoRA‑XS) can be selected via the --baseline flag.

Usage example:
    python main.py --task arithmetic --rank 64 --model_name_or_path mistralai/Mistral-7B-v0.1 --output_dir ./checkpoints
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

import torch
import wandb
import yaml

# Local imports – all modules are assumed to be in the same package
from config import ExperimentConfig
from dataset import DatasetLoader
from modeling import ModelWrapper, LoraSBLayer
from initializer import Initializer
from trainer import Trainer
from evaluate import Evaluator

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Additional deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_str: str) -> torch.device:
    """Return torch device from string."""
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    return torch.device(device_str)

def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Use float32, float16, bfloat16.")

# ---------------------------------------------------------------------------
# Baseline implementations
# ---------------------------------------------------------------------------

def apply_baseline_lora_xs(model_wrapper: ModelWrapper, config: ExperimentConfig) -> None:
    """
    Replace target linear layers with LoRA‑XS style adaptation.
    LoRA‑XS uses SVD of the original weight W0 to initialise fixed B and A,
    and a trainable R = I.
    """
    model = model_wrapper.model
    device = next(model.parameters()).device
    dtype = model_wrapper.dtype
    r = config.r

    B_dict: Dict[str, torch.Tensor] = {}
    A_dict: Dict[str, torch.Tensor] = {}
    R_dict: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if any(name.endswith("." + tgt) for tgt in config.target_modules):
            if not isinstance(module, torch.nn.Linear):
                continue
            # Get original weight (detached, cloned)
            W0 = module.weight.data.clone().to(torch.float32)
            U, S, Vh = torch.linalg.svd(W0, full_matrices=False)

            # Truncate to rank r
            r_eff = min(r, len(S))
            U_r = U[:, :r_eff]               # (m, r)
            S_r = S[:r_eff]
            Vh_r = Vh[:r_eff, :]             # (r, n)

            # LoRA‑XS initialization: B = U * S, A = Vh (row space), R = I
            # Equivalent to B * R * A = (U * S) * I * Vh = W0 (low‑rank)
            B_init = (U_r * S_r[None, :]).to(dtype)
            A_init = Vh_r.to(dtype)
            R_init = torch.eye(r_eff, device=device, dtype=dtype)

            B_dict[name] = B_init
            A_dict[name] = A_init
            R_dict[name] = R_init

    model_wrapper.apply_lora_sb(B_dict, A_dict, R_dict)


def apply_baseline_lora(model_wrapper: ModelWrapper, config: ExperimentConfig) -> None:
    """
    Apply standard LoRA using the HuggingFace PEFT library.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("peft library is required for LoRA baseline. Install with `pip install peft`")

    # Determine task type for PEFT
    if config.task.startswith("arithmetic") or config.task.startswith("commonsense"):
        task_type = TaskType.CAUSAL_LM
    else:  # GLUE
        task_type = TaskType.SEQ_CLS

    peft_config = LoraConfig(
        task_type=task_type,
        r=config.r,
        lora_alpha=config.r,      # typical alpha = r
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        bias="none",
    )

    model_wrapper.model = get_peft_model(model_wrapper.model, peft_config)
    # PEFT already freezes non‑adapter weights; ensure all trainable parameters are exposed
    # (ModelWrapper.get_trainable_parameters will return only adapted parameters)
    logger.info(f"Applied standard LoRA with rank={config.r} and alpha={config.r}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # ------ Argument parsing ------
    parser = argparse.ArgumentParser(description="LoRA‑SB reproduction")
    parser.add_argument("--task", type=str, required=True,
                        choices=["arithmetic", "commonsense", "glue"],
                        help="Benchmark family (arithmetic, commonsense, glue)")
    parser.add_argument("--rank", type=int, required=True,
                        help="Low‑rank dimension (e.g., 8, 16, 32, 64, 96)")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Override model path from config.yaml if provided")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["lora", "lora_xs"],
                        help="Run a baseline instead of LoRA‑SB")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed from config.yaml")
    parser.add_argument("--disable_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    # GLUE subtask flag
    parser.add_argument("--glue_task", type=str, default=None,
                        help="For task=glue, specify the subtask (e.g., cola, mrpc). Required for GLUE.")
    args = parser.parse_args()

    # ------ Configuration loading ------
    # Build task string: for glue, we need 'glue_cola' etc.
    if args.task == "glue":
        if args.glue_task is None:
            parser.error("--glue_task is required when task=glue")
        task_str = f"glue_{args.glue_task}"
    else:
        task_str = args.task

    # Create ExperimentConfig – YAML defaults are loaded inside __post_init__
    config_kwargs: Dict[str, Any] = {"task": task_str, "r": args.rank}
    if args.model_name_or_path:
        config_kwargs["model_name_or_path"] = args.model_name_or_path
    config = ExperimentConfig(**config_kwargs)

    # Override seed if provided
    if args.seed is not None:
        config.seed = args.seed

    # Set global reproducibility
    set_seed(config.seed)

    # Set torch default dtype if specified
    if config.dtype:
        torch.set_default_dtype(get_dtype(config.dtype))

    # Device selection
    device = get_device(config.device)
    config.device = str(device)  # update to validated device

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ------ Logging setup ------
    # Initialise wandb (unless disabled)
    if not args.disable_wandb:
        wandb_config = config.to_dict()
        run_name = f"LoRA-SB-{task_str}-r{args.rank}"
        if args.baseline:
            run_name = f"{args.baseline}-{task_str}-r{args.rank}"
        wandb.init(
            project="lora-sb",
            name=run_name,
            config=wandb_config,
            settings=wandb.Settings(start_method="fork"),
        )
    else:
        wandb.init(mode="disabled")

    # ------ Dataset ------
    logger.info(f"Loading dataset for task: {task_str}")
    dataset_loader = DatasetLoader(config)
    train_size = dataset_loader.train_dataset_size
    config.set_init_samples(train_size)  # compute number of init samples
    logger.info(f"Training set size: {train_size}, init samples: {config.num_init_samples}")

    # ------ Model ------
    logger.info(f"Loading model: {config.model_name_or_path}")
    model_wrapper = ModelWrapper(config.model_name_or_path, config)

    # ------ Initialisation (LoRA‑SB) or Baseline ------
    if args.baseline == "lora":
        logger.info("Applying standard LoRA baseline")
        apply_baseline_lora(model_wrapper, config)
    elif args.baseline == "lora_xs":
        logger.info("Applying LoRA‑XS baseline")
        apply_baseline_lora_xs(model_wrapper, config)
    else:
        # LoRA‑SB
        logger.info("Computing LoRA‑SB initialization matrices")
        init_loader = dataset_loader.get_init_dataloader(config.num_init_samples)
        initializer = Initializer(config)
        B_dict, A_dict, R_dict = initializer.compute_init(
            model_wrapper.model, init_loader, config.lr
        )
        model_wrapper.apply_lora_sb(B_dict, A_dict, R_dict)
        logger.info("LoRA‑SB layers injected successfully")

        # Free memory (gradients no longer needed)
        torch.cuda.empty_cache()

    # ------ Training ------
    logger.info("Starting training")
    trainer = Trainer(model_wrapper, dataset_loader, config)
    trainer.train()

    # ------ Evaluation ------
    logger.info("Evaluating trained model")
    evaluator = Evaluator(model_wrapper, config)
    eval_loaders = dataset_loader.get_eval_dataloader()

    all_metrics: Dict[str, Dict[str, float]] = {}
    for eval_name, loader in eval_loaders.items():
        metrics = evaluator.evaluate(loader)
        all_metrics[eval_name] = metrics
        logger.info(f"Evaluation on {eval_name}: {metrics}")
        # Log to wandb
        for k, v in metrics.items():
            wandb.log({f"{eval_name}/{k}": v})

    # ------ Save final checkpoint ------
    # For LoRA‑SB we only save the trainable R matrices and configuration
    if not args.baseline:
        checkpoint = {
            "config": config.to_dict(),
            "R_params": {name: param.data.clone() for name, param in model_wrapper.get_trainable_parameters()},
        }
        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{task_str}-r{args.rank}.pt")
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")
    else:
        # For PEFT models we can save the adapter
        model_wrapper.model.save_pretrained(args.output_dir)

    wandb.finish()
    logger.info("Experiment completed successfully.")


if __name__ == "__main__":
    main()
